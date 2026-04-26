import pandas as pd
from typing import Literal, Dict, Any, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from utils.device_registry import registry
from clients.thingsboard_client import tb_client

logger = logging.getLogger(__name__)

# Config mapping for API calls and pandas resampling
TIMEFRAME_CONFIG = {
    "2h":  {"method": "get_2h", "bin_size": "10min"},
    "24h": {"method": "get_24h", "bin_size": "2h"}, 
    "7d":  {"method": "get_7d", "bin_size": "2h"},    
    "30d": {"method": "get_30d", "bin_size": "2h"}    
}

# Semantic mapping for 0-5 scale
LIGHT_LABELS = {
    0: "Level 0 (Dark)",
    1: "Level 1 (Dim)",
    2: "Level 2 (Normal)",
    3: "Level 3 (Bright)",
    4: "Level 4 (Very Bright)",
    5: "Level 5 (Very Sunny)"
}

def get_semantic_label(val: float) -> str:
    """Safely rounds continuous data to the discrete 0-5 scale and returns its semantic label."""
    if pd.isna(val):
        return "Unknown"
    clamped_val = int(max(0, min(5, round(val))))
    return LIGHT_LABELS.get(clamped_val, f"Level {clamped_val}")

def format_distribution(series: pd.Series) -> str:
    """Helper to format a pandas series of raw ticks into a clean percentage string."""
    if series.empty:
        return "No data"
    counts = series.apply(lambda x: int(max(0, min(5, round(x))))).value_counts(normalize=True)
    dist = []
    for level_int in range(6):
        if level_int in counts:
            pct = counts[level_int] * 100
            if pct >= 1: # Ignore micro-fluctuations under 1%
                dist.append(f"{LIGHT_LABELS[level_int]}: {pct:.0f}%")
    return ", ".join(dist) if dist else "No dominant state"

def get_group_outliers(series: pd.Series, baseline_counts: pd.Series) -> List[str]:
    """Helper to find anomalous days within a specific time mask group."""
    outliers = []
    if series.empty or baseline_counts.empty: return outliers
    
    dominant_level = baseline_counts.idxmax()
    baseline_pct = baseline_counts.max()
    
    daily_groups = series.groupby(pd.Grouper(freq='D'))
    for day, day_data in daily_groups:
        if day_data.empty: continue
        
        day_counts = day_data.apply(lambda x: int(max(0, min(5, round(x))))).value_counts(normalize=True)
        day_pct = day_counts.get(dominant_level, 0)
        
        # Outlier condition: If the dominant state for this day deviates > 25% from the expected baseline
        if abs(baseline_pct - day_pct) > 0.25:
            day_str = day.strftime('%Y-%m-%d (%A)')
            outliers.append(f"        - '{day_str}': {format_distribution(day_data)}")
    
    return outliers

# Allowed rooms derived from user requirements
ROOMS = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

class LightsInput(BaseModel):
    room: ROOMS = Field(
        ..., 
        description="The specific room to check for illumination levels."
    )
    timeframe: Literal["now", "2h", "24h", "7d", "30d"] = Field(
        default="now", 
        description="The time window for the data request. 'now' provides a real-time snapshot."
    )

@tool("get_ambient_lights", args_schema=LightsInput)
def get_ambient_lights(room: str, timeframe: Literal["now", "2h", "24h", "7d", "30d"]) -> str:
    """
    Tracks indoor illumination using a discrete 0-5 scale.
    Uses state-transition logic to prevent mathematical hallucinations and maps integers to semantic labels.
    """
    # 1. Resolve Devices
    iaq_devices = registry.get_devices_by_room_and_type(room, "IAQ")
    
    if not iaq_devices:
        return f"Query_Context:\n  Room: {room}\nError: No IAQ (Light) sensors found in this room."

    sensor_names = list(iaq_devices.keys())
    active_sensors_str = f"{len(sensor_names)} ({', '.join(sensor_names)})"

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        output = [
            "Query_Context:",
            "  Domain: Ambient Light Intensity (0-5 Scale)",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Active_Sensors: {active_sensors_str}",
            "",
            "Current_State:"
        ]
        
        for device_name, device_id in iaq_devices.items():
            raw_data = tb_client.get_now(device_id, ["light_level"])
            if "light_level" in raw_data and raw_data["light_level"]:
                val = float(raw_data["light_level"][0]["value"])
                output.append(f"  {device_name}: {get_semantic_label(val)}")
            else:
                output.append(f"  {device_name}: Offline / No Data")
                
        return "\n".join(output)

    # ==========================================
    # BRANCH B: HISTORICAL DATA FETCH
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method_name = config["method"]
    fetch_method = getattr(tb_client, fetch_method_name)

    all_dataframes = []

    for device_name, device_id in iaq_devices.items():
        try:
            raw_data = fetch_method(device_id, ["light_level"])
            if "light_level" in raw_data and raw_data["light_level"]:
                df = pd.DataFrame(raw_data["light_level"])
                df['value'] = pd.to_numeric(df['value'])
                df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('datetime', inplace=True)
                df.rename(columns={'value': device_name}, inplace=True)
                df.drop(columns=['ts'], inplace=True)
                all_dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch historical light data for {device_name}: {e}")

    if not all_dataframes:
        return f"Query_Context:\n  Room: {room}\nError: No historical light data found for timeframe {timeframe}."

    combined_df = pd.concat(all_dataframes, axis=1, sort=True)
    
    # --- SENSOR SYNCHRONIZATION ---
    aligned_df = combined_df.resample('10min').median()
    aligned_df['Room_Aggregate'] = aligned_df.median(axis=1)
    raw_series = aligned_df['Room_Aggregate'].dropna()

    if raw_series.empty:
        return f"Query_Context:\n  Room: {room}\nError: Historical data was fetched but contained only invalid values."

    # ==========================================
    # BRANCH C: 30-DAY STATISTICAL PROFILE
    # ==========================================
    if timeframe == "30d":
        is_weekday = raw_series.index.dayofweek < 5
        is_weekend = raw_series.index.dayofweek >= 5
        is_working_hours = (raw_series.index.hour >= 8) & (raw_series.index.hour < 22)
        is_non_working = (raw_series.index.hour < 8) | (raw_series.index.hour >= 22)

        output = [
            "Query_Context:",
            "  Domain: Ambient Light Intensity (0-5 Scale)",
            f"  Room: {room}",
            "  Timeframe: 30d (Long-Term Statistical Profile)",
            f"  Active_Sensors: {active_sensors_str}",
            "",
            "Total_Monthly_Average:",
            f"  {format_distribution(raw_series)}",
            "",
            "Schedule_Profiling_Matrix:"
        ]
        
        def process_matrix_cell(cell_name, mask):
            cell_series = raw_series[mask]
            if cell_series.empty:
                return [f"    {cell_name}:", "      Baseline: No data", "      Outliers: None"]
            
            baseline_counts = cell_series.apply(lambda x: int(max(0, min(5, round(x))))).value_counts(normalize=True)
            dist_str = format_distribution(cell_series)
            outliers = get_group_outliers(cell_series, baseline_counts)
            
            lines = [f"    {cell_name}:"]
            lines.append(f"      Baseline: {dist_str}")
            if outliers:
                lines.append("      Outliers:")
                lines.extend(outliers)
            else:
                lines.append("      Outliers: None detected.")
            return lines

        output.append("  Weekdays (Mon-Fri):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekday & is_working_hours))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working))
        
        output.append("  Weekends (Sat-Sun):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekend & is_working_hours))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working))

        return "\n".join(output)

    # ==========================================
    # BRANCH D: 2h, 24h, 7d (PER-DAY TIMELINE LOGIC)
    # ==========================================
    output = [
        "Query_Context:",
        "  Domain: Ambient Light Intensity (0-5 Scale)",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        f"  Active_Sensors: {active_sensors_str}",
        "",
        f"Global_Illumination_Summary (Last {timeframe}):"
    ]
    
    total_raw_dist = format_distribution(raw_series)
    output.extend([f"  {item}" for item in total_raw_dist.split(", ")])
    output.append("\nTimeline_Activity:")

    daily_groups = raw_series.groupby(pd.Grouper(freq='D'))
    
    for day_start, day_series in daily_groups:
        if day_series.empty: continue
        
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        day_transitions = []
        day_stable_periods = []
        
        previous_global_state = None
        current_stable_start = None
        current_stable_state = None
        stable_bin_count = 0
        
        bucket_groups = day_series.groupby(pd.Grouper(freq=bin_size))
        
        for bucket_start, group in bucket_groups:
            if group.empty: continue
            
            bucket_end = bucket_start + pd.to_timedelta(bin_size)
            bucket_time_label = f"{bucket_start.strftime('%H:%M')} - {bucket_end.strftime('%H:%M')}"
            
            bucket_transitions = []
            seen_states = set()
            
            for exact_time, raw_val in group.items():
                current_state = get_semantic_label(raw_val)
                seen_states.add(current_state)
                
                if previous_global_state is None:
                    previous_global_state = current_state
                    current_stable_start = bucket_start.strftime('%H:%M')
                    current_stable_state = current_state
                    stable_bin_count = 0
                
                if current_state != previous_global_state:
                    time_str = exact_time.strftime('%H:%M')
                    bucket_transitions.append(f"Transition: [{previous_global_state} -> {current_state} at {time_str}].")
                    previous_global_state = current_state
            
            # Classify bucket
            if len(bucket_transitions) == 0:
                if current_stable_start is None:
                    current_stable_start = bucket_start.strftime('%H:%M')
                    current_stable_state = previous_global_state
                stable_bin_count += 1
            else:
                if stable_bin_count > 0:
                    stable_end_str = bucket_start.strftime('%H:%M')
                    day_stable_periods.append({
                        "start": current_stable_start,
                        "end": stable_end_str,
                        "intervals": stable_bin_count,
                        "state": current_stable_state
                    })
                
                if len(bucket_transitions) <= 3:
                    activity_str = " ".join(bucket_transitions)
                else:
                    states_str = " and ".join(sorted(list(seen_states)))
                    activity_str = f"Fluctuating heavily between {states_str} (Toggled {len(bucket_transitions)} times)."
                
                day_transitions.append(
                    f"      - bucket: '{bucket_time_label}'\n"
                    f"        activity: '{activity_str}'"
                )
                
                current_stable_start = bucket_end.strftime('%H:%M')
                current_stable_state = previous_global_state
                stable_bin_count = 0

        # End of the day, close out any open stable periods
        if stable_bin_count > 0:
            day_stable_periods.append({
                "start": current_stable_start,
                "end": "24:00",
                "intervals": stable_bin_count,
                "state": current_stable_state
            })

        # Append this day's activity to the global output
        output.append(f"  '{day_key}':")
        
        if not day_transitions:
            output.append("    Timeline_Transitions: None")
        else:
            output.append("    Timeline_Transitions:")
            output.extend(day_transitions)
            
        if not day_stable_periods:
            output.append("    Stable_Periods: None")
        else:
            output.append("    Stable_Periods:")
            for period in day_stable_periods:
                output.append(f"      - '{period['start']} to {period['end']}' ({period['intervals']} intervals): State: {period['state']}")

    return "\n".join(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Lights Tool...")
    print("-" * 50)
    try:
        print("\n[Testing Historical (7d)]")
        print(get_ambient_lights.invoke({"room": "4.9", "timeframe": "7d"}))
        print("\n[Testing Historical (30d)]")
        print(get_ambient_lights.invoke({"room": "4.9", "timeframe": "30d"}))
    except Exception as e:
        print(f"\nError during execution: {e}")