import pandas as pd
from typing import Literal, Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client
from src.smart_campus_assistant.clients.astral_client import astral_client

Rooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

Timeframes = Literal[
    'now', '2h', '24h', '7d', '30d', '90d'
]

logger = logging.getLogger(__name__)

# ==========================================
# DYNAMIC WEATHER STATION DISCOVERY
# ==========================================
def find_weather_station_id() -> Optional[str]:
    """Sweeps the DeviceRegistry to find the ThingsBoard UUID for the Weather Station."""
    for room in registry.get_available_rooms():
        devices = registry.get_all_devices_in_room(room)
        for name, d_val in devices.items():
            if "WEATHER" in name.upper():
                return d_val.get("id") if isinstance(d_val, dict) else d_val
    logger.error("Could not find a Weather Station in the campus_topology.json!")
    return None

WEATHER_STATION_ID = find_weather_station_id()

# Config mapping for API calls and pandas resampling
TIMEFRAME_CONFIG = {
    "2h":  {"method": "get_2h", "bin_size": "10min"},
    "24h": {"method": "get_24h", "bin_size": "2h"}, 
    "7d":  {"method": "get_7d", "bin_size": "2h"},    
    "30d": {"method": "get_30d", "bin_size": "2h"},
    "90d": {"method": "get_90d", "bin_size": "2h"}    
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

def get_group_outliers(series: pd.Series, baseline_counts: pd.Series, solar_series: pd.Series = None, solar_baseline: float = None) -> List[str]:
    """Helper to find anomalous days within a specific time mask group, including solar spikes/drops."""
    outliers = []
    if series.empty or baseline_counts.empty: return outliers
    
    dominant_level = baseline_counts.idxmax()
    baseline_pct = baseline_counts.max()
    
    daily_groups = series.groupby(pd.Grouper(freq='D'))
    for day, day_data in daily_groups:
        if day_data.empty: continue
        
        day_counts = day_data.apply(lambda x: int(max(0, min(5, round(x))))).value_counts(normalize=True)
        day_pct = day_counts.get(dominant_level, 0)
        
        is_light_outlier = False
        if abs(baseline_pct - day_pct) > 0.25:
            is_light_outlier = True
            
        solar_msg = ""
        is_solar_outlier = False
        if solar_series is not None and not solar_series.empty and solar_baseline is not None and pd.notna(solar_baseline):
            day_solar = solar_series[solar_series.index.normalize() == day.normalize()]
            if not day_solar.empty:
                day_solar_mean = day_solar.mean()
                diff = day_solar_mean - solar_baseline
                # Changed to absolute check to catch highly cloudy/stormy days (negative diff) as well
                if abs(diff) >= 200.0:
                    is_solar_outlier = True
                    sign = "+" if diff > 0 else ""
                    solar_msg = f" | Solar_Anomaly: {day_solar_mean:.1f} W/m² ({sign}{diff:.1f})"
                    
        if is_light_outlier or is_solar_outlier:
            day_str = day.strftime('%Y-%m-%d (%A)')
            outlier_str = format_distribution(day_data)
            outliers.append(f"        - '{day_str}': Light_Dist: {outlier_str}{solar_msg}")
            
    return outliers

class LightsInput(BaseModel):
    room: Rooms = Field(
        ..., 
        description="The specific room to check for illumination levels. MUST be one of the exact allowed room names."
    )
    timeframe: Timeframes = Field(
        ..., 
        description="The time window for the data request. 'now' provides a real-time snapshot. '2h', '24h', '7d' provides data for that timeframe in smaller buckets. '30d' and '90d' provide long-term statistics."
    )

@tool("get_ambient_lights", args_schema=LightsInput)
def get_ambient_lights(room: Rooms, timeframe: Timeframes) -> str:
    """
    Tracks indoor illumination using a discrete 0-5 scale.
    Uses state-transition logic to prevent mathematical hallucinations and maps integers to semantic labels.
    """
    # 1. Resolve All Devices in Room
    all_iaq_devices = registry.get_devices_by_room_and_type(room, "IAQ")
    
    if not all_iaq_devices:
        return f"Query_Context:\n  Room: {room}\nError: No IAQ (Light) sensors found in this room."

    # 2. Check Active Status via Server Attributes
    active_iaq_devices = {}
    offline_sensors = []
    
    for device_name, device_data in all_iaq_devices.items():
        device_id = device_data.get("id")
        if not device_id: 
            offline_sensors.append(device_name)
            continue
            
        try:
            attrs = tb_client.get_server_attributes(device_id, ["active"])
            
            is_active = False
            for attr in attrs:
                if attr.get("key") == "active" and str(attr.get("value")).lower() == "true":
                    is_active = True
                    break
                    
            if is_active:
                active_iaq_devices[device_name] = device_data
            else:
                offline_sensors.append(device_name)
                
        except Exception as e:
            logger.warning(f"Could not fetch active status for {device_name}: {e}")
            offline_sensors.append(device_name)

    if not active_iaq_devices:
        return f"Query_Context:\n  Room: {room}\nError: Found {len(all_iaq_devices)} IAQ sensors, but all are currently offline."

    # 3. Build the Active_Sensors reporting lines
    total_count = len(all_iaq_devices)
    active_count = len(active_iaq_devices)
    
    active_sensors_lines = [f"  Active_Sensors: {active_count}/{total_count} Online"]
    for device_name, device_data in active_iaq_devices.items():
        z = device_data.get("zone", "Unspecified")
        t = device_data.get("tag", "Unspecified")
        active_sensors_lines.append(f"    - {device_name}: Place: {z}, Room: {t}")
        
    if offline_sensors:
        active_sensors_lines.append(f"  Offline_Sensors: {', '.join(offline_sensors)}")

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        solar = astral_client.get_current_solar_context()
        solar_rad = "N/A"
        
        if WEATHER_STATION_ID:
            raw_w = tb_client.get_now(WEATHER_STATION_ID, ["solar_radiation"])
            if "solar_radiation" in raw_w and raw_w["solar_radiation"]:
                solar_rad = f"{float(raw_w['solar_radiation'][0]['value']):.1f} W/m²"

        output = [
            "Query_Context:",
            "  Domain: Ambient Light Intensity (0-5 Scale)",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)"
        ]
        output.extend(active_sensors_lines)
        output.extend([
            "  Solar_Context:",
            f"    - Average_Daylight_Window: {solar['sunrise']} to {solar['sunset']}",
            f"    - Current_Sun_Azimuth: Sun is currently facing {solar['horizontal'].split()[-1]}",
            f"    - Vertical_Angle: {solar['vertical']}",
            "",
            "Current_State:"
        ])
        if solar_rad != "N/A":
            output.append(f"  Solar_Radiation: {solar_rad}")
        
        # Only process active devices
        for device_name, device_data in active_iaq_devices.items():
            device_id = device_data.get("id")
            
            raw_data = tb_client.get_now(device_id, ["light_level"])
            if "light_level" in raw_data and raw_data["light_level"]:
                val = float(raw_data["light_level"][0]["value"])
                output.append(f"  {device_name}: {get_semantic_label(val)}")
            else:
                output.append(f"  {device_name}: No Data (Despite being marked Online)")
                
        return "\n".join(output)

    # ==========================================
    # BRANCH B: HISTORICAL DATA FETCH
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method_name = config["method"]
    fetch_method = getattr(tb_client, fetch_method_name)

    # 1. Fetch Historical Weather (Solar Radiation Series)
    solar_rad_series = pd.Series(dtype=float)
    if WEATHER_STATION_ID:
        try:
            w_raw = fetch_method(WEATHER_STATION_ID, ["solar_radiation"])
            if "solar_radiation" in w_raw and w_raw["solar_radiation"]:
                w_df = pd.DataFrame(w_raw["solar_radiation"])
                w_df['value'] = pd.to_numeric(w_df['value'])
                w_df['datetime'] = pd.to_datetime(w_df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(settings.TIMEZONE)
                w_df.set_index('datetime', inplace=True)
                solar_rad_series = w_df['value']
        except Exception as e:
            logger.warning(f"Failed to fetch historical solar radiation: {e}")

    def get_contextual_solar(is_wkday_req: bool, is_work_req: bool) -> str:
        """Calculates mean solar radiation specifically matching the schedule matrix."""
        if solar_rad_series.empty: return "N/A"
        idx = solar_rad_series.index
        is_wk = idx.dayofweek < 5
        is_work = (idx.hour >= 8) & (idx.hour < 22)
        
        mask = pd.Series(True, index=idx)
        mask = mask & is_wk if is_wkday_req else mask & ~is_wk
        mask = mask & is_work if is_work_req else mask & ~is_work
        
        val = solar_rad_series[mask].mean()
        return f"{val:.1f} W/m²" if pd.notna(val) else "N/A"

    # 2. Fetch Historical Room Data
    all_dataframes = []
    for device_name, device_data in active_iaq_devices.items():
        device_id = device_data.get("id")
        
        try:
            raw_data = fetch_method(device_id, ["light_level"])
            if "light_level" in raw_data and raw_data["light_level"]:
                df = pd.DataFrame(raw_data["light_level"])
                df['value'] = pd.to_numeric(df['value'])
                # Localize exactly to the specified timezone
                df['datetime'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(settings.TIMEZONE)
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

    # Map timeframe strings to integers for astral calculation
    days_map = {"2h": 1, "24h": 1, "7d": 7, "30d": 30, "90d": 90}
    days_back = days_map.get(timeframe, 1)
    solar_hist = astral_client.get_historical_solar_context(days_back)

    # ==========================================
    # BRANCH C: 30-DAY & 90-DAY STATISTICAL PROFILE
    # ==========================================
    if timeframe in ["30d", "90d"]:
        is_weekday = raw_series.index.dayofweek < 5
        is_weekend = raw_series.index.dayofweek >= 5
        is_working_hours = (raw_series.index.hour >= 8) & (raw_series.index.hour < 22)
        is_non_working = (raw_series.index.hour < 8) | (raw_series.index.hour >= 22)

        output = [
            "Query_Context:",
            "  Domain: Ambient Light Intensity (0-5 Scale)",
            f"  Room: {room}",
            f"  Timeframe: {timeframe} (Long-Term Statistical Profile)",
        ]
        output.extend(active_sensors_lines)
        output.extend([
            "  Solar_Context:",
            f"    - Average_Daylight_Window: {solar_hist['avg_sunrise']} to {solar_hist['avg_sunset']}",
            f"    - Daily_Sun_Trajectory: {solar_hist['trajectory']}",
            "",
            "Total_Monthly_Average:" if timeframe == "30d" else "Total_Quarterly_Average:",
            f"  {format_distribution(raw_series)}",
            "",
            "Schedule_Profiling_Matrix:"
        ])
        
        def process_matrix_cell(cell_name, mask, is_wkday_req, is_work_req):
            cell_series = raw_series[mask]
            
            ctx_solar_val = None
            ctx_solar_str = "N/A"
            s_mask = None
            if not solar_rad_series.empty:
                idx = solar_rad_series.index
                is_wk = idx.dayofweek < 5
                is_work = (idx.hour >= 8) & (idx.hour < 22)
                s_mask = pd.Series(True, index=idx)
                s_mask = s_mask & is_wk if is_wkday_req else s_mask & ~is_wk
                s_mask = s_mask & is_work if is_work_req else s_mask & ~is_work
                ctx_solar_val = solar_rad_series[s_mask].mean()
                if pd.notna(ctx_solar_val):
                    ctx_solar_str = f"{ctx_solar_val:.1f} W/m²"
            
            if cell_series.empty:
                return [
                    f"    {cell_name}:", 
                    "      Baseline: No data",
                    f"      Average_Solar_Radiation: {ctx_solar_str}",
                    "      Outliers: None"
                ]
            
            baseline_counts = cell_series.apply(lambda x: int(max(0, min(5, round(x))))).value_counts(normalize=True)
            dist_str = format_distribution(cell_series)
            
            outliers = get_group_outliers(
                cell_series, 
                baseline_counts,
                solar_series=solar_rad_series[s_mask] if s_mask is not None and not solar_rad_series.empty else None,
                solar_baseline=ctx_solar_val
            )
            
            lines = [f"    {cell_name}:"]
            lines.append(f"      Baseline: {dist_str}")
            lines.append(f"      Average_Solar_Radiation: {ctx_solar_str}")
            
            if outliers:
                lines.append("      Outliers:")
                lines.extend(outliers)
            else:
                lines.append("      Outliers: None detected.")
            return lines

        output.append("  Weekdays (Mon-Fri):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekday & is_working_hours, True, True))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working, True, False))
        
        output.append("  Weekends (Sat-Sun):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekend & is_working_hours, False, True))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working, False, False))

        return "\n".join(output)

    # ==========================================
    # BRANCH D: 2h, 24h, 7d (PER-DAY TIMELINE LOGIC)
    # ==========================================
    is_weekday = raw_series.index.dayofweek < 5
    is_weekend = raw_series.index.dayofweek >= 5
    is_working_hours = (raw_series.index.hour >= 8) & (raw_series.index.hour < 22)
    is_non_working = (raw_series.index.hour < 8) | (raw_series.index.hour >= 22)

    solar_context_lines = [
        "  Solar_Context:",
        f"    - Average_Daylight_Window: {solar_hist['avg_sunrise']} to {solar_hist['avg_sunset']}",
        f"    - Daily_Sun_Trajectory: {solar_hist['trajectory']}"
    ]
    if timeframe == "2h":
        el_label, el_desc = astral_client.get_average_elevation_info(2)
        solar_context_lines.append(f"    - Vertical_Angle: {el_label} ({el_desc})")

    output = [
        "Query_Context:",
        "  Domain: Ambient Light Intensity (0-5 Scale)",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
    ]
    output.extend(active_sensors_lines)
    output.extend(solar_context_lines)
    output.extend([
        "",
        "Statistical_Baseline (Present Contexts):"
    ])
    
    def add_context_summary(name, mask, is_wkday_req, is_work_req):
        cell_series = raw_series[mask]
        if not cell_series.empty:
            ctx_solar_str = get_contextual_solar(is_wkday_req, is_work_req)
            output.append(f"  {name}:")
            output.append(f"    Baseline: {format_distribution(cell_series)}")
            output.append(f"    Average_Solar_Radiation: {ctx_solar_str}")

    add_context_summary("Weekdays (Mon-Fri) Working_Hours (08:00-22:00)", is_weekday & is_working_hours, True, True)
    add_context_summary("Weekdays (Mon-Fri) Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working, True, False)
    add_context_summary("Weekends (Sat-Sun) Working_Hours (08:00-22:00)", is_weekend & is_working_hours, False, True)
    add_context_summary("Weekends (Sat-Sun) Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working, False, False)
    
    output.append("")
    output.append("Timeline_Activity:")

    daily_groups = raw_series.groupby(pd.Grouper(freq='D'))
    
    for day_start, day_series in daily_groups:
        if day_series.empty: continue
        
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        day_transitions = []
        day_stable_periods = []
        day_outliers = []
        
        # Check for solar spike/drop outlier for this day
        if not solar_rad_series.empty:
            day_solar = solar_rad_series[solar_rad_series.index.normalize() == day_start.normalize()]
            if not day_solar.empty:
                day_solar_mean = day_solar.mean()
                overall_solar_mean = solar_rad_series.mean()
                if pd.notna(day_solar_mean) and pd.notna(overall_solar_mean):
                    diff = day_solar_mean - overall_solar_mean
                    if abs(diff) >= 200.0:
                        sign = "+" if diff > 0 else ""
                        day_outliers.append(f"Solar_Anomaly: {day_solar_mean:.1f} W/m² ({sign}{diff:.1f} vs avg)")

        previous_global_state = None
        current_stable_start = None
        current_stable_state = None
        stable_bin_count = 0
        last_bucket_end = None
        
        bucket_groups = day_series.groupby(pd.Grouper(freq=bin_size))
        
        for bucket_start, group in bucket_groups:
            if group.empty: continue
            
            bucket_end = bucket_start + pd.to_timedelta(bin_size)
            bucket_time_label = f"{bucket_start.strftime('%H:%M')} - {bucket_end.strftime('%H:%M')}"
            last_bucket_end = bucket_end.strftime('%H:%M')
            
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

        # Close out any open stable periods using the actual last processed time
        if stable_bin_count > 0:
            end_str = last_bucket_end or "24:00"
            if end_str == "00:00": end_str = "24:00"
            day_stable_periods.append({
                "start": current_stable_start,
                "end": end_str,
                "intervals": stable_bin_count,
                "state": current_stable_state
            })

        # Append this day's activity to the global output
        output.append(f"  '{day_key}':")
        
        if day_outliers:
            output.append("    Outliers:")
            for o in day_outliers:
                output.append(f"      - {o}")
                
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
        print("\n[Testing Historical (now)]")
        print(get_ambient_lights.invoke({"room": "restaurant", "timeframe": "now"}))
        print("\n[Testing Historical (2h)]")
        print(get_ambient_lights.invoke({"room": "restaurant", "timeframe": "2h"}))
        print("\n[Testing Historical (24h)]")
        print(get_ambient_lights.invoke({"room": "restaurant", "timeframe": "24h"}))
        print("\n[Testing Historical (7d)]")
        print(get_ambient_lights.invoke({"room": "restaurant", "timeframe": "7d"}))
        print("\n[Testing Historical (30d)]")
        print(get_ambient_lights.invoke({"room": "restaurant", "timeframe": "30d"}))
        print("\n[Testing Historical (90d)]")
        print(get_ambient_lights.invoke({"room": "restaurant", "timeframe": "90d"}))
    except Exception as e:
        print(f"\nError during execution: {e}")