import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

logger = logging.getLogger(__name__)

# ==========================================
# DYNAMIC WEATHER STATION DISCOVERY
# ==========================================
def find_weather_station_id() -> Optional[str]:
    """Sweeps the DeviceRegistry to find the ThingsBoard UUID for the Weather Station."""
    for room in registry.get_available_rooms():
        devices = registry.get_all_devices_in_room(room)
        for name, d_id in devices.items():
            if "WEATHER" in name.upper():
                return d_id
    logger.error("Could not find a Weather Station in the campus_topology.json!")
    return None

WEATHER_STATION_ID = find_weather_station_id()

TIMEFRAME_CONFIG = {
    "now": {"method": "get_now", "bin_size": None, "prev_method": "get_now_prev_30d"},
    "2h":  {"method": "get_2h", "bin_size": "10min", "prev_method": "get_2h_prev_30d"},
    "24h": {"method": "get_24h", "bin_size": "2h", "prev_method": "get_24h_prev_30d"}, 
    "7d":  {"method": "get_7d", "bin_size": "2h", "prev_method": "get_7d_prev_30d"},    
    "30d": {"method": "get_30d", "bin_size": "2h", "prev_method": None},
    "90d": {"method": "get_90d", "bin_size": "2h", "prev_method": None} 
}

# Sensor Key Configurations
IAQ_KEYS = ["temperature", "humidity", "pressure"]
WEATHER_KEYS = [
    "air_temperature", "relative_humidity", "atmospheric_pressure", 
    "wind_speed", "maximum_wind_speed", "wind_direction", 
    "north_wind_speed", "east_wind_speed",
    "precipitation", "solar_radiation", "vapor_pressure",
    "lightning_strike_count", "lightning_average_distance"
]

# Baseline Deviations (Relative Deltas)
THRESHOLDS = {
    "temperature": 1.5,
    "humidity": 5.0,
    "pressure": 5.0,
    "air_temperature": 2.0,
    "relative_humidity": 5.0,
    "precipitation": 1.0,
    "wind_speed": 10.0
}

# Absolute Extreme Limits (Hard Limits)
ABSOLUTE_LIMITS = {
    "temperature": {"min": 17.0, "max": 28.0}, # Standard Indoor Temp
    "humidity": {"min": 30.0, "max": 65.0},    # Standard Indoor Humidity
    "air_temperature": {"min": 0.0, "max": 38.0} # Outdoor Temp
}

UNITS = {
    "temperature": "°C", "humidity": "%", "pressure": "hPa",
    "air_temperature": "°C", "relative_humidity": "%", 
    "solar_radiation": " W/m²", "precipitation": "mm/hr", "wind_speed": "km/h"
}

DISPLAY_NAMES = {
    "temperature": "Temp", "humidity": "Hum", "pressure": "Pres",
    "air_temperature": "Out_Temp", "relative_humidity": "Out_Hum", 
    "solar_radiation": "Average_Solar", "precipitation": "Precip", "wind_speed": "Wind"
}

Rooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

Timeframes = Literal[
    'now', '2h', '24h', '7d', '30d', '90d'
]

class TempHumidityInput(BaseModel):
    room: Rooms = Field(..., description="The specific room to check.")
    timeframe: Timeframes = Field(..., description="The time window. 'now', '2h', '24h', '7d', '30d', '90d'.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_time_context(dt: pd.Timestamp) -> str:
    """Classifies a timestamp into the 4-cell schedule matrix."""
    is_weekend = dt.dayofweek >= 5
    is_work = 8 <= dt.hour < 22
    if not is_weekend and is_work: return "weekday_work"
    if not is_weekend and not is_work: return "weekday_nonwork"
    if is_weekend and is_work: return "weekend_work"
    return "weekend_nonwork"

def get_limit(key: str, room: str) -> Optional[Dict[str, float]]:
    """Fetches absolute limits, with specific room overrides (e.g. Data Center)."""
    if key == "temperature" and room.lower() == "data_center":
        return {"min": 10.0, "max": 28.0} # Data Centers should be allowed to run cold
    return ABSOLUTE_LIMITS.get(key)

def format_val(key: str, val: float, baseline: float = None, room: str = "") -> str:
    unit = UNITS.get(key, "")
    name = DISPLAY_NAMES.get(key, key)
    if pd.isna(val): return f"{name}: N/A"
    
    val_str = f"{val:.1f}" if val % 1 else f"{int(val)}"
    
    limit_tag = ""
    limit = get_limit(key, room)
    if limit:
        if val > limit["max"]: limit_tag = " [MAX_EXCEEDED]"
        elif val < limit["min"]: limit_tag = " [MIN_EXCEEDED]"
            
    if baseline is not None and not pd.isna(baseline):
        diff = val - baseline
        diff_str = f"{diff:+.1f}" if diff % 1 else f"{int(diff):+}"
        return f"{name} {val_str}{unit} ({diff_str}{unit}){limit_tag}"
        
    return f"{name}: {val_str}{unit}{limit_tag}"

def format_baseline_str(data: dict, keys: list) -> str:
    parts = []
    for k in keys:
        if k in data and data[k] is not None:
            parts.append(format_val(k, data[k], room=""))
    return " | ".join(parts) if parts else "No Baseline Data"

def process_telemetry_to_df(raw_data: Dict, keys: List[str], bin_size: str = None) -> pd.DataFrame:
    dfs = []
    for key in keys:
        if key in raw_data and raw_data[key]:
            df = pd.DataFrame(raw_data[key])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.rename(columns={'value': key}, inplace=True)
            df.drop(columns=['ts'], inplace=True)
            dfs.append(df)
            
    if not dfs: return pd.DataFrame()
    combined = pd.concat(dfs, axis=1, sort=True)
    if bin_size:
        combined = combined.resample(bin_size).median()
    return combined

def extract_current_values(raw_data: Dict, keys: List[str]) -> Dict[str, float]:
    result = {}
    for k in keys:
        if k in raw_data and raw_data[k]:
            try:
                result[k] = float(raw_data[k][0]["value"])
            except (ValueError, KeyError, IndexError):
                result[k] = None
    return result

def average_nested_baselines(raw_bases: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    contexts = ['weekday_work', 'weekday_nonwork', 'weekend_work', 'weekend_nonwork']
    result = {k: {c: [] for c in contexts} for k in keys}
    
    for base in raw_bases:
        for k in keys:
            if k in base:
                for c in contexts:
                    if c in base[k] and base[k][c] is not None:
                        result[k][c].append(base[k][c])
                        
    final_result = {}
    for k in keys:
        final_result[k] = {}
        for c in contexts:
            vals = result[k][c]
            final_result[k][c] = np.mean(vals) if vals else None
    return final_result

@tool("get_temp_humidity", args_schema=TempHumidityInput)
def get_temp_humidity(room: Rooms, timeframe: Timeframes) -> str:
    """
    Tracks indoor Temperature, Humidity, and Pressure, correlated with Outdoor Weather.
    Splits baselines via a schedule matrix and strictly enforces absolute safety limits.
    """
    iaq_devices = registry.get_devices_by_room_and_type(room, "IAQ")
    if not iaq_devices:
        return f"Query_Context:\n  Room: {room}\nError: No IAQ sensors found in this room."

    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    
    # 1. Fetch Nested Baselines (For non-30d/90d)
    indoor_baseline = {k: {} for k in IAQ_KEYS}
    weather_baseline = {k: {} for k in WEATHER_KEYS}
    
    if timeframe not in ["30d", "90d"]:
        prev_method = getattr(tb_client, config["prev_method"])
        
        if WEATHER_STATION_ID:
            try:
                weather_baseline = prev_method(WEATHER_STATION_ID, WEATHER_KEYS)
            except Exception as e:
                logger.warning(f"Failed to fetch weather baseline: {e}")
            
        raw_bases = []
        for d_id in iaq_devices.values():
            try:
                raw_bases.append(prev_method(d_id, IAQ_KEYS))
            except Exception:
                pass
        indoor_baseline = average_nested_baselines(raw_bases, IAQ_KEYS)

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        now_ts = pd.Timestamp.now()
        current_ctx = get_time_context(now_ts)
        
        ctx_w_base = {k: weather_baseline.get(k, {}).get(current_ctx) for k in WEATHER_KEYS}
        ctx_i_base = {k: indoor_baseline.get(k, {}).get(current_ctx) for k in IAQ_KEYS}

        output = [
            "Query_Context:",
            "  Domain: Climate & Weather (Indoor_IAQ)",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Active_Context: {current_ctx}",
            "",
            f"Statistical_Baseline ({current_ctx}):",
            f"  Weather_Normals: {format_baseline_str(ctx_w_base, ['air_temperature', 'relative_humidity', 'solar_radiation', 'precipitation', 'wind_speed'])}",
            f"  Indoor_Normals: {format_baseline_str(ctx_i_base, IAQ_KEYS)}",
            "",
            "Current_State_With_Diffs (vs Baseline & Limits):"
        ]
        
        if WEATHER_STATION_ID:
            w_curr = extract_current_values(tb_client.get_now(WEATHER_STATION_ID, WEATHER_KEYS), WEATHER_KEYS)
            w_parts = [format_val(k, w_curr.get(k), ctx_w_base.get(k), room) for k in ['air_temperature', 'relative_humidity', 'solar_radiation', 'precipitation', 'wind_speed'] if w_curr.get(k) is not None]
            output.append(f"  Weather_Current: {' | '.join(w_parts) if w_parts else 'Offline'}")
        else:
            output.append("  Weather_Current: Not Configured in Topology")
        
        output.append("  Indoor_Current (Room Sensors):")
        for name, d_id in iaq_devices.items():
            i_curr = extract_current_values(tb_client.get_now(d_id, IAQ_KEYS), IAQ_KEYS)
            i_parts = [format_val(k, i_curr.get(k), ctx_i_base.get(k), room) for k in IAQ_KEYS if i_curr.get(k) is not None]
            output.append(f"    - {name}: {' | '.join(i_parts) if i_parts else 'Offline / No Data'}")
            
        return "\n".join(output)

    # ==========================================
    # HISTORICAL DATA FETCHING
    # ==========================================
    fetch_method = getattr(tb_client, config["method"])
    weather_df = pd.DataFrame()
    if WEATHER_STATION_ID:
        weather_df = process_telemetry_to_df(fetch_method(WEATHER_STATION_ID, WEATHER_KEYS), WEATHER_KEYS, bin_size)
    
    indoor_dfs = []
    for d_id in iaq_devices.values():
        df = process_telemetry_to_df(fetch_method(d_id, IAQ_KEYS), IAQ_KEYS, bin_size)
        if not df.empty: indoor_dfs.append(df)
        
    if not indoor_dfs:
        return f"Query_Context:\n  Room: {room}\nError: No historical IAQ data found for timeframe {timeframe}."
        
    indoor_df = pd.concat(indoor_dfs).groupby(level=0).median()
    master_df = indoor_df.join(weather_df, how='outer') if not weather_df.empty else indoor_df

    # ==========================================
    # BRANCH B: 30-DAY / 90-DAY STATISTICAL PROFILE
    # ==========================================
    if timeframe in ["30d", "90d"]:
        is_weekday = master_df.index.dayofweek < 5
        is_weekend = master_df.index.dayofweek >= 5
        is_working = (master_df.index.hour >= 8) & (master_df.index.hour < 22)
        is_non_working = (master_df.index.hour < 8) | (master_df.index.hour >= 22)
        
        output = [
            "Query_Context:",
            "  Domain: Climate & Weather (Indoor_IAQ)",
            f"  Room: {room}",
            f"  Timeframe: {timeframe} (Long-Term Matrix Profile)",
            "",
            "Schedule_Profiling_Matrix:"
        ]
        
        def process_matrix_cell(name: str, mask: pd.Series):
            cell_df = master_df[mask]
            if cell_df.empty: return [f"  {name}:", "    No data."]
            
            # Calculate baselines for the cell
            cell_base_i = cell_df[IAQ_KEYS].mean().to_dict() if not cell_df[IAQ_KEYS].empty else {}
            w_cols = [k for k in WEATHER_KEYS if k in cell_df.columns]
            cell_base_w = cell_df[w_cols].mean().to_dict() if w_cols and not cell_df[w_cols].empty else {}
            
            lines = [f"  {name}:"]
            
            # 1. Print Baselines FIRST
            lines.append("    Statistical_Baseline (Background):")
            lines.append(f"      Weather_Normals: {format_baseline_str(cell_base_w, ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed'])}")
            lines.append(f"      Indoor_Normals: {format_baseline_str(cell_base_i, IAQ_KEYS)}")
            
            # 2. Process Outliers
            outliers = []
            daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
            for day, day_data in daily_groups:
                if day_data.empty: continue
                day_mean = day_data.mean()
                spikes, drivers = [], []
                
                for k in IAQ_KEYS:
                    val = day_mean.get(k)
                    base = cell_base_i.get(k)
                    is_spike = False
                    if pd.notna(val):
                        if base is not None and abs(val - base) >= THRESHOLDS.get(k, 999): is_spike = True
                        
                        limit_info = get_limit(k, room)
                        if limit_info and (val < limit_info["min"] or val > limit_info["max"]): is_spike = True
                        
                        if is_spike: spikes.append(format_val(k, val, base, room))
                            
                for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']:
                    val = day_mean.get(k)
                    base = cell_base_w.get(k)
                    is_spike = False
                    if pd.notna(val):
                        if base is not None:
                            diff = val - base
                            if k == 'solar_radiation':
                                if diff >= 400.0: is_spike = True
                            elif k == 'precipitation':
                                if val > 0.5: is_spike = True
                            elif abs(diff) >= THRESHOLDS.get(k, 999): 
                                is_spike = True
                                
                        limit_info = get_limit(k, room)
                        if limit_info and (val < limit_info["min"] or val > limit_info["max"]): is_spike = True
                        
                        if is_spike: drivers.append(format_val(k, val, base, room))
                            
                if spikes:
                    day_str = day.strftime('%Y-%m-%d (%A)')
                    outliers.append(f"      - '{day_str}': Room_Spikes: {' | '.join(spikes)} | Weather_Drivers: {' | '.join(drivers) if drivers else 'None'}")
            
            # Print Outliers SECOND
            if outliers:
                lines.append("    Outliers (Priority):")
                lines.extend(outliers)
            else:
                lines.append("    Outliers (Priority): None")
                
            return lines

        output.extend(process_matrix_cell("Weekdays (Mon-Fri) Working_Hours (08:00-22:00)", is_weekday & is_working))
        output.extend(process_matrix_cell("Weekdays (Mon-Fri) Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working))
        output.extend(process_matrix_cell("Weekends (Sat-Sun) Working_Hours (08:00-22:00)", is_weekend & is_working))
        output.extend(process_matrix_cell("Weekends (Sat-Sun) Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working))
        
        return "\n".join(output)

    # ==========================================
    # BRANCH C: TIMELINE ACTIVITY (2h, 24h, 7d)
    # ==========================================
    
    # 1. Provide Contextual Baselines
    present_contexts = sorted(list(set(get_time_context(dt) for dt in master_df.index)))
    output = [
        "Query_Context:",
        "  Domain: Climate & Weather (Indoor_IAQ)",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        "",
        "Statistical_Baseline (Present Contexts):"
    ]
    
    for ctx in present_contexts:
        ctx_w_base = {k: weather_baseline.get(k, {}).get(ctx) for k in WEATHER_KEYS}
        ctx_i_base = {k: indoor_baseline.get(k, {}).get(ctx) for k in IAQ_KEYS}
        output.append(f"  {ctx}:")
        output.append(f"    Weather_Normals: {format_baseline_str(ctx_w_base, ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed'])}")
        output.append(f"    Indoor_Normals: {format_baseline_str(ctx_i_base, IAQ_KEYS)}")
    output.append("")
    
    # 2. Calculate true contextual average deviations (with Absolute Avg Val)
    period_i_deltas = {k: [] for k in IAQ_KEYS}
    period_i_vals = {k: [] for k in IAQ_KEYS}
    period_w_deltas = {k: [] for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']}
    period_w_vals = {k: [] for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']}
    
    for exact_time, row in master_df.iterrows():
        ctx = get_time_context(exact_time)
        
        for k in IAQ_KEYS:
            if pd.notna(row.get(k)) and indoor_baseline.get(k, {}).get(ctx) is not None:
                period_i_deltas[k].append(row[k] - indoor_baseline[k][ctx])
                period_i_vals[k].append(row[k])
                
        for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']:
            if pd.notna(row.get(k)) and weather_baseline.get(k, {}).get(ctx) is not None:
                period_w_deltas[k].append(row[k] - weather_baseline[k][ctx])
                period_w_vals[k].append(row[k])
                
    p_i_shifts = []
    for k in IAQ_KEYS:
        if period_i_deltas[k]:
            avg_delta = np.mean(period_i_deltas[k])
            avg_val = np.mean(period_i_vals[k])
            if abs(avg_delta) >= THRESHOLDS.get(k, 0):
                p_i_shifts.append(f"{DISPLAY_NAMES.get(k, k)} Avg_Shift {avg_val:.1f}{UNITS.get(k, '')} ({avg_delta:+.1f}{UNITS.get(k, '')})")
                
    p_w_shifts = []
    for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']:
        if period_w_deltas[k]:
            avg_delta = np.mean(period_w_deltas[k])
            avg_val = np.mean(period_w_vals[k])
            
            is_shift = False
            if k == 'solar_radiation':
                if avg_delta >= 400.0: is_shift = True
            elif k == 'precipitation':
                if avg_val > 0.5: is_shift = True
            elif abs(avg_delta) >= THRESHOLDS.get(k, 0):
                is_shift = True
                
            if is_shift:
                p_w_shifts.append(f"{DISPLAY_NAMES.get(k, k)} Avg_Shift {avg_val:.1f}{UNITS.get(k, '')} ({avg_delta:+.1f}{UNITS.get(k, '')})")

    output.append(f"Period_Deviations (Last {timeframe} True Contextual Shift):")
    output.append(f"  Weather_Shifts: {' | '.join(p_w_shifts) if p_w_shifts else 'None (Consistent with baselines)'}")
    output.append(f"  Indoor_Shifts: {' | '.join(p_i_shifts) if p_i_shifts else 'None (Consistent with baselines)'}")
    output.append("")
    output.append("Timeline_Activity:")

    daily_groups = master_df.groupby(pd.Grouper(freq='D'))
    
    for day_start, day_df in daily_groups:
        if day_df.empty: continue
        
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        anomalies = []
        stable_intervals = 0
        stable_start = None
        stable_periods = []
        
        for exact_time, row in day_df.iterrows():
            ctx = get_time_context(exact_time)
            time_str = exact_time.strftime('%H:%M')
            bucket_end = (exact_time + pd.to_timedelta(bin_size)).strftime('%H:%M')
            
            spikes, drivers = [], []
            
            # Check Indoor Spikes vs precise Context Baseline + Absolute Limits
            for k in IAQ_KEYS:
                val = row.get(k)
                base = indoor_baseline.get(k, {}).get(ctx)
                is_spike = False
                
                if pd.notna(val):
                    if base is not None and abs(val - base) >= THRESHOLDS.get(k, 999): is_spike = True
                    
                    limit_info = get_limit(k, room)
                    if limit_info and (val < limit_info["min"] or val > limit_info["max"]): is_spike = True
                    
                    if is_spike: spikes.append(format_val(k, val, base, room))
            
            # Check Weather Drivers
            for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']:
                val = row.get(k)
                base = weather_baseline.get(k, {}).get(ctx)
                is_spike = False
                
                if pd.notna(val):
                    if base is not None:
                        diff = val - base
                        if k == 'solar_radiation':
                            if diff >= 400.0: is_spike = True
                        elif k == 'precipitation':
                            if val > 0.5: is_spike = True
                        elif abs(diff) >= THRESHOLDS.get(k, 999): 
                            is_spike = True
                            
                    limit_info = get_limit(k, room)
                    if limit_info and (val < limit_info["min"] or val > limit_info["max"]): is_spike = True
                    
                    if is_spike: drivers.append(format_val(k, val, base, room))

            if spikes:
                if stable_intervals > 0:
                    stable_periods.append(f"      - '{stable_start} to {time_str}' ({stable_intervals} intervals): State matched Baseline.")
                    stable_intervals = 0
                    stable_start = None
                
                anomalies.append(f"      - bucket: '{time_str} to {bucket_end}' (Context: {ctx})")
                anomalies.append(f"        Room_Spikes: {' | '.join(spikes)}")
                anomalies.append(f"        Weather_Drivers: {' | '.join(drivers) if drivers else 'None'}")
            else:
                if stable_start is None: stable_start = time_str
                stable_intervals += 1
                
        if stable_intervals > 0:
            end_of_day = (day_start + pd.Timedelta(days=1)).strftime('%H:%M')
            if end_of_day == "00:00": end_of_day = "24:00"
            stable_periods.append(f"      - '{stable_start} to {end_of_day}' ({stable_intervals} intervals): State matched Baseline.")

        output.append(f"  '{day_key}':")
        if anomalies:
            output.append("    Anomalies (Priority):")
            output.extend(anomalies)
        else:
            output.append("    Anomalies (Priority): None")
            
        if stable_periods:
            output.append("    Stable_Periods (Background):")
            output.extend(stable_periods)
        else:
            output.append("    Stable_Periods (Background): None")

    return "\n".join(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Testing Temp & Humidity Tool...")
    print("-" * 50)
    
    try:
        print("\n[Testing Historical (now)]")
        print(get_temp_humidity.invoke({"room": "restaurant", "timeframe": "now"}))
        
        print("\n" + "="*50)
        
        print("\n[Testing Historical (2h)]")
        print(get_temp_humidity.invoke({"room": "restaurant", "timeframe": "2h"}))
        
        print("\n" + "="*50)
        
        print("\n[Testing Historical (24h)]")
        print(get_temp_humidity.invoke({"room": "restaurant", "timeframe": "24h"}))
        
        print("\n" + "="*50)
        
        print("\n[Testing Historical (7d)]")
        print(get_temp_humidity.invoke({"room": "data_center", "timeframe": "7d"}))
        
        print("\n" + "="*50)
        
        print("\n[Testing Historical (30d)]")
        print(get_temp_humidity.invoke({"room": "restaurant", "timeframe": "30d"}))

        print("\n" + "="*50)
        
        print("\n[Testing Historical (90d)]")
        print(get_temp_humidity.invoke({"room": "restaurant", "timeframe": "90d"}))
        
        print("\n" + "-"*50)
        print("All Temp & Humidity tool tests completed successfully.")
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)