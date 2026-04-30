import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

# Import Schemas
from src.smart_campus_assistant.tools.schemas import CampusRooms, Timeframes

logger = logging.getLogger(__name__)

TIMEFRAME_CONFIG = {
    "now": {"method": "get_now", "bin_size": None, "prev_method": "get_now_prev_30d_full"},
    "2h":  {"method": "get_2h", "bin_size": "10min", "prev_method": "get_2h_prev_30d_full"},
    "24h": {"method": "get_24h", "bin_size": "2h", "prev_method": "get_24h_prev_30d_full"}, 
    "7d":  {"method": "get_7d", "bin_size": "2h", "prev_method": "get_7d_prev_30d_full"},    
    "30d": {"method": "get_30d", "bin_size": "2h", "prev_method": None},
    "90d": {"method": "get_90d", "bin_size": "2h", "prev_method": None} 
}

# Sensor Key Configurations
IAQ_KEYS = ["co2", "pm2_5", "pm10", "tvoc"]

# Baseline Deviations (Relative Deltas for Anomaly Detection)
THRESHOLDS = {
    "co2": 150.0,
    "pm2_5": 5.0,
    "pm10": 10.0,
    "tvoc": 100.0
}

# Absolute Extreme Limits (Health Limits)
ABSOLUTE_LIMITS = {
    "co2": 1000.0,
    "pm2_5": 15.0,
    "pm10": 45.0,
    "tvoc": 500.0
}

UNITS = {
    "co2": "ppm", 
    "pm2_5": "µg/m³", 
    "pm10": "µg/m³", 
    "tvoc": "ppb"
}

DISPLAY_NAMES = {
    "co2": "CO2", 
    "pm2_5": "PM2_5", 
    "pm10": "PM10", 
    "tvoc": "TVOC"
}

class AirQualityInput(BaseModel):
    room: CampusRooms = Field(..., description="The specific room to check.")
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

def clean_iaq_value(key: str, val: float) -> float:
    """Sanitizes incoming telemetry to fix multiplier bugs and error codes."""
    if pd.isna(val): 
        return val
    # Filter out sensor errors (e.g. 65535 or similar huge values)
    if key == "co2" and val >= 65000:
        return np.nan
    # Correct PM multiplier bugs
    if key in ["pm2_5", "pm10"]:
        return val / 100.0
    return val

def format_val(key: str, val: float, baseline: float = None) -> str:
    unit = UNITS.get(key, "")
    name = DISPLAY_NAMES.get(key, key)
    if pd.isna(val): return f"{name}: N/A"
    
    val_str = f"{val:.1f}" if val % 1 else f"{int(val)}"
    
    limit_tag = ""
    limit = ABSOLUTE_LIMITS.get(key)
    if limit and val > limit:
        limit_tag = " [LIMIT_EXCEEDED]"
            
    if baseline is not None and not pd.isna(baseline):
        diff = val - baseline
        diff_str = f"{diff:+.1f}" if diff % 1 else f"{int(diff):+}"
        return f"{name} {val_str}{unit} ({diff_str}{unit}){limit_tag}"
        
    return f"{name}: {val_str}{unit}{limit_tag}"

def format_baseline_str(data: dict, keys: list) -> str:
    parts = []
    for k in keys:
        if k in data and data[k] is not None:
            parts.append(format_val(k, data[k]))
    return " | ".join(parts) if parts else "No Baseline Data"

def process_telemetry_to_df(raw_data: Dict, keys: List[str], bin_size: str = None) -> pd.DataFrame:
    dfs = []
    for key in keys:
        if key in raw_data and raw_data[key]:
            data_list = raw_data[key]
            
            # Failsafe: If TB returns a single dict instead of a list, wrap it
            if isinstance(data_list, dict):
                data_list = [data_list]
                
            # Safely extract ONLY ts and value to prevent Pandas array-length crashes
            records = []
            if isinstance(data_list, list):
                for item in data_list:
                    if isinstance(item, dict) and 'ts' in item and 'value' in item:
                        records.append({'ts': item['ts'], 'value': item['value']})
            
            if not records:
                continue
                
            df = pd.DataFrame(records)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Sanitize Data
            df['value'] = df['value'].apply(lambda x: clean_iaq_value(key, x))
            df.dropna(subset=['value'], inplace=True)
            
            if df.empty: continue
            
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
                val = float(raw_data[k][0]["value"])
                val = clean_iaq_value(k, val)
                result[k] = val if not pd.isna(val) else None
            except (ValueError, KeyError, IndexError):
                result[k] = None
    return result

def parse_full_nested_baselines(raw_bases: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Parses pre-aggregated full baseline data (lists of values categorized by time-context) 
    and applies data sanitization before averaging.
    """
    contexts = ['weekday_work', 'weekday_nonwork', 'weekend_work', 'weekend_nonwork']
    result = {k: {c: [] for c in contexts} for k in keys}
    
    for base in raw_bases:
        for k in keys:
            if k in base and isinstance(base[k], dict):
                for c in contexts:
                    if c in base[k]:
                        data = base[k][c]
                        if not isinstance(data, list):
                            data = [data]  # Fallback if a single value is returned
                            
                        for item in data:
                            val = None
                            if isinstance(item, dict) and 'value' in item:
                                val = item['value']
                            elif isinstance(item, (int, float, str)):
                                val = item
                                
                            if val is not None:
                                try:
                                    v = float(val)
                                    v = clean_iaq_value(k, v)
                                    if not pd.isna(v):
                                        result[k][c].append(v)
                                except (ValueError, TypeError):
                                    pass
                                    
    final_result = {}
    for k in keys:
        final_result[k] = {}
        for c in contexts:
            vals = result[k][c]
            final_result[k][c] = float(np.mean(vals)) if vals else None
            
    return final_result


@tool("get_air_quality", args_schema=AirQualityInput)
def get_air_quality(room: CampusRooms, timeframe: Timeframes) -> str:
    """
    Tracks indoor Air Quality (CO2, PM2.5, PM10, TVOC).
    Focuses on absolute health limits and deviations from period averages.
    """
    iaq_devices = registry.get_devices_by_room_and_type(room, "IAQ")
    if not iaq_devices:
        return f"Query_Context:\n  Room: {room}\nError: No IAQ sensors found in this room."

    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    
    # 1. Fetch Baselines dynamically from the previous 30 days full telemetry
    indoor_baseline = {k: {} for k in IAQ_KEYS}
    
    if timeframe not in ["30d", "90d"]:
        prev_method = getattr(tb_client, config["prev_method"])
        raw_bases = []
        for d_id in iaq_devices.values():
            try:
                raw_bases.append(prev_method(d_id, IAQ_KEYS))
            except Exception:
                pass
        indoor_baseline = parse_full_nested_baselines(raw_bases, IAQ_KEYS)

    health_limits_str = (
        "Health_Limits (Absolute):\n"
        f"  CO2: {int(ABSOLUTE_LIMITS['co2'])}ppm | "
        f"PM2_5: {int(ABSOLUTE_LIMITS['pm2_5'])}µg/m³ | "
        f"PM10: {int(ABSOLUTE_LIMITS['pm10'])}µg/m³ | "
        f"TVOC: {int(ABSOLUTE_LIMITS['tvoc'])}ppb"
    )

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        now_ts = pd.Timestamp.now()
        current_ctx = get_time_context(now_ts)
        ctx_i_base = {k: indoor_baseline.get(k, {}).get(current_ctx) for k in IAQ_KEYS}

        output = [
            "Query_Context:",
            "  Domain: Health & Safety (Indoor_IAQ)",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Active_Context: {current_ctx}",
            "",
            health_limits_str,
            "",
            f"Statistical_Baseline ({current_ctx}):",
            f"  Indoor_Normals: {format_baseline_str(ctx_i_base, IAQ_KEYS)}",
            "",
            "Current_State_With_Diffs (vs Baseline & Limits):",
            "  Indoor_Current (Room Sensors):"
        ]
        
        for name, d_id in iaq_devices.items():
            i_curr = extract_current_values(tb_client.get_now(d_id, IAQ_KEYS), IAQ_KEYS)
            i_parts = [format_val(k, i_curr.get(k), ctx_i_base.get(k)) for k in IAQ_KEYS if i_curr.get(k) is not None]
            output.append(f"    - {name}: {' | '.join(i_parts) if i_parts else 'Offline / No Data'}")
            
        return "\n".join(output)

    # ==========================================
    # HISTORICAL DATA FETCHING
    # ==========================================
    fetch_method = getattr(tb_client, config["method"])
    indoor_dfs = []
    for d_id in iaq_devices.values():
        df = process_telemetry_to_df(fetch_method(d_id, IAQ_KEYS), IAQ_KEYS, bin_size)
        if not df.empty: indoor_dfs.append(df)
        
    if not indoor_dfs:
        return f"Query_Context:\n  Room: {room}\nError: No historical IAQ data found for timeframe {timeframe}."
        
    master_df = pd.concat(indoor_dfs).groupby(level=0).median()

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
            "  Domain: Health & Safety (Indoor_IAQ)",
            f"  Room: {room}",
            f"  Timeframe: {timeframe} (Long-Term Matrix Profile)",
            "",
            health_limits_str,
            "",
            "Schedule_Profiling_Matrix:"
        ]
        
        def process_matrix_cell(name: str, mask: pd.Series):
            cell_df = master_df[mask]
            if cell_df.empty: return [f"  {name}:", "    No data."]
            
            cell_base_i = cell_df[IAQ_KEYS].mean().to_dict() if not cell_df[IAQ_KEYS].empty else {}
            
            lines = [f"  {name}:"]
            lines.append("    Statistical_Baseline (Background):")
            lines.append(f"      Indoor_Normals: {format_baseline_str(cell_base_i, IAQ_KEYS)}")
            
            outliers = []
            daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
            for day, day_data in daily_groups:
                if day_data.empty: continue
                day_mean = day_data.mean()
                spikes = []
                
                for k in IAQ_KEYS:
                    val = day_mean.get(k)
                    base = cell_base_i.get(k)
                    is_spike = False
                    if pd.notna(val):
                        if base is not None and abs(val - base) >= THRESHOLDS.get(k, 999): is_spike = True
                        if val > ABSOLUTE_LIMITS.get(k, 99999): is_spike = True
                        
                        if is_spike: 
                            spikes.append(format_val(k, val, base))
                            
                if spikes:
                    day_str = day.strftime('%Y-%m-%d (%A)')
                    outliers.append(f"      - '{day_str}': Room_Spikes: {' | '.join(spikes)}")
            
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
    present_contexts = sorted(list(set(get_time_context(dt) for dt in master_df.index)))
    output = [
        "Query_Context:",
        "  Domain: Health & Safety (Indoor_IAQ)",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        "",
        health_limits_str,
        "",
        "Statistical_Baseline (Present Contexts):"
    ]
    
    for ctx in present_contexts:
        ctx_i_base = {k: indoor_baseline.get(k, {}).get(ctx) for k in IAQ_KEYS}
        output.append(f"  {ctx}:")
        output.append(f"    Indoor_Normals: {format_baseline_str(ctx_i_base, IAQ_KEYS)}")
    output.append("")
    
    period_i_deltas = {k: [] for k in IAQ_KEYS}
    period_i_vals = {k: [] for k in IAQ_KEYS}
    
    for exact_time, row in master_df.iterrows():
        ctx = get_time_context(exact_time)
        for k in IAQ_KEYS:
            if pd.notna(row.get(k)) and indoor_baseline.get(k, {}).get(ctx) is not None:
                period_i_deltas[k].append(row[k] - indoor_baseline[k][ctx])
                period_i_vals[k].append(row[k])
                
    p_i_shifts = []
    for k in IAQ_KEYS:
        if period_i_deltas[k]:
            avg_delta = np.mean(period_i_deltas[k])
            avg_val = np.mean(period_i_vals[k])
            if abs(avg_delta) >= THRESHOLDS.get(k, 0):
                p_i_shifts.append(f"{DISPLAY_NAMES.get(k, k)} Avg_Shift {avg_val:.1f}{UNITS.get(k, '')} ({avg_delta:+.1f}{UNITS.get(k, '')})")

    output.append(f"Period_Deviations (Last {timeframe} True Contextual Shift):")
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
            
            spikes = []
            
            for k in IAQ_KEYS:
                val = row.get(k)
                base = indoor_baseline.get(k, {}).get(ctx)
                is_spike = False
                
                if pd.notna(val):
                    if base is not None and abs(val - base) >= THRESHOLDS.get(k, 999): is_spike = True
                    if val > ABSOLUTE_LIMITS.get(k, 99999): is_spike = True
                    
                    if is_spike: 
                        spikes.append(format_val(k, val, base))

            if spikes:
                if stable_intervals > 0:
                    stable_periods.append(f"      - '{stable_start} to {time_str}' ({stable_intervals} intervals): State matched Baseline.")
                    stable_intervals = 0
                    stable_start = None
                
                anomalies.append(f"      - bucket: '{time_str} to {bucket_end}' (Context: {ctx})")
                anomalies.append(f"        Room_Spikes: {' | '.join(spikes)}")
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
    print("Testing Air Quality Tool...")
    print("-" * 50)
    
    try:
        print("\n[Testing Historical (now)]")
        print(get_air_quality.invoke({"room": "restaurant", "timeframe": "now"}))
        
        print("\n" + "="*50)
        
        print("\n[Testing Historical (24h)]")
        print(get_air_quality.invoke({"room": "restaurant", "timeframe": "24h"}))
        
        print("\n" + "="*50)
        
        print("\n[Testing Historical (30d)]")
        print(get_air_quality.invoke({"room": "restaurant", "timeframe": "30d"}))

        print("\n" + "-"*50)
        print("All Air Quality tool tests completed successfully.")
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)