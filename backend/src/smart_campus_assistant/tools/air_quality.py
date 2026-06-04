import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, List, Optional, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

logger = logging.getLogger(__name__)

# ==========================================
# OUTDOOR PM SENSOR DISCOVERY
# ==========================================
_pm_devices = registry.get_all_devices_by_type("PM")
OUTDOOR_PM_NAME = next(iter(_pm_devices.keys())) if _pm_devices else None
OUTDOOR_PM_DATA = _pm_devices[OUTDOOR_PM_NAME] if OUTDOOR_PM_NAME else {}
OUTDOOR_PM_ID = OUTDOOR_PM_DATA.get("id") if isinstance(OUTDOOR_PM_DATA, dict) else OUTDOOR_PM_DATA

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
OUTDOOR_KEYS = ["pm1_0", "pm2_5", "pm10"]

# Absolute Extreme Limits (Health Limits)
ABSOLUTE_LIMITS = {
    "co2": 1000.0,
    "pm1_0": 10.0,
    "pm2_5": 25.0,
    "pm10": 45.0,
    "tvoc": 500.0
}

UNITS = {
    "co2": "ppm",
    "pm1_0": "µg/m³",
    "pm2_5": "µg/m³", 
    "pm10": "µg/m³", 
    "tvoc": "ppb"
}

DISPLAY_NAMES = {
    "co2": "CO2",
    "pm1_0": "PM1_0",
    "pm2_5": "PM2_5", 
    "pm10": "PM10", 
    "tvoc": "TVOC"
}

Rooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

Timeframes = Literal[
    'now', '2h', '24h', '7d', '30d', '90d'
]

class AirQualityInput(BaseModel):
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

def clean_iaq_value(key: str, val: float, is_iaq: bool = True) -> float:
    """Sanitizes incoming telemetry to fix multiplier bugs and error codes."""
    if pd.isna(val): 
        return val
    if key == "co2" and val >= 65000:
        return np.nan
    # ONLY apply the division hack to indoor IAQ sensors
    if is_iaq and key in ["pm1_0", "pm2_5", "pm10"]:
        if val > 500:
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
        return f"{name}: {val_str}{unit} ({diff_str}{unit}){limit_tag}"
        
    return f"{name}: {val_str}{unit}{limit_tag}"

def format_baseline_str(data: dict, keys: list) -> str:
    parts = []
    for k in keys:
        if k in data and data[k] is not None:
            parts.append(format_val(k, data[k]))
    return " | ".join(parts) if parts else "No Baseline Data"

def process_telemetry_to_df(raw_data: Dict, keys: List[str], bin_size: str = None, is_iaq: bool = True) -> pd.DataFrame:
    dfs = []
    for key in keys:
        if key in raw_data and raw_data[key]:
            data_list = raw_data[key]
            
            if isinstance(data_list, dict):
                data_list = [data_list]
                
            records = []
            if isinstance(data_list, list):
                for item in data_list:
                    if isinstance(item, dict) and 'ts' in item and 'value' in item:
                        records.append({'ts': item['ts'], 'value': item['value']})
            
            if not records: continue
                
            df = pd.DataFrame(records)
            
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            df['value'] = df['value'].apply(lambda x: clean_iaq_value(key, x, is_iaq))
            df.dropna(subset=['value'], inplace=True)

            if df.empty: continue

            df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert(settings.TIMEZONE)
            df.set_index('datetime', inplace=True)
            df.rename(columns={'value': key}, inplace=True)
            df.drop(columns=['ts'], inplace=True)
            dfs.append(df)
            
    if not dfs: return pd.DataFrame()
    combined = pd.concat(dfs, axis=1, sort=True)
    if bin_size:
        # Added numeric_only=True to prevent future Pandas warnings
        combined = combined.resample(bin_size).median(numeric_only=True)
    return combined

def extract_current_values(raw_data: Dict, keys: List[str], is_iaq: bool = True) -> Dict[str, float]:
    result = {}
    for k in keys:
        if k in raw_data and raw_data[k]:
            try:
                val = float(raw_data[k][0]["value"])
                val = clean_iaq_value(k, val, is_iaq)
                result[k] = val if not pd.isna(val) else None
            except (ValueError, KeyError, IndexError):
                result[k] = None
    return result

def parse_full_nested_baselines(raw_bases: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    contexts = ['weekday_work', 'weekday_nonwork', 'weekend_work', 'weekend_nonwork']
    result = {k: {c: [] for c in contexts} for k in keys}
    
    for base in raw_bases:
        for k in keys:
            if k in base and isinstance(base[k], dict):
                for c in contexts:
                    if c in base[k]:
                        data = base[k][c]
                        if not isinstance(data, list): data = [data]
                            
                        for item in data:
                            val = None
                            if isinstance(item, dict) and 'value' in item:
                                val = item['value']
                            elif isinstance(item, (int, float, str)):
                                val = item
                                
                            if val is not None:
                                try:
                                    v = float(val)
                                    v = clean_iaq_value(k, v, is_iaq=True) # Baselines are always IAQ
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


@tool("get_air_quality", args_schema=AirQualityInput, response_format="content_and_artifact")
def get_air_quality(room: Rooms, timeframe: Timeframes) -> Tuple[str, dict]:
    """
    Tracks indoor Air Quality (CO2, PM2.5, PM10, TVOC).
    Focuses on absolute health limits and deviations from period averages.
    """
    floor_val = str(room)[0] if str(room)[0].isdigit() else "0"
    
    all_iaq_devices = registry.get_devices_by_room_and_type(room, "IAQ")
    if not all_iaq_devices:
        error_msg = f"Query_Context:\n  Room: {room}\nError: No IAQ sensors found in this room."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "error",
                "domain": "Air Quality",
                "floor": floor_val,
                "room_id": str(room),
                "message": "No IAQ sensors found"
            }
        }

    # ==========================================
    # SERVER ATTRIBUTE ACTIVE/OFFLINE CHECK
    # ==========================================
    active_iaq_devices = {}
    offline_sensors = []
    
    for device_name, device_data in all_iaq_devices.items():
        device_id = device_data.get("id") if isinstance(device_data, dict) else device_data
        if not device_id: 
            offline_sensors.append(device_name)
            continue
            
        try:
            attrs = tb_client.get_server_attributes(device_id, ["active"])
            is_active = any(attr.get("key") == "active" and str(attr.get("value")).lower() == "true" for attr in attrs)
            if is_active:
                active_iaq_devices[device_name] = device_data
            else:
                offline_sensors.append(device_name)
        except Exception as e:
            logger.warning(f"Could not fetch active status for {device_name}: {e}")
            offline_sensors.append(device_name)

    is_outdoor_active = False
    if OUTDOOR_PM_ID:
        try:
            w_attrs = tb_client.get_server_attributes(OUTDOOR_PM_ID, ["active"])
            is_outdoor_active = any(attr.get("key") == "active" and str(attr.get("value")).lower() == "true" for attr in w_attrs)
        except Exception:
            pass
            
    if not is_outdoor_active and OUTDOOR_PM_NAME:
        if OUTDOOR_PM_NAME not in offline_sensors:
            offline_sensors.append(OUTDOOR_PM_NAME)

    if not active_iaq_devices:
        error_msg = f"Query_Context:\n  Room: {room}\nError: Found {len(all_iaq_devices)} IAQ sensors, but all are currently offline."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "error",
                "domain": "Air Quality",
                "floor": floor_val,
                "room_id": str(room),
                "message": "All sensors offline"
            }
        }

    # Build Active Sensors Block
    total_relevant = len(all_iaq_devices) + (1 if OUTDOOR_PM_ID else 0)
    active_count = len(active_iaq_devices) + (1 if is_outdoor_active else 0)
    
    sensor_info_lines = [f"  Active_Sensors: {active_count}/{total_relevant} Online"]
    
    if is_outdoor_active and OUTDOOR_PM_NAME:
        sensor_info_lines.append(f"    - {OUTDOOR_PM_NAME} (Outdoor PM)")
        
    for name, data in active_iaq_devices.items():
        if isinstance(data, dict):
            z = data.get("zone", "Unspecified")
            t = data.get("tag", "Unspecified")
            place = f"Zone: {z}, Tag: {t}"
        else:
            place = "Unspecified"
        sensor_info_lines.append(f"    - {name} (IAQ): {place}")

    if offline_sensors:
        sensor_info_lines.append(f"  Offline_Sensors: {', '.join(offline_sensors)}")

    # ==========================================
    # CORE LOGIC
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    
    indoor_baseline = {k: {} for k in IAQ_KEYS}
    if timeframe not in ["30d", "90d"]:
        prev_method = getattr(tb_client, config["prev_method"])
        raw_bases = []
        for d_data in active_iaq_devices.values():
            d_id = d_data.get("id") if isinstance(d_data, dict) else d_data
            try:
                raw_bases.append(prev_method(d_id, IAQ_KEYS))
            except Exception:
                pass
        indoor_baseline = parse_full_nested_baselines(raw_bases, IAQ_KEYS)

    health_limits_str = (
        "Health_Limits (Absolute):\n"
        f"  CO2: {int(ABSOLUTE_LIMITS['co2'])}ppm | "
        f"PM1_0: {int(ABSOLUTE_LIMITS['pm1_0'])}µg/m³ | "
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

        outdoor_curr = {}
        if is_outdoor_active and OUTDOOR_PM_ID:
            try:
                outdoor_curr = extract_current_values(tb_client.get_now(OUTDOOR_PM_ID, OUTDOOR_KEYS), OUTDOOR_KEYS, is_iaq=False)
            except Exception:
                pass
                
        outdoor_pm_str = format_baseline_str(outdoor_curr, OUTDOOR_KEYS) if outdoor_curr else "Offline / No Data"

        output = [
            "Query_Context:",
            "  Domain: Health & Safety (Indoor_IAQ)",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Current_Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Active_Context: {current_ctx}",
        ]
        output.extend(sensor_info_lines)
        output.extend([
            "",
            health_limits_str,
            "",
            "Outdoor_PM_Context (Now):",
            f"  {outdoor_pm_str}",
            "",
            f"Statistical_Baseline ({current_ctx}):",
            f"  Indoor_Normals: {format_baseline_str(ctx_i_base, IAQ_KEYS)}",
            "",
            "Current_State_With_Diffs (vs Baseline & Limits):",
            "  Indoor_Current (Room Sensors):"
        ])
        
        # --- NEW NESTED ARTIFACT LOGIC ---
        ui_aggregates = {}
        ui_sensors = {}
        
        # 1. Outdoor Sensor processing
        if OUTDOOR_PM_NAME:
            ui_sensors[OUTDOOR_PM_NAME] = {
                "status": "good" if is_outdoor_active else "error",
                "category": "OUTDOOR",
                "readings": outdoor_curr if (is_outdoor_active and outdoor_curr) else None
            }
            
        # 2. Offline Sensors processing
        for device_name in offline_sensors:
            if device_name != OUTDOOR_PM_NAME:
                ui_sensors[device_name] = {
                    "status": "error",
                    "category": "IAQ",
                    "readings": None
                }
                
        # 3. Active Sensors processing & actual values
        i_curr_list = []
        for name, data in active_iaq_devices.items():
            d_id = data.get("id") if isinstance(data, dict) else data
            zone = data.get("zone", "Unspecified") if isinstance(data, dict) else "Unspecified"
            
            i_curr = extract_current_values(tb_client.get_now(d_id, IAQ_KEYS), IAQ_KEYS, is_iaq=True)
            i_curr_list.append(i_curr)
            
            # Check if reading is valid (e.g. not 65535 nan error)
            has_valid_data = any(v is not None for v in i_curr.values())
            sensor_status = "error"
            
            if has_valid_data:
                sensor_status = "good"
                for k, v in i_curr.items():
                    if v is None: continue
                    limit = ABSOLUTE_LIMITS.get(k)
                    if limit:
                        if v > limit:
                            sensor_status = "critical"
                            break  # Highest severity met
                        elif v > limit * 0.8:
                            sensor_status = "warning"
                            
            ui_sensors[name] = {
                "status": sensor_status,
                "category": "IAQ",
                "readings": i_curr if has_valid_data else None
            }
            
            # Keep LLM text output intact
            i_parts = [format_val(k, i_curr.get(k), ctx_i_base.get(k)) for k in IAQ_KEYS if i_curr.get(k) is not None]
            output.append(f"    - {name} (Zone: {zone}): {' | '.join(i_parts) if i_parts else 'Offline / No Data'}")
            
        # 4. Aggregate IAQ for Room Level
        for k in IAQ_KEYS:
            vals = [i_curr.get(k) for i_curr in i_curr_list if i_curr.get(k) is not None]
            if vals:
                ui_aggregates[k] = sum(vals) / len(vals)
                
        # 5. Determine Overall Room Status
        if not ui_aggregates:
            overall_status = "error"
        else:
            overall_status = "good"
            for k, v in ui_aggregates.items():
                limit = ABSOLUTE_LIMITS.get(k)
                if limit:
                    if v > limit:
                        overall_status = "critical"
                        break
                    elif v > limit * 0.8:
                        overall_status = "warning"

        artifact = {
            "type": "map_update",
            "artifact": {
                "view_type": "snapshot",
                "domain": "Air Quality",
                "floor": floor_val,
                "room_id": str(room),
                "status": overall_status,
                "room_aggregates": ui_aggregates,
                "sensors": ui_sensors
            }
        }
            
        return "\n".join(output), artifact

    # ==========================================
    # HISTORICAL DATA FETCHING
    # ==========================================
    fetch_method = getattr(tb_client, config["method"])
    
    outdoor_df = pd.DataFrame()
    if is_outdoor_active and OUTDOOR_PM_ID:
        try:
            outdoor_df = process_telemetry_to_df(fetch_method(OUTDOOR_PM_ID, OUTDOOR_KEYS), OUTDOOR_KEYS, bin_size, is_iaq=False)
            # Rename columns immediately so they do not collide with indoor sensors
            if not outdoor_df.empty:
                outdoor_df = outdoor_df.rename(columns={k: f"outdoor_{k}" for k in OUTDOOR_KEYS})
        except Exception as e:
            logger.warning(f"Failed to fetch historical outdoor PM data: {e}")
            
    indoor_dfs = []
    for d_data in active_iaq_devices.values():
        d_id = d_data.get("id") if isinstance(d_data, dict) else d_data
        df = process_telemetry_to_df(fetch_method(d_id, IAQ_KEYS), IAQ_KEYS, bin_size, is_iaq=True)
        if not df.empty: indoor_dfs.append(df)
        
    if not indoor_dfs:
        error_msg = f"Query_Context:\n  Room: {room}\nError: No historical IAQ data found for timeframe {timeframe}."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "graph",
                "domain": "Air Quality",
                "floor": floor_val,
                "room_id": str(room),
                "timeframe": timeframe,
                "online_sensors": list(active_iaq_devices.keys()) + ([OUTDOOR_PM_NAME] if is_outdoor_active and OUTDOOR_PM_NAME else []),
                "offline_sensors": offline_sensors,
                "series": [],
                "metadata": {}
            }
        }
        
    indoor_df = pd.concat(indoor_dfs).groupby(level=0).median()
    master_df = indoor_df.join(outdoor_df, how='outer') if not outdoor_df.empty else indoor_df
    
    # --- BUILD THE GRAPH ARTIFACT ---
    if timeframe in ["30d", "90d"]:
        artifact_df = master_df.resample('1D').median(numeric_only=True)
    else:
        artifact_df = master_df
        
    series_data = []
    for dt, row in artifact_df.iterrows():
        point = {"timestamp": dt.isoformat()}
        for col in artifact_df.columns:
            val = row[col]
            if pd.notna(val):
                point[col] = float(val)
        if len(point) > 1:
            series_data.append(point)
            
    metadata = {}
    for col in artifact_df.columns:
        base_key = col.replace("outdoor_", "")
        metadata[col] = UNITS.get(base_key, "")
            
    online_sensor_names = list(active_iaq_devices.keys())
    if is_outdoor_active and OUTDOOR_PM_NAME:
        online_sensor_names.append(OUTDOOR_PM_NAME)

    graph_artifact = {
        "type": "map_update",
        "artifact": {
            "view_type": "graph",
            "domain": "Air Quality",
            "floor": floor_val,
            "room_id": str(room),
            "timeframe": timeframe,
            "online_sensors": online_sensor_names,
            "offline_sensors": offline_sensors,
            "series": series_data,
            "metadata": metadata
        }
    }
    
    overall_outdoor_mean = {}
    if not outdoor_df.empty:
        for k in OUTDOOR_KEYS:
            if f"outdoor_{k}" in outdoor_df.columns:
                overall_outdoor_mean[k] = outdoor_df[f"outdoor_{k}"].mean()

    outdoor_pm_str = format_baseline_str(overall_outdoor_mean, OUTDOOR_KEYS) if overall_outdoor_mean else "Offline / No Data"

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
            f"  Current_Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        output.extend(sensor_info_lines)
        output.extend([
            "",
            health_limits_str,
            "",
            "Outdoor_PM_Context (Timeframe Average):",
            f"  {outdoor_pm_str}",
            "",
            "Schedule_Profiling_Matrix:"
        ])
        
        def process_matrix_cell(name: str, mask: pd.Series):
            cell_df = master_df[mask]
            if cell_df.empty: return [f"    {name}:", "      Baseline: No data.", "      Outliers: None detected."]
            
            cell_base_i = cell_df[IAQ_KEYS].mean().to_dict() if not cell_df[IAQ_KEYS].empty else {}
            
            lines = [f"    {name}:"]
            # Standardized Baseline Key
            lines.append(f"      Baseline: {format_baseline_str(cell_base_i, IAQ_KEYS)}")
            
            outliers = []
            daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
            for day, day_data in daily_groups:
                if day_data.empty: continue
                day_mean = day_data.mean()
                i_spikes = []
                o_spikes = []
                
                for k in IAQ_KEYS:
                    val = day_mean.get(k)
                    base = cell_base_i.get(k)
                    if pd.notna(val) and val > ABSOLUTE_LIMITS.get(k, 99999):
                        i_spikes.append(format_val(k, val, base))
                        
                for k in OUTDOOR_KEYS:
                    # Map back to the renamed outdoor key
                    val = day_mean.get(f"outdoor_{k}")
                    if pd.notna(val) and val > ABSOLUTE_LIMITS.get(k, 99999):
                        o_spikes.append(format_val(k, val, None))
                            
                if i_spikes or o_spikes:
                    day_str = day.strftime('%Y-%m-%d (%A)')
                    parts = []
                    if i_spikes: parts.append(f"Room: {' | '.join(i_spikes)}")
                    if o_spikes: parts.append(f"Outdoor: {' | '.join(o_spikes)}")
                    # Indented for Nested Matrix
                    outliers.append(f"        - '{day_str}': Spikes: {' | '.join(parts)}")
            
            # Standardized Outliers Key
            if outliers:
                lines.append("      Outliers:")
                lines.extend(outliers)
            else:
                lines.append("      Outliers: None detected.")
                
            return lines

        # Applied Nested YAML Layout Structure
        output.append("  Weekdays (Mon-Fri):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekday & is_working))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working))
        output.append("  Weekends (Sat-Sun):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekend & is_working))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working))
        
        return "\n".join(output), graph_artifact

    # ==========================================
    # BRANCH C: TIMELINE ACTIVITY (2h, 24h, 7d)
    # ==========================================
    present_contexts = sorted(list(set(get_time_context(dt) for dt in master_df.index)))
    output = [
        "Query_Context:",
        "  Domain: Health & Safety (Indoor_IAQ)",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        f"  Current_Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    output.extend(sensor_info_lines)
    output.extend([
        "",
        health_limits_str,
        "",
        "Outdoor_PM_Context (Timeframe Average):",
        f"  {outdoor_pm_str}",
        "",
        "Statistical_Baseline (Present Contexts):"
    ])
    
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
            if avg_val > ABSOLUTE_LIMITS.get(k, 99999):
                p_i_shifts.append(f"{DISPLAY_NAMES.get(k, k)}: {avg_val:.1f}{UNITS.get(k, '')} ({avg_delta:+.1f}{UNITS.get(k, '')}) [LIMIT_EXCEEDED]")

    output.append(f"Period_Deviations (Last {timeframe}):")
    output.append(f"  Indoor_Shifts: {' | '.join(p_i_shifts) if p_i_shifts else 'None (Consistent with baselines / Limits not passed)'}")
    output.append("")
    output.append("Timeline_Activity:")

    daily_groups = master_df.groupby(pd.Grouper(freq='D'))
    
    for day_start, day_df in daily_groups:
        if day_df.empty: continue
        
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        anomalies = []
        stable_intervals = 0
        stable_start = None
        stable_end = None
        stable_periods = []
        
        for exact_time, row in day_df.iterrows():
            ctx = get_time_context(exact_time)
            time_str = exact_time.strftime('%H:%M')
            bucket_end = (exact_time + pd.to_timedelta(bin_size)).strftime('%H:%M')
            if bucket_end == "00:00": bucket_end = "24:00"
            
            i_spikes = []
            o_spikes = []
            
            for k in IAQ_KEYS:
                val = row.get(k)
                base = indoor_baseline.get(k, {}).get(ctx)
                if pd.notna(val) and val > ABSOLUTE_LIMITS.get(k, 99999):
                    i_spikes.append(format_val(k, val, base))
                    
            for k in OUTDOOR_KEYS:
                # Map back to the renamed outdoor key
                val = row.get(f"outdoor_{k}")
                if pd.notna(val) and val > ABSOLUTE_LIMITS.get(k, 99999):
                    o_spikes.append(format_val(k, val, None))

            if i_spikes or o_spikes:
                if stable_intervals > 0:
                    stable_periods.append(f"      - '{stable_start} to {time_str}' ({stable_intervals} intervals): State below Health_Limits.")
                    stable_intervals = 0
                    stable_start = None
                    stable_end = None
                
                anomalies.append(f"      - bucket: '{time_str} - {bucket_end}' (Context: {ctx})")
                parts = []
                if i_spikes: parts.append(f"Indoor: {' | '.join(i_spikes)}")
                if o_spikes: parts.append(f"Outdoor: {' | '.join(o_spikes)}")
                anomalies.append(f"        Spikes: {' | '.join(parts)}")
            else:
                if stable_start is None: stable_start = time_str
                stable_end = bucket_end
                stable_intervals += 1
                
        if stable_intervals > 0:
            if stable_end == "00:00": stable_end = "24:00"
            stable_periods.append(f"      - '{stable_start} to {stable_end}' ({stable_intervals} intervals): State below Health_Limits.")

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

    return "\n".join(output), graph_artifact

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Testing Air Quality Tool...")
    print("-" * 50)
    
    try:
        print("\n[Testing]")
        summary, raw_data = get_air_quality.func(room="2.4", timeframe="now")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "="*50)
        
        print("\n[Testing]")
        summary, raw_data = get_air_quality.func(room="2.3", timeframe="24h")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "="*50)
        
        print("\n[Testing]")
        summary, raw_data = get_air_quality.func(room="2.3", timeframe="30d")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)