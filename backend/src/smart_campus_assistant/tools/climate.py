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
from src.smart_campus_assistant.clients.astral_client import astral_client

logger = logging.getLogger(__name__)

# ==========================================
# OUTDOOR WEATHER STATION DISCOVERY
# ==========================================
_weather_devices = registry.get_all_devices_by_type("WEATHERSTATION")
WEATHER_STATION_NAME = next(iter(_weather_devices.keys())) if _weather_devices else None
WEATHER_STATION_DATA = _weather_devices[WEATHER_STATION_NAME] if WEATHER_STATION_NAME else {}
WEATHER_STATION_ID = WEATHER_STATION_DATA.get("id") if isinstance(WEATHER_STATION_DATA, dict) else WEATHER_STATION_DATA

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
    "solar_radiation": "Avg_Solar", "precipitation": "Precip", "wind_speed": "Wind"
}

Rooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7', 'building'
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
        if val > limit["max"]: limit_tag = " [MAX!]"
        elif val < limit["min"]: limit_tag = " [MIN!]"
            
    if baseline is not None and not pd.isna(baseline):
        diff = val - baseline
        diff_str = f"{diff:+.1f}" if diff % 1 else f"{int(diff):+}"
        return f"{name}: {val_str}{unit} ({diff_str}{unit}){limit_tag}"
        
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
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert(settings.TIMEZONE)
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

@tool("get_climate", args_schema=TempHumidityInput, response_format="content_and_artifact")
def get_climate(room: Rooms, timeframe: Timeframes) -> Tuple[str, dict]:
    """
    Tracks indoor Temperature, Humidity, and Pressure, correlated with Outdoor Weather.
    Splits baselines via a schedule matrix and strictly enforces absolute safety limits.
    Groups consecutive anomalous intervals into blocks to preserve LLM token context.
    """
    room_str = str(room).lower()
    
    if room_str == 'building':
        floor_val = "B"
        all_iaq_devices = registry.get_all_devices_by_type("IAQ")
    else:
        floor_val = registry.get_floor_for_room(room) or (str(room)[0] if str(room)[0].isdigit() else "0")
        all_iaq_devices = registry.get_devices_by_room_and_type(room, "IAQ")
        
    if not all_iaq_devices:
        error_msg = f"Query_Context:\n  Room: {room}\nError: No IAQ sensors found in this target."
        return error_msg, {
            "type": "map_update", 
            "artifact": {
                "view_type": "error", 
                "domain": "Climate",
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

    is_weather_active = False
    if WEATHER_STATION_ID:
        try:
            w_attrs = tb_client.get_server_attributes(WEATHER_STATION_ID, ["active"])
            is_weather_active = any(attr.get("key") == "active" and str(attr.get("value")).lower() == "true" for attr in w_attrs)
        except Exception:
            pass
            
    if not is_weather_active and WEATHER_STATION_NAME:
        offline_sensors.append(WEATHER_STATION_NAME)

    if not active_iaq_devices:
        error_msg = f"Query_Context:\n  Room: {room}\nError: Found {len(all_iaq_devices)} IAQ sensors, but all are currently offline."
        return error_msg, {
            "type": "map_update", 
            "artifact": {
                "view_type": "error", 
                "domain": "Climate",
                "floor": floor_val,
                "room_id": str(room),
                "message": "All sensors offline"
            }
        }

    # Build Active Sensors Block
    total_relevant = len(all_iaq_devices) + (1 if WEATHER_STATION_ID else 0)
    active_count = len(active_iaq_devices) + (1 if is_weather_active else 0)
    
    active_sensors_lines = [f"  Active_Sensors: {active_count}/{total_relevant} Online"]
    
    if is_weather_active and WEATHER_STATION_NAME:
        active_sensors_lines.append(f"    - {WEATHER_STATION_NAME} (Outdoor Weather)")
        
    for name, data in active_iaq_devices.items():
        if isinstance(data, dict):
            z = data.get("zone", "Unspecified")
            t = data.get("tag", "Unspecified")
            if room_str == 'building':
                r = data.get("room", "Unknown")
                place = f"Room: {r}, Zone: {z}, Tag: {t}"
            else:
                place = f"Zone: {z}, Tag: {t}"
        else:
            place = "Unspecified"
        active_sensors_lines.append(f"    - {name} (IAQ): {place}")

    if offline_sensors:
        active_sensors_lines.append(f"  Offline_Sensors: {', '.join(offline_sensors)}")

    # ==========================================
    # CORE LOGIC
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    
    # 1. Fetch Nested Baselines (For non-30d/90d)
    indoor_baseline = {k: {} for k in IAQ_KEYS}
    weather_baseline = {k: {} for k in WEATHER_KEYS}
    
    if timeframe not in ["30d", "90d"]:
        prev_method = getattr(tb_client, config["prev_method"])
        
        if is_weather_active and WEATHER_STATION_ID:
            try:
                weather_baseline = prev_method(WEATHER_STATION_ID, WEATHER_KEYS)
            except Exception as e:
                logger.warning(f"Failed to fetch weather baseline: {e}")
            
        raw_bases = []
        for d_data in active_iaq_devices.values():
            d_id = d_data.get("id") if isinstance(d_data, dict) else d_data
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
        
        solar = astral_client.get_current_solar_context()

        output = [
            "Query_Context:",
            "  Domain: Climate & Weather (Indoor_IAQ)",
            f"  Room: {room.upper()}",
            "  Timeframe: Now (Snapshot)",
            f"  Current_Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Active_Context: {current_ctx}"
        ]
        output.extend(active_sensors_lines)
        
        output.extend([
            "  Solar_Context (Heat Gain Potential):",
            f"    - Daylight_Window: {solar['sunrise']} to {solar['sunset']}",
            f"    - Current_Sun_Azimuth: {solar['horizontal']}",
            f"    - Vertical_Angle: {solar['thermal_vertical']}",
            "",
            f"Statistical_Baseline ({current_ctx}):",
            f"  Weather: {format_baseline_str(ctx_w_base, ['air_temperature', 'relative_humidity', 'solar_radiation', 'precipitation', 'wind_speed'])}",
            f"  Indoor: {format_baseline_str(ctx_i_base, IAQ_KEYS)}",
            "",
            "Current_State_With_Diffs (vs Baseline & Limits):"
        ])
        
        # --- NEW NESTED ARTIFACT LOGIC ---
        ui_aggregates = {}
        ui_sensors = {}
        
        # 1. Weather Station processing
        if WEATHER_STATION_NAME:
            if is_weather_active:
                w_curr = extract_current_values(tb_client.get_now(WEATHER_STATION_ID, WEATHER_KEYS), WEATHER_KEYS)
                
                ui_w_curr = {k: v for k, v in w_curr.items() if k in ["air_temperature", "relative_humidity", "atmospheric_pressure"]}
                
                ui_sensors[WEATHER_STATION_NAME] = {
                    "status": "good",
                    "category": "WEATHER",
                    "readings": ui_w_curr
                }
                # Keep text generation exactly the same for the LLM
                w_parts = [format_val(k, w_curr.get(k), ctx_w_base.get(k), room) for k in ['air_temperature', 'relative_humidity', 'solar_radiation', 'precipitation', 'wind_speed'] if w_curr.get(k) is not None]
                output.append(f"  Weather: {' | '.join(w_parts) if w_parts else 'Offline / No Data'}")
            else:
                ui_sensors[WEATHER_STATION_NAME] = {
                    "status": "error",
                    "category": "WEATHER",
                    "readings": None
                }
                output.append("  Weather: Offline / No Data")
        else:
            output.append("  Weather: Offline / Not Configured")
            
        # 2. Offline Sensors processing
        for device_name in offline_sensors:
            if device_name != WEATHER_STATION_NAME:
                ui_sensors[device_name] = {
                    "status": "error",
                    "category": "IAQ",
                    "readings": None
                }
        
        output.append("  Indoor (Room Sensors):")
        
        # 3. Active Sensors processing & actual values
        i_curr_list = []
        for name, data in active_iaq_devices.items():
            d_id = data.get("id") if isinstance(data, dict) else data
            
            # Ensure correct limit parsing and display for building-wide requests
            sensor_room = data.get("room", room) if isinstance(data, dict) else room
            
            if room_str == 'building' and isinstance(data, dict):
                r_label = data.get("room", "Unknown")
                z_label = data.get("zone", "Unspecified")
                place_label = f"Room: {r_label}, Zone: {z_label}"
            else:
                z_label = data.get("zone", "Unspecified") if isinstance(data, dict) else "Unspecified"
                place_label = f"Zone: {z_label}"

            i_curr = extract_current_values(tb_client.get_now(d_id, IAQ_KEYS), IAQ_KEYS)
            i_curr_list.append(i_curr)
            
            has_valid_data = any(v is not None for v in i_curr.values())
            sensor_status = "error"
            
            if has_valid_data:
                sensor_status = "good"
                for k, v in i_curr.items():
                    if v is None: continue
                    lim = get_limit(k, sensor_room)
                    if lim:
                        span = lim["max"] - lim["min"]
                        margin = span * 0.1 # 10% tolerance for warning
                        if v < lim["min"] or v > lim["max"]:
                            sensor_status = "critical"
                            break
                        elif v < (lim["min"] + margin) or v > (lim["max"] - margin):
                            sensor_status = "warning"
                    
                    base = ctx_i_base.get(k)
                    thresh = THRESHOLDS.get(k)
                    if base is not None and thresh:
                        diff = abs(v - base)
                        if diff >= thresh:
                            sensor_status = "critical"
                            break
                        elif diff >= thresh * 0.8:
                            if sensor_status == "good": sensor_status = "warning"
                            
            ui_sensors[name] = {
                "status": sensor_status,
                "category": "IAQ",
                "readings": i_curr if has_valid_data else None
            }
            
            # Keep text intact
            i_parts = [format_val(k, i_curr.get(k), ctx_i_base.get(k), sensor_room) for k in IAQ_KEYS if i_curr.get(k) is not None]
            output.append(f"    - {name} ({place_label}): {' | '.join(i_parts) if i_parts else 'Offline / No Data'}")
            
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
                lim = get_limit(k, room)
                if lim:
                    span = lim["max"] - lim["min"]
                    margin = span * 0.1
                    if v < lim["min"] or v > lim["max"]:
                        overall_status = "critical"
                        break
                    elif v < (lim["min"] + margin) or v > (lim["max"] - margin):
                        overall_status = "warning"
                        
                base = ctx_i_base.get(k)
                thresh = THRESHOLDS.get(k)
                if base is not None and thresh:
                    diff = abs(v - base)
                    if diff >= thresh:
                        overall_status = "critical"
                        break
                    elif diff >= thresh * 0.8:
                        if overall_status == "good": overall_status = "warning"

        artifact = {
            "type": "map_update",
            "artifact": {
                "view_type": "snapshot",
                "domain": "Climate",
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
    weather_df = pd.DataFrame()
    if is_weather_active and WEATHER_STATION_ID:
        weather_df = process_telemetry_to_df(fetch_method(WEATHER_STATION_ID, WEATHER_KEYS), WEATHER_KEYS, bin_size)
    
    indoor_dfs = []
    for d_data in active_iaq_devices.values():
        d_id = d_data.get("id") if isinstance(d_data, dict) else d_data
        df = process_telemetry_to_df(fetch_method(d_id, IAQ_KEYS), IAQ_KEYS, bin_size)
        if not df.empty: indoor_dfs.append(df)
        
    if not indoor_dfs:
        error_msg = f"Query_Context:\n  Room: {room}\nError: No historical IAQ data found for timeframe {timeframe}."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "graph",
                "domain": "Climate",
                "floor": floor_val,
                "room_id": str(room),
                "timeframe": timeframe,
                "online_sensors": list(active_iaq_devices.keys()) + ([WEATHER_STATION_NAME] if is_weather_active and WEATHER_STATION_NAME else []),
                "offline_sensors": offline_sensors,
                "series": [],
                "metadata": {}
            }
        }
        
    indoor_df = pd.concat(indoor_dfs).groupby(level=0).median()
    master_df = indoor_df.join(weather_df, how='outer') if not weather_df.empty else indoor_df
    
    # --- BUILD THE GRAPH ARTIFACT ---
    if timeframe in ["30d", "90d"]:
        artifact_df = master_df.resample('1D').median(numeric_only=True)
    else:
        artifact_df = master_df
        
    allowed_ui_cols = IAQ_KEYS + ["air_temperature", "relative_humidity", "atmospheric_pressure"]
    ui_df = artifact_df[[c for c in artifact_df.columns if c in allowed_ui_cols]]
        
    series_data = []
    for dt, row in ui_df.iterrows():
        point = {"timestamp": dt.isoformat()}
        for col in ui_df.columns:
            val = row[col]
            if pd.notna(val):
                point[col] = float(val)
        if len(point) > 1:
            series_data.append(point)
            
    online_sensor_names = list(active_iaq_devices.keys())
    if is_weather_active and WEATHER_STATION_NAME:
        online_sensor_names.append(WEATHER_STATION_NAME)

    graph_artifact = {
        "type": "map_update",
        "artifact": {
            "view_type": "graph",
            "domain": "Climate",
            "floor": floor_val,
            "room_id": str(room),
            "timeframe": timeframe,
            "online_sensors": online_sensor_names,
            "offline_sensors": offline_sensors,
            "series": series_data,
            "metadata": {col: UNITS.get(col, "") for col in ui_df.columns}
        }
    }
    
    days_map = {"2h": 1, "24h": 1, "7d": 7, "30d": 30, "90d": 90}
    days_back = days_map.get(timeframe, 1)
    solar_hist = astral_client.get_historical_solar_context(days_back)

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
            f"  Room: {room.upper()}",
            f"  Timeframe: {timeframe} (Long-Term Matrix Profile)",
            f"  Current_Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        output.extend(active_sensors_lines)
        
        output.extend([
            "  Solar_Context (Heat Gain Potential):",
            f"    - Average_Daylight_Window: {solar_hist['avg_sunrise']} to {solar_hist['avg_sunset']}",
            f"    - Daily_Sun_Trajectory: {solar_hist['trajectory']}",
            "", 
            "Schedule_Profiling_Matrix:"
        ])
        
        def process_matrix_cell(name: str, mask: pd.Series):
            cell_df = master_df[mask]
            if cell_df.empty: return [f"    {name}:", "      Baseline: No data.", "      Outliers: None detected."]
            
            cell_base_i = cell_df[IAQ_KEYS].mean().to_dict() if not cell_df[IAQ_KEYS].empty else {}
            w_cols = [k for k in WEATHER_KEYS if k in cell_df.columns]
            cell_base_w = cell_df[w_cols].mean().to_dict() if w_cols and not cell_df[w_cols].empty else {}
            
            lines = [f"    {name}:"]
            # Standardized Baseline Key
            lines.append("      Baseline:")
            lines.append(f"        Weather: {format_baseline_str(cell_base_w, ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed'])}")
            lines.append(f"        Indoor: {format_baseline_str(cell_base_i, IAQ_KEYS)}")
            
            outliers = []
            daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
            
            prev_sig = None
            streak_start = None
            streak_end = None
            streak_days = []
            
            def flush_streak():
                s_keys_with_sign, d_keys_with_sign = prev_sig
                s_keys = [k for k, sign in s_keys_with_sign]
                d_keys = [k for k, sign in d_keys_with_sign]
                
                streak_df = pd.concat([cell_df[cell_df.index.normalize() == d.normalize()] for d in streak_days])
                streak_mean = streak_df.mean()
                
                spikes = [format_val(k, streak_mean.get(k), cell_base_i.get(k), room) for k in s_keys]
                drivers = [format_val(k, streak_mean.get(k), cell_base_w.get(k), room) for k in d_keys]
                
                # Added Day of Week to Date Formatting
                date_str = streak_start.strftime('%Y-%m-%d (%A)')
                if streak_start != streak_end:
                    date_str += f" to {streak_end.strftime('%Y-%m-%d (%A)')}"
                    
                day_word = "day" if len(streak_days) == 1 else "days"
                
                parts = []
                if spikes: parts.append(f"Indoor: {' | '.join(spikes)}")
                if drivers: parts.append(f"Outdoor: {' | '.join(drivers)}")
                
                # Indented for Nested Matrix
                outliers.append(f"        - '{date_str}' ({len(streak_days)} {day_word}): {' | '.join(parts)}")

            for day, day_data in daily_groups:
                if day_data.empty: continue
                day_mean = day_data.mean()
                spike_keys, driver_keys = [], []
                
                for k in IAQ_KEYS:
                    val = day_mean.get(k)
                    base = cell_base_i.get(k)
                    is_spk = False
                    sign = 0
                    if pd.notna(val):
                        if base is not None and abs(val - base) >= THRESHOLDS.get(k, 999):
                            is_spk = True
                            sign = 1 if (val - base) > 0 else -1
                        lim = get_limit(k, room)
                        if lim:
                            if val < lim["min"]: 
                                is_spk = True
                                sign = -1
                            elif val > lim["max"]: 
                                is_spk = True
                                sign = 1
                        if is_spk: spike_keys.append((k, sign))
                            
                for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']:
                    val = day_mean.get(k)
                    base = cell_base_w.get(k)
                    is_spk = False
                    sign = 0
                    if pd.notna(val):
                        if base is not None:
                            diff = val - base
                            if k == 'solar_radiation' and diff >= 400.0: 
                                is_spk = True
                                sign = 1
                            elif k == 'precipitation' and val > 0.5: 
                                is_spk = True
                                sign = 1
                            elif abs(diff) >= THRESHOLDS.get(k, 999): 
                                is_spk = True
                                sign = 1 if diff > 0 else -1
                        lim = get_limit(k, room)
                        if lim:
                            if val < lim["min"]: 
                                is_spk = True
                                sign = -1
                            elif val > lim["max"]: 
                                is_spk = True
                                sign = 1
                        if is_spk: driver_keys.append((k, sign))
                            
                state_key = (tuple(spike_keys), tuple(driver_keys))
                
                if not spike_keys and not driver_keys:
                    if prev_sig is not None:
                        flush_streak()
                        prev_sig = None
                    continue
                    
                if prev_sig is None:
                    prev_sig = state_key
                    streak_start = day
                    streak_end = day
                    streak_days = [day]
                elif state_key == prev_sig:
                    streak_end = day
                    streak_days.append(day)
                else:
                    flush_streak()
                    prev_sig = state_key
                    streak_start = day
                    streak_end = day
                    streak_days = [day]
                    
            if prev_sig is not None:
                flush_streak()
            
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
    
    solar_context_lines = [
        "  Solar_Context (Heat Gain Potential):",
        f"    - Average_Daylight_Window: {solar_hist['avg_sunrise']} to {solar_hist['avg_sunset']}",
        f"    - Daily_Sun_Trajectory: {solar_hist['trajectory']}"
    ]
    if timeframe == "2h":
        el_label, el_desc = astral_client.get_average_thermal_elevation_info(2)
        solar_context_lines.append(f"    - Vertical_Angle: {el_label} ({el_desc})")
        
    output = [
        "Query_Context:",
        "  Domain: Climate & Weather (Indoor_IAQ)",
        f"  Room: {room.upper()}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        f"  Current_Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    output.extend(active_sensors_lines)
    output.extend(solar_context_lines)
    output.extend(["", "Statistical_Baseline (Present Contexts):"])
    
    for ctx in present_contexts:
        ctx_w_base = {k: weather_baseline.get(k, {}).get(ctx) for k in WEATHER_KEYS}
        ctx_i_base = {k: indoor_baseline.get(k, {}).get(ctx) for k in IAQ_KEYS}
        output.append(f"  {ctx}:")
        output.append(f"    Weather: {format_baseline_str(ctx_w_base, ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed'])}")
        output.append(f"    Indoor: {format_baseline_str(ctx_i_base, IAQ_KEYS)}")
    output.append("")
    
    # Calculate true contextual average deviations
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
                p_i_shifts.append(f"{DISPLAY_NAMES.get(k, k)}: {avg_val:.1f}{UNITS.get(k, '')} ({avg_delta:+.1f}{UNITS.get(k, '')})")
                
    p_w_shifts = []
    for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']:
        if period_w_deltas[k]:
            avg_delta = np.mean(period_w_deltas[k])
            avg_val = np.mean(period_w_vals[k])
            is_shift = False
            if k == 'solar_radiation' and avg_delta >= 400.0: is_shift = True
            elif k == 'precipitation' and avg_val > 0.5: is_shift = True
            elif abs(avg_delta) >= THRESHOLDS.get(k, 0): is_shift = True
                
            if is_shift:
                p_w_shifts.append(f"{DISPLAY_NAMES.get(k, k)}: {avg_val:.1f}{UNITS.get(k, '')} ({avg_delta:+.1f}{UNITS.get(k, '')})")

    output.append(f"Period_Deviations (Last {timeframe}):")
    output.append(f"  Weather_Shifts: {' | '.join(p_w_shifts) if p_w_shifts else 'None'}")
    output.append(f"  Indoor_Shifts: {' | '.join(p_i_shifts) if p_i_shifts else 'None'}")
    output.append("")
    output.append("Timeline_Activity:")

    daily_groups = master_df.groupby(pd.Grouper(freq='D'))
    
    for day_start, day_df in daily_groups:
        if day_df.empty: continue
        
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        anomalies = []
        stable_periods = []
        
        prev_state = None
        current_start = None
        current_end = None
        current_intervals = 0
        current_ctx = None
        period_timestamps = []
        
        def flush_timeline():
            s_keys_with_sign, d_keys_with_sign, ctx_val = prev_state
            if not s_keys_with_sign and not d_keys_with_sign:
                stable_periods.append(f"      - '{current_start} to {current_end}' ({current_intervals} int): Matched Baseline")
                return
                
            s_keys = [k for k, sign in s_keys_with_sign]
            d_keys = [k for k, sign in d_keys_with_sign]
                
            period_df = day_df.loc[period_timestamps]
            period_mean = period_df.mean()
            
            spikes = [format_val(k, period_mean.get(k), indoor_baseline.get(k, {}).get(ctx_val), room) for k in s_keys]
            drivers = [format_val(k, period_mean.get(k), weather_baseline.get(k, {}).get(ctx_val), room) for k in d_keys]
            
            parts = []
            if spikes: parts.append(f"Indoor: {' | '.join(spikes)}")
            if drivers: parts.append(f"Outdoor: {' | '.join(drivers)}")
            
            anomalies.append(f"      - bucket: '{current_start} - {current_end}' (Context: {ctx_val})\n        Spikes: {' | '.join(parts)}")
        
        for exact_time, row in day_df.iterrows():
            ctx = get_time_context(exact_time)
            time_str = exact_time.strftime('%H:%M')
            bucket_end = (exact_time + pd.to_timedelta(bin_size)).strftime('%H:%M')
            if bucket_end == "00:00": bucket_end = "24:00"
            
            spike_keys, driver_keys = [], []
            
            for k in IAQ_KEYS:
                val = row.get(k)
                base = indoor_baseline.get(k, {}).get(ctx)
                is_spike = False
                sign = 0
                if pd.notna(val):
                    if base is not None and abs(val - base) >= THRESHOLDS.get(k, 999):
                        is_spike = True
                        sign = 1 if (val - base) > 0 else -1
                    limit_info = get_limit(k, room)
                    if limit_info:
                        if val < limit_info["min"]: 
                            is_spike = True
                            sign = -1
                        elif val > limit_info["max"]: 
                            is_spike = True
                            sign = 1
                    if is_spike: spike_keys.append((k, sign))
            
            for k in ['air_temperature', 'solar_radiation', 'precipitation', 'wind_speed']:
                val = row.get(k)
                base = weather_baseline.get(k, {}).get(ctx)
                is_spike = False
                sign = 0
                if pd.notna(val):
                    if base is not None:
                        diff = val - base
                        if k == 'solar_radiation' and diff >= 400.0: 
                            is_spike = True
                            sign = 1
                        elif k == 'precipitation' and val > 0.5: 
                            is_spike = True
                            sign = 1
                        elif abs(diff) >= THRESHOLDS.get(k, 999): 
                            is_spike = True
                            sign = 1 if diff > 0 else -1
                    limit_info = get_limit(k, room)
                    if limit_info:
                        if val < limit_info["min"]: 
                            is_spike = True
                            sign = -1
                        elif val > limit_info["max"]: 
                            is_spike = True
                            sign = 1
                    if is_spike: driver_keys.append((k, sign))

            state_key = (tuple(spike_keys), tuple(driver_keys), ctx)
            
            if prev_state is None:
                prev_state = state_key
                current_start = time_str
                current_end = bucket_end
                current_ctx = ctx
                current_intervals = 1
                period_timestamps = [exact_time]
            elif state_key == prev_state:
                current_intervals += 1
                current_end = bucket_end
                period_timestamps.append(exact_time)
            else:
                flush_timeline()
                prev_state = state_key
                current_start = time_str
                current_end = bucket_end
                current_ctx = ctx
                current_intervals = 1
                period_timestamps = [exact_time]
                
        if current_intervals > 0:
            flush_timeline()

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
    print("Testing Temp & Humidity Tool...")
    print("-" * 50)
    
    try:
        print("\n[Testing]")
        summary, raw_data = get_climate.func(room="parkin.b", timeframe="now")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "="*50)
        
        print("\n[Testing]")
        summary, raw_data = get_climate.func(room="building", timeframe="24h")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "="*50)
        
        print("\n[Testing]")
        summary, raw_data = get_climate.func(room="building", timeframe="30d")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)