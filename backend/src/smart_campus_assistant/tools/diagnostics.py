import time
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
from typing import Literal, Dict, Any, List, Optional, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

logger = logging.getLogger(__name__)

# Target list updated to use only 'building' for the campus-wide view
Targets = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7', 'roof', 'infrastructure',
    'building'
]

Timeframes = Literal[
    'now', '2h', '24h', '7d', '30d', '90d'
]

# Config mapping for API calls and pandas grouping
TIMEFRAME_CONFIG = {
    "now": {"method": "get_now", "bin_size": None},
    "2h":  {"method": "get_2h", "bin_size": "10min"},
    "24h": {"method": "get_24h", "bin_size": "2h"}, 
    "7d":  {"method": "get_7d", "bin_size": "2h"},    
    "30d": {"method": "get_30d", "bin_size": "1D"},
    "90d": {"method": "get_90d", "bin_size": "1D"}    
}

# ==========================================
# INTERNAL DIAGNOSTIC ENGINE
# ==========================================

def _safe_extract_float(data_dict: dict, keys_to_check: list) -> Optional[float]:
    for k in keys_to_check:
        if k in data_dict and data_dict[k]:
            val = data_dict[k][0].get('value')
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    return None

def _get_device_attributes(device_id: str) -> dict:
    try:
        endpoint = f"/api/plugins/telemetry/DEVICE/{device_id}/values/attributes/SERVER_SCOPE"
        response = tb_client._request("GET", endpoint)
        attr_list = response.json()
        
        attrs = {}
        if isinstance(attr_list, list):
            for item in attr_list:
                attrs[item["key"]] = item["value"]
        return attrs
    except Exception as e:
        logger.error(f"Failed to fetch attributes for {device_id}: {e}")
        return {}

def _format_meta(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""
    parts = []
    if meta.get("zone"): parts.append(f"Zone: {meta['zone']}")
    if meta.get("tag"): parts.append(f"Tag: {meta['tag']}")
    if meta.get("group"): parts.append(f"Group: {meta['group']}")
    return f" [{', '.join(parts)}]" if parts else ""

def _audit_device(device_name: str, device_id: str) -> dict:
    now_ts = int(time.time() * 1000)
    
    is_pc_or_wo = "-PC" in device_name.upper() or "-WO" in device_name.upper()
    is_weather = "WEATHER" in device_name.upper()
    
    bat_keys = ["battery"]
    other_keys = [
        "rssi", "loRaSNR", "tamper_alarm", "tamper", "tamper_status", 
        "temperature", "humidity", "co2", "air_temperature",
        "line_1_period_in", "line_1_period_out", "people_count_max", "buzzer_status"
    ]
    
    attrs = _get_device_attributes(device_id)
    is_online = attrs.get("active", False)
    if "active" not in attrs:
        is_online = True
        
    last_seen_str = "Unknown"
    offline_duration_str = ""
    last_ts = attrs.get("lastDisconnectTime") or attrs.get("inactivityAlarmTime") or attrs.get("lastActivityTime")
    
    if not is_online:
        if last_ts:
            dt_last = datetime.fromtimestamp(last_ts / 1000.0)
            now_dt = datetime.now()
            diff_hours = (now_dt - dt_last).total_seconds() / 3600
            last_seen_str = dt_last.strftime("%Y-%m-%d %H:%M:%S EEST")
            if diff_hours > 48:
                offline_duration_str = f"> {int(diff_hours/24)} days"
            else:
                offline_duration_str = f"> {int(diff_hours)}h"
        else:
            offline_duration_str = "Unknown duration"

    current_battery = None
    drain_per_day = 0.0
    est_days = 999
    anomalies = []
    tamper = False
    tamper_time = ""

    try:
        latest_data = tb_client.get_now(device_id, bat_keys + other_keys)
    except Exception:
        latest_data = {}

    try:
        other_data = tb_client.get_7d_2h_splits(device_id, other_keys)
    except Exception:
        other_data = {}

    if not is_pc_or_wo:
        current_battery = _safe_extract_float(latest_data, bat_keys)
        try:
            battery_data = tb_client.get_7d(device_id, bat_keys)
        except Exception:
            battery_data = {}

        bat_key = "battery_level" if "battery_level" in battery_data and battery_data["battery_level"] else "battery"
        if bat_key in battery_data and battery_data[bat_key]:
            df_bat = pd.DataFrame(battery_data[bat_key])
            df_bat['value'] = pd.to_numeric(df_bat['value'], errors='coerce')
            df_bat.dropna(inplace=True)
            
            if not df_bat.empty:
                if current_battery is None:
                    current_battery = df_bat.iloc[-1]['value']
                if len(df_bat) > 5: 
                    max_b = df_bat['value'].max()
                    min_b = df_bat['value'].min()
                    days_span = (df_bat['ts'].max() - df_bat['ts'].min()) / (1000 * 3600 * 24)
                    if days_span > 1:
                        drain_per_day = (max_b - min_b) / days_span
                        if drain_per_day > 0 and current_battery is not None:
                            est_days = current_battery / drain_per_day

    current_rssi = _safe_extract_float(latest_data, ["rssi"])
    if current_rssi is not None and current_rssi < -105:
        anomalies.append(f"[WEAK_SIGNAL] RSSI verified at {int(current_rssi)} dBm")
        
    current_snr = _safe_extract_float(latest_data, ["loRaSNR"])
    if current_snr is not None and current_snr < 0:
        anomalies.append(f"[POOR_SNR] Signal-to-Noise Ratio at {current_snr}")
    
    t_keys = ["tamper_alarm", "tamper", "tamper_status"]
    for t_key in t_keys:
        if t_key in other_data and other_data[t_key]:
            df_t = pd.DataFrame(other_data[t_key])
            df_t['value'] = pd.to_numeric(df_t['value'], errors='coerce')
            recent_t = df_t[df_t['ts'] > (now_ts - 24*3600*1000)]
            if (recent_t['value'] > 0).any():
                tamper = True
                t_ts = recent_t[recent_t['value'] > 0].iloc[0]['ts']
                tamper_time = datetime.fromtimestamp(t_ts / 1000.0).strftime("%A, %B %d, %Y %H:%M:%S")
                break
        if not tamper and t_key in latest_data and latest_data[t_key]:
            val = _safe_extract_float(latest_data, [t_key])
            if val is not None and val > 0:
                tamper = True
                t_ts = latest_data[t_key][0]['ts']
                tamper_time = datetime.fromtimestamp(t_ts / 1000.0).strftime("%A, %B %d, %Y %H:%M:%S")
                break

    for k in ["temperature", "humidity", "co2", "air_temperature"]:
        curr_val = _safe_extract_float(latest_data, [k])
        if curr_val == 65535.0:
            anomalies.append(f"[{k.upper()}_HARDWARE_FAULT] Error Code 65535")
            continue
        
        if k in other_data and other_data[k]:
            df_k = pd.DataFrame(other_data[k])
            df_k['value'] = pd.to_numeric(df_k['value'], errors='coerce')
            recent_k = df_k[df_k['ts'] > (now_ts - 24*3600*1000)]
            if len(recent_k) > 5 and recent_k['value'].max() == recent_k['value'].min():
                locked_val = recent_k['value'].iloc[0]
                df_k_sorted = df_k.sort_values(by='ts', ascending=False)
                diff_mask = df_k_sorted['value'] != locked_val
                
                if not diff_mask.any():
                    duration_str = "> 7 days"
                else:
                    last_good_ts = df_k_sorted[diff_mask].iloc[0]['ts']
                    duration_hours = (now_ts - last_good_ts) / (1000 * 3600)
                    if duration_hours >= 48:
                        duration_str = f"{int(duration_hours / 24)} days"
                    else:
                        duration_str = f"{int(duration_hours)}h"
                anomalies.append(f"[{k.upper()}_FLATLINE] locked at {locked_val:.1f} for {duration_str}")

    return {
        "name": device_name,
        "is_online": is_online,
        "last_seen_str": last_seen_str,
        "offline_duration_str": offline_duration_str,
        "last_ts": last_ts,
        "battery": current_battery,
        "drain_per_day": drain_per_day,
        "est_days": est_days,
        "anomalies": list(set(anomalies)),
        "tamper": tamper,
        "tamper_time": tamper_time,
        "is_plugged_in": is_pc_or_wo,
        "is_weather": is_weather
    }

def _fetch_historical_context(name: str, uid: str, method_name: str) -> dict:
    audit = _audit_device(name, uid)
    fetch_method = getattr(tb_client, method_name)
    
    df_bat = pd.DataFrame()
    hist_errors = []
    
    try:
        data = fetch_method(uid, ["battery", "battery_level", "tamper", "tamper_alarm"])
        bat_key = "battery_level" if "battery_level" in data and data["battery_level"] else "battery"
        if bat_key in data and data[bat_key]:
            df_bat = pd.DataFrame(data[bat_key])
            df_bat['value'] = pd.to_numeric(df_bat['value'], errors='coerce')
            df_bat['datetime'] = pd.to_datetime(df_bat['ts'], unit='ms', utc=True).dt.tz_convert(settings.TIMEZONE)
            df_bat.set_index('datetime', inplace=True)
            df_bat.rename(columns={'value': name}, inplace=True)
            df_bat.drop(columns=['ts'], inplace=True)
            df_bat = df_bat.sort_index()

        for t_key in ["tamper", "tamper_alarm"]:
            if t_key in data and data[t_key]:
                df_t = pd.DataFrame(data[t_key])
                df_t['value'] = pd.to_numeric(df_t['value'], errors='coerce')
                if (df_t['value'] > 0).any():
                    hist_errors.append("Tamper alarm triggered historically")
                    break
    except Exception:
        pass

    if audit["battery"] is not None:
        inject_ts = None
        if audit["is_online"]:
            inject_ts = pd.Timestamp.now(tz=settings.TIMEZONE)
        elif audit.get("last_ts"):
            inject_ts = pd.to_datetime(audit["last_ts"], unit='ms', utc=True).tz_convert(settings.TIMEZONE)
            
        if inject_ts:
            latest_df = pd.DataFrame({name: [audit["battery"]]}, index=[inject_ts])
            if df_bat.empty:
                df_bat = latest_df
            else:
                df_bat = pd.concat([df_bat, latest_df])
                df_bat = df_bat[~df_bat.index.duplicated(keep='last')].sort_index()

    return {
        "name": name,
        "df": df_bat,
        "hist_errors": hist_errors,
        "audit": audit
    }

# ==========================================
# UNIFIED TOOL
# ==========================================

class DiagnosticsInput(BaseModel):
    target: Targets = Field(..., description="The room or 'building' to run diagnostics on.")
    timeframe: Timeframes = Field(..., description="The time window. 'now' for a snapshot, else a timeline profile.")

@tool("get_diagnostics", args_schema=DiagnosticsInput, response_format="content_and_artifact")
def get_diagnostics(target: Targets, timeframe: Timeframes) -> Tuple[str, dict]:
    """
    Unified Diagnostic System. Checks Connectivity (Offline), Power (Battery Drain/Levels), and Hardware Health.
    """
    if target == 'building':
        floor_val = "B"
        room_id = "building"
        target_rooms = registry.get_available_rooms()
    else:
        floor_val = registry.get_floor_for_room(target) or (str(target)[0] if str(target)[0].isdigit() else "0")
        room_id = str(target)
        target_rooms = [target]

    tasks = []
    for room in target_rooms:
        devices = registry.get_all_devices_in_room(room)
        for name, meta in devices.items():
            uid = meta.get("id") if isinstance(meta, dict) else meta
            tasks.append((room, name, uid, meta))

    if not tasks:
        error_msg = f"Error: No devices found for target '{target}'."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "error",
                "domain": "Diagnostics",
                "floor": floor_val,
                "room_id": room_id,
                "message": error_msg
            }
        }

    current_time_str = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
    total_scanned = len(tasks)

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        online_count = 0
        offline_lines = []
        power_warnings = []
        anomaly_lines = []
        tamper_lines = []
        battery_estimates_lines = []
        
        ui_sensors = {}
        room_status_counts = {"error": 0, "critical": 0, "warning": 0, "good": 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            future_to_task = {executor.submit(_audit_device, name, uid): (room, name, meta) for room, name, uid, meta in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                room, name, meta = future_to_task[future]
                meta_str = _format_meta(meta)
                name_disp = f"'{name}'{meta_str}" if target == 'building' else f"{name}{meta_str}"
                
                try:
                    data = future.result()
                    
                    sensor_status = "error"
                    reason = "Unknown"
                    has_major_fault = any("HARDWARE_FAULT" in a or "FLATLINE" in a for a in data["anomalies"])
                    has_signal_issue = any("WEAK_SIGNAL" in a or "POOR_SNR" in a for a in data["anomalies"])
                    
                    bat_pct = 100.0
                    if not data["is_plugged_in"] and data["battery"] is not None:
                        bat_pct = data["battery"]
                        if data["is_weather"]:
                            bat_pct = max(0, min(100, (data["battery"] - 2.4) / (3.0 - 2.4) * 100))

                    if not data["is_online"]:
                        sensor_status = "error"
                        reason = f"Dead/Offline ({data['offline_duration_str']})"
                    elif data["tamper"]:
                        sensor_status = "error"
                        reason = "Tamper Alarm Triggered"
                    elif has_major_fault:
                        sensor_status = "error"
                        reason = "Error Readings / Hardware Fault"
                    else:
                        reasons = []
                        if data["is_plugged_in"]:
                            sensor_status = "good"
                        else:
                            if bat_pct > 40: sensor_status = "good"
                            elif bat_pct >= 16: 
                                sensor_status = "warning"
                                reasons.append("Low Battery")
                            elif bat_pct >= 1: 
                                sensor_status = "critical"
                                reasons.append("Extremely Low Battery")
                            else: 
                                sensor_status = "error"
                                reasons.append("Battery Depleted")
                                
                        if has_signal_issue:
                            if sensor_status == "good": sensor_status = "warning"
                            reasons.append("Low Signal")
                            
                        if not reasons:
                            reason = "Operating Normally"
                        else:
                            reason = " & ".join(reasons)
                    
                    room_status_counts[sensor_status] += 1
                    
                    if not data["is_online"]:
                        battery_val = 0.0 
                    elif data["is_plugged_in"]:
                        battery_val = "Plugged In"
                    else:
                        battery_val = data["battery"] if data["battery"] is not None else "No Data"
                        
                    ui_sensors[name] = {
                        "status": sensor_status,
                        "category": "DIAGNOSTIC",
                        "reason": reason,
                        "readings": {
                            "battery": battery_val,
                            "est_days": int(data["est_days"]) if (data["battery"] is not None and not data["is_plugged_in"]) else None,
                            "is_online": data["is_online"],
                            "tamper_alarm": data["tamper"]
                        }
                    }

                    if data["is_online"]:
                        online_count += 1
                    else:
                        offline_lines.append(f"    - {name_disp} (Offline {data['offline_duration_str']})")
                        if data["is_weather"]:
                            last_bat = f"Last known: {data['battery']:.2f}V" if data['battery'] is not None else "Unknown Voltage"
                        else:
                            last_bat = f"Last known: {data['battery']:.1f}%" if data['battery'] is not None else "Unknown Battery"
                        battery_estimates_lines.append(f"    - {name_disp}: 0.0 (Dead/Offline. {last_bat})")
                        continue
                        
                    if data["is_plugged_in"]:
                        battery_estimates_lines.append(f"    - {name_disp}: Plugged In (Unlimited)")
                    elif data["battery"] is not None:
                        unit = "V" if data["is_weather"] else "%"
                        val_format = ".2f" if data["is_weather"] else ".1f"
                        
                        battery_estimates_lines.append(f"    - {name_disp}: {data['battery']:{val_format}}{unit} | Est. {int(data['est_days'])} days remaining")
                        
                        if bat_pct < 16:
                            power_warnings.append(f"    - {name_disp}: [CRITICAL_BATTERY] {data['battery']:{val_format}}{unit} remaining")
                        elif bat_pct <= 40:
                            power_warnings.append(f"    - {name_disp}: [LOW_BATTERY] {data['battery']:{val_format}}{unit} remaining")
                        elif data["est_days"] < 14:
                            power_warnings.append(f"    - {name_disp}: [HIGH_DRAIN_RATE_ANOMALY] Est. {int(data['est_days'])} days remaining")
                    else:
                        # Sensor doesn't have battery telemetry
                        battery_estimates_lines.append(f"    - {name_disp}: No Battery Telemetry")
                        
                    if data["anomalies"]:
                        for a in data["anomalies"]: anomaly_lines.append(f"    - {name_disp} {a}")
                    if data["tamper"]:
                        tamper_lines.append(f"    - {name_disp} (Casing opened at {data['tamper_time']})")
                        
                except Exception as e:
                    logger.error(f"Concurrent execution failed for {name} in {room}: {e}")

        offline_lines.sort()
        anomaly_lines.sort()
        tamper_lines.sort()
        power_warnings.sort()
        battery_estimates_lines.sort()

        overall_status = "error"
        if room_status_counts["error"] > 0: overall_status = "error"
        elif room_status_counts["critical"] > 0: overall_status = "critical"
        elif room_status_counts["warning"] > 0: overall_status = "warning"
        else: overall_status = "good"

        uptime_pct = (online_count / total_scanned * 100) if total_scanned > 0 else 0
        output = [
            "Query_Context:",
            "  Domain: Diagnostics (System Health Audit)",
            f"  Target: {target.upper()}",
            "  Timeframe: Now (Snapshot)",
            f"  Total_Devices_Scanned: {total_scanned}",
            f"  Current_Time: {current_time_str}",
            "",
            "Connectivity_Audit:",
            f"  Status: {online_count}/{total_scanned} Online ({uptime_pct:.1f}% Uptime)"
        ]
        
        if offline_lines:
            output.append("  Offline_Devices:")
            output.extend(offline_lines)
        else:
            output.append("  Offline_Devices: None")
            
        output.append("\nHardware_Health_Audit:")
        if anomaly_lines:
            output.append("  Anomalies_Detected:")
            output.extend(anomaly_lines)
        else:
            output.append("  Anomalies_Detected: None")
            
        if tamper_lines:
            output.append("  Tamper_Alarms:")
            output.extend(tamper_lines)
        else:
            output.append("  Tamper_Alarms: None")

        output.append("\nPower_Depletion_Warnings:")
        if power_warnings:
            output.extend(power_warnings)
        else:
            output.append("  None")
            
        output.append("\nBattery_Life_Estimates:")
        output.extend(battery_estimates_lines)

        artifact = {
            "type": "map_update",
            "artifact": {
                "view_type": "snapshot",
                "domain": "Diagnostics",
                "floor": floor_val,
                "room_id": room_id,
                "status": overall_status,
                "room_aggregates": {
                    "total": total_scanned,
                    "good": room_status_counts["good"],
                    "warning": room_status_counts["warning"],
                    "critical": room_status_counts["critical"],
                    "error": room_status_counts["error"]
                }
            }
        }
        
        if target != 'building':
            artifact["artifact"]["sensors"] = ui_sensors

        return "\n".join(output), artifact

    # ==========================================
    # BRANCH B: HISTORICAL TIMELINE
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    method_name = config["method"]
    bin_size = config["bin_size"]
    
    historical_contexts = {}
    all_dfs = []
    all_sensor_names = [name for _, name, _, _ in tasks]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_task = {executor.submit(_fetch_historical_context, name, uid, method_name): name for _, name, uid, _ in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            name = future_to_task[future]
            try:
                res = future.result()
                historical_contexts[name] = res
                if not res["df"].empty:
                    all_dfs.append(res["df"])
            except Exception:
                pass

    online_sensors_list = []
    offline_sensors_list = []
    for name in all_sensor_names:
        ctx = historical_contexts.get(name, {})
        audit = ctx.get("audit", {})
        if audit.get("is_online", True):
            online_sensors_list.append(name)
        else:
            offline_sensors_list.append(name)

    if not all_dfs:
        error_msg = f"Query_Context:\n  Target: {target}\nError: No historical diagnostic data found for timeframe {timeframe}."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "graph",
                "domain": "Diagnostics",
                "floor": floor_val,
                "room_id": room_id,
                "timeframe": timeframe,
                "online_sensors": online_sensors_list,
                "offline_sensors": offline_sensors_list,
                "series": [],
                "metadata": {}
            }
        }

    combined_df = pd.concat(all_dfs, axis=1, sort=True)
    combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)

    # Resolve massive time gap issue by securely trimming the dataframe to the requested timeframe bounds
    if not combined_df.empty:
        # 1. Forward-fill to propagate values from older dead/offline timestamps 
        combined_df = combined_df.ffill()
        
        # 2. Extract boundaries
        now_dt = pd.Timestamp.now(tz=settings.TIMEZONE)
        td_map = {
            "2h": pd.Timedelta(hours=2), 
            "24h": pd.Timedelta(hours=24), 
            "7d": pd.Timedelta(days=7), 
            "30d": pd.Timedelta(days=30), 
            "90d": pd.Timedelta(days=90)
        }
        start_dt = now_dt - td_map.get(timeframe, pd.Timedelta(hours=2))
        
        # 3. Clip timestamps before the timeframe boundary
        if combined_df.index[0] < start_dt:
            pre_start_data = combined_df[combined_df.index <= start_dt]
            if not pre_start_data.empty:
                last_pre_start_vals = pre_start_data.iloc[-1]
                combined_df = combined_df[combined_df.index >= start_dt]
                combined_df.loc[start_dt] = last_pre_start_vals
                
        # 4. Enforce that the dataframe correctly covers the entire timeframe to now
        if not combined_df.empty:
            combined_df.loc[now_dt] = combined_df.iloc[-1]
            
        combined_df = combined_df.sort_index()
        combined_df = combined_df.resample(bin_size).median().ffill().bfill()


    # --- BUILD THE GRAPH ARTIFACT (WITH DELTA-ONLY LOGIC) ---
    series_data = []
    
    if target == 'building':
        last_agg = None
        for dt, row in combined_df.iterrows():
            counts = {"good": 0, "warning": 0, "critical": 0, "error": 0}
            bat_sum = 0.0
            bat_count = 0
            
            for name in all_sensor_names:
                ctx = historical_contexts.get(name, {})
                audit = ctx.get("audit", {})
                is_weather = audit.get("is_weather", False)
                is_plugged = audit.get("is_plugged_in", False)
                
                is_online_now = True
                if not audit.get("is_online", True) and audit.get("last_ts"):
                    death_dt = pd.to_datetime(audit["last_ts"], unit='ms', utc=True).tz_convert(settings.TIMEZONE)
                    if dt > death_dt:
                        is_online_now = False

                has_major_fault = any("HARDWARE_FAULT" in a or "FLATLINE" in a for a in audit.get("anomalies", []))
                has_signal_issue = any("WEAK_SIGNAL" in a or "POOR_SNR" in a for a in audit.get("anomalies", []))

                # Battery logic for building average
                if not is_plugged:
                    if not is_online_now:
                        bat_sum += 0.0
                        bat_count += 1
                    else:
                        val = row.get(name) if name in combined_df.columns else audit.get("battery")
                        if pd.notna(val) and val is not None:
                            bat_pct = float(val)
                            if is_weather:
                                bat_pct = max(0, min(100, (bat_pct - 2.4) / (3.0 - 2.4) * 100))
                            bat_sum += bat_pct
                            bat_count += 1

                if not is_online_now or audit.get("tamper", False) or has_major_fault:
                    counts["error"] += 1
                else:
                    status = "error"
                    if is_plugged:
                        status = "warning" if has_signal_issue else "good"
                    else:
                        val = row.get(name) if name in combined_df.columns else audit.get("battery")
                        if pd.isna(val) or val is None:
                            status = "error"
                        else:
                            bat_pct = float(val)
                            if is_weather:
                                bat_pct = max(0, min(100, (bat_pct - 2.4) / (3.0 - 2.4) * 100))
                                
                            if bat_pct > 40: status = "good"
                            elif bat_pct >= 16: status = "warning"
                            elif bat_pct >= 1: status = "critical"
                            else: status = "error"
                            
                            if has_signal_issue and status == "good":
                                status = "warning"
                            
                    counts[status] += 1

            avg_bat = round(bat_sum / bat_count, 1) if bat_count > 0 else 0.0
            current_agg = (counts["good"], counts["warning"], counts["critical"], counts["error"], avg_bat)
            
            if last_agg is None or current_agg != last_agg:
                point = {"timestamp": dt.isoformat()}
                if last_agg is None:
                    point["total"] = total_scanned
                point.update(counts)
                point["average_battery"] = avg_bat
                series_data.append(point)
                last_agg = current_agg

        graph_artifact = {
            "type": "map_update",
            "artifact": {
                "view_type": "graph",
                "domain": "Diagnostics",
                "floor": floor_val,
                "room_id": room_id,
                "timeframe": timeframe,
                "online_sensors": online_sensors_list,
                "offline_sensors": offline_sensors_list,
                "series": series_data,
                "metadata": {
                    "good": "Healthy",
                    "warning": "Warning",
                    "critical": "Critical",
                    "error": "Offline / Error",
                    "average_battery": "Avg Battery %"
                }
            }
        }
    else:
        # Standard Single Room Logic
        last_sent_values = {col: None for col in all_sensor_names}
        for dt, row in combined_df.iterrows():
            point = {"timestamp": dt.isoformat()}
            for col in all_sensor_names:
                ctx = historical_contexts.get(col, {})
                audit = ctx.get("audit", {})
                is_online = audit.get("is_online", True)
                is_plugged = audit.get("is_plugged_in", False)
                
                if is_plugged:
                    val = 100.0
                else:
                    if col in combined_df.columns:
                        val = row[col]
                    else:
                        # Fallback for sensors entirely missing from telemetry history
                        val = audit.get("battery") 

                if not is_online and audit.get("last_ts"):
                    death_dt = pd.to_datetime(audit["last_ts"], unit='ms', utc=True).tz_convert(settings.TIMEZONE)
                    if dt > death_dt:
                        val = 0.0
                        
                if pd.notna(val) and val is not None:
                    val = round(float(val), 2 if audit.get("is_weather", False) else 1)
                    if last_sent_values[col] is None or val != last_sent_values[col]:
                        point[col] = val
                        last_sent_values[col] = val
                        
            if len(point) > 1:
                series_data.append(point)

        meta_dict = {}
        for col in all_sensor_names:
            audit = historical_contexts.get(col, {}).get("audit", {})
            if audit.get("is_plugged_in", False):
                meta_dict[col] = "Plugged In"
            elif "WEATHER" in col.upper():
                meta_dict[col] = "Battery (V)"
            else:
                meta_dict[col] = "Battery %"

        graph_artifact = {
            "type": "map_update",
            "artifact": {
                "view_type": "graph",
                "domain": "Diagnostics",
                "floor": floor_val,
                "room_id": room_id,
                "timeframe": timeframe,
                "online_sensors": online_sensors_list,
                "offline_sensors": offline_sensors_list,
                "series": series_data,
                "metadata": meta_dict
            }
        }

    # Build Text Output for LLM
    output = [
        "Query_Context:",
        "  Domain: Diagnostics (Historical Timeline)",
        f"  Target: {target.upper()}",
        f"  Timeframe: {timeframe}",
        f"  Total_Devices_Scanned: {total_scanned}",
        "",
        "Historical_Warnings:"
    ]
    
    err_lines = []
    # Process ALL sensors, missing dataframe columns included
    for col in all_sensor_names:
        ctx = historical_contexts.get(col, {})
        audit = ctx.get("audit", {})
        hist_errors = ctx.get("hist_errors", [])
        
        sensor_warnings = []
        if not audit.get("is_online", True):
            sensor_warnings.append(f"[OFFLINE] {audit.get('offline_duration_str', 'Unknown')}")
            
        if audit.get("tamper", False):
            sensor_warnings.append(f"[TAMPER] Triggered at {audit.get('tamper_time')}")
            
        for err in hist_errors:
            if "Tamper" in err and not audit.get("tamper", False):
                sensor_warnings.append(f"[{err}]")
                
        for a in audit.get("anomalies", []):
            sensor_warnings.append(a)
            
        bat = audit.get("battery")
        if bat is not None and not audit.get("is_plugged_in", False):
            bat_pct = bat
            unit = "V" if audit.get("is_weather", False) else "%"
            val_format = ".2f" if audit.get("is_weather", False) else ".1f"
            
            if audit.get("is_weather", False):
                bat_pct = max(0, min(100, (bat - 2.4) / (3.0 - 2.4) * 100))
                
            if bat_pct < 16:
                sensor_warnings.append(f"[CRITICAL_BATTERY] {bat:{val_format}}{unit} remaining")
            elif bat_pct <= 40:
                sensor_warnings.append(f"[LOW_BATTERY] {bat:{val_format}}{unit} remaining")
                
        if sensor_warnings:
            err_lines.append(f"  - {col}: {', '.join(sensor_warnings)}")

    if err_lines:
        err_lines.sort()
        output.extend(err_lines)
    else:
        output.append("  - None detected in this timeframe.")
        
    output.append("")
    output.append("Battery_Drain_Summary (Over Timeframe):")
    
    drain_lines = []
    
    # Process ALL sensors, evaluating sensors missing from history
    for col in all_sensor_names:
        ctx = historical_contexts.get(col, {})
        audit = ctx.get("audit", {})
        is_plugged = audit.get("is_plugged_in", False)
        
        if is_plugged:
            drain_lines.append(f"  - {col}: Plugged In")
            continue
            
        raw_df = ctx.get("df", pd.DataFrame())
        
        if col in raw_df.columns and not raw_df[col].dropna().empty:
            start_val = raw_df[col].dropna().iloc[0]
            last_recorded = raw_df[col].dropna().iloc[-1]
            diff = start_val - last_recorded
            
            is_online = audit.get("is_online", True)
            is_weather = audit.get("is_weather", False)
            
            unit = "V" if is_weather else "%"
            val_format = ".2f" if is_weather else ".1f"
            
            days_span = (raw_df.index[-1] - raw_df.index[0]).total_seconds() / (3600 * 24)
            span_str = f" over {int(days_span)} days" if days_span >= 1 else ""
            
            est_str = ""
            if is_online and diff > 0 and days_span > 1:
                drain_per_day = diff / days_span
                if drain_per_day > 0:
                    est_days = int(last_recorded / drain_per_day)
                    est_str = f" | Est. {est_days} days remaining"
            
            if diff < 0:
                event_str = f"Battery Replaced (Jumped from {start_val:{val_format}}{unit} to {last_recorded:{val_format}}{unit})"
            elif diff > 0:
                event_str = f"Dropped {diff:{val_format}}{unit} (From {start_val:{val_format}}{unit} to {last_recorded:{val_format}}{unit}){span_str}{est_str}"
            else:
                event_str = f"Stable at {last_recorded:{val_format}}{unit}"
                
            if not is_online:
                off_str = audit.get('offline_duration_str', 'Unknown')
                drain_lines.append(f"  - {col}: {event_str} | Currently 0.0{unit} [Dead/Offline: {off_str}]")
            else:
                drain_lines.append(f"  - {col}: {event_str}")
        else:
            is_online = audit.get("is_online", True)
            if not is_online:
                off_str = audit.get('offline_duration_str', 'Unknown')
                drain_lines.append(f"  - {col}: No Battery Telemetry [Dead/Offline: {off_str}]")
            else:
                drain_lines.append(f"  - {col}: No Battery Telemetry")
                
    if drain_lines:
        drain_lines.sort()
        output.extend(drain_lines)
    else:
        output.append("  - No significant battery data detected.")

    return "\n".join(output), graph_artifact

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Testing Diagnostics Tool...")
    print("-" * 50)

    try:
        print("\n[Testing]")
        summary, raw_data = get_diagnostics.func(target="2.4", timeframe="2h")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "-"*50)
        
        print("\n[Testing]")
        summary, raw_data = get_diagnostics.func(target="2.4", timeframe="24h")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "-"*50)
        
        print("\n[Testing]")
        summary, raw_data = get_diagnostics.func(target="2.4", timeframe="30d")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("\n" + "-"*50)


    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)