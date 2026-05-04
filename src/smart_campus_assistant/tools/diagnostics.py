import time
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
from typing import Literal, Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

logger = logging.getLogger(__name__)

# Allowed rooms derived from registry
CampusRooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7', 'roof'
]

# ==========================================
# INTERNAL DIAGNOSTIC ENGINE
# ==========================================

def _safe_extract_float(data_dict: dict, keys_to_check: list) -> Optional[float]:
    """Safely extracts a float value from ThingsBoard's timeseries list format, handling None values."""
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
    """Fetches SERVER_SCOPE attributes (like 'active', 'lastDisconnectTime') to reliably check connectivity."""
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

def _audit_device(device_name: str, device_id: str) -> dict:
    """
    Fetches diagnostic data using optimized methods. 
    Uses TB Server Attributes for definitive Online/Offline status.
    """
    now_ts = int(time.time() * 1000)
    
    is_pc_or_wo = "-PC" in device_name.upper() or "-WO" in device_name.upper()
    is_weather = "WEATHER" in device_name.upper()
    
    bat_keys = ["battery_level", "battery"]
    other_keys = [
        "rssi", "loRaSNR", "tamper_alarm", "tamper", "tamper_status", 
        "temperature", "humidity", "co2", "air_temperature",
        "line_1_period_in", "line_1_period_out", "people_count_max"
    ]
    
    # 1. Connectivity Audit via Server Attributes (Bulletproof method)
    attrs = _get_device_attributes(device_id)
    is_online = attrs.get("active", False)
    
    # Fallback just in case ThingsBoard drops the attribute entirely
    if "active" not in attrs:
        is_online = True
        
    last_seen_str = "Unknown"
    offline_duration_str = ""
    
    if not is_online:
        last_ts = attrs.get("lastDisconnectTime") or attrs.get("inactivityAlarmTime") or attrs.get("lastActivityTime")
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

    # Initialize defaults
    current_battery = None
    drain_per_day = 0.0
    est_days = 999
    anomalies = []
    tamper = False
    tamper_time = ""

    # IF ONLINE: Run Power and Anomaly Audits
    if is_online:
        # 2. Fetch Absolute Latest Data (Solves blank/None values)
        try:
            latest_data = tb_client.get_now(device_id, bat_keys + other_keys)
        except Exception as e:
            logger.error(f"Failed to fetch latest data for {device_name}: {e}")
            latest_data = {}

        # 3. Fetch Aggregated Split Data (7 days in 2h intervals) for flatline checks
        try:
            other_data = tb_client.get_7d_2h_splits(device_id, other_keys)
        except Exception:
            other_data = {}

        # 4. Power Audit (Only for battery devices)
        if not is_pc_or_wo:
            current_battery = _safe_extract_float(latest_data, bat_keys)
            
            # Fetch high-res history strictly for drain calculation
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
                        
                    # Calculate drain if we have points spanning at least 1 day
                    if len(df_bat) > 5: 
                        max_b = df_bat['value'].max()
                        min_b = df_bat['value'].min()
                        days_span = (df_bat['ts'].max() - df_bat['ts'].min()) / (1000 * 3600 * 24)
                        if days_span > 1:
                            drain_per_day = (max_b - min_b) / days_span
                            if drain_per_day > 0 and current_battery is not None:
                                est_days = current_battery / drain_per_day

        # 5. Hardware / Signal Anomalies Audit
        current_rssi = _safe_extract_float(latest_data, ["rssi"])
        if current_rssi is not None and current_rssi < -105:
            anomalies.append(f"[WEAK_SIGNAL] RSSI verified at {int(current_rssi)} dBm")
            
        current_snr = _safe_extract_float(latest_data, ["loRaSNR"])
        if current_snr is not None and current_snr < 0:
            anomalies.append(f"[POOR_SNR] Signal-to-Noise Ratio at {current_snr}")
        
        # Tamper Checks
        t_keys = ["tamper_alarm", "tamper", "tamper_status"]
        for t_key in t_keys:
            # Check historical splits first
            if t_key in other_data and other_data[t_key]:
                df_t = pd.DataFrame(other_data[t_key])
                df_t['value'] = pd.to_numeric(df_t['value'], errors='coerce')
                recent_t = df_t[df_t['ts'] > (now_ts - 24*3600*1000)]
                if (recent_t['value'] > 0).any():
                    tamper = True
                    t_ts = recent_t[recent_t['value'] > 0].iloc[0]['ts']
                    tamper_time = datetime.fromtimestamp(t_ts / 1000.0).strftime("%H:%M:%S EEST")
                    break
            # Fallback to latest_data
            if not tamper and t_key in latest_data and latest_data[t_key]:
                val = _safe_extract_float(latest_data, [t_key])
                if val is not None and val > 0:
                    tamper = True
                    t_ts = latest_data[t_key][0]['ts']
                    tamper_time = datetime.fromtimestamp(t_ts / 1000.0).strftime("%H:%M:%S EEST")
                    break

        # Flatline Checks (Trace duration backwards through the full 7-day data)
        for k in ["temperature", "humidity", "co2", "air_temperature"]:
            if k in other_data and other_data[k]:
                df_k = pd.DataFrame(other_data[k])
                df_k['value'] = pd.to_numeric(df_k['value'], errors='coerce')
                
                # Check the last 24 hours first
                recent_k = df_k[df_k['ts'] > (now_ts - 24*3600*1000)]
                if len(recent_k) > 5 and recent_k['value'].max() == recent_k['value'].min():
                    locked_val = recent_k['value'].iloc[0]
                    
                    # Trace back to see exactly how long it's been locked
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
        "battery": current_battery,
        "drain_per_day": drain_per_day,
        "est_days": est_days,
        "anomalies": anomalies,
        "tamper": tamper,
        "tamper_time": tamper_time,
        "is_plugged_in": is_pc_or_wo,
        "is_weather": is_weather
    }

# ==========================================
# TOOL 1: TARGETED DEVICE AUDIT
# ==========================================

class DeviceAuditInput(BaseModel):
    target: CampusRooms = Field(..., description="The specific room to run diagnostics on.")
    sensor_type: Optional[str] = Field(None, description="Optional. Filter by sensor type (e.g., 'IAQ', 'PC', 'DESK').")

@tool("run_device_diagnostic_audit", args_schema=DeviceAuditInput)
def run_device_diagnostic_audit(target: CampusRooms, sensor_type: Optional[str] = None) -> str:
    """
    Deep-dive diagnostic audit for a specific room. 
    Checks Connectivity (Offline/RSSI), Power (Battery Drain), and Hardware Health (Flatlines/Tampers).
    """
    if sensor_type:
        devices = registry.get_devices_by_room_and_type(target, sensor_type)
        filter_str = f"  Sensor_Filter: {sensor_type.upper()}"
    else:
        devices = registry.get_all_devices_in_room(target)
        filter_str = "  Sensor_Filter: None"
        
    if not devices:
        return f"Error: No devices found in room '{target}'."

    current_time_str = datetime.now().strftime("%A, %b %d, %Y at %I:%M:%S %p EEST")
    total_scanned = len(devices)
    
    online_count = 0
    offline_lines = []
    
    battery_vals = []
    plugged_in_count = 0
    power_warnings = []
    
    operational_count = 0
    anomaly_lines = []
    tamper_lines = []

    for name, uid in devices.items():
        data = _audit_device(name, uid)
        
        # 1. Connectivity
        if data["is_online"]:
            online_count += 1
        else:
            offline_lines.append(f"    - {name}: (Offline {data['offline_duration_str']})")
            continue # If offline, suppress all other false-positive warnings
            
        # 2. Power
        if data["is_plugged_in"]:
            plugged_in_count += 1
        elif data["battery"] is not None:
            if data["is_weather"]:
                if data["battery"] < 2.4:
                    power_warnings.append(f"    - {name}: {data['battery']:.2f}V remaining | Est. {int(data['est_days'])} days")
            else:
                battery_vals.append(data["battery"])
                if data["battery"] < 15 or data["est_days"] < 14:
                    power_warnings.append(f"    - {name}: {data['battery']:.1f}% remaining | Est. {int(data['est_days'])} days")
                
        # 3. Hardware
        has_anomaly = False
        if data["anomalies"]:
            has_anomaly = True
            for a in data["anomalies"]:
                anomaly_lines.append(f"    - {name}: {a}")
                
        if data["tamper"]:
            tamper_lines.append(f"    - {name}: Tamper alarm triggered at {data['tamper_time']}")
            
        if not has_anomaly and not data["tamper"]:
            operational_count += 1

    avg_bat = int(np.mean(battery_vals)) if battery_vals else 0
    plugged_str = f" | {plugged_in_count} Plugged In" if plugged_in_count > 0 else ""

    # Format Output
    output = [
        "Query_Context:",
        "  Domain: Diagnostics (Targeted Audit)",
        f"  Target_Room: {target}",
        filter_str,
        f"  Total_Devices_Scanned: {total_scanned}",
        f"  Current_Time: {current_time_str}",
        "",
        "Connectivity_Audit:",
        f"  Status: {online_count}/{total_scanned} Online"
    ]
    
    if offline_lines:
        output.append("  Offline_Devices:")
        output.extend(offline_lines)
    else:
        output.append("  Offline_Devices: None")
        
    output.extend([
        "",
        "Power_Audit:",
        f"  Aggregate_Health: {len(battery_vals)} Reporting (Avg Battery: {avg_bat}%){plugged_str}"
    ])
    
    if power_warnings:
        output.append("  Depletion_Warnings:")
        output.extend(power_warnings)
    else:
        output.append("  Depletion_Warnings: None")
        
    output.extend([
        "",
        "Hardware_Health_Audit:",
        f"  Aggregate_Health: {operational_count}/{online_count} Operational"
    ])
    
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

    return "\n".join(output)

# ==========================================
# TOOL 2: CAMPUS-WIDE SUMMARY
# ==========================================

class EmptyInput(BaseModel):
    pass

@tool("run_building_diagnostic_summary", args_schema=EmptyInput)
def run_building_diagnostic_summary() -> str:
    """
    Triage report for the entire campus. Aggregates all healthy data into single lines 
    and explicitly lists ONLY the devices that require human intervention.
    Runs concurrently for high performance.
    """
    current_time_str = datetime.now().strftime("%A, %b %d, %Y at %I:%M:%S %p EEST")
    
    all_rooms = registry.get_available_rooms()
    
    # Setup batch processing tasks
    tasks = []
    for room in all_rooms:
        for name, uid in registry.get_all_devices_in_room(room).items():
            tasks.append((room, name, uid))
            
    total_scanned = len(tasks)
    
    online_count = 0
    battery_vals = []
    
    offline_list = []
    power_list = []
    anomaly_list = []
    tamper_list = []

    # Execute all device checks simultaneously using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_task = {executor.submit(_audit_device, name, uid): (room, name) for room, name, uid in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            room, name = future_to_task[future]
            try:
                data = future.result()
                
                # Connectivity
                if data["is_online"]:
                    online_count += 1
                else:
                    offline_list.append(f"    - {room}: '{name}' (Offline {data['offline_duration_str']})")
                    continue # Skip power/hardware alerts for offline devices
                    
                # Power
                if not data["is_plugged_in"] and data["battery"] is not None:
                    if data["is_weather"]:
                        if data["battery"] < 2.4:
                            power_list.append(f"    - {room}: '{name}' ({data['battery']:.2f}V remaining | Est. {int(data['est_days'])} days)")
                    else:
                        battery_vals.append(data["battery"])
                        if data["battery"] < 15 or data["est_days"] < 14:
                            power_list.append(f"    - {room}: '{name}' ({data['battery']:.1f}% remaining | Est. {int(data['est_days'])} days)")
                        
                # Hardware
                if data["anomalies"]:
                    for a in data["anomalies"]:
                        anomaly_list.append(f"    - {room}: '{name}' {a}")
                        
                if data["tamper"]:
                    tamper_list.append(f"    - {room}: '{name}' (Casing opened at {data['tamper_time']})")
                    
            except Exception as e:
                logger.error(f"Concurrent execution failed for {name} in {room}: {e}")

    uptime_pct = (online_count / total_scanned * 100) if total_scanned > 0 else 0
    avg_bat = int(np.mean(battery_vals)) if battery_vals else 0
    overall_status = "ATTENTION_REQUIRED" if (offline_list or power_list or anomaly_list or tamper_list) else "HEALTHY"

    # Format Output
    output = [
        "Query_Context:",
        "  Domain: Diagnostics (Campus-Wide Summary)",
        "  Target: Entire Campus",
        f"  Total_Devices_Scanned: {total_scanned}",
        f"  Current_Time: {current_time_str}",
        "",
        "Global_Health_Overview:",
        f"  Total_Active: {online_count}/{total_scanned} ({uptime_pct:.1f}% Uptime)",
        f"  Campus_Average_Battery: {avg_bat}%",
        f"  Overall_Status: [{overall_status}]",
        "",
        "Actionable_Maintenance_Queue:"
    ]

    if offline_list:
        output.append(f"  Offline_Sensors ({len(offline_list)}):")
        output.extend(offline_list)
    else:
        output.append("  Offline_Sensors: None")
        
    if power_list:
        output.append(f"  Power_Depletion_Warnings ({len(power_list)}):")
        output.extend(power_list)
    else:
        output.append("  Power_Depletion_Warnings: None")
        
    if anomaly_list:
        output.append(f"  Hardware_Anomalies ({len(anomaly_list)}):")
        output.extend(anomaly_list)
    else:
        output.append("  Hardware_Anomalies: None")
        
    if tamper_list:
        output.append(f"  Tamper_Alarms ({len(tamper_list)}):")
        output.extend(tamper_list)
    else:
        output.append("  Tamper_Alarms: None")

    return "\n".join(output)

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Testing Diagnostics Tool...")
    print("-" * 50)
    
    try:
        print("\n[Testing Targeted Audit (Room 1.2)]")
        print(run_device_diagnostic_audit.invoke({"target": "1.2"}))

        print("\n[Testing Targeted Audit (Room entrance, PC sensors)]")
        print(run_device_diagnostic_audit.invoke({"target": "entrance", "sensor_type": "PC"}))

        print("\n[Testing Campus-Wide Summary]")
        print(run_building_diagnostic_summary.invoke({}))
        
        print("\n" + "-"*50)
        print("All Diagnostics tool tests completed successfully.")
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)