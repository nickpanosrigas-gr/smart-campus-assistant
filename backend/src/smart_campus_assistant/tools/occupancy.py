import pandas as pd
from typing import Literal, Dict, Any, List, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

Rooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7', 'building'
]

Timeframes = Literal[
    'now', '2h', '24h', '7d', '30d', '90d'
]

logger = logging.getLogger(__name__)

# Full-room capacities for threshold calculations
CAPACITIES = {
    'building': 600,
    'restaurant': 30,
    '1.2': 120,
    '2.3': 40,
    '2.4': 32,
    '3.9': 50
}

# Config mapping for API calls and pandas resampling
TIMEFRAME_CONFIG = {
    "now": {"method": "get_now", "bin_size": None, "prev_method": "get_now_prev_30d_full"},
    "2h":  {"method": "get_2h", "bin_size": "10min", "prev_method": "get_2h_prev_30d_full"},
    "24h": {"method": "get_24h", "bin_size": "2h", "prev_method": "get_24h_prev_30d_full"},
    "7d":  {"method": "get_7d", "bin_size": "2h", "prev_method": "get_7d_prev_30d_full"},    
    "30d": {"method": "get_30d", "bin_size": "2h", "prev_method": None},
    "90d": {"method": "get_90d", "bin_size": "2h", "prev_method": None}
}

CONTEXT_NAMES = {
    "weekday_work": "Weekdays (Mon-Fri) Working_Hours (08:00-22:00)",
    "weekday_nonwork": "Weekdays (Mon-Fri) Non-Working_Hours (22:00-08:00)",
    "weekend_work": "Weekends (Sat-Sun) Working_Hours (08:00-22:00)",
    "weekend_nonwork": "Weekends (Sat-Sun) Non-Working_Hours (22:00-08:00)"
}

def get_time_context(dt: pd.Timestamp) -> str:
    is_weekend = dt.dayofweek >= 5
    is_work = 8 <= dt.hour < 22
    if not is_weekend and is_work: return "weekday_work"
    if not is_weekend and not is_work: return "weekday_nonwork"
    if is_weekend and is_work: return "weekend_work"
    return "weekend_nonwork"

class OccupancyInput(BaseModel):
    room: Rooms = Field(
        ..., 
        description="The specific room to check for occupancy levels. MUST be one of the exact allowed room names. 'building' checks overall campus occupancy."
    )
    timeframe: Timeframes = Field(
        ..., 
        description="The time window for the data request. 'now' provides a real-time snapshot. '2h', '24h', '7d' provides data for that timeframe in smaller buckets. '30d' and '90d' provide long-term statistics."
    )

def fetch_and_resample(devices: Dict[str, Any], keys: List[str], fetch_method, bin_size: str, sensor_type: str, timeframe: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Helper to fetch telemetry for multiple devices, combine them, and resample.
    Includes a fallback for sparse sensors (like People Counters) that only report on state-change.
    Returns the aggregated series AND the detailed aligned dataframe.
    """
    all_dfs = []
    for device_name, device_data in devices.items():
        device_id = device_data.get("id") if isinstance(device_data, dict) else device_data
        if not device_id: continue
        
        try:
            raw_data = fetch_method(device_id, keys)
            for key in keys:
                if key in raw_data and raw_data[key]:
                    df = pd.DataFrame(raw_data[key])
                    df['value'] = pd.to_numeric(df['value'])
                    # Convert raw UTC milliseconds to Local Greek Time exactly as configured
                    df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert(settings.TIMEZONE).dt.tz_localize(None)
                    df.set_index('datetime', inplace=True)
                    df = df[['value']].rename(columns={'value': f"{device_name}_{key}"})
                    all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch data for {device_name}: {e}")

    # Ensure timeframe boundaries are captured
    end_ts = pd.Timestamp.now(tz=settings.TIMEZONE).tz_localize(None)
    td_map = {
        "2h": pd.Timedelta(hours=2), 
        "24h": pd.Timedelta(hours=24), 
        "7d": pd.Timedelta(days=7), 
        "30d": pd.Timedelta(days=30),
        "90d": pd.Timedelta(days=90)
    }
    start_ts = end_ts - td_map.get(timeframe, pd.Timedelta(hours=2))

    # SPARSE DATA FALLBACK
    if not all_dfs:
        if sensor_type == 'motion':
            return pd.Series(dtype=float), pd.DataFrame()
            
        last_known = {}
        for device_name, device_data in devices.items():
            device_id = device_data.get("id") if isinstance(device_data, dict) else device_data
            if not device_id: continue
            
            try:
                raw_now = tb_client.get_now(device_id, keys)
                for key in keys:
                    if key in raw_now and raw_now[key]:
                        last_known[f"{device_name}_{key}"] = float(raw_now[key][0]["value"])
            except Exception:
                pass
                
        if not last_known:
            return pd.Series(dtype=float), pd.DataFrame()
            
        if sensor_type == 'pc':
            last_known = {k: 0 for k in last_known}
            
        combined_df = pd.DataFrame([last_known, last_known], index=[start_ts, end_ts])
    else:
        combined_df = pd.concat(all_dfs, axis=1, sort=True)
        
        # Stretch timeframe to guarantee 100% of intervals show up in timeline
        if not combined_df.empty and timeframe != "now":
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            boundary_idx = pd.DatetimeIndex([start_ts, end_ts])
            full_idx = combined_df.index.union(boundary_idx).sort_values()
            combined_df = combined_df.reindex(full_idx)
            # Fill boundaries
            if sensor_type != 'pc':
                combined_df.ffill(inplace=True)
                combined_df.bfill(inplace=True) # Catch anything before the first datapoint

    if sensor_type != 'pc':
        combined_df.ffill(inplace=True)

    if sensor_type == 'pc':
        aligned_df = combined_df.resample(bin_size).sum() 
    else:
        aligned_df = combined_df.resample(bin_size).max()
        aligned_df.ffill(inplace=True)
        
    aligned_df.fillna(0, inplace=True)

    if sensor_type == 'desk':
        return aligned_df.sum(axis=1), aligned_df
    elif sensor_type == 'wo':
        return aligned_df.sum(axis=1), aligned_df
    elif sensor_type == 'motion':
        return aligned_df.max(axis=1), aligned_df
    elif sensor_type == 'pc':
        in_cols = [c for c in aligned_df.columns if 'period_in' in c]
        out_cols = [c for c in aligned_df.columns if 'period_out' in c]
        
        total_in = aligned_df[in_cols].sum(axis=1) if in_cols else pd.Series(0, index=aligned_df.index)
        total_out = aligned_df[out_cols].sum(axis=1) if out_cols else pd.Series(0, index=aligned_df.index)
        
        net_change = total_in - total_out
        
        # BOUNDED CUMULATIVE SUM WITH MIDNIGHT RESET (Fixes Multi-Day Sensor Drift)
        occ_series = pd.Series(0.0, index=net_change.index)
        if not net_change.empty:
            for day_date, day_group in net_change.groupby(net_change.index.date):
                current_occ = 0
                for idx, change in day_group.items():
                    current_occ = max(0, current_occ + change)
                    occ_series[idx] = current_occ
                    
        return occ_series, aligned_df

    return pd.Series(dtype=float), pd.DataFrame()

@tool("get_occupancy", args_schema=OccupancyInput, response_format="content_and_artifact")
def get_occupancy(room: Rooms, timeframe: Timeframes) -> Tuple[str, dict]:
    """
    Tracks room occupancy using a polymorphic schema. Automatically detects if the room uses 
    Desk Sensors, People Counters (PC), Area Wait Counters (WO), or Motion Sensors (IAQ).
    """
    room_key = str(room).strip().lower()
    
    # ==========================================
    # SENSOR TARGETING LOGIC
    # ==========================================
    if room_key == "building":
        # For the whole building, use Entrance PC as primary, and ALL campus IAQ as secondary
        entrance_devices = registry.get_all_devices_in_room("entrance")
        pc_devices = {k: v for k, v in entrance_devices.items() if "-PC" in k.upper()}
        wo_devices = {}
        desk_devices = {}
        iaq_devices = registry.get_all_devices_by_type("IAQ")
        
        if not pc_devices and not iaq_devices:
            error_msg = f"Query_Context:\n  Room: {room}\nError: No building-level sensors found."
            return error_msg, {"view_type": "error", "message": "No sensors found"}
    else:
        devices = registry.get_all_devices_in_room(room_key)
        if not devices:
            error_msg = f"Query_Context:\n  Room: {room}\nError: Room not found or has no devices."
            return error_msg, {"view_type": "error", "message": "Room empty"}

        pc_devices = {k: v for k, v in devices.items() if "-PC" in k.upper()}
        wo_devices = {k: v for k, v in devices.items() if "-WO" in k.upper()}
        desk_devices = {k: v for k, v in devices.items() if "-DESK" in k.upper()}
        iaq_devices = {k: v for k, v in devices.items() if "-IAQ" in k.upper()}
        
        # NICHE CASE: "entrance" should prioritize DESK sensors and ignore its PC sensor
        if room_key == "entrance":
            pc_devices = {}

    # CRITICAL: Precedence shifted so PC overrides Desk.
    if pc_devices:
        primary_type = "People_Counter (Synthesized from In/Out Traffic)"
        primary_devs = pc_devices
        primary_keys = ["line_1_period_in", "line_1_period_out"]
        sensor_category = "pc"
    elif wo_devices:
        primary_type = "Area_Wait_Counter (Continuous Count)"
        primary_devs = wo_devices
        primary_keys = ["people_count_max"]
        sensor_category = "wo"
    elif desk_devices:
        primary_type = "Desk_Contact (Binary)"
        primary_devs = desk_devices
        primary_keys = ["occupancy"]
        sensor_category = "desk"
    elif iaq_devices:
        primary_type = "IAQ_Motion (Binary Activity)"
        primary_devs = iaq_devices
        primary_keys = ["pir"]
        sensor_category = "motion"
    else:
        error_msg = f"Query_Context:\n  Room: {room}\nError: No occupancy or motion sensors found."
        return error_msg, {"view_type": "error", "message": "No valid sensors"}

    total_primary_sensors = len(primary_devs)
    
    # Bundle non-primary sensors into the support string
    support_parts = []
    if sensor_category != "motion" and iaq_devices:
        support_cnt = len(iaq_devices)
        support_parts.append(f"{support_cnt} Motion Sensor{'s' if support_cnt != 1 else ''}")
    if sensor_category != "desk" and desk_devices:
        support_cnt = len(desk_devices)
        support_parts.append(f"{support_cnt} Desk Sensor{'s' if support_cnt != 1 else ''}")
    
    support_sensors_str = ", ".join(support_parts) if support_parts else "None"

    # ==========================================
    # DESK ZONE EXTRACTION
    # ==========================================
    desk_zones_total = {}
    desk_device_to_zone = {}
    if desk_devices:
        for name, data in desk_devices.items():
            z = data.get("zone", "Unspecified")
            desk_zones_total[z] = desk_zones_total.get(z, 0) + 1
            desk_device_to_zone[name] = z

    # ==========================================
    # SERVER ATTRIBUTE ACTIVE/OFFLINE CHECK
    # ==========================================
    devices_to_check = {**primary_devs}
    if sensor_category != "motion":
        devices_to_check.update(iaq_devices)
    if sensor_category != "desk":
        devices_to_check.update(desk_devices)

    active_devices = {}
    offline_sensors = []
    
    for device_name, device_data in devices_to_check.items():
        device_id = device_data.get("id") if isinstance(device_data, dict) else device_data
        if not device_id:
            offline_sensors.append(device_name)
            continue
            
        try:
            attrs = tb_client.get_server_attributes(device_id, ["active"])
            is_active = any(attr.get("key") == "active" and str(attr.get("value")).lower() == "true" for attr in attrs)
            if is_active:
                active_devices[device_name] = device_data
            else:
                offline_sensors.append(device_name)
        except Exception as e:
            logger.warning(f"Could not fetch active status for {device_name}: {e}")
            offline_sensors.append(device_name)

    active_primary_devs = {k: v for k, v in primary_devs.items() if k in active_devices}
    active_iaq_devices = {k: v for k, v in iaq_devices.items() if k in active_devices}
    active_secondary_desk_devices = {k: v for k, v in desk_devices.items() if k in active_devices} if sensor_category != "desk" else {}

    desk_zones_active = {}
    if sensor_category == "desk":
        for name, data in active_primary_devs.items():
            z = desk_device_to_zone.get(name, "Unspecified")
            desk_zones_active[z] = desk_zones_active.get(z, 0) + 1

    # Format Active Sensor Context Block
    total_relevant = len(devices_to_check)
    active_count = len(active_devices)
    active_sensors_lines = []
    
    if sensor_category == "desk":
        active_sensors_lines.append(f"  Active_Sensors: {active_count}/{total_relevant} Online")
        for z, count in desk_zones_active.items():
            tot = desk_zones_total.get(z, count)
            active_sensors_lines.append(f"    - Group: {z} (DESK): {count}/{tot} Online")
            
        # Add Supporting IAQ sensors to Active Sensors
        for name, data in active_iaq_devices.items():
            z = data.get("zone", "Unspecified") if isinstance(data, dict) else "Unspecified"
            t = data.get("tag", "Unspecified") if isinstance(data, dict) else "Unspecified"
            active_sensors_lines.append(f"    - {name} (IAQ): Zone: {z}, Tag: {t}")
            
        if offline_sensors:
            active_sensors_lines.append("  Offline_Sensors:")
            for off_s in offline_sensors:
                if off_s in desk_device_to_zone:
                    z = desk_device_to_zone[off_s]
                    active_sensors_lines.append(f"    - {off_s} (DESK): Zone: {z}")
                elif off_s in iaq_devices:
                    data = iaq_devices[off_s]
                    z = data.get("zone", "Unspecified") if isinstance(data, dict) else "Unspecified"
                    active_sensors_lines.append(f"    - {off_s} (IAQ): Zone: {z}")
                else:
                    active_sensors_lines.append(f"    - {off_s}")
    else:
        active_sensors_lines.append(f"  Active_Sensors: {active_count}/{total_relevant} Online")
        for name, data in active_devices.items():
            if isinstance(data, dict):
                z = data.get("zone", "Unspecified")
                t = data.get("tag", "Unspecified")
            else:
                z, t = "Unspecified", "Unspecified"
            
            sens_type = "PC" if "-PC" in name.upper() else "WO" if "-WO" in name.upper() else "IAQ" if "-IAQ" in name.upper() else "SENSOR"
            active_sensors_lines.append(f"    - {name} ({sens_type}): Zone: {z}, Tag: {t}")

        if offline_sensors:
            active_sensors_lines.append(f"  Offline_Sensors: {', '.join(offline_sensors)}")

    now_ts = pd.Timestamp.now(tz=settings.TIMEZONE).tz_localize(None)
    has_active_motion = len(active_iaq_devices) > 0

    # ==========================================
    # DEEP HISTORICAL BASELINE FETCHING
    # ==========================================
    baselines = {c: "No data" for c in ['weekday_work', 'weekday_nonwork', 'weekend_work', 'weekend_nonwork']}
    
    if timeframe not in ["30d", "90d"]:
        try:
            # We fetch 30 days of data in 2h bins to construct a lightweight baseline DataFrame
            base_prim_series, base_prim_df = fetch_and_resample(active_primary_devs, primary_keys, tb_client.get_30d, "2h", sensor_type=sensor_category, timeframe="30d")
            base_mot_series, _ = fetch_and_resample(active_iaq_devices, ["pir"], tb_client.get_30d, "2h", sensor_type="motion", timeframe="30d")
            
            if not base_prim_series.empty:
                b_df = pd.DataFrame({"primary": base_prim_series})
                if not base_mot_series.empty and sensor_category != "motion":
                    b_df = b_df.join(base_mot_series.rename("motion"), how="outer")
                else:
                    b_df["motion"] = 0.0
                b_df.fillna(0, inplace=True)
                
                b_group_df = pd.DataFrame()
                if sensor_category == "desk" and not base_prim_df.empty:
                    desk_group_series = {}
                    for z in desk_zones_active.keys():
                        z_cols = [f"{d}_{primary_keys[0]}" for d, dz in desk_device_to_zone.items() if dz == z and d in active_primary_devs]
                        valid_z_cols = [c for c in z_cols if c in base_prim_df.columns]
                        if valid_z_cols:
                            desk_group_series[z] = base_prim_df[valid_z_cols].sum(axis=1)
                        else:
                            desk_group_series[z] = pd.Series(0, index=base_prim_series.index)
                    b_group_df = pd.DataFrame(desk_group_series)
                    if not b_group_df.empty:
                        b_group_df = b_group_df.reindex(b_df.index).fillna(0)

                idx = b_df.index
                is_wk = idx.dayofweek < 5
                is_work = (idx.hour >= 8) & (idx.hour < 22)
                
                masks = {
                    'weekday_work': is_wk & is_work,
                    'weekday_nonwork': is_wk & ~is_work,
                    'weekend_work': ~is_wk & is_work,
                    'weekend_nonwork': ~is_wk & ~is_work
                }
                
                for c_name, c_mask in masks.items():
                    c_df = b_df[c_mask]
                    if c_df.empty: continue
                    
                    if sensor_category == 'motion':
                        motion_pct = (c_df['primary'] > 0).mean() * 100
                        baselines[c_name] = f"Active {motion_pct:.0f}% of the time"
                    else:
                        p_occ = c_df['primary'].max()
                        a_occ = c_df['primary'].mean()
                        if pd.isna(p_occ): p_occ = 0
                        if pd.isna(a_occ): a_occ = 0
                        
                        if has_active_motion:
                            motion_pct = (c_df['motion'] > 0).mean() * 100
                            motion_ctx_str = f" | Motion Active: {motion_pct:.0f}%"
                        else:
                            motion_ctx_str = " | Motion: Offline"
                        
                        if sensor_category == "desk":
                            base_str = f"Peak: {p_occ:.0f}/{total_primary_sensors} Desks | Avg: {a_occ:.1f}/{total_primary_sensors} Desks{motion_ctx_str}"
                            if not b_group_df.empty:
                                c_group_df = b_group_df[c_mask]
                                group_avgs = c_group_df.mean()
                                g_strs = [f"{z}: {avg:.1f}/{desk_zones_total[z]}" for z, avg in group_avgs.items() if pd.notna(avg)]
                                if g_strs:
                                    base_str += f"\nGroup_Averages: {', '.join(g_strs)}"
                            baselines[c_name] = base_str
                        else:
                            baselines[c_name] = f"Peak: {p_occ:.0f} people | Avg: {a_occ:.1f} people{motion_ctx_str}"
        except Exception as e:
            logger.warning(f"Failed to fetch 30-day baseline for occupancy: {e}")

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        current_ctx = get_time_context(now_ts)
        output = [
            "Query_Context:",
            "  Domain: Occupancy",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Current_Time: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Active_Context: {current_ctx}",
            f"  Primary_Sensor: {primary_type}",
            f"  Supporting_Sensors: {support_sensors_str}"
        ]
        output.extend(active_sensors_lines)
        
        output.append("")
        output.append(f"Statistical_Baseline ({current_ctx}):")
        if baselines[current_ctx] == "No data":
            output.append("  Baseline: No data")
        else:
            b_lines = baselines[current_ctx].split('\n')
            output.append(f"  Baseline: {b_lines[0]}")
            if len(b_lines) > 1:
                output.append(f"  {b_lines[1]}")
                
        output.extend(["", "Current_State:"])
        
        primary_val = 0
        has_data = False
        group_counts = {z: 0 for z in desk_zones_active.keys()}
        
        ui_current_values = {}
        for off_s in offline_sensors:
            ui_current_values[f"{off_s}_status_color"] = "red"
            ui_current_values[off_s] = None
        
        # Primary Data Collection
        if active_primary_devs and sensor_category == "pc":
            recent_series, _ = fetch_and_resample(active_primary_devs, primary_keys, tb_client.get_24h, "10min", "pc", "24h")
            if not recent_series.empty:
                has_data = True
                primary_val = recent_series.iloc[-1]
                for name in active_primary_devs:
                    ui_current_values[name] = primary_val
                    ui_current_values[f"{name}_status_color"] = "green"
        else:
            for name, data in active_primary_devs.items():
                device_id = data.get("id") if isinstance(data, dict) else data
                raw = tb_client.get_now(device_id, primary_keys)
                key = primary_keys[0]
                if key in raw and raw[key]:
                    has_data = True
                    val = float(raw[key][0]["value"])
                    
                    if sensor_category == "wo":
                        ui_current_values[f"{name}_status_color"] = "green"
                    elif sensor_category == "desk":
                        if room_key == "entrance":
                            ui_current_values[f"{name}_status_color"] = "green" if val > 0 else "orange"
                        else:
                            ui_current_values[f"{name}_status_color"] = "orange" if val > 0 else "green"
                    elif sensor_category == "motion":
                        if room_key == "entrance":
                            ui_current_values[f"{name}_status_color"] = "green" if val > 0 else "orange"
                        else:
                            ui_current_values[f"{name}_status_color"] = "orange" if val > 0 else "green"
                    
                    if desk_devices:
                        ui_current_values[name] = "Occupied" if val > 0 else "Empty"
                        if val > 0: 
                            primary_val += 1
                            z = desk_device_to_zone.get(name, "Unspecified")
                            if z in group_counts:
                                group_counts[z] += 1
                    elif sensor_category == "motion":
                        ui_current_values[name] = "Active" if val > 0 else "Idle"
                        if val > 0: primary_val = 1
                    else:
                        ui_current_values[name] = val
                        primary_val += val
                else:
                    ui_current_values[f"{name}_status_color"] = "red"
                    ui_current_values[name] = None
                
        if has_data:
            if sensor_category == "desk":
                output.append(f"  Current_Occupancy: {int(primary_val)}/{total_primary_sensors} Desks Occupied")
                group_strs = []
                for z, c in group_counts.items():
                    tot = desk_zones_total.get(z, 0)
                    group_strs.append(f"{z}: {c}/{tot}")
                if group_strs:
                    output.append(f"  Group_Details: {', '.join(group_strs)}")
            elif sensor_category == "motion":
                output.append(f"  Current_State: {'Active' if primary_val > 0 else 'Idle'}")
            else:
                output.append(f"  Current_Occupancy: {int(primary_val)} people")
        else:
            output.append("  Primary_Status: Offline / No Data")

        # Secondary Sensors Evaluation
        if sensor_category != "motion":
            output.append("  Motion_Status:")
            if not active_iaq_devices:
                output.append("    No Active Motion Sensors")
            else:
                for name, data in active_iaq_devices.items():
                    device_id = data.get("id") if isinstance(data, dict) else data
                    raw = tb_client.get_now(device_id, ["pir"])
                    if "pir" in raw and raw["pir"]:
                        val = float(raw["pir"][0]["value"])
                        state_str = "Active" if val > 0 else "Idle"
                        output.append(f"    - {name}: {state_str}")
                        ui_current_values[name] = state_str
                        if room_key == "entrance":
                            ui_current_values[f"{name}_status_color"] = "green" if val > 0 else "orange"
                        else:
                            ui_current_values[f"{name}_status_color"] = "orange" if val > 0 else "green"
                    else:
                        output.append(f"    - {name}: Offline / No Data")
                        ui_current_values[name] = None
                        ui_current_values[f"{name}_status_color"] = "red"
                        
        if active_secondary_desk_devices:
            output.append("  Secondary_Desk_Status:")
            desk_occ = 0
            for name, data in active_secondary_desk_devices.items():
                device_id = data.get("id") if isinstance(data, dict) else data
                raw = tb_client.get_now(device_id, ["occupancy"])
                if "occupancy" in raw and raw["occupancy"]:
                    val = float(raw["occupancy"][0]["value"])
                    ui_current_values[name] = "Occupied" if val > 0 else "Empty"
                    ui_current_values[f"{name}_status_color"] = "orange" if val > 0 else "green"
                    if val > 0: desk_occ += 1
                else:
                    ui_current_values[name] = None
                    ui_current_values[f"{name}_status_color"] = "red"
            output.append(f"    {desk_occ}/{len(active_secondary_desk_devices)} Desks Occupied")
            
        # Determine Overall Room Status Color based on Capacity
        status_color = "green"
        if room_key == "entrance":
            if primary_val == 0:
                status_color = "orange"
        else:
            capacity = CAPACITIES.get(room_key) if sensor_category != "desk" else total_primary_sensors
            if capacity and capacity > 0:
                ratio = primary_val / capacity
                if ratio > 0.85:
                    status_color = "red"
                elif ratio > 0.60:
                    status_color = "orange"
                    
        artifact = {
            "view_type": "snapshot",
            "current_values": ui_current_values,
            "status_color": status_color
        }
        
        return "\n".join(output), artifact

    # ==========================================
    # BRANCH B: HISTORICAL DATA FETCH
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method = getattr(tb_client, config["method"])

    primary_series, primary_aligned_df = fetch_and_resample(active_primary_devs, primary_keys, fetch_method, bin_size, sensor_type=sensor_category, timeframe=timeframe)
    motion_series, _ = fetch_and_resample(active_iaq_devices, ["pir"], fetch_method, bin_size, sensor_type="motion", timeframe=timeframe)

    if primary_series.empty:
        error_msg = f"Query_Context:\n  Room: {room}\nError: No historical data found for timeframe {timeframe}. Check if sensor is actively transmitting."
        return error_msg, {"view_type": "graph", "series": [], "metadata": {}}

    df = pd.DataFrame({"primary": primary_series})
    if not motion_series.empty and sensor_category != "motion":
        df = df.join(motion_series.rename("motion"), how="outer")
    else:
        df["motion"] = 0.0

    df.fillna(0, inplace=True)
    
    # --- BUILD THE GRAPH ARTIFACT ---
    series_data = []
    for dt, row in df.iterrows():
        point = {"timestamp": dt.isoformat()}
        if pd.notna(row.get('primary')):
            point['Occupancy'] = float(row['primary'])
        if pd.notna(row.get('motion')):
            point['Motion'] = 1 if row['motion'] > 0 else 0
        if 'Occupancy' in point or 'Motion' in point:
            series_data.append(point)
            
    graph_artifact = {
        "view_type": "graph",
        "series": series_data,
        "metadata": {
            "Occupancy": "Count" if sensor_category != "motion" else "Active (1/0)",
            "Motion": "Active (1/0)"
        }
    }

    # Extract grouped DataFrames for Desks
    group_df = pd.DataFrame()
    if sensor_category == "desk" and not primary_aligned_df.empty:
        desk_group_series = {}
        for z in desk_zones_active.keys():
            z_cols = [f"{d}_{primary_keys[0]}" for d, dz in desk_device_to_zone.items() if dz == z and d in active_primary_devs]
            valid_z_cols = [c for c in z_cols if c in primary_aligned_df.columns]
            if valid_z_cols:
                desk_group_series[z] = primary_aligned_df[valid_z_cols].sum(axis=1)
            else:
                desk_group_series[z] = pd.Series(0, index=primary_series.index)
        group_df = pd.DataFrame(desk_group_series)
        
        # Align group_df index with the master df index
        if not group_df.empty:
            group_df = group_df.reindex(df.index).fillna(0)

    # ==========================================
    # BRANCH C: 30-DAY & 90-DAY STATISTICAL PROFILE
    # ==========================================
    if timeframe in ["30d", "90d"]:
        output = [
            "Query_Context:",
            "  Domain: Occupancy",
            f"  Room: {room}",
            f"  Timeframe: {timeframe} (Long-Term Statistical Profile)",
            f"  Current_Time: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Primary_Sensor: {primary_type}",
            f"  Supporting_Sensors: {support_sensors_str}"
        ]
        output.extend(active_sensors_lines)
        output.extend(["", "Schedule_Profiling_Matrix:"])
        
        is_weekday = df.index.dayofweek < 5
        is_weekend = df.index.dayofweek >= 5
        is_working_hours = (df.index.hour >= 8) & (df.index.hour < 22)
        is_non_working = (df.index.hour < 8) | (df.index.hour >= 22)
        
        def process_longterm_cell(cell_name, mask):
            cell_df = df[mask]
            if cell_df.empty:
                return [f"    {cell_name}:", "      Baseline: No data", "      Outliers: None"]
            
            lines = [f"    {cell_name}:"]
            outliers = []
            
            if sensor_category == 'motion':
                motion_pct = (cell_df['primary'] > 0).mean() * 100
                lines.append(f"      Baseline: Active {motion_pct:.0f}% of timeframe")
                
                daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
                for day, day_data in daily_groups:
                    if day_data.empty: continue
                    day_motion = (day_data['primary'] > 0).mean() * 100
                    
                    if abs(day_motion - motion_pct) >= 25:
                        day_str = day.strftime('%Y-%m-%d (%A)')
                        outliers.append(f"        - '{day_str}': Activity level at {day_motion:.0f}%")
            else:
                daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
                daily_peaks = {day: day_data['primary'].max() for day, day_data in daily_groups if not day_data.empty}
                daily_motions = {day: (day_data['motion'] > 0).mean() * 100 for day, day_data in daily_groups if not day_data.empty}
                
                avg_peak = sum(daily_peaks.values()) / len(daily_peaks) if daily_peaks else 0
                max_peak = max(daily_peaks.values()) if daily_peaks else 0

                if has_active_motion:
                    motion_pct = (cell_df['motion'] > 0).mean() * 100
                    motion_ctx_str = f" | Motion Active: {motion_pct:.0f}%"
                else:
                    motion_pct = 0
                    motion_ctx_str = " | Motion: Offline"
                
                if sensor_category == "desk":
                    stats = f"Avg Daily Peak: {avg_peak:.1f}/{total_primary_sensors} Desks | Max Peak: {max_peak:.0f}/{total_primary_sensors} Desks{motion_ctx_str}"
                    lines.append(f"      Baseline: {stats}")
                    
                    if not group_df.empty:
                        cell_group_df = group_df[mask]
                        group_avgs = cell_group_df.mean()
                        g_strs = [f"{z}: {avg:.1f}/{desk_zones_total[z]}" for z, avg in group_avgs.items() if pd.notna(avg)]
                        if g_strs:
                            lines.append(f"      Group_Averages: {', '.join(g_strs)}")
                    
                    for day, peak in daily_peaks.items():
                        day_motion = daily_motions.get(day, 0)
                        is_peak_outlier = abs(peak - avg_peak) >= max(3, avg_peak * 0.5)
                        is_motion_outlier = has_active_motion and abs(day_motion - motion_pct) >= 25
                        
                        if is_peak_outlier or is_motion_outlier:
                            day_str = day.strftime('%Y-%m-%d (%A)')
                            outlier_parts = []
                            if is_peak_outlier: outlier_parts.append(f"Peak reached {peak:.0f}/{total_primary_sensors} Desks")
                            if is_motion_outlier: outlier_parts.append(f"Motion Active: {day_motion:.0f}%")
                            outliers.append(f"        - '{day_str}': " + " | ".join(outlier_parts))
                else:
                    stats = f"Avg Daily Peak: {avg_peak:.1f} people | Max Peak: {max_peak:.0f} people{motion_ctx_str}"
                    lines.append(f"      Baseline: {stats}")
                    
                    for day, peak in daily_peaks.items():
                        day_motion = daily_motions.get(day, 0)
                        is_peak_outlier = abs(peak - avg_peak) >= max(5, avg_peak * 0.5)
                        is_motion_outlier = has_active_motion and abs(day_motion - motion_pct) >= 25
                        
                        if is_peak_outlier or is_motion_outlier:
                            day_str = day.strftime('%Y-%m-%d (%A)')
                            outlier_parts = []
                            if is_peak_outlier: outlier_parts.append(f"Peak reached {peak:.0f} people")
                            if is_motion_outlier: outlier_parts.append(f"Motion Active: {day_motion:.0f}%")
                            outliers.append(f"        - '{day_str}': " + " | ".join(outlier_parts))
                            
            if outliers:
                lines.append("      Outliers:")
                lines.extend(outliers)
            else:
                lines.append("      Outliers: None detected.")
                
            return lines

        output.append("  Weekdays (Mon-Fri):")
        output.extend(process_longterm_cell("Working_Hours (08:00-22:00)", is_weekday & is_working_hours))
        output.extend(process_longterm_cell("Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working))
        
        output.append("  Weekends (Sat-Sun):")
        output.extend(process_longterm_cell("Working_Hours (08:00-22:00)", is_weekend & is_working_hours))
        output.extend(process_longterm_cell("Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working))

        return "\n".join(output), graph_artifact

    # ==========================================
    # BRANCH D: 2h, 24h, 7d (TIMELINE LOGIC)
    # ==========================================
    peak_occ = df['primary'].max()
    avg_occ = df['primary'].mean()
    
    if sensor_category == "motion":
        global_motion_pct = (df['primary'] > 0).mean() * 100
        global_summary = f"Activity_Profile: Active {global_motion_pct:.0f}% of the time"
    else:
        if has_active_motion:
            global_motion_pct = (df['motion'] > 0).mean() * 100
            motion_ctx_str = f"Active {global_motion_pct:.0f}% / Idle {100-global_motion_pct:.0f}%"
        else:
            motion_ctx_str = "Offline / Unavailable"

        if sensor_category == "desk":
            global_summary = f"Peak_Occupancy: {peak_occ:.0f}/{total_primary_sensors} Desks | Avg_Occupancy: {avg_occ:.1f}/{total_primary_sensors} Desks\n  Motion_Context: {motion_ctx_str}"
            if not group_df.empty:
                group_avgs = group_df.mean()
                g_strs = [f"{z}: {avg:.1f}/{desk_zones_total[z]}" for z, avg in group_avgs.items() if pd.notna(avg)]
                if g_strs:
                    global_summary += f"\n  Group_Averages: {', '.join(g_strs)}"
        else:
            global_summary = f"Peak_Occupancy: {peak_occ:.0f} people | Avg_Occupancy: {avg_occ:.1f} people\n  Motion_Context: {motion_ctx_str}"

    output = [
        "Query_Context:",
        "  Domain: Occupancy",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        f"  Current_Time: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Primary_Sensor: {primary_type}",
        f"  Supporting_Sensors: {support_sensors_str}",
    ]
    output.extend(active_sensors_lines)
    output.extend(["", "Statistical_Baseline (Present Contexts):"])
    
    present_contexts = sorted(list(set(get_time_context(dt) for dt in df.index)))
    if not present_contexts:
        present_contexts = [get_time_context(now_ts)]
        
    for ctx in present_contexts:
        output.append(f"  {CONTEXT_NAMES.get(ctx, ctx)}:")
        if baselines[ctx] == "No data":
            output.append("    Baseline: No data")
        else:
            b_lines = baselines[ctx].split('\n')
            output.append(f"    Baseline: {b_lines[0]}")
            if len(b_lines) > 1:
                output.append(f"    {b_lines[1]}")

    output.extend(["", f"Global_Occupancy_Summary (Last {timeframe}):"])
    g_lines = global_summary.split('\n')
    for g_line in g_lines:
        if g_line.startswith("  "):
            output.append(g_line)
        else:
            output.append(f"  {g_line}")

    output.extend(["", "Timeline_Activity:"])

    daily_groups = df.groupby(pd.Grouper(freq='D'))
    for day_start, day_series in daily_groups:
        if day_series.empty: continue
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        
        transitions = []
        stable_periods = []
        
        current_stable_start = None
        current_stable_state = None
        stable_bins = 0
        prev_state = None
        prev_prim_state = None
        b_end = None
        
        # We process the pre-resampled DataFrame using itertuples/iterrows instead of grouping again
        for exact_time, row in day_series.iterrows():
            b_end = exact_time + pd.to_timedelta(bin_size)
            time_str = exact_time.strftime('%H:%M')
            bucket_end_str = b_end.strftime('%H:%M')
            
            if bucket_end_str == "00:00":
                bucket_end_str = "24:00"
            
            if sensor_category == "motion":
                motion_active = row['primary'] > 0
                state_str = "Active" if motion_active else "Idle"
                combined_state = state_str
                current_prim_state = state_str
            else:
                p_val = row['primary']
                
                if has_active_motion:
                    motion_active = row['motion'] > 0
                    motion_str = "Active" if motion_active else "Idle"
                else:
                    motion_str = "Offline"
                
                if sensor_category == "desk":
                    current_prim_state = f"{p_val:.0f}/{total_primary_sensors} Desks"
                    
                    if not group_df.empty and exact_time in group_df.index:
                        g_row = group_df.loc[exact_time]
                        g_strs = [f"{z}: {val:.0f}/{desk_zones_total[z]}" for z, val in g_row.items() if pd.notna(val)]
                        if g_strs:
                            current_prim_state += f" [{', '.join(g_strs)}]"
                else:
                    current_prim_state = f"{p_val:.0f} people"
                    
                combined_state = f"Status: {current_prim_state}. Motion: {motion_str}."

            if prev_state is None:
                current_stable_start = time_str
                current_stable_state = combined_state
                prev_state = combined_state
                prev_prim_state = current_prim_state
                stable_bins = 1
            elif combined_state != prev_state:
                # Flush previous stable period up to the start of this transition
                if stable_bins > 0:
                    stable_periods.append(f"      - '{current_stable_start} to {time_str}' ({stable_bins} intervals): {current_stable_state}")
                    
                transitions.append(f"      - bucket: '{time_str} - {bucket_end_str}'")
                if sensor_category == "motion":
                    transitions.append(f"        activity: 'Transitioned to {state_str}'")
                else:
                    if current_prim_state != prev_prim_state:
                        transitions.append(f"        activity: 'Transitioned to {current_prim_state}'")
                        transitions.append(f"        motion_state: '{motion_str}'")
                    else:
                        transitions.append(f"        motion_state: '{motion_str}'")
                
                # Setup the next stable period to start AFTER this transition bucket
                current_stable_start = bucket_end_str
                current_stable_state = combined_state
                prev_state = combined_state
                prev_prim_state = current_prim_state
                stable_bins = 0
            else:
                if current_stable_start is None:
                    current_stable_start = time_str
                stable_bins += 1

        if stable_bins > 0:
            end_str = b_end.strftime('%H:%M') if b_end is not None else "24:00"
            if end_str == "00:00": end_str = "24:00"
            stable_periods.append(f"      - '{current_stable_start} to {end_str}' ({stable_bins} intervals): {current_stable_state}")

        output.append(f"  '{day_key}':")
        if not transitions:
            output.append("    Timeline_Transitions: None (State was stable)")
        else:
            output.append("    Timeline_Transitions:")
            output.extend(transitions)
            
        if not stable_periods:
            output.append("    Stable_Periods: None")
        else:
            output.append("    Stable_Periods:")
            output.extend(stable_periods)

    return "\n".join(output), graph_artifact

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Occupancy Tool...")
    print("-" * 50)
    try:
        print("\n[Testing]")
        summary, raw_data = get_occupancy.func(room="1.2", timeframe="30d")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError during execution: {e}")