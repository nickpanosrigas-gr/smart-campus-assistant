import pandas as pd
from typing import Literal, Dict, Any, List, Optional, Tuple
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

# Config mapping for API calls and pandas grouping
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

def parse_magnet_status(val: Any) -> bool:
    """Parses raw telemetry into a boolean True (Open) or False (Closed)."""
    if pd.isna(val):
        return False
    if str(val).lower() in ["true", "1", "open"]:
        return True
    return False

def get_time_context(dt: pd.Timestamp) -> str:
    is_weekend = dt.dayofweek >= 5
    is_work = 8 <= dt.hour < 22
    if not is_weekend and is_work: return "weekday_work"
    if not is_weekend and not is_work: return "weekday_nonwork"
    if is_weekend and is_work: return "weekend_work"
    return "weekend_nonwork"

def get_state_label(is_open: bool) -> str:
    return "Open" if is_open else "Closed"

def format_binary_distribution(series: pd.Series) -> str:
    """Calculates the percentage of Open vs Closed for a boolean series."""
    if series.empty:
        return "No data"
    counts = series.value_counts(normalize=True) * 100
    closed_pct = counts.get(False, 0.0)
    open_pct = counts.get(True, 0.0)
    
    if closed_pct == 100: return "Closed: 100%, Open: 0%"
    if open_pct == 100: return "Closed: 0%, Open: 100%"
    
    return f"Closed: {closed_pct:.0f}%, Open: {open_pct:.0f}%"

class DoorWindowInput(BaseModel):
    room: Rooms = Field(
        ..., 
        description="The specific room to check for door/window access states. MUST be one of the exact allowed room names."
    )
    timeframe: Timeframes = Field(
        ..., 
        description="The time window for the data request. 'now' provides a real-time snapshot. '2h', '24h', '7d' provide timelines. '30d', '90d' provide long-term profiling."
    )

@tool("get_door_window_status", args_schema=DoorWindowInput, response_format="content_and_artifact")
def get_door_window_status(room: Rooms, timeframe: Timeframes) -> Tuple[str, dict]:
    """
    Tracks physical access points (Doors/Windows) using Magnetic Contact (MC) sensors.
    Reports Open/Closed states, timelines of physical entry, and long-term anomalies.
    """
    room_str = str(room).lower()
    
    if room_str == 'building':
        floor_val = "B"
        all_mc_devices = registry.get_all_devices_by_type("MC")
    else:
        floor_val = str(room)[0] if str(room)[0].isdigit() else "0"
        all_mc_devices = registry.get_devices_by_room_and_type(room, "MC")
    
    if not all_mc_devices:
        error_msg = f"Query_Context:\n  Room: {room}\nError: No MC (Door/Window) sensors found in this target."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "error",
                "domain": "Doors/Windows",
                "floor": floor_val,
                "room_id": str(room),
                "message": "No MC sensors found"
            }
        }

    active_mc_devices = {}
    offline_sensors = []
    
    for device_name, device_data in all_mc_devices.items():
        device_id = device_data.get("id")
        if not device_id: 
            offline_sensors.append(device_name)
            continue
            
        try:
            attrs = tb_client.get_server_attributes(device_id, ["active"])
            is_active = any(attr.get("key") == "active" and str(attr.get("value")).lower() == "true" for attr in attrs)
                    
            if is_active:
                active_mc_devices[device_name] = device_data
            else:
                offline_sensors.append(device_name)
        except Exception as e:
            logger.warning(f"Could not fetch active status for {device_name}: {e}")
            offline_sensors.append(device_name)

    if not active_mc_devices:
        error_msg = f"Query_Context:\n  Room: {room}\nError: Found {len(all_mc_devices)} MC sensors, but all are offline."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "error",
                "domain": "Doors/Windows",
                "floor": floor_val,
                "room_id": str(room),
                "message": "All sensors offline"
            }
        }

    total_count = len(all_mc_devices)
    active_count = len(active_mc_devices)
    
    header_lines = [
        "Query_Context:",
        "  Domain: Door & Window Access State (Open/Closed)",
        f"  Room: {room.upper()}",
    ]
    
    if timeframe == "now": header_lines.append("  Timeframe: Now (Snapshot)")
    elif timeframe in ["30d", "90d"]: header_lines.append(f"  Timeframe: {timeframe} (Long-Term Statistical Profile)")
    else: header_lines.append(f"  Timeframe: {timeframe} ({TIMEFRAME_CONFIG[timeframe]['bin_size']} intervals)")

    now_ts_for_ctx = pd.Timestamp.now(tz=settings.TIMEZONE)
    header_lines.append(f"  Current_Time: {now_ts_for_ctx.strftime('%Y-%m-%d %H:%M:%S')}")
    header_lines.append(f"  Active_Context: {get_time_context(now_ts_for_ctx)}")
    header_lines.append(f"  Active_Sensors: {active_count}/{total_count} Online")
    
    doors = []
    windows = []
    for d_name, d_data in active_mc_devices.items():
        grp = d_data.get("group", "Unknown")
        tag = d_data.get("tag", "Unspecified")
        
        # Add room labels if evaluating the whole building
        room_label = d_data.get("room", "Unknown") if room_str == "building" else "Unspecified"
        context_str = f"Room: {room_label}, Tag: {tag}" if room_str == "building" else f"Zone: Unspecified, Tag: {tag}"
        
        header_lines.append(f"    - {d_name} ({grp}): {context_str}")
        if "door" in grp.lower(): doors.append(d_name)
        if "window" in grp.lower(): windows.append(d_name)
        
    if offline_sensors:
        header_lines.append(f"  Offline_Sensors: {', '.join(offline_sensors)}")

    # ==========================================
    # HISTORICAL BASELINE FETCHING
    # ==========================================
    baselines = {c: {'doors': "No data", 'windows': "No data"} for c in ['weekday_work', 'weekday_nonwork', 'weekend_work', 'weekend_nonwork']}
    if timeframe not in ["30d", "90d"]:
        prev_method_name = TIMEFRAME_CONFIG[timeframe]["prev_method"]
        if prev_method_name and hasattr(tb_client, prev_method_name):
            fetch_prev = getattr(tb_client, prev_method_name)
            raw_bases = []
            for d_name, d_data in active_mc_devices.items():
                try:
                    raw_bases.append(fetch_prev(d_data.get("id"), ["magnet_status"]))
                except Exception:
                    raw_bases.append({})
            
            contexts = ['weekday_work', 'weekday_nonwork', 'weekend_work', 'weekend_nonwork']
            collected = {c: {'doors': [], 'windows': []} for c in contexts}
            
            for base_idx, d_name in enumerate(active_mc_devices.keys()):
                if base_idx >= len(raw_bases) or not raw_bases[base_idx]: continue
                base = raw_bases[base_idx]
                
                grp = active_mc_devices[d_name].get("group", "Unknown").lower()
                cat = "doors" if "door" in grp else "windows" if "window" in grp else None
                if not cat: continue
                
                if "magnet_status" in base and isinstance(base["magnet_status"], dict):
                    for c in contexts:
                        if c in base["magnet_status"]:
                            data = base["magnet_status"][c]
                            if not isinstance(data, list): data = [data]
                            for item in data:
                                val = item.get('value') if isinstance(item, dict) else item
                                if val is not None:
                                    collected[c][cat].append(parse_magnet_status(val))
            
            for c in contexts:
                baselines[c]['doors'] = format_binary_distribution(pd.Series(collected[c]['doors']))
                baselines[c]['windows'] = format_binary_distribution(pd.Series(collected[c]['windows']))

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        current_ctx = get_time_context(now_ts_for_ctx)
        output = list(header_lines)
        output.extend([
            "",
            f"Statistical_Baseline ({current_ctx}):",
            f"  Doors Baseline: {baselines[current_ctx]['doors']}",
            f"  Windows Baseline: {baselines[current_ctx]['windows']}",
            "",
            "Current_State:"
        ])
        
        # --- NEW NESTED ARTIFACT LOGIC ---
        ui_sensors = {}
        
        # 1. Offline Sensors processing
        for device_name in offline_sensors:
            ui_sensors[device_name] = {
                "status": "error",
                "category": "MC",
                "readings": None
            }
        
        open_count = 0
        open_doors_count = 0
        open_windows_count = 0
        
        # 2. Active Sensors processing & text output
        for device_name, device_data in active_mc_devices.items():
            device_id = device_data.get("id")
            tag = device_data.get("tag", "Unspecified")
            
            raw_data = tb_client.get_now(device_id, ["magnet_status"])
            if "magnet_status" in raw_data and raw_data["magnet_status"]:
                is_open = parse_magnet_status(raw_data["magnet_status"][0]["value"])
                
                # Keep text intact for LLM
                output.append(f"  {device_name} ({tag}): {get_state_label(is_open)}")
                
                # Update counters and status (Open = good, Closed = critical)
                if is_open:
                    open_count += 1
                    if device_name in doors: open_doors_count += 1
                    if device_name in windows: open_windows_count += 1
                    sensor_status = "good"
                else:
                    sensor_status = "critical"
                    
                ui_sensors[device_name] = {
                    "status": sensor_status,
                    "category": "MC",
                    "readings": {"magnet_status": "Open" if is_open else "Closed"}
                }
            else:
                # Keep text intact for LLM
                output.append(f"  {device_name} ({tag}): No Data")
                ui_sensors[device_name] = {
                    "status": "error",
                    "category": "MC",
                    "readings": None
                }
                
        ui_aggregates = {
            "total_count": total_count,
            "open_count": open_count,
            "total_doors": len(doors),
            "open_doors": open_doors_count,
            "total_windows": len(windows),
            "open_windows": open_windows_count
        }
                
        # 3. Overall Room Status Logic 
        if total_count > 0:
            closed_count = total_count - open_count
            if open_count >= closed_count:
                overall_status = "good"
            else:
                overall_status = "critical"
        else:
            overall_status = "error" 

        artifact = {
            "type": "map_update",
            "artifact": {
                "view_type": "snapshot",
                "domain": "Doors/Windows",
                "floor": floor_val,
                "room_id": str(room),
                "status": overall_status,
                "room_aggregates": ui_aggregates,
                "sensors": ui_sensors
            }
        }
                
        return "\n".join(output), artifact

    # ==========================================
    # BRANCH B: HISTORICAL DATA FETCH & ALIGNMENT
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method = getattr(tb_client, config["method"])

    now_ts = pd.Timestamp.now(tz=settings.TIMEZONE)
    if timeframe == "2h": start_ts = now_ts - pd.Timedelta(hours=2)
    elif timeframe == "24h": start_ts = now_ts - pd.Timedelta(hours=24)
    elif timeframe == "7d": start_ts = now_ts - pd.Timedelta(days=7)
    elif timeframe == "30d": start_ts = now_ts - pd.Timedelta(days=30)
    elif timeframe == "90d": start_ts = now_ts - pd.Timedelta(days=90)

    all_dfs = []
    for device_name, device_data in active_mc_devices.items():
        device_id = device_data.get("id")
        try:
            # Always get current state as fallback boundary
            now_raw = tb_client.get_now(device_id, ["magnet_status"])
            current_state = False
            if "magnet_status" in now_raw and now_raw["magnet_status"]:
                current_state = parse_magnet_status(now_raw["magnet_status"][0]["value"])

            raw_data = fetch_method(device_id, ["magnet_status"])
            df = pd.DataFrame()
            if "magnet_status" in raw_data and raw_data["magnet_status"]:
                df = pd.DataFrame(raw_data["magnet_status"])
                df['value'] = df['value'].apply(parse_magnet_status)
                df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert(settings.TIMEZONE)
                df.set_index('datetime', inplace=True)
                df.rename(columns={'value': device_name}, inplace=True)
                df.drop(columns=['ts'], inplace=True)
                df = df.sort_index()

            first_state = df.iloc[0][device_name] if not df.empty else current_state
            
            boundary_df = pd.DataFrame({
                device_name: [first_state, current_state]
            }, index=[start_ts, now_ts])
            
            dev_df = pd.concat([boundary_df, df])
            dev_df = dev_df.sort_index()
            dev_df = dev_df[~dev_df.index.duplicated(keep='last')]
            all_dfs.append(dev_df)
            
        except Exception as e:
            logger.warning(f"Failed to fetch historical MC data for {device_name}: {e}")

    if not all_dfs:
        error_msg = "\n".join(header_lines) + f"\n\nError: No historical data found for {timeframe}."
        return error_msg, {
            "type": "map_update",
            "artifact": {
                "view_type": "graph",
                "domain": "Doors/Windows",
                "floor": floor_val,
                "room_id": str(room),
                "timeframe": timeframe,
                "online_sensors": list(active_mc_devices.keys()),
                "offline_sensors": offline_sensors,
                "series": [],
                "metadata": {}
            }
        }

    # Merge all sensors
    combined_df = pd.concat(all_dfs, axis=1, sort=True)
    combined_df = combined_df.ffill().bfill()
    combined_df = combined_df.resample('1min').ffill()
    
    # Create the timeline Graph Artifact
    # Drop calculated groups for the pure visual payload, only keep sensor columns
    ui_df = combined_df[[col for col in combined_df.columns if col in active_mc_devices.keys()]].copy()
    
    # --- NEW: Binning & Delta (Change-Only) logic for the graph artifact ---
    if room_str == 'building':
        artifact_df = pd.DataFrame(index=ui_df.index)
        d_cols = [c for c in doors if c in ui_df.columns]
        w_cols = [c for c in windows if c in ui_df.columns]
        
        artifact_df['open_doors'] = ui_df[d_cols].sum(axis=1) if d_cols else 0
        artifact_df['open_windows'] = ui_df[w_cols].sum(axis=1) if w_cols else 0
        
        if timeframe in ["30d", "90d"]:
            artifact_df = artifact_df.resample('1D').max()
            
        metadata = {"open_doors": "Open Doors Count", "open_windows": "Open Windows Count"}
    else:
        if timeframe in ["30d", "90d"]:
            # For 30d/90d, show 1 if it was opened at any point during that day
            artifact_df = ui_df.resample('1D').max()
        else:
            # Use the base DataFrame for short timeframes to preserve exact minute changes
            artifact_df = ui_df
            
        metadata = {col: "State (1=Open, 0=Closed)" for col in artifact_df.columns}
        
    series_data = []
    # Track the last value sent to the frontend for each sensor/aggregate to apply delta logic
    last_sent_values = {col: None for col in artifact_df.columns}
    
    for dt, row in artifact_df.iterrows():
        point = {"timestamp": dt.isoformat()}
        
        for col in artifact_df.columns:
            val = row[col]
            if pd.notna(val):
                if room_str == 'building':
                    mapped_val = int(val) # Send the summed integer
                else:
                    mapped_val = 1 if val else 0  # Map boolean to 1 (Open) or 0 (Closed)
                
                # Only include this metric in the payload if its value CHANGED
                if mapped_val != last_sent_values[col]:
                    point[col] = mapped_val
                    last_sent_values[col] = mapped_val
                    
        # Only append the timestamp to the array if at least ONE metric changed state
        if len(point) > 1:
            series_data.append(point)
            
    graph_artifact = {
        "type": "map_update",
        "artifact": {
            "view_type": "graph",
            "domain": "Doors/Windows",
            "floor": floor_val,
            "room_id": str(room),
            "timeframe": timeframe,
            "online_sensors": list(active_mc_devices.keys()),
            "offline_sensors": offline_sensors,
            "series": series_data,
            "metadata": metadata
        }
    }

    # Create Logical Groups
    if doors: combined_df['Group_Doors'] = combined_df[doors].any(axis=1)
    else: combined_df['Group_Doors'] = pd.Series(False, index=combined_df.index)
        
    if windows: combined_df['Group_Windows'] = combined_df[windows].any(axis=1)
    else: combined_df['Group_Windows'] = pd.Series(False, index=combined_df.index)

    idx = combined_df.index
    is_wk = idx.dayofweek < 5
    is_work = (idx.hour >= 8) & (idx.hour < 22)

    # ==========================================
    # BRANCH C: 30D / 90D PROFILE
    # ==========================================
    if timeframe in ["30d", "90d"]:
        output = list(header_lines)
        avg_title = "Total_Monthly_Average:" if timeframe == "30d" else "Total_Quarterly_Average:"
        
        output.extend([
            "", avg_title,
            f"  Doors: {format_binary_distribution(combined_df['Group_Doors'].dropna())}",
            f"  Windows: {format_binary_distribution(combined_df['Group_Windows'].dropna())}",
            "", "Schedule_Profiling_Matrix:"
        ])
        
        def process_profile_cell(name: str, mask: pd.Series, is_non_working: bool):
            cell_doors = combined_df.loc[mask, 'Group_Doors'].dropna()
            cell_windows = combined_df.loc[mask, 'Group_Windows'].dropna()
            
            lines = [f"    {name}:"]
            if cell_doors.empty and cell_windows.empty:
                lines.append("      Baseline: No data")
                lines.append("      Outliers: None detected.")
                return lines
                
            lines.append("      Baseline:")
            lines.append(f"        Doors: {format_binary_distribution(cell_doors)}")
            lines.append(f"        Windows: {format_binary_distribution(cell_windows)}")
            
            outliers = []
            if is_non_working and not combined_df[mask].empty:
                daily = combined_df[mask].groupby(pd.Grouper(freq='D'))
                for day, day_data in daily:
                    if len(day_data) < 120: continue
                    day_str = day.strftime('%Y-%m-%d (%A)')
                    
                    for dev_name in active_mc_devices.keys():
                        if dev_name not in day_data: continue
                        
                        open_pct = day_data[dev_name].mean()
                        tag = active_mc_devices[dev_name].get('tag', dev_name)
                        is_door = "door" in active_mc_devices[dev_name].get("group", "").lower()
                        
                        toggles = (day_data[dev_name] != day_data[dev_name].shift()).sum()
                        
                        if is_door and toggles > 0:
                            outliers.append(f"        - '{day_str}': Security_Flag: [{tag}] accessed ({toggles} times).")
                        elif not is_door and open_pct > 0.5:
                            outliers.append(f"        - '{day_str}': Energy_Flag: [{tag}] left Open deeply into the night.")

            if outliers:
                lines.append("      Outliers:")
                lines.extend(outliers)
            else:
                lines.append("      Outliers: None detected.")
            return lines

        output.append("  Weekdays (Mon-Fri):")
        output.extend(process_profile_cell("Working_Hours (08:00-22:00)", is_wk & is_work, False))
        output.extend(process_profile_cell("Non-Working_Hours (22:00-08:00)", is_wk & ~is_work, True))
        
        output.append("  Weekends (Sat-Sun):")
        output.extend(process_profile_cell("Working_Hours (08:00-22:00)", ~is_wk & is_work, False))
        output.extend(process_profile_cell("Non-Working_Hours (22:00-08:00)", ~is_wk & ~is_work, True))
        
        return "\n".join(output), graph_artifact

    # ==========================================
    # BRANCH D: TIMELINES (2h, 24h, 7d)
    # ==========================================
    output = list(header_lines)
    output.extend(["", "Statistical_Baseline (Present Contexts):"])

    present_contexts = sorted(list(set(get_time_context(dt) for dt in combined_df.index)))
    if not present_contexts:
        present_contexts = [get_time_context(pd.Timestamp.now(tz=settings.TIMEZONE))]
        
    for ctx in present_contexts:
        output.append(f"  {CONTEXT_NAMES.get(ctx, ctx)}:")
        output.append(f"    Doors Baseline: {baselines[ctx]['doors']}")
        output.append(f"    Windows Baseline: {baselines[ctx]['windows']}")

    output.extend(["", "Timeline_Activity:"])

    combined_df['All_Closed'] = ~(combined_df['Group_Doors'] | combined_df['Group_Windows'])
    sensor_states = {dev: None for dev in active_mc_devices.keys()}
    
    daily_groups = combined_df.groupby(pd.Grouper(freq='D'))
    for day_start, day_series in daily_groups:
        if day_series.empty: continue
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        
        day_transitions = []
        day_stable = []
        day_outliers = []
        
        nw_mask = (day_series.index.hour < 8) | (day_series.index.hour >= 22)
        nw_data = day_series[nw_mask]
        
        if not nw_data.empty and len(nw_data) >= 120:
            for dev in active_mc_devices.keys():
                if dev not in nw_data: continue
                
                tag = active_mc_devices[dev].get('tag', dev)
                is_door = "door" in active_mc_devices[dev].get("group", "").lower()
                open_pct = nw_data[dev].mean()
                
                if open_pct > 0:
                    toggles = (nw_data[dev] != nw_data[dev].shift()).sum()
                    if is_door and toggles > 0:
                        day_outliers.append(f"Security_Flag: [{tag}] accessed during non-working hours ({toggles} transitions).")
                    elif not is_door and open_pct > 0.5:
                        day_outliers.append(f"Energy_Flag: [{tag}] left Open for >50% of Non-Working Hours.")
                        
        stable_count = 0
        current_stable_start = None
        current_stable_state = None
        b_end = None
        
        bucket_groups = day_series.groupby(pd.Grouper(freq=bin_size))
        
        for bucket_start, group in bucket_groups:
            if group.empty: continue
            
            b_end = bucket_start + pd.to_timedelta(bin_size)
            b_label = f"{bucket_start.strftime('%H:%M')} - {b_end.strftime('%H:%M')}"
            
            bucket_activity = []
            bucket_toggled_any = False
            
            for dev_name, dev_data in active_mc_devices.items():
                if dev_name not in group: continue
                tag = dev_data.get('tag', dev_name)
                dev_transitions = []
                toggle_count = 0
                
                for exact_time, is_open in group[dev_name].items():
                    if pd.isna(is_open): continue
                    current_s = "Open" if is_open else "Closed"
                    
                    if sensor_states[dev_name] is None:
                        sensor_states[dev_name] = current_s
                        
                    if current_s != sensor_states[dev_name]:
                        t_str = exact_time.strftime('%H:%M')
                        dev_transitions.append(f"Transition: [{sensor_states[dev_name]} -> {current_s} at {t_str}].")
                        sensor_states[dev_name] = current_s
                        toggle_count += 1
                        bucket_toggled_any = True
                
                if toggle_count > 4:
                    bucket_activity.append(f"[{tag}]: Fluctuating heavily (Toggled {toggle_count} times).")
                elif dev_transitions:
                    bucket_activity.append(f"[{tag}]: " + " ".join(dev_transitions))

            if bucket_activity:
                day_transitions.append(f"      - bucket: '{b_label}'\n        activity: '{' '.join(bucket_activity)}'")
                
            is_all_closed_bucket = group['All_Closed'].all()
            bucket_state_str = "All Closed" if is_all_closed_bucket else "Partial Open/Activity"
            
            if not bucket_toggled_any:
                if current_stable_start is None:
                    current_stable_start = bucket_start.strftime('%H:%M')
                    current_stable_state = bucket_state_str
                stable_count += 1
            else:
                if stable_count > 0:
                    day_stable.append({
                        "start": current_stable_start,
                        "end": bucket_start.strftime('%H:%M'),
                        "intervals": stable_count,
                        "state": current_stable_state
                    })
                stable_count = 0
                current_stable_start = None

        if stable_count > 0:
            end_time_str = b_end.strftime('%H:%M') if b_end is not None else "24:00"
            if end_time_str == "00:00": end_time_str = "24:00"
            
            day_stable.append({
                "start": current_stable_start,
                "end": end_time_str,
                "intervals": stable_count,
                "state": current_stable_state
            })

        output.append(f"  '{day_key}':")
        if day_outliers:
            output.append("    Outliers:")
            for o in day_outliers: output.append(f"      - {o}")
            
        if not day_transitions:
            output.append("    Timeline_Transitions: None")
        else:
            output.append("    Timeline_Transitions:")
            output.extend(day_transitions)
            
        if not day_stable:
            output.append("    Stable_Periods: None")
        else:
            output.append("    Stable_Periods:")
            for p in day_stable:
                output.append(f"      - '{p['start']} to {p['end']}' ({p['intervals']} intervals): State: {p['state']}")

    return "\n".join(output), graph_artifact

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Door & Window Tool...")
    print("-" * 50)
    try:
        print("\n[Testing]")
        summary, raw_data = get_door_window_status.func(room="2.3", timeframe="now")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("-" * 50)
        
        print("\n[Testing]")
        summary, raw_data = get_door_window_status.func(room="2.3", timeframe="24h")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("-" * 50)
        
        print("\n[Testing]")
        summary, raw_data = get_door_window_status.func(room="2.3", timeframe="30d")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError during execution: {e}")