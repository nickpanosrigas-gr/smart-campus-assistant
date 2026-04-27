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

AVAILABLE_ROOMS = registry.get_available_rooms()

class OccupancyInput(BaseModel):
    room: str = Field(
        ...,
        description=f"The specific room to check for occupancy."
    )
    timeframe: Literal["now", "2h", "24h", "7d", "30d"] = Field(
        default="now", 
        description="The time window for the data request. 'now' provides a real-time snapshot."
    )

def fetch_and_resample(devices: Dict[str, str], keys: List[str], fetch_method, bin_size: str, sensor_type: str, timeframe: str) -> pd.Series:
    """
    Helper to fetch telemetry for multiple devices, combine them, and resample.
    Includes a fallback for sparse sensors (like People Counters) that only report on state-change.
    """
    all_dfs = []
    for device_name, device_id in devices.items():
        try:
            raw_data = fetch_method(device_id, keys)
            for key in keys:
                if key in raw_data and raw_data[key]:
                    df = pd.DataFrame(raw_data[key])
                    df['value'] = pd.to_numeric(df['value'])
                    df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert('Europe/Athens').dt.tz_localize(None)
                    df.set_index('datetime', inplace=True)
                    df = df[['value']].rename(columns={'value': f"{device_name}_{key}"})
                    all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch data for {device_name}: {e}")

    # SPARSE DATA FALLBACK
    if not all_dfs:
        if sensor_type == 'motion':
            return pd.Series(dtype=float) 
            
        last_known = {}
        for device_name, device_id in devices.items():
            try:
                raw_now = tb_client.get_now(device_id, keys)
                for key in keys:
                    if key in raw_now and raw_now[key]:
                        last_known[f"{device_name}_{key}"] = float(raw_now[key][0]["value"])
            except Exception:
                pass
                
        if not last_known:
            return pd.Series(dtype=float)
            
        end_ts = pd.Timestamp.now('UTC').tz_localize(None)
        td_map = {
            "2h": pd.Timedelta(hours=2), 
            "24h": pd.Timedelta(hours=24), 
            "7d": pd.Timedelta(days=7), 
            "30d": pd.Timedelta(days=30)
        }
        start_ts = end_ts - td_map.get(timeframe, pd.Timedelta(hours=2))
        
        if sensor_type == 'pc':
            last_known = {k: 0 for k in last_known}
            
        combined_df = pd.DataFrame([last_known, last_known], index=[start_ts, end_ts])
    else:
        combined_df = pd.concat(all_dfs, axis=1, sort=True)

    if sensor_type != 'pc':
        combined_df.ffill(inplace=True)

    if sensor_type == 'pc':
        aligned_df = combined_df.resample(bin_size).sum() 
    else:
        aligned_df = combined_df.resample(bin_size).max()
        aligned_df.ffill(inplace=True)
        
    aligned_df.fillna(0, inplace=True)

    if sensor_type == 'desk':
        return aligned_df.sum(axis=1)
    elif sensor_type == 'wo':
        return aligned_df.sum(axis=1)
    elif sensor_type == 'motion':
        return aligned_df.max(axis=1)
    elif sensor_type == 'pc':
        in_cols = [c for c in aligned_df.columns if 'period_in' in c]
        out_cols = [c for c in aligned_df.columns if 'period_out' in c]
        
        total_in = aligned_df[in_cols].sum(axis=1) if in_cols else pd.Series(0, index=aligned_df.index)
        total_out = aligned_df[out_cols].sum(axis=1) if out_cols else pd.Series(0, index=aligned_df.index)
        
        net_change = total_in - total_out
        
        # BOUNDED CUMULATIVE SUM (Fixes 30d Negative Debt Drift)
        current_occ = 0
        occ_list = []
        for change in net_change:
            current_occ = max(0, current_occ + change)
            occ_list.append(current_occ)
            
        return pd.Series(occ_list, index=net_change.index)

    return pd.Series(dtype=float)

@tool("get_occupancy", args_schema=OccupancyInput)
def get_occupancy(room: str, timeframe: Literal["now", "2h", "24h", "7d", "30d"]) -> str:
    """
    Tracks room occupancy using a polymorphic schema. Automatically detects if the room uses 
    Desk Sensors, People Counters (PC), or Area Wait Counters (WO), and cross-validates 
    the primary count with secondary IAQ Motion (PIR) sensors.
    """
    room_key = str(room).strip().lower()
    devices = registry.get_all_devices_in_room(room_key)
    
    if not devices:
        return f"Query_Context:\n  Room: {room}\nError: Room not found or has no devices."

    pc_devices = {k: v for k, v in devices.items() if "-PC" in k.upper()}
    wo_devices = {k: v for k, v in devices.items() if "-WO" in k.upper()}
    desk_devices = {k: v for k, v in devices.items() if "-DESK" in k.upper()}
    iaq_devices = {k: v for k, v in devices.items() if "-IAQ" in k.upper()}

    if desk_devices:
        primary_type = "Desk_Contact (Binary)"
        primary_devs = desk_devices
        primary_keys = ["occupancy"]
        sensor_category = "desk"
    elif pc_devices:
        primary_type = "People_Counter (Synthesized from In/Out Traffic)"
        primary_devs = pc_devices
        primary_keys = ["line_1_period_in", "line_1_period_out"]
        sensor_category = "pc"
    elif wo_devices:
        primary_type = "Area_Wait_Counter (Continuous Count)"
        primary_devs = wo_devices
        primary_keys = ["people_count_max"]
        sensor_category = "wo"
    else:
        return f"Query_Context:\n  Room: {room}\nError: No primary occupancy sensor (PC, WO, or Desk) found."

    total_primary_sensors = len(primary_devs)
    iaq_sensor_names = list(iaq_devices.keys())
    support_sensors_str = f"{len(iaq_sensor_names)} ({', '.join(iaq_sensor_names)} Motion)" if iaq_devices else "None"

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        output = [
            "Query_Context:",
            "  Domain: Occupancy",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Primary_Sensor: {primary_type}",
            f"  Supporting_Sensors: {support_sensors_str}",
            "",
            "Current_State:"
        ]
        
        primary_val = 0
        has_data = False
        
        if pc_devices:
            recent_series = fetch_and_resample(primary_devs, primary_keys, tb_client.get_24h, "10min", "pc", "24h")
            if not recent_series.empty:
                has_data = True
                primary_val = recent_series.iloc[-1]
        else:
            for name, uid in primary_devs.items():
                raw = tb_client.get_now(uid, primary_keys)
                key = primary_keys[0]
                if key in raw and raw[key]:
                    has_data = True
                    val = float(raw[key][0]["value"])
                    if desk_devices:
                        if val > 0: primary_val += 1
                    else:
                        primary_val += val
                
        if has_data:
            if desk_devices:
                output.append(f"  Current_Occupancy: {int(primary_val)}/{total_primary_sensors} Desks Occupied")
            else:
                output.append(f"  Current_Occupancy: {int(primary_val)} people")
        else:
            output.append("  Primary_Status: Offline / No Data")

        motion_active = False
        for name, uid in iaq_devices.items():
            raw = tb_client.get_now(uid, ["pir"])
            if "pir" in raw and raw["pir"]:
                if float(raw["pir"][0]["value"]) > 0:
                    motion_active = True
                    
        motion_str = "Active (Validates occupancy)" if motion_active else "Idle"
        output.append(f"  Motion_Status: {motion_str}")
        
        return "\n".join(output)

    # ==========================================
    # BRANCH B: HISTORICAL DATA FETCH
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method = getattr(tb_client, config["method"])

    primary_series = fetch_and_resample(primary_devs, primary_keys, fetch_method, bin_size, sensor_type=sensor_category, timeframe=timeframe)
    motion_series = fetch_and_resample(iaq_devices, ["pir"], fetch_method, bin_size, sensor_type="motion", timeframe=timeframe)

    if primary_series.empty:
        return f"Query_Context:\n  Room: {room}\nError: No historical occupancy data found for timeframe {timeframe}. Check if sensor is actively transmitting."

    df = pd.DataFrame({"primary": primary_series})
    if not motion_series.empty:
        df = df.join(motion_series.rename("motion"), how="outer")
    else:
        df["motion"] = 0.0

    df.fillna(0, inplace=True)

    # ==========================================
    # BRANCH C: 30-DAY STATISTICAL PROFILE WITH OUTLIERS
    # ==========================================
    if timeframe == "30d":
        output = [
            "Query_Context:",
            "  Domain: Occupancy",
            f"  Room: {room}",
            "  Timeframe: 30d (Long-Term Statistical Profile)",
            f"  Primary_Sensor: {primary_type}",
            "",
            "Schedule_Profiling_Matrix:"
        ]
        
        is_weekday = df.index.dayofweek < 5
        is_weekend = df.index.dayofweek >= 5
        is_working_hours = (df.index.hour >= 8) & (df.index.hour < 22)
        is_non_working = (df.index.hour < 8) | (df.index.hour >= 22)
        
        def process_30d_cell(cell_name, mask):
            cell_df = df[mask]
            if cell_df.empty:
                return [f"    {cell_name}:", "      Baseline: No data", "      Outliers: None"]
            
            motion_pct = (cell_df['motion'] > 0).mean() * 100
            outliers = []
            lines = [f"    {cell_name}:"]
            
            if desk_devices:
                baseline_util = (cell_df['primary'] > 0).mean() * 100
                stats = f"Utilization: {baseline_util:.0f}% | Motion Active: {motion_pct:.0f}%"
                lines.append(f"      Baseline: {stats}")
                
                # Check Outliers for Desks (Utilization deviates > 25%)
                daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
                for day, day_data in daily_groups:
                    if day_data.empty: continue
                    day_util = (day_data['primary'] > 0).mean() * 100
                    if abs(day_util - baseline_util) > 25:
                        day_str = day.strftime('%Y-%m-%d (%A)')
                        outliers.append(f"        - '{day_str}': Utilization {day_util:.0f}%")
            else:
                # Find the maximum peak for each day within this specific mask
                daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
                daily_peaks = {day: day_data['primary'].max() for day, day_data in daily_groups if not day_data.empty}
                
                avg_peak = sum(daily_peaks.values()) / len(daily_peaks) if daily_peaks else 0
                max_peak = max(daily_peaks.values()) if daily_peaks else 0
                
                stats = f"Avg Daily Peak: {avg_peak:.1f} people | Max Peak: {max_peak:.0f} people | Motion Active: {motion_pct:.0f}%"
                lines.append(f"      Baseline: {stats}")
                
                # Check Outliers for PCs/WOs (Peak deviates significantly from average)
                for day, peak in daily_peaks.items():
                    # Flags if peak is 50% larger/smaller than average AND absolute difference is at least 5 people
                    if abs(peak - avg_peak) >= max(5, avg_peak * 0.5):
                        day_str = day.strftime('%Y-%m-%d (%A)')
                        outliers.append(f"        - '{day_str}': Peak reached {peak:.0f} people")
                        
            if outliers:
                lines.append("      Outliers:")
                lines.extend(outliers)
            else:
                lines.append("      Outliers: None detected.")
                
            return lines

        output.append("  Weekdays (Mon-Fri):")
        output.extend(process_30d_cell("Working_Hours (08:00-22:00)", is_weekday & is_working_hours))
        output.extend(process_30d_cell("Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working))
        
        output.append("  Weekends (Sat-Sun):")
        output.extend(process_30d_cell("Working_Hours (08:00-22:00)", is_weekend & is_working_hours))
        output.extend(process_30d_cell("Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working))

        return "\n".join(output)

    # ==========================================
    # BRANCH D: 2h, 24h, 7d (PER-DAY TIMELINE LOGIC)
    # ==========================================
    global_motion_pct = (df['motion'] > 0).mean() * 100
    peak_occ = df['primary'].max()
    avg_occ = df['primary'].mean()
    
    if desk_devices:
        global_summary = f"Peak_Occupancy: {peak_occ:.0f}/{total_primary_sensors} Desks | Avg_Occupancy: {avg_occ:.1f}/{total_primary_sensors} Desks"
    else:
        global_summary = f"Peak_Occupancy: {peak_occ:.0f} people | Avg_Occupancy: {avg_occ:.1f} people"

    output = [
        "Query_Context:",
        "  Domain: Occupancy",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        f"  Primary_Sensor: {primary_type}",
        f"  Supporting_Sensors: {support_sensors_str}",
        "",
        f"Global_Occupancy_Summary (Last {timeframe}):",
        f"  {global_summary}",
        f"  Motion_Context: Active {global_motion_pct:.0f}% / Idle {100-global_motion_pct:.0f}%",
        "",
        "Timeline_Activity:"
    ]

    daily_groups = df.groupby(pd.Grouper(freq='D'))
    
    for day_start, day_df in daily_groups:
        if day_df.empty: continue
        
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        output.append(f"  '{day_key}':")
        
        transitions = []
        stable_periods = []
        
        prev_state = None
        current_stable_start = None
        current_stable_state = None
        stable_bins = 0
        
        for ts, row in day_df.iterrows():
            time_str = ts.strftime('%H:%M')
            
            # Dynamically calculate the end time of this specific bucket
            bucket_end_ts = ts + pd.to_timedelta(bin_size)
            bucket_end_str = bucket_end_ts.strftime('%H:%M')
            if bucket_end_str == "00:00": 
                bucket_end_str = "24:00" # Formatting for midnight
            
            motion_str = "Active" if row['motion'] > 0 else "Idle"
            if desk_devices:
                prim_state = f"{int(row['primary'])}/{total_primary_sensors} Desks Occupied"
            else:
                prim_state = f"{int(row['primary'])} people"
                
            combined_state = f"Status: {prim_state}. Motion: {motion_str}."
            
            if prev_state is None:
                prev_state = combined_state
                current_stable_start = time_str
                current_stable_state = combined_state
                stable_bins = 0
                
            if combined_state != prev_state:
                transitions.append(f"      - bucket: '{time_str}'")
                transitions.append(f"        activity: 'Transitioned to {prim_state}'")
                transitions.append(f"        motion_state: '{motion_str}'")
                
                if stable_bins > 1:
                    stable_periods.append(f"      - '{current_stable_start} to {time_str}' ({stable_bins} intervals): {current_stable_state}")
                
                current_stable_start = time_str
                current_stable_state = combined_state
                prev_state = combined_state
                stable_bins = 1
            else:
                stable_bins += 1

        # Close out any remaining stable periods dynamically using the actual end of the last bucket
        if stable_bins > 0:
            stable_periods.append(f"      - '{current_stable_start} to {bucket_end_str}' ({stable_bins} intervals): {current_stable_state}")

        if not transitions:
            output.append("    Timeline_Transitions: None (State was stable)")
        else:
            output.append("    Timeline_Transitions:")
            output.extend(transitions)
            
        if stable_periods:
            output.append("    Stable_Periods:")
            output.extend(stable_periods)

    return "\n".join(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Occupancy Tool...")
    print("-" * 50)
    try:
        print("\n[Testing Historical (now)]")
        print(get_occupancy.invoke({"room": "1.2", "timeframe": "now"}))
        print("\n[Testing Historical (2h)]")
        print(get_occupancy.invoke({"room": "1.2", "timeframe": "2h"}))
        print("\n[Testing Historical (24h)]")
        print(get_occupancy.invoke({"room": "1.2", "timeframe": "24h"}))
        print("\n[Testing Historical (7d)]")
        print(get_occupancy.invoke({"room": "1.2", "timeframe": "7d"}))
        print("\n[Testing Historical (30d)]")
        print(get_occupancy.invoke({"room": "1.2", "timeframe": "30d"}))
    except Exception as e:
        print(f"\nError during execution: {e}")