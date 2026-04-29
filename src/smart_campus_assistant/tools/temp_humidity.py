import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, List, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

# Import Schemas
from src.smart_campus_assistant.tools.schemas import CampusRooms, Timeframes

logger = logging.getLogger(__name__)

# Config mapping for API calls and pandas resampling
TIMEFRAME_CONFIG = {
    "2h":  {"method": "get_2h", "bin_size": "10min"},
    "24h": {"method": "get_24h", "bin_size": "2h"}, 
    "7d":  {"method": "get_7d", "bin_size": "2h"},    
    "30d": {"method": "get_30d", "bin_size": "2h"}    
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

# Tolerances for grouping continuous data into "Stable Periods"
TOLERANCE = {
    "temp": 1.0,      # Group if temp changes by less than 1.0°C
    "hum": 5.0,       # Group if humidity changes by less than 5%
    "wind": 10.0      # Group if wind speed changes by less than 10 km/h
}

class ClimateInput(BaseModel):
    room: CampusRooms = Field(
        ..., 
        description="The room to check (e.g., '1.1', 'restaurant') OR 'roof' to access the Campus Weather Station."
    )
    timeframe: Timeframes = Field(
        ..., 
        description="The time window. 'now' for real-time. '2h', '24h', '7d' for timelines. '30d' for statistics."
    )

def format_climate_state(row: pd.Series, is_outdoor: bool) -> str:
    """Creates a highly dense, token-efficient string from raw telemetry."""
    if row.empty or row.isna().all():
        return "Offline/No Data"
        
    if not is_outdoor:
        t = f"{row.get('temperature', 0):.1f}°C" if pd.notna(row.get('temperature')) else "N/A"
        h = f"{row.get('humidity', 0):.0f}%" if pd.notna(row.get('humidity')) else "N/A"
        p = f"{row.get('pressure', 0):.0f}hPa" if pd.notna(row.get('pressure')) else "N/A"
        return f"Temp: {t} | Hum: {h} | Pres: {p}"
    
    # Weather Station Formatting (Smart grouping)
    t = f"{row.get('air_temperature', 0):.1f}°C" if pd.notna(row.get('air_temperature')) else "N/A"
    h = f"{row.get('relative_humidity', 0):.0f}%" if pd.notna(row.get('relative_humidity')) else "N/A"
    
    wind_spd = row.get('wind_speed', 0)
    wind_max = row.get('maximum_wind_speed', wind_spd)
    wind_dir = row.get('wind_direction', 0)
    wind_str = f"{wind_spd:.1f}km/h (Max: {wind_max:.1f}, Dir: {wind_dir:.0f}°)" if pd.notna(wind_spd) else "N/A"
    
    rain = row.get('precipitation', 0)
    rain_str = f"{rain:.1f}mm" if rain > 0 else "None"
    
    strikes = row.get('lightning_strike_count', 0)
    lightning_str = f"{strikes:.0f} strikes (Avg Dist: {row.get('lightning_average_distance', 0):.1f}km)" if strikes > 0 else "None"
    
    solar = f"{row.get('solar_radiation', 0):.0f}W/m²" if pd.notna(row.get('solar_radiation')) else "N/A"
    
    return f"Temp: {t} | Hum: {h} | Wind: {wind_str} | Rain: {rain_str} | Lightning: {lightning_str} | Solar: {solar}"

def is_state_similar(state1: pd.Series, state2: pd.Series, is_outdoor: bool) -> bool:
    """Determines if two chronological buckets are mathematically similar enough to group."""
    if state1.empty or state2.empty: return False
    
    t_key = 'air_temperature' if is_outdoor else 'temperature'
    h_key = 'relative_humidity' if is_outdoor else 'humidity'
    
    if abs(state1.get(t_key, 0) - state2.get(t_key, 0)) > TOLERANCE["temp"]: return False
    if abs(state1.get(h_key, 0) - state2.get(h_key, 0)) > TOLERANCE["hum"]: return False
    
    if is_outdoor:
        if abs(state1.get('wind_speed', 0) - state2.get('wind_speed', 0)) > TOLERANCE["wind"]: return False
        # Any change in rain or lightning breaks the stable period immediately
        if state1.get('precipitation', 0) != state2.get('precipitation', 0): return False
        if state1.get('lightning_strike_count', 0) != state2.get('lightning_strike_count', 0): return False
        
    return True

@tool("get_climate_and_weather", args_schema=ClimateInput)
def get_climate_and_weather(room: CampusRooms, timeframe: Timeframes) -> str:
    """
    Tracks indoor HVAC conditions (Temperature/Humidity) OR outdoor Campus Weather.
    Compresses data to highlight extreme weather events, anomalies, and schedule baselines.
    """
    loc_key = str(room).strip().lower()
    is_outdoor = loc_key == 'roof'
    
    if is_outdoor:
        devices = registry.get_devices_by_type("WEATHERSTATION")
        keys_to_fetch = WEATHER_KEYS
        sensor_type = "Weather_Station"
    else:
        devices = registry.get_devices_by_room_and_type(loc_key, "IAQ")
        keys_to_fetch = IAQ_KEYS
        sensor_type = "Indoor_IAQ"

    if not devices:
        return f"Query_Context:\n  Room: {room}\nError: No {sensor_type} sensors found for this location."

    active_sensors_str = f"{len(devices)} ({', '.join(devices.keys())})"

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        output = [
            "Query_Context:",
            f"  Domain: Climate & Weather ({sensor_type})",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Active_Sensors: {active_sensors_str}",
            "",
            "Current_State:"
        ]
        
        for device_name, device_id in devices.items():
            raw_data = tb_client.get_now(device_id, keys_to_fetch)
            parsed_row = {}
            for k in keys_to_fetch:
                if k in raw_data and raw_data[k]:
                    parsed_row[k] = float(raw_data[k][0]["value"])
            
            output.append(f"  {device_name}: {format_climate_state(pd.Series(parsed_row), is_outdoor)}")
                
        return "\n".join(output)

    # ==========================================
    # BRANCH B: HISTORICAL DATA FETCH
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method = getattr(tb_client, config["method"])

    all_dataframes = []
    for device_name, device_id in devices.items():
        try:
            raw_data = fetch_method(device_id, keys_to_fetch)
            for key in keys_to_fetch:
                if key in raw_data and raw_data[key]:
                    df = pd.DataFrame(raw_data[key])
                    df['value'] = pd.to_numeric(df['value'])
                    df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert('Europe/Athens').dt.tz_localize(None)
                    df.set_index('datetime', inplace=True)
                    df = df[['value']].rename(columns={'value': key})
                    all_dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch historical climate data for {device_name}: {e}")

    if not all_dataframes:
        return f"Query_Context:\n  Room: {room}\nError: No historical data found for timeframe {timeframe}."

    # FIX HERE: Group by index and mean to aggregate multiple devices/keys
    # Added `sort=True` to silence the warning, and updated the groupby to avoid `axis=1`
    combined_df = pd.concat(all_dataframes, axis=1, sort=True)
    combined_df = combined_df.T.groupby(level=0).mean().T
    
    # Custom Resampling: Max for Rain/Lightning/Wind, Mean for Temp/Hum
    resample_rules = {k: 'max' if k in ['precipitation', 'lightning_strike_count', 'maximum_wind_speed'] else 'mean' for k in keys_to_fetch if k in combined_df.columns}
    aligned_df = combined_df.resample(bin_size).agg(resample_rules).ffill().fillna(0)

    # ==========================================
    # BRANCH C: 30-DAY STATISTICAL PROFILE
    # ==========================================
    if timeframe == "30d":
        is_weekday = aligned_df.index.dayofweek < 5
        is_weekend = aligned_df.index.dayofweek >= 5
        is_working_hours = (aligned_df.index.hour >= 8) & (aligned_df.index.hour < 22)
        is_non_working = (aligned_df.index.hour < 8) | (aligned_df.index.hour >= 22)

        output = [
            "Query_Context:",
            f"  Domain: Climate & Weather ({sensor_type})",
            f"  Room: {room}",
            "  Timeframe: 30d (Long-Term Statistical Profile)",
            "",
            "Schedule_Profiling_Matrix:"
        ]
        
        def process_matrix_cell(cell_name, mask):
            cell_df = aligned_df[mask]
            if cell_df.empty:
                return [f"    {cell_name}:", "      Baseline: No data", "      Outliers: None"]
            
            baseline = cell_df.mean()
            lines = [f"    {cell_name}:"]
            lines.append(f"      Baseline: {format_climate_state(baseline, is_outdoor)}")
            
            # --- OUTLIER DETECTION ---
            outliers = []
            daily_groups = cell_df.groupby(pd.Grouper(freq='D'))
            
            t_key = 'air_temperature' if is_outdoor else 'temperature'
            t_mean, t_std = cell_df[t_key].mean(), cell_df[t_key].std()
            
            for day, day_df in daily_groups:
                if day_df.empty: continue
                day_str = day.strftime('%Y-%m-%d (%A)')
                day_max = day_df.max()
                
                triggers = []
                # Temp Spikes/Drops (2 Standard Deviations)
                if abs(day_max.get(t_key, t_mean) - t_mean) > (t_std * 2) and t_std > 0.5:
                    triggers.append(f"Extreme Temp ({day_max[t_key]:.1f}°C)")
                
                if is_outdoor:
                    if day_max.get('precipitation', 0) > 0:
                        triggers.append(f"Rain Event ({day_max['precipitation']:.1f}mm)")
                    if day_max.get('lightning_strike_count', 0) > 0:
                        triggers.append(f"Lightning Storm ({day_max['lightning_strike_count']:.0f} strikes)")
                    if day_max.get('wind_speed', 0) > 30: # 30km/h threshold
                        triggers.append(f"High Winds ({day_max['wind_speed']:.1f}km/h)")
                else:
                    if day_max.get('humidity', 0) > 75:
                        triggers.append(f"High Humidity ({day_max['humidity']:.0f}%)")

                if triggers:
                    outliers.append(f"        - '{day_str}': {', '.join(triggers)}")
            
            if outliers:
                lines.append("      Outliers (Anomalies/Weather Events):")
                lines.extend(outliers)
            else:
                lines.append("      Outliers: None detected (Conditions stable).")
            return lines

        output.append("  Weekdays (Mon-Fri):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekday & is_working_hours))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekday & is_non_working))
        
        output.append("  Weekends (Sat-Sun):")
        output.extend(process_matrix_cell("Working_Hours (08:00-22:00)", is_weekend & is_working_hours))
        output.extend(process_matrix_cell("Non-Working_Hours (22:00-08:00)", is_weekend & is_non_working))

        return "\n".join(output)

    # ==========================================
    # BRANCH D: 2h, 24h, 7d (TIMELINE COMPRESSION)
    # ==========================================
    t_key = 'air_temperature' if is_outdoor else 'temperature'
    global_max = aligned_df[t_key].max()
    global_min = aligned_df[t_key].min()
    
    output = [
        "Query_Context:",
        f"  Domain: Climate & Weather ({sensor_type})",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        "",
        f"Global_Summary (Last {timeframe}):",
        f"  Temp Range: {global_min:.1f}°C to {global_max:.1f}°C",
        "Timeline_Activity:"
    ]

    daily_groups = aligned_df.groupby(pd.Grouper(freq='D'))
    
    for day_start, day_df in daily_groups:
        if day_df.empty: continue
        
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        output.append(f"  '{day_key}':")
        
        timeline = []
        prev_row = None
        stable_start_ts = None
        stable_bins = 0
        
        for ts, row in day_df.iterrows():
            time_str = ts.strftime('%H:%M')
            bucket_end_ts = ts + pd.to_timedelta(bin_size)
            bucket_end_str = bucket_end_ts.strftime('%H:%M')
            if bucket_end_str == "00:00": bucket_end_str = "24:00"
            
            if prev_row is None:
                prev_row = row
                stable_start_ts = time_str
                stable_bins = 1
                continue
                
            if is_state_similar(prev_row, row, is_outdoor):
                stable_bins += 1
            else:
                # Flush stable period
                state_str = format_climate_state(prev_row, is_outdoor)
                if stable_bins > 1:
                    timeline.append(f"      - '{stable_start_ts} to {time_str}' ({stable_bins} intervals): {state_str}")
                else:
                    timeline.append(f"      - '{stable_start_ts}': {state_str} (Shift detected)")
                
                stable_start_ts = time_str
                prev_row = row
                stable_bins = 1

        # Close final period
        if stable_bins > 0:
            state_str = format_climate_state(prev_row, is_outdoor)
            timeline.append(f"      - '{stable_start_ts} to {bucket_end_str}' ({stable_bins} intervals): {state_str}")

        if not timeline:
            output.append("    No data for this day.")
        else:
            output.extend(timeline)

    return "\n".join(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Temp & Humidity Tool...")
    print("-" * 50)
    try:
        print("\n[Testing Historical (now)]")
        print(get_climate_and_weather.invoke({"room": "restaurant", "timeframe": "now"}))
        print("\n[Testing Historical (2h)]")
        print(get_climate_and_weather.invoke({"room": "restaurant", "timeframe": "2h"}))
        print("\n[Testing Historical (24h)]")
        print(get_climate_and_weather.invoke({"room": "restaurant", "timeframe": "24h"}))
        print("\n[Testing Historical (7d)]")
        print(get_climate_and_weather.invoke({"room": "restaurant", "timeframe": "7d"}))
        print("\n[Testing Historical (30d)]")
        print(get_climate_and_weather.invoke({"room": "restaurant", "timeframe": "30d"}))
        
        print("\n[Testing Roof/Weather Station (now)]")
        print(get_climate_and_weather.invoke({"room": "roof", "timeframe": "now"}))
    except Exception as e:
        print(f"\nError during execution: {e}")