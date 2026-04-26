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
    "7d":  {"method": "get_7d", "bin_size": "12h"},  
    "30d": {"method": "get_30d", "bin_size": "2d"}   
}

# Semantic mapping for 0-5 scale
LIGHT_LABELS = {
    0: "Level 0 (Dark)",
    1: "Level 1 (Dim)",
    2: "Level 2 (Normal)",
    3: "Level 3 (Bright)",
    4: "Level 4 (Very Bright)",
    5: "Level 5 (Very Sunny)"
}

def get_semantic_label(val: float) -> str:
    """Safely rounds continuous data to the discrete 0-5 scale and returns its semantic label."""
    if pd.isna(val):
        return "Unknown"
    clamped_val = int(max(0, min(5, round(val))))
    return LIGHT_LABELS.get(clamped_val, f"Level {clamped_val}")

# Allowed rooms derived from user requirements
ROOMS = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

class LightsInput(BaseModel):
    room: ROOMS = Field(
        ..., 
        description="The specific room to check for illumination levels."
    )
    timeframe: Literal["now", "2h", "24h", "7d", "30d"] = Field(
        default="now", 
        description="The time window for the data request. 'now' provides a real-time snapshot."
    )

@tool("get_ambient_lights", args_schema=LightsInput)
def get_ambient_lights(room: str, timeframe: Literal["now", "2h", "24h", "7d", "30d"]) -> str:
    """
    Tracks indoor illumination using a discrete 0-5 scale.
    Uses state-transition logic to prevent mathematical hallucinations and maps integers to semantic labels.
    """
    # 1. Resolve Devices (Only IAQ monitors containing light sensors)
    iaq_devices = registry.get_devices_by_room_and_type(room, "IAQ")
    
    if not iaq_devices:
        return f"Query_Context:\n  Room: {room}\nError: No IAQ (Light) sensors found in this room."

    sensor_names = list(iaq_devices.keys())
    active_sensors_str = f"{len(sensor_names)} ({', '.join(sensor_names)})"

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        output = [
            "Query_Context:",
            "  Domain: Ambient Light Intensity (0-5 Scale)",
            f"  Room: {room}",
            "  Timeframe: Now (Snapshot)",
            f"  Active_Sensors: {active_sensors_str}",
            "",
            "Current_State:"
        ]
        
        for device_name, device_id in iaq_devices.items():
            raw_data = tb_client.get_now(device_id, ["light_level"])
            if "light_level" in raw_data and raw_data["light_level"]:
                val = float(raw_data["light_level"][0]["value"])
                output.append(f"  {device_name}: {get_semantic_label(val)}")
            else:
                output.append(f"  {device_name}: Offline / No Data")
                
        return "\n".join(output)

    # ==========================================
    # BRANCH B: HISTORICAL TIMELINE & TRANSITIONS
    # ==========================================
    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method_name = config["method"]
    fetch_method = getattr(tb_client, fetch_method_name)

    all_dataframes = []

    # 2. Fetch and format Raw Data
    for device_name, device_id in iaq_devices.items():
        try:
            raw_data = fetch_method(device_id, ["light_level"])
            if "light_level" in raw_data and raw_data["light_level"]:
                df = pd.DataFrame(raw_data["light_level"])
                df['value'] = pd.to_numeric(df['value'])
                df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('datetime', inplace=True)
                df.rename(columns={'value': device_name}, inplace=True)
                df.drop(columns=['ts'], inplace=True)
                all_dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch historical light data for {device_name}: {e}")

    if not all_dataframes:
        return f"Query_Context:\n  Room: {room}\nError: No historical light data found for timeframe {timeframe}."

    # Combine all sensors, resample into chunks, and calculate the median room state per chunk
    combined_df = pd.concat(all_dataframes, axis=1)
    binned_df = combined_df.resample(bin_size).median()
    
    # Calculate Room Aggregate per bin (median across all active sensors in the room)
    binned_df['Room_Aggregate'] = binned_df.median(axis=1)
    binned_df.dropna(subset=['Room_Aggregate'], inplace=True)

    # 3. Calculate Global Illumination Summary
    total_bins = len(binned_df)
    summary_counts = binned_df['Room_Aggregate'].apply(lambda x: int(max(0, min(5, round(x))))).value_counts()
    
    global_summary = []
    for level_int in range(6):
        count = summary_counts.get(level_int, 0)
        percentage = (count / total_bins * 100) if total_bins > 0 else 0
        if percentage > 0:
            global_summary.append(f"  {LIGHT_LABELS[level_int]}: {percentage:.0f}%")

    # 4. Identify Transitions and Stable Periods
    transitions = []
    stable_periods = []
    
    current_stable_start = None
    current_stable_state = None
    stable_bin_count = 0

    previous_state = None

    for timestamp, row in binned_df.iterrows():
        current_state_int = int(max(0, min(5, round(row['Room_Aggregate']))))
        current_state_label = LIGHT_LABELS[current_state_int]
        bucket_label = timestamp.strftime('%Y-%m-%d %H:%M')

        # First bucket initialization
        if previous_state is None:
            previous_state = current_state_label
            current_stable_start = bucket_label
            current_stable_state = current_state_label
            stable_bin_count = 1
            continue

        # Check for State Transition
        if current_state_label != previous_state:
            # 1. Close out the previous stable period
            stable_periods.append({
                "start": current_stable_start,
                "end": bucket_label,
                "intervals": stable_bin_count,
                "state": current_stable_state
            })
            
            # 2. Record the transition event
            time_of_change = timestamp.strftime('%H:%M')
            transitions.append(
                f"- bucket: '{bucket_label}'\n"
                f"  activity:\n"
                f"    Room_Aggregate: 'Transition: [{previous_state} -> {current_state_label} at {time_of_change}].'"
            )

            # 3. Reset stable tracker
            previous_state = current_state_label
            current_stable_start = bucket_label
            current_stable_state = current_state_label
            stable_bin_count = 1
        else:
            stable_bin_count += 1

    # Close out the final stable period
    if stable_bin_count > 0:
        stable_periods.append({
            "start": current_stable_start,
            "end": "Present",
            "intervals": stable_bin_count,
            "state": current_stable_state
        })

    # 5. Build Final YAML-style Output
    output = [
        "Query_Context:",
        "  Domain: Ambient Light Intensity (0-5 Scale)",
        f"  Room: {room}",
        f"  Timeframe: {timeframe} ({bin_size} intervals)",
        f"  Active_Sensors: {active_sensors_str}",
        "",
        f"Global_Illumination_Summary (Last {timeframe}):"
    ]
    
    output.extend(global_summary)
    output.append("")
    
    output.append("Timeline_Transitions:")
    if not transitions:
        output.append("  No lighting state transitions detected.")
    else:
        output.extend(transitions)
        
    output.append("")
    output.append("Stable_Periods (No State Changes):")
    for period in stable_periods:
        output.append(f"  - '{period['start']} to {period['end']}' ({period['intervals']} intervals):")
        output.append(f"      State: {period['state']}")

    return "\n".join(output)

if __name__ == "__main__":
    # Test execution block
    logging.basicConfig(level=logging.INFO)
    print("Testing Lights Tool...")
    print("-" * 50)
    try:
        # Assuming you have a device in the restaurant for testing
        print("\n[Testing Snapshot (Now)]")
        print(get_ambient_lights.invoke({"room": "restaurant", "timeframe": "now"}))
        
        print("\n[Testing Historical (2h)]")
        print(get_ambient_lights.invoke({"room": "entrance", "timeframe": "24h"}))
    except Exception as e:
        print(f"\nError during execution: {e}")