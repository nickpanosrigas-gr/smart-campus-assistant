import pandas as pd
from datetime import datetime, timedelta
from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import your singletons
from utils.device_registry import registry
from clients.thingsboard_client import tb_client

# Define the timeframe mapping to Pandas resample frequencies
TIMEFRAME_CONFIG = {
    "1h":  {"delta": timedelta(hours=1), "bin_size": "5min"},
    "24h": {"delta": timedelta(hours=24), "bin_size": "2h"}, 
    "7d":  {"delta": timedelta(days=7), "bin_size": "12h"},  
    "30d": {"delta": timedelta(days=30), "bin_size": "2d"}   
}

AVAILABLE_ROOMS = list(registry.devices_by_room.keys())

class AirQualityInput(BaseModel):
    room_key: str = Field(
        ..., 
        description=f"The specific room to check. Must be one of: {', '.join(AVAILABLE_ROOMS)}"
    )
    timeframe: Literal["1h", "24h", "7d", "30d"] = Field(
        default="1h", 
        description="The time window for the data request."
    )

@tool("get_air_quality", args_schema=AirQualityInput)
def get_air_quality(room_key: str, timeframe: Literal["1h", "24h", "7d", "30d"]) -> str:
    """
    Fetches Air Quality metrics (CO2, TVOC, PM10, PM2.5) for a specific room.
    Returns highly processed timeline data: global baselines, 'NOW' context, and historical anomalies grouped by time.
    """
    # 1. Resolve Devices
    room_devices = registry.get_devices_in_room(*room_key.split("_", 1)) 
    iaq_devices = {name: dev_id for name, dev_id in room_devices.items() if "IAQ" in name}
    
    if not iaq_devices:
        return f"Target_Room: {room_key}\nError: No IAQ (Air Quality) sensors found in this room."

    # 2. Calculate Timestamps
    config = TIMEFRAME_CONFIG[timeframe]
    end_time = datetime.now()
    start_time = end_time - config["delta"]
    
    end_ts = int(end_time.timestamp() * 1000)
    start_ts = int(start_time.timestamp() * 1000)
    
    # Updated metrics array
    metrics = ["co2", "tvoc", "pm10", "pm2.5"]
    
    room_data_per_metric = {m: [] for m in metrics}

    # 3. Fetch and pre-process Data into Pandas DataFrames
    for device_name, device_id in iaq_devices.items():
        telemetry = tb_client.get_historical_telemetry(device_id, ",".join(metrics), start_ts, end_ts)
        if not telemetry: 
            continue
            
        for metric in metrics:
            if metric in telemetry and telemetry[metric]:
                df = pd.DataFrame(telemetry[metric])
                df['value'] = pd.to_numeric(df['value'])
                df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('datetime', inplace=True)
                
                # Resample into bins
                binned = df['value'].resample(config["bin_size"]).agg(['min', 'max', 'mean']).dropna()
                if not binned.empty:
                    room_data_per_metric[metric].append({
                        "sensor": device_name,
                        "binned": binned
                    })

    # 4. Process Baselines and Timeline
    global_baselines = {}
    timeline = {} # Dictionary mapping timestamp -> metric -> sensor -> values
    all_seen_timestamps = set()

    for metric, sensors_data in room_data_per_metric.items():
        if not sensors_data: 
            continue

        # Calculate Global Baseline for this metric across ALL sensors in the room
        all_means = pd.concat([sd["binned"]['mean'] for sd in sensors_data])
        global_mean = all_means.mean()
        global_std = all_means.std()
        
        # Determine anomaly threshold (1 standard dev, or 10% if std is weird)
        threshold = global_std if pd.notna(global_std) and global_std > 0 else global_mean * 0.1
        
        # Save baseline to output structure
        global_baselines[f"{metric.upper()}_avg"] = f"{global_mean:.2f}"

        for sd in sensors_data:
            sensor = sd["sensor"]
            binned = sd["binned"]
            all_seen_timestamps.update(binned.index)

            # Identify anomalies for this specific sensor
            anomalies = binned[
                (binned['max'] > global_mean + threshold) | 
                (binned['min'] < global_mean - threshold)
            ]

            # Enforce the "NOW" Principle: Always include the latest bucket with ALL sensors/values
            now_ts = None
            if not binned.empty:
                now_ts = binned.index[-1]
                now_row = binned.loc[now_ts]
                
                val_str = f"min {now_row['min']:.1f}, avr {now_row['mean']:.1f}, max {now_row['max']:.1f}"
                
                if now_ts not in timeline:
                    timeline[now_ts] = {"is_now": True, "data": {}}
                if metric.upper() not in timeline[now_ts]["data"]:
                    timeline[now_ts]["data"][metric.upper()] = {}
                    
                timeline[now_ts]["data"][metric.upper()][sensor] = val_str

            # Add historical anomalies to the timeline
            for ts, row in anomalies.iterrows():
                # Skip if it's the "Now" bin, we already added it above
                if ts == now_ts:
                    continue 

                val_str = f"min {row['min']:.1f}, avr {row['mean']:.1f}, max {row['max']:.1f}"

                if ts not in timeline:
                    timeline[ts] = {"is_now": False, "data": {}}
                if metric.upper() not in timeline[ts]["data"]:
                    timeline[ts]["data"][metric.upper()] = {}
                    
                timeline[ts]["data"][metric.upper()][sensor] = val_str

    # 5. Build the Final String Output (YAML style)
    output = []
    output.append(f"Target_Room: {room_key}")
    output.append(f"Timeframe: {timeframe} ({config['bin_size']} intervals)")
    
    output.append("Global_Baseline:")
    if not global_baselines:
        output.append("  No data available.")
    for k, v in global_baselines.items():
        output.append(f"  {k}: {v}")

    output.append("Timeline:")
    if not timeline:
         output.append("  []")
         
    # Sort timestamps chronologically descending (NEWEST first -> OLDEST last)
    for ts in sorted(timeline.keys(), reverse=True):
        bucket_info = timeline[ts]
        is_now = bucket_info["is_now"]

        bucket_label = ts.strftime('%Y-%m-%d %H:%M:%S')
        if is_now:
            bucket_label = f"NOW ({bucket_label})"

        output.append(f"- bucket: '{bucket_label}'")
        output.append("  anomalies:")

        for metric_name, sensors_dict in bucket_info["data"].items():
            output.append(f"    {metric_name}:")
            for sens, val in sensors_dict.items():
                output.append(f"      {sens}: {val}")

    # Calculate exactly how many bins we hid from the LLM
    ignored_count = len(all_seen_timestamps) - len(timeline)
    output.append(f"Ignored_Buckets: {ignored_count} intervals omitted (Normal behavior).")

    return "\n".join(output)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Air Quality Tool for the Restaurant...")
    print("-" * 50)
    
    try:
        result = get_air_quality.invoke({
            "room_key": "F0_restaurant",
            "timeframe": "1h"
        })
        
        print("\n📊 --- TOOL OUTPUT --- 📊\n")
        print(result)
        
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")