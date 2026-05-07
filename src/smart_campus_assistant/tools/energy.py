import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

# Import project singletons
from src.smart_campus_assistant.utils.device_registry import registry
from src.smart_campus_assistant.clients.thingsboard_client import tb_client

logger = logging.getLogger(__name__)

# Configs matching the other tools
TIMEFRAME_CONFIG = {
    "now": {"method": "get_now", "bin_size": None, "prev_method": "get_now_prev_30d_full"},
    "2h":  {"method": "get_2h", "bin_size": "10min", "prev_method": "get_2h_prev_30d_full"},
    "24h": {"method": "get_24h", "bin_size": "2h", "prev_method": "get_24h_prev_30d_full"}, 
    "7d":  {"method": "get_7d", "bin_size": "2h", "prev_method": "get_7d_prev_30d_full"},    
    "30d": {"method": "get_30d", "bin_size": "24h", "prev_method": None},
    "90d": {"method": "get_90d", "bin_size": "24h", "prev_method": None} 
}

# Eastron Meter Keys
ENERGY_KEYS = [
    "f1_var1", "f1_var2", "f1_var3", "f1_var4", "f1_var5", "f1_var6", # kWh, V1, V2, V3, A1, A2
    "f2_var1", "f2_var2", "f2_var3", "f2_var4", "f2_var5"             # A3, PF1, PF2, PF3, Hz
]

# UNIFIED ROOM INPUT (Matches Telemetry Tools exactly)
Rooms = Literal[
    'car_lift', 'front_lift', 'back_lift', 'hvac', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

Timeframes = Literal['now', '2h', '24h', '7d', '30d', '90d']

class EnergyInput(BaseModel):
    room: Rooms = Field(..., description="The room or infrastructure target to check.")
    timeframe: Timeframes = Field(..., description="The time window. 'now', '2h', '24h', '7d', '30d', '90d'.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_time_context(dt: pd.Timestamp) -> str:
    is_weekend = dt.dayofweek >= 5
    is_work = 8 <= dt.hour < 22
    if not is_weekend and is_work: return "Weekday (Mon-Fri) Working_Hours (08:00-22:00)"
    if not is_weekend and not is_work: return "Weekday (Mon-Fri) Non-Working_Hours (22:00-08:00)"
    if is_weekend and is_work: return "Weekend (Sat-Sun) Working_Hours (08:00-22:00)"
    return "Weekend (Sat-Sun) Non-Working_Hours (22:00-08:00)"

def process_energy_telemetry(raw_data: Dict, bin_size: str = None) -> pd.DataFrame:
    dfs = []
    for key in ENERGY_KEYS:
        if key in raw_data and raw_data[key]:
            df = pd.DataFrame(raw_data[key])
            if df.empty or 'value' not in df.columns: continue
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.rename(columns={'value': key}, inplace=True)
            df.drop(columns=['ts'], inplace=True)
            dfs.append(df)
            
    if not dfs: return pd.DataFrame()
    
    combined = pd.concat(dfs, axis=1, sort=True).ffill().bfill()
    
    if bin_size:
        agg_funcs = {k: 'median' for k in ENERGY_KEYS if k in combined.columns}
        if 'f1_var1' in combined.columns: agg_funcs['f1_var1'] = 'max'
        combined = combined.resample(bin_size).agg(agg_funcs)
        
        if 'f1_var1' in combined.columns:
            combined['kwh_consumed'] = combined['f1_var1'].diff().fillna(0)
            combined.loc[combined['kwh_consumed'] < 0, 'kwh_consumed'] = 0
            
    v_cols, a_cols = ['f1_var2', 'f1_var3', 'f1_var4'], ['f1_var5', 'f1_var6', 'f2_var1']
    for c in v_cols + a_cols:
        if c not in combined.columns: combined[c] = 0
        
    combined['total_kw'] = ((combined['f1_var2'] * combined['f1_var5']) + 
                            (combined['f1_var3'] * combined['f1_var6']) + 
                            (combined['f1_var4'] * combined['f2_var1'])) * 0.9 / 1000
                            
    return combined

def detect_anomalies(row, is_gen: bool, baseline_kw: float = 0) -> List[str]:
    anomalies = []
    hz = row.get('f2_var5', 0)
    kw = row.get('total_kw', 0)
    
    is_live = hz > 45
    if not is_live:
        return ["Power Offline"]
        
    if is_gen: anomalies.append("GENERATOR ACTIVE (Blackout)")
        
    v1, v2, v3 = row.get('f1_var2', 0), row.get('f1_var3', 0), row.get('f1_var4', 0)
    a1, a2, a3 = row.get('f1_var5', 0), row.get('f1_var6', 0), row.get('f2_var1', 0)
    
    if is_live and (v1 < 207 or v2 < 207 or v3 < 207):
        anomalies.append(f"Voltage Sag Detected (V1:{v1:.0f} V2:{v2:.0f} V3:{v3:.0f})")
        
    amps = [a1, a2, a3]
    if max(amps) > 2.0 and min(amps) <= 0.1:
        anomalies.append(f"Phase Drop Detected (A1:{a1:.1f} A2:{a2:.1f} A3:{a3:.1f})")
        
    if baseline_kw > 0 and kw > (baseline_kw * 1.8) and kw > 1.0:
        anomalies.append(f"Load Spike ({kw:.1f}kW vs {baseline_kw:.1f}kW base)")
        
    return anomalies

@tool("get_energy_infrastructure", args_schema=EnergyInput)
def get_energy_infrastructure(room: Rooms, timeframe: Timeframes) -> str:
    """Tracks Electrical Power (Grid vs Generator), Current Load (kW), Consumption (kWh), and Phase Faults."""
    
    # NEW REGISTRY MAPPING CALL
    meters = registry.get_energy_meters_for_target(room)
    if not meters: return f"Query_Context:\n  Target: {room}\nError: No energy meters mapped to this room/infrastructure."

    config = TIMEFRAME_CONFIG[timeframe]
    bin_size = config["bin_size"]
    fetch_method = getattr(tb_client, config["method"])

    meter_dfs = {}
    for name, d_id in meters.items():
        raw = fetch_method(d_id, ENERGY_KEYS)
        df = process_energy_telemetry(raw, bin_size)
        if not df.empty: meter_dfs[name] = df

    if not meter_dfs: return f"Error: No historical energy data found for timeframe {timeframe}."

    # ==========================================
    # BRANCH A: REAL-TIME SNAPSHOT ("NOW")
    # ==========================================
    if timeframe == "now":
        output = [
            "Query_Context:", "  Domain: Energy & Infrastructure",
            f"  Target: {room.upper()}", "  Timeframe: Now (Snapshot)", ""
        ]
        
        active_source = "NONE (BLACKOUT)"
        live_kw = 0.0
        
        output.append("Current_State:")
        for name, df in meter_dfs.items():
            if df.empty: continue
            curr = df.iloc[-1]
            hz = curr.get('f2_var5', 0)
            is_gen = "GEN" in name.upper()
            
            if hz > 45:
                active_source = f"{name} [{'GENERATOR' if is_gen else 'GRID'}]"
                live_kw = curr.get('total_kw', 0)
                faults = detect_anomalies(curr, is_gen)
                
                output.append(f"  Active_Source: {active_source}")
                output.append(f"  Grid_Health: {'NORMAL' if not faults else ' | '.join(faults)}")
                output.append(f"  Voltage (L1|L2|L3): {curr.get('f1_var2',0):.1f}V | {curr.get('f1_var3',0):.1f}V | {curr.get('f1_var4',0):.1f}V")
                output.append(f"  Current (L1|L2|L3): {curr.get('f1_var5',0):.1f}A | {curr.get('f1_var6',0):.1f}A | {curr.get('f2_var1',0):.1f}A")
                output.append(f"  Estimated_Live_Load: {live_kw:.2f} kW")
        
        if active_source == "NONE (BLACKOUT)":
            output.append("  Status: OFFLINE / TOTAL POWER LOSS DETECTED")
            
        return "\n".join(output)

    # Master DataFrame logic (combine PPC and GEN)
    master_df = pd.DataFrame()
    for name, df in meter_dfs.items():
        is_gen = "GEN" in name.upper()
        df['source'] = name
        df['is_gen'] = is_gen
        master_df = pd.concat([master_df, df])
        
    master_df.sort_index(inplace=True)
    
    # ==========================================
    # BRANCH B: LONG-TERM MATRIX (30d / 90d)
    # ==========================================
    if timeframe in ["30d", "90d"]:
        output = [
            "Query_Context:", "  Domain: Energy & Infrastructure",
            f"  Target: {room.upper()}", f"  Timeframe: {timeframe} (Statistical Profile)", "",
            "Schedule_Profiling_Matrix:"
        ]
        
        def process_matrix_cell(name: str, mask: pd.Series):
            cell_df = master_df[mask]
            if cell_df.empty: return [f"  {name}:", "    No data."]
            
            avg_daily_kwh = cell_df['kwh_consumed'].sum() / (len(cell_df.index.unique().date) or 1)
            avg_peak_kw = cell_df.groupby(cell_df.index.date)['total_kw'].max().mean()
            
            lines = [
                f"  {name}:",
                f"    Baseline: Avg Daily Consumption: {avg_daily_kwh:.1f} kWh | Avg Peak Load: {avg_peak_kw:.1f} kW"
            ]
            
            outliers = []
            for day, day_df in cell_df.groupby(cell_df.index.date):
                day_kwh = day_df['kwh_consumed'].sum()
                day_peak = day_df['total_kw'].max()
                day_gen = day_df[day_df['is_gen'] & (day_df['f2_var5'] > 45)]
                
                day_notes = []
                if not day_gen.empty: day_notes.append("GENERATOR ACTIVE (Blackout/Test)")
                if avg_peak_kw > 0 and day_peak > (avg_peak_kw * 1.5): day_notes.append(f"Load Spike ({day_peak:.1f}kW)")
                
                if day_notes:
                    outliers.append(f"      - '{day}': {', '.join(day_notes)} | Consumed: {day_kwh:.1f} kWh")
                    
            lines.append("    Outliers:")
            lines.extend(outliers) if outliers else lines.append("      None detected.")
            return lines

        is_wkdy = master_df.index.dayofweek < 5
        is_work = (master_df.index.hour >= 8) & (master_df.index.hour < 22)
        
        output.extend(process_matrix_cell("Weekdays (Mon-Fri) Working_Hours (08:00-22:00)", is_wkdy & is_work))
        output.extend(process_matrix_cell("Weekdays (Mon-Fri) Non-Working_Hours (22:00-08:00)", is_wkdy & ~is_work))
        output.extend(process_matrix_cell("Weekends (Sat-Sun) Working_Hours (08:00-22:00)", ~is_wkdy & is_work))
        output.extend(process_matrix_cell("Weekends (Sat-Sun) Non-Working_Hours (22:00-08:00)", ~is_wkdy & ~is_work))
        
        return "\n".join(output)

    # ==========================================
    # BRANCH C: TIMELINE ACTIVITY (2h, 24h, 7d)
    # ==========================================
    output = [
        "Query_Context:", "  Domain: Energy & Infrastructure",
        f"  Target: {room.upper()}", f"  Timeframe: {timeframe} ({bin_size} intervals)", ""
    ]
    
    total_kwh = master_df['kwh_consumed'].sum()
    peak_kw = master_df['total_kw'].max()
    gen_time = master_df[master_df['is_gen'] & (master_df['f2_var5'] > 45)].shape[0] * pd.Timedelta(bin_size).total_seconds() / 3600
    
    output.append(f"Global_Energy_Summary (Last {timeframe}):")
    output.append(f"  Total Consumption: {total_kwh:.1f} kWh")
    output.append(f"  Peak Load: {peak_kw:.1f} kW")
    if gen_time > 0: output.append(f"  WARNING: Generator Active for ~{gen_time:.1f} hours during period.")
    output.append("")
    output.append("Timeline_Activity:")

    for day_start, day_df in master_df.groupby(pd.Grouper(freq='D')):
        if day_df.empty: continue
        day_key = day_start.strftime('%Y-%m-%d (%A)')
        
        transitions = []
        stable_intervals, stable_start = 0, None
        
        rolling_baseline_kw = master_df['total_kw'].mean()

        for exact_time, row in day_df.iterrows():
            time_str = exact_time.strftime('%H:%M')
            bucket_end = (exact_time + pd.to_timedelta(bin_size)).strftime('%H:%M')
            
            anomalies = detect_anomalies(row, row.get('is_gen', False), rolling_baseline_kw)
            
            if anomalies:
                if stable_intervals > 0:
                    transitions.append(f"      - '{stable_start} to {time_str}' ({stable_intervals} intervals): Stable Grid Power. Load ~{rolling_baseline_kw:.1f}kW")
                    stable_intervals, stable_start = 0, None
                    
                transitions.append(f"      - bucket: '{time_str}' -> FAULT DETECTED: {' | '.join(anomalies)} (Source: {row['source']})")
            else:
                if stable_start is None: stable_start = time_str
                stable_intervals += 1
                
        if stable_intervals > 0:
            end_of_day = "24:00" if (day_start + pd.Timedelta(days=1)).strftime('%H:%M') == "00:00" else (day_start + pd.Timedelta(days=1)).strftime('%H:%M')
            transitions.append(f"      - '{stable_start} to {end_of_day}' ({stable_intervals} intervals): Stable Grid Power.")

        output.append(f"  '{day_key}':")
        output.extend(transitions)

    return "\n".join(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Testing Energy Infrastructure Tool...")
    print("=" * 60)

    test_cases = [
        {"room": "hvac", "timeframe": "now"},          
        {"room": "car_lift", "timeframe": "2h"},      
        {"room": "3.8", "timeframe": "24h"},          # Testing standard room mapping to floor
        {"room": "restaurant", "timeframe": "7d"},    # Testing standard room mapping to floor
        {"room": "hvac", "timeframe": "30d"},         
        {"room": "front_lift", "timeframe": "90d"}    
    ]

    for case in test_cases:
        room_val = case["room"]
        tf = case["timeframe"]
        
        print(f"\n[TEST] Target: {room_val.upper()} | Timeframe: {tf}")
        print("-" * 40)
        
        try:
            result = get_energy_infrastructure.invoke({"room": room_val, "timeframe": tf})
            print(result)
        except Exception as e:
            print(f"Error executing test for {room_val}/{tf}: {e}")
        
        print("\n" + "=" * 60)

    print("\nTesting Complete.")