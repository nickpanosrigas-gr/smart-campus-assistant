# Smart Campus Intelligent Assistant

## Overview
This repository contains the prototype for an LLM-based Intelligent Assistant designed for Smart Campus management, developed as part of a university thesis. The system allows building managers to interact with building infrastructure and IoT systems (lighting, HVAC, occupancy, access control) using natural language queries.

By integrating Google's Gemini 3.1-Flash via LangGraph, the assistant translates natural language into precise data requests, performs anomaly detection, queries semantic knowledge, and proposes automated infrastructure rules via ThingsBoard.

## Architecture

The system employs a **Hybrid Agentic Supervisor** architecture built on **LangGraph**:

1. **Supervisor Node (Gemini 3.1-Flash):** The central router that analyzes user intent and delegates tasks to the appropriate specialized sub-agent or tool.
2. **Query & Analytics Node (Read-Only):** Handles telemetry checks and time-series aggregations. It relies on highly specific, domain-segregated tools to interact with the ThingsBoard REST API.
3. **Action & Rule Agent (Write-Access):** Proposes infrastructure changes and constructs ThingsBoard Rule Chain JSONs. **Safety constraint:** Includes a Human-in-the-Loop (HITL) approval step before deploying any changes to the active ThingsBoard instance.
4. **Semantic Memory:** Utilizes a Vector Database to store and retrieve unstructured campus knowledge, building topology, and Standard Operating Procedures (SOPs).

## Core System Clients & Utilities

To decouple the LLM logic from the IoT infrastructure, the system utilizes highly optimized singletons for API communication and campus topology resolution:

* **ThingsBoard Client (`clients/thingsboard_client.py`):** A robust REST API wrapper that handles JWT authentication and automatic token refreshing. Crucially, it fetches **RAW telemetry data** with massive limits (up to 100k points) rather than relying on server-side aggregation, allowing the LangGraph tools to perform custom, domain-specific Pandas operations.
* **Device Registry (`utils/device_registry.py`):** An in-memory cache layer loaded from `campus_topology.json`. It traverses the complex building hierarchy and flattens it into a fast, case-insensitive O(1) lookup table. It maps human-readable rooms (e.g., "Restaurant") to ThingsBoard device UUIDs based on dynamic sensor types (e.g., "-IAQ", "-PC"), allowing the LLM to query physical spaces without knowing database IDs.

## Data Processing Strategy

To prevent LLM context-window bloat and reduce hallucinations (especially "mathematical hallucinations" on discrete data), the architecture utilizes advanced Pandas processing before passing data to the LLM:

* **Sensor Synchronization Grid:** When fusing data from multiple sensors in the same room, raw ticks are aligned to a uniform grid (e.g., 10-minute intervals) *before* aggregation. This prevents the "ping-pong" effect where staggered sensor transmission times cause the room's median state to rapidly fluctuate.
* **Bucket Activity Classification:** For discrete states (like lights or door sensors), standard mathematical downsampling (like taking the median of a 2-hour block) destroys historical fidelity. Instead, data is chunked into logical buckets, and the internal ticks are chronologically scanned to classify the *volume* of activity:
  * *Stable:* Added to a clean list of stable periods.
  * *Clear Transitions (1-3 changes):* The exact minute of the state change is extracted and reported.
  * *High Volatility (4+ changes):* The noise is compressed into a single token-efficient summary (e.g., "Fluctuating heavily").
* **Long-Term Statistical Profiling (30d):** To prevent LLM attention degradation over massive timeframes, 30-day queries discard event logs entirely. Instead, data is shaped into a 2x2 Statistical Matrix (Weekdays vs. Weekends / Working vs. Non-Working hours), and only specific days that deviate violently from the monthly baseline are reported as anomalies.
* **Real-Time Snapshots ("Now"):** When immediate situational awareness is needed, the `now` timeframe is explicitly called. This bypasses all historical bucketing and returns a zero-latency, lightweight snapshot of the current state.

## Tool Output Architectures

To maximize token efficiency and cross-tool correlation, the system returns YAML-formatted exception summaries. 

### 1. Temperature & Humidity (`temp_humidity.py`)
Fuses multiple room sensors and incorporates dynamic baseline deviations, fixed limits, and advanced external weather context (including solar heat gain and wind).

```yaml
Query_Context:
  Domain: Temperature & Humidity
  Room: Restaurant
  Timeframe: 24h (2h intervals)
  Active_Sensors: 3 (F0_Restaurant-IAQ-1 to 3) + 1 (Roof_Weather_Station)

External_Weather_Context (Last 24h):
  Temp_avg: 16.3°C
  Hum_avg: 46.3%
  Solar_Radiation_Peak: 750 W/m² (High Solar Heat Gain)
  Wind_Speed_avg: 1.2 m/s (Peak: 4.5 m/s)
  Precipitation: 0 mm

Global_Room_Baseline (Last 24h):
  Temp_avg: 22.4°C (Normal: 20°C - 25°C)
  Hum_avg: 45.2% (Normal: 30% - 60%)

Timeline_Exceptions:
- bucket: '2026-04-26 12:00 - 14:00'
  trigger: 'Baseline Deviation (> +2.0°C)'
  details: 'Room Temp spiked to 26.5°C despite mild outdoor temp (16°C). Correlates directly with Peak Solar Radiation and South-Facing windows.'
  room_aggregate:
    Temp_max: 26.5°C
    Temp_avg: 25.0°C

Stable_Periods: 
  - '2026-04-25 16:00 to 2026-04-26 12:00' (10 intervals)
  - '2026-04-26 14:00 to Present' (1 interval)
  Status: Values stable. HVAC maintained optimal conditions successfully.
```

### 2. Ambient Lights (`lights.py`)
Tracks indoor illumination using a discrete 0-5 scale. Uses state-transition logic mapped to semantic labels. It dynamically alters its output format based on the timeframe size.

#### Scenario A: Weekly Event Log (7d Timeframe)
Groups chronological events hierarchically by day, nesting the Stable Periods directly inside the day to prevent the LLM from losing context.

```yaml
Query_Context:
  Domain: Ambient Light Intensity (0-5 Scale)
  Room: restaurant
  Timeframe: 7d (2h intervals)
  Active_Sensors: 3 (F0_Restaurant-IAQ-1, F0_Restaurant-IAQ-2, F0_Restaurant-IAQ-3)

Global_Illumination_Summary (Last 7d):
  Level 0 (Dark): 46%
  Level 1 (Dim): 28%
  Level 2 (Normal): 26%

Timeline_Activity:
  '2026-04-24 (Friday)':
    Timeline_Transitions:
      - bucket: '08:00 - 10:00'
        activity: 'Transition: [Level 0 (Dark) -> Level 2 (Normal) at 08:30].'
      - bucket: '14:00 - 16:00'
        activity: 'Fluctuating heavily between Level 1 (Dim) and Level 2 (Normal) (Toggled 5 times).'
      - bucket: '16:00 - 18:00'
        activity: 'Transition: [Level 1 (Dim) -> Level 0 (Dark) at 17:50].'
    Stable_Periods:
      - '00:00 to 08:00' (4 intervals): State: Level 0 (Dark)
      - '10:00 to 14:00' (2 intervals): State: Level 2 (Normal)
      - '18:00 to 24:00' (3 intervals): State: Level 0 (Dark)
```

#### Scenario B: Long-Term Profiling (30d Timeframe)
Drops the event log entirely to prevent token bloat. Generates a 2x2 statistical matrix (Weekdays vs. Weekends / Working vs. Non-Working Hours) and extracts anomalous days to instantly flag energy waste or unusual usage to the LLM.

```yaml
Query_Context:
  Domain: Ambient Light Intensity (0-5 Scale)
  Room: restaurant
  Timeframe: 30d (Long-Term Statistical Profile)
  Active_Sensors: 3

Total_Monthly_Average:
  Level 0 (Dark): 65%, Level 1 (Dim): 10%, Level 2 (Normal): 25%

Schedule_Profiling_Matrix:
  Weekdays (Mon-Fri):
    Working_Hours (08:00-22:00):
      Baseline: Level 1 (Dim): 20%, Level 2 (Normal): 80%
      Outliers: None detected.
    Non-Working_Hours (22:00-08:00):
      Baseline: Level 0 (Dark): 100%
      Outliers:
        - '2026-04-14 (Tuesday)': Level 0 (Dark): 40%, Level 2 (Normal): 60%  # Anomaly: Lights left on overnight
  Weekends (Sat-Sun):
    Working_Hours (08:00-22:00):
      Baseline: Level 0 (Dark): 95%, Level 1 (Dim): 5%
      Outliers: None detected.
    Non-Working_Hours (22:00-08:00):
      Baseline: Level 0 (Dark): 100%
      Outliers: None detected.
```

### 3. Air Quality (`air_quality.py`)
Utilizes fixed biological safety thresholds and uses contextual injects to determine source attribution (indoor vs outdoor pollution).

```yaml
Query_Context:
  Domain: Air Quality (CO2, TVOC, PM2.5, PM10)
  Room: Restaurant
  Timeframe: 7d (12h intervals)
  Active_Sensors: 3 (F0_Restaurant-IAQ) + 1 (Roof_Weather_Station)

External_Weather_Context (Last 7d):
  PM2.5_avg: 42 µg/m³ (Poor Outdoor Air Quality)
  PM10_avg: 65 µg/m³ (Elevated Dust/Smog)

Global_Room_Baseline (Last 7d):
  CO2_avg: 550 ppm (Normal < 800)
  TVOC_avg: 180 ppb (Normal < 250)
  PM2.5_avg: 14 µg/m³ (Normal < 25)
  PM10_avg: 22 µg/m³ (Normal < 50)

Timeline_Exceptions:
- bucket: '2026-04-20 12:00 - 24:00'
  trigger: 'Threshold Breach (Dinner Rush Spike)'
  room_aggregate:
    CO2_max: 1650 ppm
    TVOC_max: 420 ppb
- bucket: '2026-04-23 00:00 - 12:00'
  trigger: 'Particulate Spike (Source: Likely External)'
  details: 'Indoor PM2.5 reached 48 µg/m³. Correlates directly with severe Outdoor PM2.5 spike (85 µg/m³) during this window.'
  room_aggregate:
    PM2.5_max: 48 µg/m³
    PM10_max: 75 µg/m³

Stable_Periods: 
  - '2026-04-20 00:00 to 12:00' (1 interval)
  - '2026-04-21 00:00 to 2026-04-23 00:00' (4 intervals)
  - '2026-04-23 12:00 to Present' (3 intervals)
  Status: All metrics normal. Filtration successfully scrubbing poor outdoor air during these periods.
```

### 4. Doors & Windows (`door_window.py`)
Treats continuous states (like open windows for ventilation) as stable and only reports chronological state-transitions, aligning seamlessly with the continuous variables of other tools.

```yaml
Query_Context:
  Domain: Doors & Windows (Binary Contact)
  Room: Classroom_101
  Timeframe: 24h (2h intervals)
  Active_Sensors: 4 (Door_Main, Win_Front, Win_Mid, Win_Back)

Global_State_Summary (Last 24h):
  Door_Main: Open 5% / Closed 95%
  Win_Front: Open 100% / Closed 0%
  Win_Mid: Open 35% / Closed 65%
  Win_Back: Open 0% / Closed 100%

Timeline_Transitions:
- bucket: '2026-04-24 08:00 - 10:00'
  activity:
    Door_Main: 'Started Closed. Toggled 12 times. Ended Closed.'
    Win_Mid: 'Transition: [Closed -> Open at 08:15].'
- bucket: '2026-04-24 14:00 - 16:00'
  activity:
    Door_Main: 'Transition: [Closed -> Open at 14:15]. Transition: [Open -> Closed at 15:30].'
- bucket: '2026-04-24 16:00 - 18:00'
  activity:
    Win_Mid: 'Transition: [Open -> Closed at 17:45].'

Stable_Periods (No State Changes):
  - '2026-04-23 16:00 to 2026-04-24 08:00' (10 intervals): 
      State: Door_Main Closed, Win_Front Open, Win_Mid Closed, Win_Back Closed
  - '2026-04-24 10:00 to 14:00' (2 intervals): 
      State: Door_Main Closed, Win_Front Open, Win_Mid Open, Win_Back Closed
  - '2026-04-24 18:00 to Present' (3 intervals): 
      State: Door_Main Closed, Win_Front Open, Win_Mid Closed, Win_Back Closed
```

### 5. Occupancy (`occupancy.py`)
Utilizes a **Polymorphic Schema** to handle different data shapes (Binary Desk Sensors, Continuous People Counters, and Queue Trackers). Crucially, it fuses primary counting sensors with secondary binary **Motion Sensors** (from the IAQ monitors) to cross-validate presence and detect anomalies like "Ghost Occupancy" or security bypasses.

#### Scenario A: Desk Sensor + Motion (Detecting Ghost Occupancy)
```yaml
Query_Context:
  Domain: Occupancy
  Room: Library_Study_Pod_1
  Timeframe: 24h (2h intervals)
  Primary_Sensor: Desk_Contact (Binary)
  Supporting_Sensors: 1 (F0_LibraryPod1-IAQ Motion)

Global_Occupancy_Summary (Last 24h):
  State: Occupied 25% / Available 75%
  Total_Sessions: 2
  Motion_Context: Active 20% / Idle 80%

Timeline_Activity:
- bucket: '2026-04-24 10:00 - 12:00'
  activity: 'Transition: [Available -> Occupied at 10:15].'
  motion_state: 'Transition: [Idle -> Sustained Active]. (Validates human presence)'
- bucket: '2026-04-24 14:00 - 16:00'
  activity: 'Transition: [Occupied -> Available at 14:30]. Transition: [Available -> Occupied at 15:45].'
  motion_state: 'Idle from 14:30 to 15:45. Active upon re-entry.'
- bucket: '2026-04-24 18:00 - 20:00'
  activity: 'Sustained Occupied'
  motion_state: 'Anomaly: Sustained Idle (100% of bucket). (Warning: No movement detected, desk may be ghost-occupied)'

Stable_Periods (No State Changes):
  - '2026-04-23 16:00 to 2026-04-24 10:00' (9 intervals): 
      State: Available. Motion: Idle.
  - '2026-04-24 16:00 to 18:00' (1 interval): 
      State: Occupied. Motion: Active.
  - '2026-04-24 20:00 to Present' (2 intervals):
      State: Occupied. Motion: Active.
```

#### Scenario B: People Counter + Motion (Detecting Unregistered Entry)
```yaml
Query_Context:
  Domain: Occupancy
  Room: Classroom_101
  Timeframe: 24h (2h intervals)
  Primary_Sensor: People_Counter (Continuous Count)
  Supporting_Sensors: 2 (F0_Class101-IAQ-1 Motion, F0_Class101-IAQ-2 Motion)

Global_Occupancy_Summary (Last 24h):
  Peak_Occupancy: 45 people
  Utilization_Profile: Active 6h / Empty 18h
  Motion_Context: Active 28% / Idle 72%

Timeline_Activity (Significant Movements):
- bucket: '2026-04-24 08:00 - 10:00'
  activity: 'Mass Arrival'
  metrics:
    Net_Change: +42 people
    Peak_in_Bucket: 42
  motion_state: 'Transition: [Idle -> Sustained Active at 08:15]'
- bucket: '2026-04-24 10:00 - 12:00'
  activity: 'Mass Departure'
  metrics:
    Net_Change: -42 people (Room emptied at 10:15)
  motion_state: 'Transition: [Sustained Active -> Idle at 10:20]'
- bucket: '2026-04-24 22:00 - 24:00'
  activity: 'Anomaly: Unregistered Motion'
  metrics:
    Net_Change: 0 people (Main counter registered no entry)
  motion_state: 'Room_Aggregate: Toggled 14 times. (Possible security patrol or cleaning crew)'

Stable_Periods:
  - '2026-04-23 16:00 to 2026-04-24 08:00' (10 intervals): 
      Status: Empty (0 people). Motion: Idle.
  - '2026-04-24 12:00 to 22:00' (5 intervals): 
      Status: Empty (0 people). Motion: Idle.
```

#### Scenario C: Line Counter + Motion (Distinguishing Staff vs Queue)
```yaml
Query_Context:
  Domain: Occupancy
  Room: Restaurant_Main_Counter
  Timeframe: 24h (2h intervals)
  Primary_Sensor: Line_Counter (Queue Length)
  Supporting_Sensors: 1 (F0_Restaurant-IAQ-1 Motion)

Global_Occupancy_Summary (Last 24h):
  Avg_Line_Length: 2 people
  Peak_Line_Length: 28 people (Threshold Breach: > 15 people)
  Motion_Context: Active 45% / Idle 55%

Timeline_Activity (Queue Spikes & Anomalies):
- bucket: '2026-04-24 10:00 - 12:00'
  activity: 'Pre-Shift Activity'
  metrics:
    Max_Queue: 0 people
  motion_state: 'Transition: [Idle -> Sustained Active]. (Likely staff prep, no queue formed)'
- bucket: '2026-04-24 12:00 - 14:00'
  activity: 'Severe Queue Formation (Lunch Rush)'
  metrics:
    Max_Queue: 28 people
    Sustained_Over_Threshold: 45 minutes
  motion_state: 'Sustained Active (Validates high traffic)'

Stable_Periods:
  - '2026-04-23 16:00 to 2026-04-24 10:00' (9 intervals):
      Status: Queue 0. Motion: Idle.
  - '2026-04-24 14:00 to Present' (5 intervals):
      Status: Queue remained short/manageable (< 5 people). Motion: Intermittent Active.
```

### 6. Real-Time Snapshots ("Now" Timeframe)
When the LLM Orchestrator needs instant situational awareness, it passes `timeframe: "now"` to any tool. This branches the backend logic to skip historical aggregation entirely and return a token-efficient, zero-latency state check.

```yaml
Query_Context:
  Domain: Occupancy
  Room: Classroom_101
  Timeframe: Now (Snapshot)
  Primary_Sensor: People_Counter
  Supporting_Sensors: Motion (Binary)

Current_State:
  Current_Occupancy: 12 people
  Motion_Status: Active (Validates occupancy)
```

## Technology Stack

* **Framework:** Python, LangChain, LangGraph
* **LLM:** Google Gemini 3.1-Flash
* **Embeddings:** Google Text Embedding Model
* **Vector Database:** Qdrant (Local/Docker or Cloud)
* **IoT Platform:** ThingsBoard (REST API & Rule Engine)

## Project Structure

```text
smart-campus-assistant/
│
├── src/
│   └── smart_campus_assistant/
│       ├── __init__.py
│       │
│       ├── agents/                     # LangGraph Agent Definitions
│       │   ├── __init__.py
│       │   ├── supervisor.py           # The main routing logic
│       │   ├── query_agent.py          # Read-only API and Analytics logic
│       │   └── action_agent.py         # Write-access and Rule generation logic
│       │
│       ├── tools/                      # Domain-Specific Tool Definitions
│       │   ├── __init__.py
│       │   ├── temp_humidity.py        # Combines Temp & Humidity with Indoor/Outdoor delta
│       │   ├── air_quality.py          # Oxygen, CO2, TVOC, Particulate Matter
│       │   ├── occupancy.py            # Polymorphic logic (People/Desk/Line) with Motion fusion
│       │   ├── door_window.py          # Binary state transitions (Open/Closed)
│       │   ├── lights.py               # Multimodal timelines (Daily Events vs. 30d Statistical Profile)
│       │   └── knowledge.py            # Qdrant Vector DB semantic search tool
│       │
│       ├── clients/                    # Tool clients
│       │   ├── __init__.py
│       │   └── thingsboard_client.py   # JWT Auth & High-Volume Raw Telemetry fetcher
│       │
│       ├── database/                   # Vector DB Management
│       │   ├── __init__.py
│       │   ├── qdrant_client.py        # Qdrant connection and initialization
│       │   └── document_loader.py      # Scripts to chunk and embed PDFs/Manuals
│       │
│       ├── graph/                      # LangGraph Setup
│       │   ├── __init__.py
│       │   └── workflow.py             # Compiles nodes, edges, and HITL breakpoints
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   └── device_registry.py      # O(1) in-memory Campus Topology resolver
│       │
│       └── config/                     # Configuration
│           ├── __init__.py
│           └── settings.py             # Environment variables mapping
│
├── data/                       # Local Knowledge Base (Unstructured Data)
│   ├── manuals/                # HVAC/Sensor PDFs
│   └── campus_topology.json    # Spatial definitions for resolving Room IDs
│
├── .env.example                # API keys template (Gemini, ThingsBoard, Qdrant)
├── pyproject.toml              # Modern Python packaging and build configuration
├── uv.lock                     # Locked dependency resolution (uv)
├── main.py                     # Entry point (CLI or API wrapper)
├── requirements.txt            # Exported Python dependencies
└── README.md                   # Project documentation
```