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

## Data Processing Strategy
To prevent LLM context-window bloat and reduce hallucinations, this architecture does not feed raw sensor data to the LLM. Instead, data is pre-processed using the following strategy:
* **Timeframe Chunking:** Data is requested in strict historical timeframes (`1h`, `24h`, `7d`, `30d`) and downsampled into 12-14 bins (e.g., a 24h request is chunked into 2-hour bins).
* **Anomaly Highlighting:** For historical chunks, average data points are omitted. Only anomalous spikes (min/max deviations) or state transitions are passed to the LLM.
* **Real-Time Snapshots ("Now"):** When immediate situational awareness is needed, the `now` timeframe is explicitly called. This completely bypasses the historical timeline/bucket architecture and returns a lightweight, real-time snapshot, saving tokens and reducing latency.

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

### 2. Air Quality (`air_quality.py`)
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

### 3. Doors & Windows (`door_window.py`)
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

### 4. Ambient Lights (`lights.py`)
Tracks indoor illumination using a discrete 0-5 scale. Uses state-transition logic to prevent mathematical hallucinations and maps integers to semantic labels (e.g., "Level 3 - Bright").

```yaml
Query_Context:
  Domain: Ambient Light Intensity (0-5 Scale)
  Room: Classroom_101
  Timeframe: 24h (2h intervals)
  Active_Sensors: 2 (F0_Class101-IAQ-1, F0_Class101-IAQ-2)

Global_Illumination_Summary (Last 24h):
  Level 0 (Dark): 60%
  Level 3 (Bright): 25%
  Level 4 (Very Bright): 15%
  (Levels 1, 2, 5: 0%)

Timeline_Transitions:
- bucket: '2026-04-24 08:00 - 10:00'
  activity:
    Room_Aggregate: 'Transition: [Level 0 (Dark) -> Level 3 (Bright) at 08:15].'
- bucket: '2026-04-24 12:00 - 14:00'
  activity:
    F0_Class101-IAQ-1: 'Transition: [Level 3 -> Level 5 (Overcast/Sunny)]. (Likely direct sunlight)'
    F0_Class101-IAQ-2: 'Remained Level 3 (Bright).'
- bucket: '2026-04-24 16:00 - 18:00'
  activity:
    Room_Aggregate: 'Transition: [Level 3/5 -> Level 0 (Dark) at 17:30].'

Stable_Periods (No State Changes):
  - '2026-04-23 16:00 to 2026-04-24 08:00' (10 intervals): 
      State: Level 0 (Dark)
  - '2026-04-24 10:00 to 12:00' (1 interval): 
      State: Level 3 (Bright)
  - '2026-04-24 18:00 to Present' (3 intervals): 
      State: Level 0 (Dark)
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
├── agents/                     # LangGraph Agent Definitions
│   ├── __init__.py
│   ├── supervisor.py           # The main routing logic
│   ├── query_agent.py          # Read-only API and Analytics logic
│   └── action_agent.py         # Write-access and Rule generation logic
│
├── tools/                      # Domain-Specific Tool Definitions
│   ├── __init__.py
│   ├── temp_humidity.py        # Combines Temp & Humidity with Indoor/Outdoor delta
│   ├── air_quality.py          # Oxygen, CO2, TVOC, Particulate Matter
│   ├── occupancy.py            # Polymorphic logic (People/Desk/Line) with Motion fusion
│   ├── door_window.py          # Binary state transitions (Open/Closed)
│   ├── lights.py               # Ambient light state-transitions (0-5 scale)
│   └── knowledge.py            # Qdrant Vector DB semantic search tool
│
├── clients/                    # Tool clients
│   ├── __init__.py
│   └── thingsboard_client.py   # Shared logic for ThingsBoard Tools
│
├── database/                   # Vector DB Management
│   ├── __init__.py
│   ├── qdrant_client.py        # Qdrant connection and initialization
│   └── document_loader.py      # Scripts to chunk and embed PDFs/Manuals
│
├── graph/                      # LangGraph Setup
│   ├── __init__.py
│   └── workflow.py             # Compiles nodes, edges, and HITL breakpoints
│
├── data/                       # Local Knowledge Base (Unstructured Data)
│   ├── manuals/                # HVAC/Sensor PDFs
│   └── campus_topology.json    # Spatial definitions for resolving Room IDs
│
├── config/                     # Configuration
│   ├── settings.py             # Environment variables mapping
│   └── .env.example            # API keys (Gemini, ThingsBoard, Qdrant)
│
├── main.py                     # Entry point (CLI or API wrapper)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```