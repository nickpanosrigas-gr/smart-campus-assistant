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
Fuses multiple room sensors and incorporates dynamic baseline deviations, fixed limits, and external weather context.

```yaml
Query_Context:
  Domain: Temperature & Humidity
  Room: Restaurant
  Timeframe: 24h (2h intervals)
  Active_Sensors: 3 (F0_Restaurant-IAQ-1 to 3) + 1 (Roof_Weather_Station)

External_Weather_Context (Last 24h):
  Temp_avg: 31.5°C (Extreme Heat)
  Hum_avg: 65.0%

Global_Room_Baseline (Last 24h):
  Temp_avg: 22.4°C (Normal: 20°C - 25°C)
  Hum_avg: 45.2% (Normal: 30% - 60%)

Timeline_Exceptions:
- bucket: '2026-04-24 08:00 - 10:00'
  trigger: 'Baseline Deviation (> +2.0°C)'
  room_aggregate:
    Temp_max: 25.8°C
    Temp_avg: 24.5°C
- bucket: '2026-04-24 14:00 - 16:00'
  trigger: 'Threshold Breach'
  details: 'Room Temp reached 27.2°C. Correlates with Peak Outdoor Temp (34.0°C). Likely HVAC under-capacity or open doors.'
  room_aggregate:
    Temp_max: 27.2°C
    Hum_max: 68%

Stable_Periods: 
  - '2026-04-23 16:00 to 2026-04-24 08:00' (10 intervals)
  - '2026-04-24 10:00 to 14:00' (2 intervals)
  Status: Values stable. HVAC maintained ~10°C delta from outdoor temperature successfully.
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

### 4. Occupancy (`occupancy.py`)
Utilizes a Polymorphic Schema to handle different data shapes (Binary Desk Sensors, Continuous People Counters, and Queue Trackers) and incorporates Auxiliary Motion Sensors to cross-validate data and detect anomalies (like "Ghost Occupancy" or stuck sensors) without breaking the established YAML structure.

```yaml
Query_Context:
  Domain: Occupancy & Presence
  Room: Classroom_101
  Timeframe: 24h (2h intervals)
  Primary_Sensor: People_Counter (Continuous Count)
  Auxiliary_Sensors: 2 (F0_Classroom_101-IAQ_Main-Motion, F0_Classroom_101-IAQ_Back-Motion)

Global_Occupancy_Summary (Last 24h):
  Peak_Occupancy: 45 people
  Utilization_Profile: Active 6h / Empty 18h
  Aux_Motion_Profile: Active 7h / Idle 17h (High Correlation)

Timeline_Activity (Significant Movements & Anomalies):
- bucket: '2026-04-24 08:00 - 10:00'
  activity: 'Mass Arrival & Motion Sync'
  metrics:
    Net_Change: +42 people (Peak: 42)
    Aux_Motion: 'Transition: [Idle -> Active at 08:05]. Sustained Active.'
- bucket: '2026-04-24 10:00 - 12:00'
  activity: 'Mass Departure'
  metrics:
    Net_Change: -42 people (Room emptied at 10:15)
    Aux_Motion: 'Transition: [Active -> Idle at 10:20].'
- bucket: '2026-04-24 22:00 - 24:00'
  activity: 'Anomaly: Uncorrelated Presence (Ghost Occupancy)'
  details: 'Primary counter reads 0, but sustained motion detected. Likely cleaning crew, security patrol, or people counter missed entry.'
  metrics:
    Net_Change: 0 people (Primary reads 0)
    Aux_Motion: 'Transition: [Idle -> Active at 22:15]. Toggled 14 times. Ended Idle at 23:30.'

Stable_Periods:
  - '2026-04-23 16:00 to 2026-04-24 08:00' (10 intervals): 
      Status: Empty (0 people) | Motion: Idle (0 toggles)
  - '2026-04-24 12:00 to 22:00' (5 intervals): 
      Status: Empty (0 people) | Motion: Idle (0 toggles)
```

### 5. Illumination (lights.py)
Translates categorical 0-5 light intensity indices into semantic states (Dark to Overcast/Sunny). It focuses on significant state transitions (e.g., Sudden Spikes vs. Gradual Increases) to provide context-agnostic activity cues to the LLM.

```yaml
Query_Context:
  Domain: Illumination (0-5 Index)
  Room: Classroom_101
  Timeframe: 24h (2h intervals)
  Active_Sensors: 2 (F0_Classroom_101-IAQ_Main, F0_Classroom_101-IAQ_Back)

Global_Illumination_Summary (Last 24h):
  Peak_Level: 4 (Very Bright)
  Lowest_Level: 0 (Dark)
  Profile: Dark 14h / Illuminated 10h

Timeline_Transitions:
- bucket: '2026-04-24 06:00 - 08:00'
  activity: 'Gradual Illumination Increase'
  metrics:
    Transition: '0 (Dark) -> 2 (Normal)'
    Details: 'Steady rise across the bucket. (Typical of sunrise).'
- bucket: '2026-04-24 08:00 - 10:00'
  activity: 'Sudden Illumination Spike'
  metrics:
    Transition: '2 (Normal) -> 4 (Very Bright)'
    Details: 'Rapid jump at 08:15. Sustained at level 4.'
- bucket: '2026-04-24 18:00 - 20:00'
  activity: 'Sudden Illumination Drop'
  metrics:
    Transition: '3 (Bright) -> 0 (Dark)'
    Details: 'Immediate drop to 0 (Dark) at 18:30.'

Stable_Periods:
  - '2026-04-23 20:00 to 2026-04-24 06:00' (5 intervals): 
      State: Maintained 0 (Dark)
  - '2026-04-24 10:00 to 18:00' (4 intervals): 
      State: Fluctuated between 3 (Bright) and 4 (Very Bright)
```

### 6. Real-Time Snapshots ("Now" Timeframe)
When the LLM Orchestrator needs instant situational awareness, it passes `timeframe: "now"` to any tool. This branches the backend logic to skip historical aggregation entirely and return a token-efficient, zero-latency state check.

```yaml
Query_Context:
  Domain: Doors & Windows
  Room: Classroom_101
  Timeframe: Now (Snapshot)

Current_State:
  Door_Main: Closed
  Win_Front: Open
  Win_Mid: Closed
  Win_Back: Closed
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
│   ├── occupancy.py            # Polymorphic logic for people/desk/line counters + Aux Motion
│   ├── door_window.py          # Binary state transitions (Open/Closed)
│   ├── lights.py               # 0-5 Categorical illumination states and transitions
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