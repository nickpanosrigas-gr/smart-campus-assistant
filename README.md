# Smart Campus Intelligent Assistant 🏛️🤖

## Overview
This repository contains the prototype for an LLM-based Intelligent Assistant designed for Smart Campus management, developed as part of a university thesis. The system allows building managers to interact with building infrastructure and IoT systems (lighting, HVAC, occupancy, access control) using natural language queries.

By integrating Google's Gemini 3.1-Flash via LangGraph, the assistant translates natural language into precise data requests, performs anomaly detection, queries semantic knowledge, and proposes automated infrastructure rules via ThingsBoard.

## Architecture 🏗️

The system employs a **Hybrid Agentic Supervisor** architecture built on **LangGraph**:

1. **Supervisor Node (Gemini 3.1-Flash):** The central router that analyzes user intent and delegates tasks to the appropriate specialized sub-agent or tool.
2. **Query & Analytics Node (Read-Only):** Handles telemetry checks and time-series aggregations. It relies on highly specific, domain-segregated tools to interact with the ThingsBoard REST API.
3. **Action & Rule Agent (Write-Access):** Proposes infrastructure changes and constructs ThingsBoard Rule Chain JSONs. **Safety constraint:** Includes a Human-in-the-Loop (HITL) approval step before deploying any changes to the active ThingsBoard instance.
4. **Semantic Memory:** Utilizes a Vector Database to store and retrieve unstructured campus knowledge, building topology, and Standard Operating Procedures (SOPs).

## Data Processing Strategy 📊
To prevent LLM context-window bloat and reduce hallucinations, this architecture does not feed raw sensor data to the LLM. Instead, data is pre-processed using the following strategy:
* **Timeframe Chunking:** Data is requested in strict timeframes (`1h`, `24h`, `7d`, `30d`) and downsampled into 12-14 bins (e.g., a 24h request is chunked into 2-hour bins).
* **Anomaly Highlighting:** For historical chunks, average data points are omitted. Only anomalous spikes (min/max deviations) are passed to the LLM.
* **The "Now" Principle:** Live data is not handled by a separate tool. Instead, whenever the `1h` timeframe is requested, the very first 5-minute chunk represents "Now" and is **always** provided to the LLM, ensuring real-time context is never omitted.

## Technology Stack 💻

* **Framework:** Python, LangChain, LangGraph
* **LLM:** Google Gemini 3.1-Flash-Preview
* **Embeddings:** Google Text Embedding Model
* **Vector Database:** Qdrant (Local/Docker or Cloud)
* **IoT Platform:** ThingsBoard (REST API & Rule Engine)

## Project Structure 📁

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
│   ├── temp_humidity_api.py    # Combines Temp & Humidity with Indoor/Outdoor delta
│   ├── air_quality_api.py      # Oxygen, CO2, TVOC, Particulate Matter
│   ├── occupancy_api.py        # People counter data logic
│   ├── door_window_api.py      # Binary state sensors (Open/Closed)
│   └── knowledge_query.py      # Qdrant Vector DB semantic search tool
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