# Smart Campus Intelligent Assistant 🏛️🤖

## Overview
This repository contains the prototype for an LLM-based Intelligent Assistant designed for Smart Campus management, developed as part of a university thesis. The system allows building managers to interact with building infrastructure and IoT systems (lighting, HVAC, occupancy, access control) using natural language queries.

By integrating Google's Gemini 3.1-Flash via LangGraph, the assistant translates natural language into precise data requests, performs anomaly detection, queries semantic knowledge, and proposes automated infrastructure rules via ThingsBoard.

## Architecture 🏗️

The system employs an **Agentic Supervisor (Multi-Agent)** architecture built on **LangGraph**:

1. **Supervisor Node (Gemini 3.1-Flash):** The central router that analyzes user intent and delegates tasks to the appropriate specialized sub-agent.
2. **Query & Analytics Agent (Read-Only):** Handles real-time telemetry checks (e.g., "What is the CO2 level in Room 204?") and time-series aggregations (e.g., "Give me yesterday's occupancy stats"). Interfaces directly with the ThingsBoard REST API.
3. **Action & Rule Agent (Write-Access):** Proposes infrastructure changes and constructs ThingsBoard Rule Chain JSONs. **Safety constraint:** Includes a Human-in-the-Loop (HITL) approval step before deploying any changes to the active ThingsBoard instance.
4. **Semantic Memory (Qdrant Vector DB):** Utilizes Google's Embedding models to store and retrieve unstructured campus knowledge, including HVAC manuals, building topology (e.g., "Which rooms face South?"), and Standard Operating Procedures (SOPs).

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
├── tools/                      # Tool Definitions for Agents
│   ├── __init__.py
│   ├── thingsboard_api.py      # Raw REST API calls to ThingsBoard
│   ├── analytics.py            # Pandas/Stats processing for time-series
│   └── vector_search.py        # Qdrant querying logic
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
│   └── campus_topology.json    # Spatial definitions
│
├── config/                     # Configuration
│   ├── settings.py             # Environment variables mapping
│   └── .env.example            # API keys (Gemini, ThingsBoard, Qdrant)
│
├── main.py                     # Entry point (CLI or API wrapper)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
