import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
import random

from transformers import AutoTokenizer

from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.agents.supervisor import supervisor_llm, all_campus_tools, supervisor_prompt

logger = logging.getLogger(__name__)

# Pre-load Qwen Tokenizer for Exact Context Math
try:
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Chat")
except Exception as e:
    logger.warning(f"Could not load Qwen tokenizer, falling back to tiktoken: {e}")
    qwen_tokenizer = None

# ==========================================
# STATUS PHRASE DICTIONARIES
# ==========================================
INITIAL_THINKING_PHRASES = [
    "Interpreting your request...",
    "Checking data availability...",
    "Identifying necessary insights...",
    "Defining search parameters..."
]

PROCESSING_PHRASES = [
    "Interpreting sensor feedback...",
    "Correlating data trends...",
    "Validating threshold status...",
    "Drafting final summary..."
]

TOOL_PHRASES = {
    "get_air_quality": [
        "Checking air quality in {room} for {timeframe}...",
        "Synthesizing {timeframe} air quality report for {room}..."
    ],
    "get_campus_diagnostics": [
        "Running health audit on {room} devices ({timeframe})...",
        "Analyzing hardware connectivity for {room} ({timeframe})..."
    ],
    "get_door_window_status": [
        "Scanning access states for {room} ({timeframe})...",
        "Reviewing {timeframe} entry logs for {room}..."
    ],
    "get_ambient_lights": [
        "Measuring ambient light in {room} for {timeframe}...",
        "Processing {timeframe} illumination trends in {room}..."
    ],
    "get_occupancy": [
        "Calculating occupancy density for {room} ({timeframe})...",
        "Evaluating {timeframe} usage patterns in {room}..."
    ],
    # Map all schedule tools to the schedule phrases
    "get_room_schedule": [
        "Loading {timeframe} schedule data for {room}...",
        "Verifying course occupancy for {room} ({timeframe})..."
    ],
    "get_temp_humidity": [
        "Checking climate stats for {room} ({timeframe})...",
        "Analyzing {timeframe} climate stability in {room}..."
    ]
}

# ==========================================
# CONSTANTS & MAPPINGS
# ==========================================
BACKEND_TO_UI_TOOLS = {
    "get_room_schedule": "Schedule",
    "get_course_schedule": "Schedule", 
    "get_instructor_schedule": "Schedule", 
    "get_semester_schedule": "Schedule",
    "get_temp_humidity": "Climate",
    "get_air_quality": "Air Quality",
    "get_occupancy": "Occupancy",
    "get_door_window_status": "Doors/Windows",
    "get_ambient_lights": "Lights",
    "get_campus_diagnostics": "Diagnostics"
}

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    map_context: dict 

# ==========================================
# 2. DEFINE THE NODES
# ==========================================
async def call_supervisor(state: GraphState, config: RunnableConfig):
    messages = state["messages"]
    map_context = state.get("map_context", {})
    
    dynamic_prompt = supervisor_prompt
    if map_context:
        dynamic_prompt += f"\n\n[SYSTEM LOG]: The user is currently viewing the following map data: {map_context}"
        
    full_context = [SystemMessage(content=dynamic_prompt)] + messages
    logger.info("Supervisor LLM is evaluating the state...")
    
    response = await supervisor_llm.ainvoke(full_context, config=config)
    return {"messages": [response]}

base_tool_node = ToolNode(all_campus_tools)

async def safe_tool_node(state: GraphState, config: RunnableConfig):
    """Wraps the standard ToolNode to ensure outputs are msgpack serializable for MemorySaver."""
    result = await base_tool_node.ainvoke(state, config)
    
    # Scrub un-serializable artifacts from the ToolMessages before saving to graph state memory
    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            if hasattr(msg, "artifact") and msg.artifact is not None:
                msg.artifact = "Artifact removed for memory serialization"
                
    return result

def should_continue(state: GraphState):
    last_message = state["messages"][-1]
    
    # Safely check if the message possesses the tool_calls attribute first
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# ==========================================
# 3. BUILD AND COMPILE THE GRAPH
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("supervisor", call_supervisor)
workflow.add_node("tools", safe_tool_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", should_continue, ["tools", END])
workflow.add_edge("tools", "supervisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================
# 4. FASTAPI WEBSOCKET HANDLERS
# ==========================================

async def process_chat_message(user_query: str, thread_id: str, websocket):
    """Handles the user query directly, tracks tools dynamically, and streams results."""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    ui_sync_data = {"tools": set(), "rooms": set()}
    accumulated_text = ""
    has_called_tools = False # NEW: Tracks if we are in the 'Initial' or 'Processing' phase
    
    try:
        async for event in app.astream_events(
            {"messages": [HumanMessage(content=user_query)]}, 
            config=config, 
            version="v2"
        ):
            kind = event["event"]
            
            # A. Catch LLM Start (Initial Thinking vs. Data Processing)
            if kind == "on_chat_model_start":
                # If tools were already used, the LLM is now generating the final answer
                status_msg = random.choice(PROCESSING_PHRASES) if has_called_tools else random.choice(INITIAL_THINKING_PHRASES)
                
                await websocket.send_json({
                    "type": "llm_status",
                    "state": "thinking",
                    "message": status_msg
                })

            # B. Catch Tool Starts (Dynamic Tool Phrases)
            elif kind == "on_tool_start":
                tool_name = event["name"]
                if tool_name != "tools":
                    has_called_tools = True # Flag that we have entered the tool phase
                    
                    args = event["data"].get("input", {})
                    room_id = args.get("room_id") or args.get("room") or "selected area"
                    timeframe = args.get("timeframe") or "now"
                    
                    logger.info(f"[AGENT TOOL] Executing: {tool_name} | Target: {room_id} | Args: {args}")
                    
                    ui_tool_name = BACKEND_TO_UI_TOOLS.get(tool_name, tool_name)
                    ui_sync_data["tools"].add(ui_tool_name)
                    if room_id != "selected area":
                        ui_sync_data["rooms"].add(room_id)
                    
                    # Fetch dynamic phrase or fallback to generic
                    phrases = TOOL_PHRASES.get(tool_name, [f"Running {ui_tool_name} for {{room}}..."])
                    status_msg = random.choice(phrases).format(room=room_id, timeframe=timeframe)
                    
                    # Send unified llm_status payload
                    await websocket.send_json({
                        "type": "llm_status",
                        "state": "tool_use",
                        "tool_name": ui_tool_name,
                        "message": status_msg
                    })
            
            # C. Catch Tool Ends (Artifact Routing)
            elif kind == "on_tool_end":
                tool_name = event["name"]
                if tool_name != "tools":
                    output = event["data"].get("output")
                    
                    raw_data = None
                    if hasattr(output, 'artifact') and output.artifact:
                        raw_data = output.artifact
                    elif isinstance(output, tuple) and len(output) > 1:
                        raw_data = output[1]
                    elif isinstance(output, dict):
                        raw_data = output
                        
                    if raw_data:
                        if isinstance(raw_data, list):
                            for artifact in raw_data:
                                await websocket.send_json({
                                    "type": "map_update",
                                    "artifact": artifact
                                })
                        elif isinstance(raw_data, dict) and "view_type" in raw_data:
                            await websocket.send_json({
                                "type": "map_update",
                                "artifact": raw_data
                            })
            
            # D. Catch Chat Stream
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                
                if isinstance(chunk, str) and chunk:
                    accumulated_text += chunk
                    await websocket.send_json({
                        "type": "text",
                        "text": accumulated_text
                    })
                    
    except Exception as e:
        err_msg = str(e).lower()
        if "close message has been sent" in err_msg or "closed" in err_msg or "disconnect" in err_msg:
            logger.warning(f"[STREAM] Frontend dropped connection. Halting LLM stream gracefully.")
            return
            
        logger.error(f"[GRAPH ERROR] {e}")
        try:
            await websocket.send_json({"type": "text", "text": "\n[System Error: Unable to process request.]"})
        except Exception:
            pass # Socket is completely dead, ignore
        
    # --- CALCULATE EXACT QWEN TOKENS ---
    current_state = app.get_state(config)
    messages = current_state.values.get("messages", [])
    
    try:
        full_text = "\n".join([str(m.content) for m in messages])
        if qwen_tokenizer:
            token_count = len(qwen_tokenizer.encode(full_text))
        else:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            token_count = len(enc.encode(full_text))
    except Exception as e:
        logger.error(f"Token parsing failed: {e}")
        token_count = len(str(messages)) // 4 
        
    # --- EXTRACT ALL TOOLS (LLM + MANUAL MAP CLICKS) ---
    session_tools = []
    for msg in messages:
        # 1. LLM Executed Tools
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                t_name = BACKEND_TO_UI_TOOLS.get(tc["name"], tc["name"])
                args = tc.get("args", {})
                room = args.get("room") or args.get("room_id") or "Unknown Area"
                session_tools.append({"tool": t_name, "room": room})
        
        # 2. Map-clicked Manual Tools (Parsed from System Logs)
        msg_content = getattr(msg, "content", msg.get("content", "") if isinstance(msg, dict) else "")
        
        if isinstance(msg_content, str) and "The user clicked on the map to view" in msg_content:
            try:
                after_view = msg_content.split("view ")[1]
                domain_part = after_view.split(" for rooms: ")[0].strip()
                
                after_rooms = after_view.split(" for rooms: ")[1]
                rooms_part = after_rooms.split(". Current data:")[0].strip()
                
                for r in rooms_part.split(","):
                    session_tools.append({"tool": domain_part, "room": r.strip()})
            except Exception as e:
                logger.error(f"Failed to parse system log for tools: {e}")
                
    await websocket.send_json({
        "type": "context_update",
        "tokens": token_count,
        "session_tools": session_tools
    })
        
    await websocket.send_json({"type": "resolved"})


async def handle_map_interaction(rooms: list, floor: str, domain: str, thread_id: str, websocket):
    """Bypasses LLM, runs tool directly for multiple rooms, and silently updates graph memory."""
    
    # Reverse map the UI domain to the backend tool name
    tool_name = None
    for backend_name, ui_name in BACKEND_TO_UI_TOOLS.items():
        if ui_name == domain:
            tool_name = backend_name
            break
            
    if not tool_name:
        return
        
    target_tool = next((t for t in all_campus_tools if t.name == tool_name), None)
    if not target_tool:
        return

    # Complete Campus Floor Topology
    if "ALL" in rooms:
        if floor == "-3": target_rooms = ["parkin.c"]
        elif floor == "-2": target_rooms = ["parkin.b"]
        elif floor == "-1": target_rooms = ["data_center"]
        elif floor == "0": target_rooms = ["entrance", "restaurant"]
        elif floor == "1": target_rooms = ["1.1", "1.2", "kitchen"]
        elif floor == "2": target_rooms = ["2.1", "2.2", "2.3", "2.4"]
        elif floor == "3": target_rooms = ["3.7", "3.8", "3.9"]
        elif floor == "4": target_rooms = ["4.9"]
        elif floor == "5": target_rooms = ["5.6", "5.7"]
        elif floor == "B": target_rooms = ["building"]
        else: target_rooms = []
    else:
        target_rooms = rooms

    logger.info(f"[USER TOOL] Map Clicked: {tool_name} | Floor: {floor} | Target Rooms: {len(target_rooms)} rooms")

    combined_logs = []
    
    # Execute tools sequentially and stream artifacts directly to frontend
    for room in target_rooms:
        try:
            result = await target_tool.ainvoke({"room": room, "timeframe": "now"})
            
            yaml_summary = result[0] if isinstance(result, tuple) else str(result)
            raw_data = result[1] if isinstance(result, tuple) else result 

            combined_logs.append(f"--- Room {room} ---\n{yaml_summary}")
            
            # Broadcast the artifacts exactly the same way the LLM loop does
            if isinstance(raw_data, list):
                for artifact in raw_data:
                    await websocket.send_json({
                        "type": "map_update",
                        "artifact": artifact
                    })
            elif isinstance(raw_data, dict) and "view_type" in raw_data:
                await websocket.send_json({
                    "type": "map_update",
                    "artifact": raw_data
                })

        except Exception as e:
            logger.error(f"Direct tool execution failed for {domain} in {room}: {e}")

    if combined_logs:
        config = {"configurable": {"thread_id": thread_id}}
        full_log = "\n".join(combined_logs)
        context_msg = SystemMessage(
            content=f"[SYSTEM LOG]: The user clicked on the map to view {domain} for rooms: {', '.join(target_rooms)}. Current data:\n{full_log}"
        )
        app.update_state(config, {"messages": [context_msg]})
        
        # --- CALCULATE EXACT QWEN TOKENS AFTER MAP CLICK ---
        current_state = app.get_state(config)
        messages = current_state.values.get("messages", [])
        
        try:
            full_text = "\n".join([str(m.content) for m in messages])
            if qwen_tokenizer:
                token_count = len(qwen_tokenizer.encode(full_text))
            else:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                token_count = len(enc.encode(full_text))
        except:
            token_count = len(str(messages)) // 4
            
        # --- EXTRACT ALL TOOLS (LLM + MANUAL MAP CLICKS) ---
        session_tools = []
        for msg in messages:
            # 1. LLM Executed Tools
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    t_name = BACKEND_TO_UI_TOOLS.get(tc["name"], tc["name"]) 
                    args = tc.get("args", {})
                    room = args.get("room") or args.get("room_id") or "Unknown Area"
                    session_tools.append({"tool": t_name, "room": room})
            
            # 2. Map-clicked Manual Tools (Parsed from System Logs)
            msg_content = getattr(msg, "content", msg.get("content", "") if isinstance(msg, dict) else "")
            
            if isinstance(msg_content, str) and "The user clicked on the map to view" in msg_content:
                try:
                    after_view = msg_content.split("view ")[1]
                    domain_part = after_view.split(" for rooms: ")[0].strip()
                    
                    after_rooms = after_view.split(" for rooms: ")[1]
                    rooms_part = after_rooms.split(". Current data:")[0].strip()
                    
                    for r in rooms_part.split(","):
                        session_tools.append({"tool": domain_part, "room": r.strip()})
                except Exception as e:
                    logger.error(f"Failed to parse system log for tools: {e}")
                    
        await websocket.send_json({
            "type": "context_update",
            "tokens": token_count,
            "session_tools": session_tools
        })