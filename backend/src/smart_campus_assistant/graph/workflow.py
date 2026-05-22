import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# Import the existing LLM setup and tools from your supervisor
# (Note: Visual/UI intent pre-processing imports have been removed)
from src.smart_campus_assistant.agents.supervisor import supervisor_llm, all_campus_tools, supervisor_prompt

logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    # Rolling UI State: Overwrites itself instead of appending (no reducer)
    map_context: dict 

# ==========================================
# 2. DEFINE THE NODES
# ==========================================
async def call_supervisor(state: GraphState, config: RunnableConfig):
    messages = state["messages"]
    map_context = state.get("map_context", {})
    
    # Dynamically inject the rolling map context into the prompt if it exists
    dynamic_prompt = supervisor_prompt
    if map_context:
        dynamic_prompt += f"\n\n[SYSTEM LOG]: The user is currently viewing the following map data: {map_context}"
        
    full_context = [SystemMessage(content=dynamic_prompt)] + messages
    logger.info("Supervisor LLM is evaluating the state...")
    
    # Use ainvoke for asynchronous execution
    response = await supervisor_llm.ainvoke(full_context, config=config)
    return {"messages": [response]}

tool_node = ToolNode(all_campus_tools)

def should_continue(state: GraphState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ==========================================
# 3. BUILD AND COMPILE THE GRAPH
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("supervisor", call_supervisor)
workflow.add_node("tools", tool_node)

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
    
    # --- MAIN LANGGRAPH EXECUTION (STREAMING) ---
    config = {"configurable": {"thread_id": thread_id}}
    
    # Track backend data to send the perfect JSON UI payload without LLM hallucinations
    ui_sync_data = {"tools": set(), "rooms": set()}
    ui_sync_sent = False
    
    try:
        async for event in app.astream_events(
            {"messages": [HumanMessage(content=user_query)]}, 
            config=config, 
            version="v2"
        ):
            kind = event["event"]
            
            # A. Catch Tool Starts (Dynamic Status Text & Aggregation)
            if kind == "on_tool_start":
                tool_name = event["name"]
                args = event["data"].get("input", {})
                
                # Add to our UI JSON Tracker
                ui_sync_data["tools"].add(tool_name)
                room_id = args.get("room_id") or args.get("room")
                if room_id:
                    ui_sync_data["rooms"].add(room_id)
                
                # Format a dynamic, clean status message for the user
                clean_tool_name = tool_name.replace("get_", "").replace("_", " ").title()
                status_msg = f"Checking {clean_tool_name} for {room_id}..." if room_id else f"Running {clean_tool_name}..."
                
                await websocket.send_json({
                    "type": "status",
                    "message": status_msg
                })
            
            # B. Catch Tool Ends (Extracting Raw Artifacts for Frontend)
            elif kind == "on_tool_end":
                tool_name = event["name"]
                output = event["data"].get("output")
                
                # Safely extract artifact depending on LangChain tool setup
                raw_data = None
                if isinstance(output, ToolMessage) and hasattr(output, 'artifact') and output.artifact:
                    raw_data = output.artifact
                elif isinstance(output, tuple) and len(output) > 1:
                    raw_data = output[1]
                    
                if raw_data:
                    await websocket.send_json({
                        "type": "tool_data",
                        "tool": tool_name,
                        "raw_data": raw_data
                    })
            
            # C. Catch Chat Stream (Typewriter effect & Synchronize Map UI)
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                
                # The moment the LLM is ready to answer, emit the perfect UI layout JSON FIRST
                if not ui_sync_sent and ui_sync_data["tools"]:
                    await websocket.send_json({
                        "type": "ui_sync",
                        "data": {
                            "tools": list(ui_sync_data["tools"]),
                            "rooms": list(ui_sync_data["rooms"])
                        }
                    })
                    ui_sync_sent = True
                
                # Then stream the text naturally
                if chunk:
                    await websocket.send_json({
                        "type": "text_stream",
                        "chunk": chunk
                    })
                    
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        await websocket.send_json({
            "type": "text_stream",
            "chunk": "\n[System Error: Unable to process request.]"
        })
        
    # --- END STREAM SIGNAL ---
    await websocket.send_json({"type": "stream_end"})


async def handle_map_interaction(rooms: list, floor: str, domain: str, thread_id: str, websocket):
    """Bypasses LLM, runs tool directly for multiple rooms, and silently updates graph memory."""
    
    tool_mapping = {
        "Occupancy": "get_occupancy",
        "Air Quality": "get_air_quality",
        "Climate": "get_temp_humidity",
        "Lights": "get_lights",
        "Doors/Windows": "get_door_window",
    }
    
    tool_name = tool_mapping.get(domain)
    if not tool_name:
        return
        
    target_tool = next((t for t in all_campus_tools if t.name == tool_name), None)
    if not target_tool:
        return

    # 1. Resolve "ALL" to actual rooms based on the active floor
    if "ALL" in rooms:
        if floor == "2":
            target_rooms = ["2.1", "2.2", "2.3", "2.4"]
        elif floor == "B":
            target_rooms = ["building"]
        else:
            target_rooms = []
    else:
        target_rooms = rooms

    combined_logs = []
    room_health_updates = {}
    
    # Map raw backend UI color outputs to RoomHealth standard states for the frontend
    color_to_health = {
        "green": "Good",
        "orange": "Warning",
        "red": "Error"
    }

    # 2. Execute tools for each selected room
    for room in target_rooms:
        try:
            # CRITICAL FIX: Pydantic schema expects 'room', not 'room_id'
            result = await target_tool.ainvoke({"room": room, "timeframe": "now"})
            
            yaml_summary = result[0] if isinstance(result, tuple) else str(result)
            raw_data = result[1] if isinstance(result, tuple) else result 

            combined_logs.append(f"--- Room {room} ---\n{yaml_summary}")
            
            if isinstance(raw_data, dict) and "status_color" in raw_data:
                color = raw_data["status_color"]
                room_health_updates[room] = color_to_health.get(color, "Good")

        except Exception as e:
            logger.error(f"Direct tool execution failed for {domain} in {room}: {e}")

    # 3. Instant UI Update
    if room_health_updates:
        await websocket.send_json({
            "type": "map_update",
            "target_rooms": target_rooms,
            "room_data": room_health_updates
        })
        
    # 4. Silent Context Injection
    if combined_logs:
        config = {"configurable": {"thread_id": thread_id}}
        full_log = "\n".join(combined_logs)
        context_msg = SystemMessage(
            content=f"[SYSTEM LOG]: The user clicked on the map to view {domain} for rooms: {', '.join(target_rooms)}. Current data:\n{full_log}"
        )
        app.update_state(config, {"messages": [context_msg]})