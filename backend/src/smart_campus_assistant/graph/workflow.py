import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Import the existing LLM setup and tools from your supervisor
from src.smart_campus_assistant.agents.supervisor import supervisor_llm, all_campus_tools, supervisor_prompt
from src.smart_campus_assistant.agents.visual import get_ui_intent

logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

# ==========================================
# 2. DEFINE THE NODES
# ==========================================
async def call_supervisor(state: GraphState, config: dict):
    messages = state["messages"]
    full_context = [SystemMessage(content=supervisor_prompt)] + messages
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
    """Handles the sequential UI routing and then streams the LangGraph execution."""
    
    # --- 1. SEQUENTIAL UI ROUTING (FAST PATH) ---
    try:
        ui_intent = await get_ui_intent(user_query)
        await websocket.send_json({
            "type": "ui_layout",
            "data": ui_intent.dict()
        })
    except Exception as e:
        logger.error(f"UI Routing failed: {e}")
    
    # --- 2. MAIN LANGGRAPH EXECUTION (STREAMING) ---
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        async for event in app.astream_events(
            {"messages": [HumanMessage(content=user_query)]}, 
            config=config, 
            version="v2"
        ):
            kind = event["event"]
            
            # A. Catch Tool Starts (Status Updates & UI Correction)
            if kind == "on_tool_start":
                tool_name = event["name"]
                if tool_name == "verify_ui_state":
                    # The Supervisor is locking in the final UI view
                    args = event["data"].get("input", {})
                    await websocket.send_json({
                        "type": "ui_correction",
                        "rooms": args.get("rooms", []),
                        "domains": args.get("domains", [])
                    })
                else:
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Calling {tool_name}..."
                    })
            
            # B. Catch Tool Ends (Extracting Raw Artifacts for Frontend)
            elif kind == "on_tool_end":
                tool_name = event["name"]
                if tool_name != "verify_ui_state":
                    output = event["data"].get("output")
                    
                    # Safely extract artifact depending on LangChain tool setup
                    raw_data = None
                    if isinstance(output, ToolMessage) and output.artifact:
                        raw_data = output.artifact
                    elif isinstance(output, tuple) and len(output) > 1:
                        raw_data = output[1]
                        
                    if raw_data:
                        await websocket.send_json({
                            "type": "tool_data",
                            "tool": tool_name,
                            "raw_data": raw_data
                        })
            
            # C. Catch Chat Stream (Typewriter text effect)
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
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
        
    # --- 3. END STREAM SIGNAL ---
    await websocket.send_json({"type": "stream_end"})


async def handle_map_interaction(room: str, domain: str, thread_id: str, websocket):
    """Bypasses LLM, runs tool directly for zero-latency UI updates, and silently updates graph memory."""
    
    # Map domain strings to your actual tool function names
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
        
    # Locate the tool from your compiled list
    target_tool = next((t for t in all_campus_tools if t.name == tool_name), None)
    
    if target_tool:
        try:
            # 1. Direct Execution
            result = await target_tool.ainvoke({"room_id": room, "timeframe": "now"})
            
            # Unpack the YAML summary and raw artifact
            yaml_summary = result[0] if isinstance(result, tuple) else str(result)
            raw_data = result[1] if isinstance(result, tuple) else result 

            # 2. Instant UI Update
            await websocket.send_json({
                "type": "map_data_update",
                "room": room,
                "domain": domain,
                "data": raw_data
            })
            
            # 3. Silent Context Injection
            config = {"configurable": {"thread_id": thread_id}}
            context_msg = SystemMessage(
                content=f"[SYSTEM LOG]: The user just clicked on the map to view the {domain} for {room}. The current data is:\n{yaml_summary}"
            )
            # Update the Thread memory without generating an LLM response
            app.update_state(config, {"messages": [context_msg]})
            
        except Exception as e:
            logger.error(f"Direct tool execution failed for {domain} in {room}: {e}")