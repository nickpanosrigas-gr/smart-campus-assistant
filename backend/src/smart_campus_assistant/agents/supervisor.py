import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# Import project config
from src.smart_campus_assistant.config.settings import settings

# ==========================================
# 1. IMPORT ALL RAW TOOLS DIRECTLY
# ==========================================
from src.smart_campus_assistant.tools.topology import search_topology
from src.smart_campus_assistant.tools.schedule import (
    get_room_schedule, get_course_schedule, get_instructor_schedule, get_semester_schedule
)
from src.smart_campus_assistant.tools.temp_humidity import get_temp_humidity
from src.smart_campus_assistant.tools.air_quality import get_air_quality
from src.smart_campus_assistant.tools.occupancy import get_occupancy
from src.smart_campus_assistant.tools.door_window import get_door_window_status
from src.smart_campus_assistant.tools.lights import get_ambient_lights
from src.smart_campus_assistant.tools.energy import get_energy_infrastructure
from src.smart_campus_assistant.tools.diagnostics import get_campus_diagnostics
from src.smart_campus_assistant.tools.visual_sync import verify_ui_state

logger = logging.getLogger(__name__)

# ==========================================
# 2. CONFIGURE THE SUPERVISOR PROMPT
# ==========================================
# We must be extremely explicit here because the 4B model has to choose between ~12 tools now.
supervisor_prompt = """You are the Smart Campus Assistant.
Your job is to answer the user's request by calling the correct data tools and explaining the results.

CRITICAL INSTRUCTIONS & TOOL SELECTION RULES:
1. TOPOLOGY FIRST: If the user asks about a broad area (e.g., "second floor") and you do not know the exact room IDs, you MUST call 'search_topology' FIRST.
2. SCHEDULES: If asking 'when' or 'where' a class/teacher is, use the get_*_schedule tools. Map the timeframe to "now", "today", or "week".
3. TELEMETRY: 
   - Temperature/Humidity -> get_temp_humidity
   - CO2, TVOC, Air -> get_air_quality
   - People, desks, queue -> get_occupancy
   - Open/Closed doors -> get_door_window_status
   - Brightness/Illumination -> get_ambient_lights
4. FACILITIES:
   - Power/kWh -> get_energy_infrastructure
   - Broken sensors/offline -> get_campus_diagnostics
5. MISSING SENSORS: If a tool returns "Error: No sensors found", DO NOT RETRY. Just tell the user there are no sensors in that room.
6. NO SUMMARIZATION OF ERRORS: If a tool fails, read the error to understand what arguments you got wrong, and try ONE more time.
7. SYNTHESIS (CRITICAL): Once a tool successfully returns data, you MUST write a clear, conversational response to the user summarizing the answer. NEVER return an empty response.

DO NOT guess data. ALWAYS use tools to fetch real-time campus information."""

# ==========================================
# 3. INITIALIZE OLLAMA AND BIND ALL TOOLS
# ==========================================

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    num_ctx=settings.OLLAMA_NUM_CTX, 
    temperature=0,
    think=False,
    disable_thinking=True
)

# Combine ALL tools into one massive arsenal for the LLM
all_campus_tools = [
    search_topology,
    get_room_schedule, get_course_schedule, get_instructor_schedule, get_semester_schedule,
    get_temp_humidity, get_air_quality, get_occupancy, get_door_window_status, get_ambient_lights,
    get_energy_infrastructure, get_campus_diagnostics, verify_ui_state
]

supervisor_llm = llm.bind_tools(all_campus_tools)

# Note: We rename 'sub_systems' to 'all_campus_tools'. 
# YOU MUST UPDATE YOUR workflow.py TO IMPORT 'all_campus_tools' INSTEAD OF 'sub_systems'!

def run_supervisor(user_query: str, config: dict = None) -> str:
    """The main execution loop for the Supervisor."""
    
    messages = [
        SystemMessage(content=supervisor_prompt),
        HumanMessage(content=user_query)
    ]
    
    logger.info("Analyzing request and determining routing strategy...")
    
    ai_msg = supervisor_llm.invoke(messages, config=config)
    messages.append(ai_msg)
    
    if not ai_msg.tool_calls:
        return ai_msg.content
    
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        tool_obj = next((t for t in all_campus_tools if t.name == tool_name), None)
        if tool_obj:
            try:
                raw_data = tool_obj.invoke(tool_args, config={"callbacks": []})
                messages.append(ToolMessage(content=str(raw_data), tool_call_id=tool_id))
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                messages.append(ToolMessage(content=f"Error in {tool_name}: {e}", tool_call_id=tool_id))
        else:
            logger.warning(f"Tool {tool_name} not found.")
            messages.append(ToolMessage(content=f"Error: {tool_name} not found.", tool_call_id=tool_id))
            
    logger.info("Reading raw data and synthesizing final answer...")
    final_ai_msg = supervisor_llm.invoke(messages, config=config)
    
    return final_ai_msg.content

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(levelname)s - %(message)s')
    logger.info("Testing Supervisor Agent (Ollama)...")
    
    # Test query using the new Knowledge capabilities
    user_query = "Where is Dr. Angeliki Presvelou's office and what sensors are in it?"
    logger.info(f"User Query: {user_query}")
    
    # Run the Supervisor
    final_output = run_supervisor(user_query)
    
    logger.info("FINAL SUPERVISOR RESPONSE:")
    print(final_output)