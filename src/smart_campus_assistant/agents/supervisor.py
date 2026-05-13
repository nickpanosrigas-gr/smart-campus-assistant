import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# Import project config
from src.smart_campus_assistant.config.settings import settings

# Import Agents and Tools
from src.smart_campus_assistant.agents.telemetry import run_telemetry_agent
from src.smart_campus_assistant.agents.scheduler import run_scheduler_agent
from src.smart_campus_assistant.agents.facilities import run_facilities_agent
from src.smart_campus_assistant.agents.knowledge import run_knowledge_agent

logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE SUB-AGENTS & TOOLS 
# ==========================================

@tool
def ask_telemetry_agent(query: str) -> str:
    """
    Call this agent to fetch raw sensor data, historical metrics, or current states (occupancy, lights, temp, humidity, air quality, doors, windows).
    CRITICAL: Use this for ALL environmental and physical state queries.
    Your 'query' MUST explicitly state the target ROOM NAME and the TIMEFRAME.
    - BAD Query: 'Are the windows open?'
    - GOOD Query: 'Fetch window status and temperature for room 2.1 for timeframe: now.'
    """
    logger.info(f"[Telemetry Node]: Hitting API for query: '{query}'")
    return run_telemetry_agent(query)

@tool
def ask_scheduler_agent(query: str) -> str:
    """
    Call this agent to fetch academic schedules, class times, university programs, and holidays (useful for cross-referencing occupancy).
    CRITICAL: Your 'query' MUST explicitly state the TARGET (exact room, teacher, course) and the TIMEFRAME (now, today, week, etc.).
    - BAD Query: 'Where is the CS class?'
    - GOOD Query: 'Find the room and time for course: Intro to CS for timeframe: week.'
    """
    logger.info(f"[Scheduler Node]: Hitting Registry for query: '{query}'")
    return run_scheduler_agent(query)

@tool
def ask_facilities_agent(query: str) -> str:
    """
    Call this agent for energy infrastructure (kWh consumption, live kW load) AND hardware diagnostics (offline sensors, battery levels, network health).
    CRITICAL: DO NOT use this for environmental metrics (temperatures, windows, occupancy). Use this ONLY to check if physical sensors are broken or for power loads.
    Your 'query' MUST explicitly state the TARGET and the TIMEFRAME.
    - BAD Query: 'Are the windows open in the kitchen?'
    - GOOD Query: 'Check live energy status and run hardware diagnostics on the kitchen sensors for timeframe: now.'
    """
    logger.info(f"[Facilities Node]: Running infrastructure check for: '{query}'")
    return run_facilities_agent(query)

@tool
def ask_knowledge_agent(query: str) -> str:
    """
    Call this agent to search the Vector Database for building layouts, room topologies, faculty offices, and hardware manuals.
    CRITICAL: Your 'query' MUST explicitly state what you are looking for (e.g., 'Find all rooms on the third floor' or 'What are Dr. Smith's office hours?').
    """
    logger.info(f"[Knowledge Node]: Hitting Qdrant for query: '{query}'")
    return run_knowledge_agent(query)

@tool
def ask_rule_agent(query: str) -> str:
    """
    Call this agent to create, update, or propose automation rules for ThingsBoard.
    CRITICAL: Your 'query' MUST be a precise statement of the IF/THEN automation logic.
    - BAD Query: 'Make sure the lights turn off when empty.'
    - GOOD Query: 'Draft a Rule Chain: IF room 1.2 occupancy == 0 THEN set lights to 0.'
    """
    logger.info(f"[Rule Node]: Drafting Rule Chain for: '{query}'")
    return "MOCK_DATA: Successfully drafted rule."


# ==========================================
# 2. CONFIGURE THE SUPERVISOR
# ==========================================

supervisor_prompt = """You are the Supreme Supervisor Agent for a Smart Campus.
Your job is to route the user's request to the correct sub-agent, evaluate the raw data they return, and synthesize a clear, helpful final answer.

CRITICAL INSTRUCTIONS:
1. ORDER OF OPERATIONS: If the user asks for data about a broad area (e.g., "second floor") and you do not know the exact room IDs, you MUST use the knowledge tool FIRST to understand the building layout. ONLY AFTER you know the exact room IDs should you call the Telemetry or Facilities agents.
2. EXPLICIT INTENT ONLY (CRITICAL): DO NOT fetch telemetry (occupancy, temperature, etc.) unless the user EXPLICITLY asks for current conditions, data, or metrics. If they just ask "what rooms are on the 5th floor", ONLY use the knowledge base and STOP.
3. ACCEPT MISSING SENSORS (CRITICAL): If a telemetry or facilities tool returns "Error: No sensors found" or "No data", DO NOT RETRY. Accept that the room has no sensors for that metric and simply inform the user.
4. STRICT TOOL SEPARATION: 
   - TELEMETRY is for environmental data (temperatures, windows, doors, occupancy, air quality, lights).
   - FACILITIES is ONLY for hardware health, offline sensors, and energy power loads. Do NOT use Facilities for temperatures or windows.
5. PARAMETER EXTRACTION: You must extract concrete parameters from the user's request (specific room names, times, days) and embed them explicitly in the 'query' you send to the sub-agent.
6. REFLECTION & RETRY: If a tool call fails because you formatted the parameters incorrectly, generate a NEW tool call with adjusted parameters. However, see Rule 3: DO NOT retry if the error simply states there are no sensors.
7. SYNTHESIS: Once you have successfully gathered all necessary data, synthesize it into a clean, conversational response. Do not expose raw YAML/JSON formatting to the user."""

# ==========================================
# 3. INITIALIZE OLLAMA AND BIND TOOLS
# ==========================================

# Initialize Ollama
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    num_ctx=settings.OLLAMA_NUM_CTX, 
    temperature=0,
    think=False,
    disable_thinking=True
)

sub_systems = [
    ask_telemetry_agent, 
    ask_scheduler_agent, 
    ask_facilities_agent, 
    ask_rule_agent, 
    ask_rule_agent
]
supervisor_llm = llm.bind_tools(sub_systems)

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
        
        tool_obj = next((t for t in sub_systems if t.name == tool_name), None)
        if tool_obj:
            try:
                # EXPLICITLY BLOCK Langfuse from bleeding into the sub-agents
                raw_data = tool_obj.invoke(tool_args, config={"callbacks": []})
                messages.append(ToolMessage(content=str(raw_data), tool_call_id=tool_id))
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                messages.append(ToolMessage(content=f"Error in {tool_name}: {e}", tool_call_id=tool_id))
        else:
            logger.warning(f"Tool {tool_name} not found.")
            messages.append(ToolMessage(content=f"Error: {tool_name} not found.", tool_call_id=tool_id))
            
    logger.info("Reading raw data and synthesizing final answer...")
    
    # Pass the config parameter into the final LLM call as well
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