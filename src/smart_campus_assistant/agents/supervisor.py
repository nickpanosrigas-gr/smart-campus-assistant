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

logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE SUB-AGENTS & TOOLS 
# ==========================================

@tool
def ask_telemetry_agent(query: str) -> str:
    """
    Call this agent to fetch raw sensor data, historical metrics, or current states (occupancy, lights, temp, humidity, air quality).
    CRITICAL: Your 'query' MUST explicitly state the target ROOM NAME and the TIMEFRAME.
    - BAD Query: 'How is the air quality?'
    - GOOD Query: 'Fetch air quality for the restaurant for timeframe: now.'
    """
    logger.info(f"[Telemetry Node]: Hitting API for query: '{query}'")
    return run_telemetry_agent(query)

@tool
def ask_scheduler_agent(query: str) -> str:
    """
    Call this agent to fetch academic schedules, class times, and university programs.
    CRITICAL: Your 'query' MUST explicitly state the TARGET (exact room, exact teacher, exact course, or semester) and the TIMEFRAME (now, today, week, Monday, etc.).
    - BAD Query: 'Where is the CS class?'
    - GOOD Query: 'Find the room and time for course: Introduction to Computer Science for timeframe: week.'
    """
    logger.info(f"[Scheduler Node]: Hitting Registry for query: '{query}'")
    return run_scheduler_agent(query)

@tool
def ask_diagnostics_agent(query: str) -> str:
    """
    Call this agent for troubleshooting hardware, finding offline sensors, or checking battery levels.
    CRITICAL: Your 'query' MUST directly target specific device types or rooms.
    - BAD Query: 'Why is the AC not working?'
    - GOOD Query: 'Run diagnostic health checks on all HVAC sensors in the kitchen.'
    """
    logger.info(f"[Diagnostics Node]: Running health checks for: '{query}'")
    return "MOCK_DATA: All sensors are online. Battery levels normal."

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

@tool
def query_knowledge_base(query: str) -> str:
    """
    Call this tool to search the Vector Database for manuals, topologies, or SOPs.
    CRITICAL: Your 'query' MUST be a concise list of search keywords, not a conversational sentence.
    - BAD Query: 'How do I reset the main router in the data center?'
    - GOOD Query: 'Data center main router hard reset procedure SOP.'
    """
    logger.info(f"[Knowledge Base]: Searching Vector DB for: '{query}'")
    return "MOCK_DATA: Found manual."


# ==========================================
# 2. CONFIGURE THE SUPERVISOR
# ==========================================

supervisor_prompt = """You are the Supreme Supervisor Agent for a Smart Campus.
Your job is to route the user's request to the correct sub-agent, evaluate the raw data they return, and synthesize a clear, helpful final answer.

CRITICAL INSTRUCTIONS:
1. TRANSLATION RULE: Never pass the user's raw conversational question to a sub-agent. You must translate their intent into a highly specific, declarative data-fetching command.
2. PARAMETER EXTRACTION: You must extract concrete parameters from the user's request (e.g., specific room names, times, days, names) and embed them explicitly in the 'query' you send to the sub-agent. 
3. MULTI-ROUTING: If the user asks for multiple distinct things, you MUST trigger multiple sub-agent tools simultaneously.
4. REFLECTION & RETRY (CRITICAL): When a sub-agent returns data, evaluate if it actually answers the user's question. 
   - If the data is missing, incomplete, or returns an error (e.g., "Room not found"), DO NOT give up. 
   - You MUST generate a NEW tool call with different, adjusted parameters (e.g., try a different timeframe, check a different room, or use a broader search term). 
   - Keep retrying until you have the correct data or have exhausted logical alternatives.
5. SYNTHESIS: Once you have successfully gathered all necessary data, synthesize it into a clean, conversational response. Do not expose raw YAML/JSON formatting to the user.
6. FINAL FALLBACK: Only if you have retried multiple times and still cannot find the data, apologize to the user and explain exactly what you tried to look up and why it failed."""

# ==========================================
# 2. CONFIGURE THE SUPERVISOR
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

# Bind the sub-agents to the LLM
sub_systems = [ask_telemetry_agent, ask_scheduler_agent, ask_diagnostics_agent, ask_rule_agent, query_knowledge_base]
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
    
    # 2. Pass the config parameter into the final LLM call as well
    final_ai_msg = supervisor_llm.invoke(messages, config=config)
    
    return final_ai_msg.content

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(levelname)s - %(message)s')
    logger.info("Testing Supervisor Agent (Ollama)...")
    
    # Test query
    user_query = "what is the highest number of people in the restaurant this year"
    logger.info(f"User Query: {user_query}")
    
    # Run the Supervisor
    final_output = run_supervisor(user_query)
    
    logger.info("FINAL SUPERVISOR RESPONSE:")
    # We leave one print here just to display the final output cleanly in the terminal during testing
    print(final_output)