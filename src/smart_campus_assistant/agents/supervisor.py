import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# Import project config
from src.smart_campus_assistant.config.settings import settings

# Import Agents and Tools
from src.smart_campus_assistant.agents.telemetry import run_telemetry_agent

logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE SUB-AGENTS & TOOLS 
# ==========================================

@tool
def ask_telemetry_agent(query: str) -> str:
    """
    Call this agent to fetch raw sensor data, historical metrics, or current states.
    Use this for questions about occupancy, light levels, temperature, and humidity.
    """
    logger.info(f"[Telemetry Node]: Hitting API for query: '{query}'")
    return run_telemetry_agent(query)

@tool
def ask_diagnostics_agent(query: str) -> str:
    """
    Call this agent for troubleshooting hardware, finding offline sensors, or checking battery levels.
    CRITICAL INSTRUCTION: The 'query' MUST be a direct command targeting specific devices or rooms.
    - BAD Query: 'Why is the AC not working in the kitchen?'
    - GOOD Query: 'Run health checks on all HVAC and sensor devices in the kitchen.'
    """
    logger.info(f"[Diagnostics Node]: Running health checks for: '{query}'")
    return "MOCK_DATA: All sensors are online. Battery levels normal."

@tool
def ask_rule_agent(query: str) -> str:
    """
    Call this agent to create, update, or propose automation rules for ThingsBoard.
    CRITICAL INSTRUCTION: The 'query' MUST be a precise statement of the IF/THEN automation logic.
    - BAD Query: 'Can you make sure the lights turn off when we leave?'
    - GOOD Query: 'Draft a Rule Chain: IF room occupancy is 0 THEN turn off lights.'
    """
    logger.info(f"[Rule Node]: Drafting Rule Chain for: '{query}'")
    return "MOCK_DATA: Successfully drafted rule."

@tool
def query_knowledge_base(query: str) -> str:
    """
    Call this tool to search the Vector Database (Qdrant) for manuals, topologies, or SOPs.
    CRITICAL INSTRUCTION: The 'query' MUST be a concise set of search keywords, not a full sentence.
    - BAD Query: 'How do I reset the main router in the data center?'
    - GOOD Query: 'Data center main router hard reset procedure SOP.'
    """
    logger.info(f"[Knowledge Base]: Searching Vector DB for: '{query}'")
    return "MOCK_DATA: Found manual."


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
sub_systems = [ask_telemetry_agent, ask_diagnostics_agent, ask_rule_agent, query_knowledge_base]
supervisor_llm = llm.bind_tools(sub_systems)

supervisor_prompt = """You are the Supreme Supervisor Agent for a Smart Campus.
Your job is to route the user's request to the correct sub-agent, read the data they return, and give the user a clear, helpful final answer.

CRITICAL INSTRUCTIONS:
1. TRANSLATION RULE: Never pass the user's raw question to a sub-agent. You must translate their intent into an explicit, declarative data-fetching command.
2. If the user asks about a campus-wide metric (e.g., "brightest room in the campus"), you MUST explicitly command the sub-agent to "check ALL rooms".
3. If the user asks for multiple distinct things, you MUST trigger multiple tools simultaneously.
4. Once the tools return their data, synthesize it into a clean, conversational response. Do not expose raw YAML to the user unless necessary.
5. If a tool returns an error, apologize and explain what went wrong."""

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