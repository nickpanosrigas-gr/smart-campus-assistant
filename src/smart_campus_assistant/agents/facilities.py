import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Import project singletons and config
from src.smart_campus_assistant.config.settings import settings

# Import Infrastructure Tools
from src.smart_campus_assistant.tools.energy import get_energy_infrastructure
from src.smart_campus_assistant.tools.diagnostics import get_diagnostics, get_campus_diagnostics

logger = logging.getLogger(__name__)

# 1. Initialize the Local Ollama Model
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    num_ctx=settings.OLLAMA_NUM_CTX, 
    temperature=0,
    think=False,
    disable_thinking=True
)

# 2. Bind the tools strictly to the LLM
tools = [get_energy_infrastructure, get_diagnostics, get_campus_diagnostics]
llm_with_tools = llm.bind_tools(tools)

# 3. System Prompt (Focused strictly on routing)
system_prompt = """You are the Facilities & Infrastructure Routing Node for a Smart Campus. 
Your ONLY job is to analyze the command from the Supervisor and trigger the correct infrastructure tools (Energy or Diagnostics).

CRITICAL INSTRUCTIONS:
1. Do not attempt to answer the user yourself or summarize data; just call the tools. Your raw tool output will be sent back to the Supervisor.
2. Map the requested timeframe to one of these exact values: "now", "2h", "24h", "7d", "30d", "90d". If the query asks for "current", use "now".
3. Map the requested target to the closest valid infrastructure component (e.g., hvac, front_lift, 3rd_floor, data_center).
4. If the Supervisor commands you to check multiple targets or run both energy and diagnostic checks, you MUST trigger multiple tool calls simultaneously."""

def run_facilities_agent(query: str) -> str:
    """
    Custom agent router that forces raw tool output for Energy and Diagnostics.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query) 
    ]
    
    logger.info("Invoking LLM for facilities tool routing...")
    ai_msg = llm_with_tools.invoke(messages, config={"callbacks": []})
    
    if not ai_msg.tool_calls:
        logger.warning("LLM did not trigger any tools.")
        return f"Error: The LLM did not trigger any tools. Response: {ai_msg.content}"
    
    results = []
    
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        logger.info(f"Triggering {tool_name} with args: {tool_args}")
        
        tool_obj = next((t for t in tools if t.name == tool_name), None)
        if tool_obj:
            try:
                raw_output = tool_obj.invoke(tool_args)
                results.append(str(raw_output))
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                results.append(f"Error executing {tool_name}: {e}")
        else:
            logger.warning(f"Tool {tool_name} not found.")
            results.append(f"Error: Tool {tool_name} not found.")
    
    return "\n\n" + "="*50 + "\n\n".join(results)

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(levelname)s - %(message)s')
    logger.info("Testing Facilities Routing Node (Ollama)...")
    
    query = "check if the hvac has grid power right now and run a diagnostic check on the 3rd floor sensors"
    logger.info(f"User Query: {query}")
    
    final_raw_output = run_facilities_agent(query)
    logger.info("RAW TOOL OUTPUTS (Bypassing LLM Summary)")
    print(final_raw_output)