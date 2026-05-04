import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Import project singletons and config
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.tools.lights import get_ambient_lights
from src.smart_campus_assistant.tools.occupancy import get_occupancy
from src.smart_campus_assistant.tools.temp_humidity import get_temp_humidity
from src.smart_campus_assistant.tools.air_quality import get_air_quality

logger = logging.getLogger(__name__)

# 1. Initialize the Local Ollama Model using settings.py
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    num_ctx=settings.OLLAMA_NUM_CTX, 
    temperature=0,
    think=False,
    disable_thinking=True
)

# 2. Bind the tools strictly to the LLM
tools = [get_ambient_lights, get_occupancy, get_temp_humidity, get_air_quality]
llm_with_tools = llm.bind_tools(tools)

# 3. System Prompt (Focused strictly on routing, not conversing)
system_prompt = """You are the Telemetry Routing Node for a Smart Campus. 
Your ONLY job is to analyze the command from the Supervisor and trigger the correct telemetry tools.

CRITICAL INSTRUCTIONS:
1. Do not attempt to answer the user yourself or summarize data; just call the tools. Your raw tool output will be sent back to the Supervisor.
2. Map the requested timeframe to one of these exact values: "now", "2h", "24h", "7d", "30d", "90d". If the query says "current" or "currently", use "now".
3. Map the requested room to the closest matching valid room name in your tool schemas.
4. If the Supervisor commands you to check multiple rooms or multiple metrics, you MUST trigger multiple tool calls at the same time."""

def run_telemetry_agent(query: str) -> str:
    """
    Custom agent router that forces raw tool output.
    Bypasses the LLM's tendency to summarize data by returning the tool execution directly.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query) 
    ]
    
    logger.info("Invoking LLM for telemetry tool routing...")
    # EXPLICITLY BLOCK Langfuse from bleeding into this routing decision
    ai_msg = llm_with_tools.invoke(messages, config={"callbacks": []})
    
    # 2. Check if the LLM decided to call any tools
    if not ai_msg.tool_calls:
        logger.warning("LLM did not trigger any tools.")
        return f"Error: The LLM did not trigger any tools. Response: {ai_msg.content}"
    
    results = []
    
    # 3. Execute the tools programmatically and collect the RAW output
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        logger.info(f"Triggering {tool_name} with args: {tool_args}")
        
        # Match the requested tool to our actual Python functions
        tool_obj = next((t for t in tools if t.name == tool_name), None)
        if tool_obj:
            try:
                # Execute the tool and capture the raw YAML output
                raw_output = tool_obj.invoke(tool_args)
                results.append(str(raw_output))
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                results.append(f"Error executing {tool_name}: {e}")
        else:
            logger.warning(f"Tool {tool_name} not found.")
            results.append(f"Error: Tool {tool_name} not found.")
    
    # 4. Return the combined raw strings
    return "\n\n" + "="*50 + "\n\n".join(results)

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(levelname)s - %(message)s')
    logger.info("Testing Telemetry Routing Node (Ollama)...")
    
    # The compound double question
    query = "how many people are in the restaurant right now and can you give me some long term statistic on room 4.9 occupancy"
    logger.info(f"User Query: {query}")
    
    # Run our custom execution loop
    final_raw_output = run_telemetry_agent(query)
    
    logger.info("RAW TOOL OUTPUTS (Bypassing LLM Summary)")
    # Keep standard print for the final payload structure
    print(final_raw_output)