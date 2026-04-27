import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Import project singletons and config
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.tools.lights import get_ambient_lights
from src.smart_campus_assistant.tools.occupancy import get_occupancy

logger = logging.getLogger(__name__)

# 1. Initialize the Local Ollama Model using settings.py
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    num_ctx=settings.OLLAMA_NUM_CTX, 
    temperature=0
)

# 2. Bind the tools strictly to the LLM
tools = [get_ambient_lights, get_occupancy]
llm_with_tools = llm.bind_tools(tools)

# 3. System Prompt (Focused strictly on routing, not conversing)
system_prompt = """You are the Telemetry Routing Node for a Smart Campus. 
Your ONLY job is to analyze the user's request and trigger the correct telemetry tools.
If the user asks for multiple rooms or multiple timeframes, you MUST trigger multiple tool calls at the same time.
Do not attempt to answer the user yourself; just call the tools."""

def run_telemetry_agent(user_query: str) -> str:
    """
    Custom agent router that forces raw tool output.
    Bypasses the LLM's tendency to summarize data by returning the tool execution directly.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    # 1. Ask the LLM to decide which tools to call based on the prompt
    logger.info("Invoking LLM for telemetry tool routing...")
    ai_msg = llm_with_tools.invoke(messages)
    
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