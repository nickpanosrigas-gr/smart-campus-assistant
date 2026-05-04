import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Import project config
from src.smart_campus_assistant.config.settings import settings

# Import the 4 scheduling tools
from src.smart_campus_assistant.tools.schedule import (
    get_room_schedule,
    get_course_schedule,
    get_instructor_schedule,
    get_semester_schedule
)

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

# 2. Bind the tools to the LLM
tools = [get_room_schedule, get_course_schedule, get_instructor_schedule, get_semester_schedule]
llm_with_tools = llm.bind_tools(tools)

# 3. System Prompt (Focused strictly on data extraction and routing)
system_prompt = """You are the Schedule Routing Node for a Smart Campus. 
Your ONLY job is to analyze the command from the Supervisor and trigger the correct academic scheduling tools.

CRITICAL INSTRUCTIONS:
1. Do not attempt to answer the user yourself or summarize data; just call the tools. Your raw tool output will be sent back to the Supervisor.
2. If the Supervisor asks for multiple distinct items (e.g., a teacher AND a room), you MUST trigger multiple tool calls at the same time.
3. The timeframe argument MUST be mapped to one of: "now", "today", "week", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday". If the command implies currently, use "now".
4. The tool schemas contain strict ENUM lists for valid inputs. You MUST select the exact matching value from those predefined lists based on the Supervisor's query."""

def run_scheduler_agent(query: str) -> str:
    """
    Custom agent router that forces raw tool output.
    Bypasses the LLM's tendency to summarize data by returning the tool execution directly.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query) 
    ]
    
    logger.info("Invoking LLM for schedule tool routing...")
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
    
    # 4. Return the combined raw strings
    return "\n\n" + "="*50 + "\n\n".join(results)

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(levelname)s - %(message)s')
    logger.info("Testing Schedule Routing Node (Ollama)...")
    
    # Test a compound question to ensure it triggers multiple tools or precise lookups
    query = "Where is Dr. Turing teaching right now, and what is happening in room 1.2 on Friday?"
    logger.info(f"User Query: {query}")
    
    final_raw_output = run_scheduler_agent(query)
    
    logger.info("RAW TOOL OUTPUTS (Bypassing LLM Summary)")
    print(final_raw_output)