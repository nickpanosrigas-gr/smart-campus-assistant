import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.tools.knowledge import search_knowledge_base

logger = logging.getLogger(__name__)

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    num_ctx=settings.OLLAMA_NUM_CTX, 
    temperature=0,
    think=False,
    disable_thinking=True
)

tools = [search_knowledge_base]
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are the Knowledge Routing Node for a Smart Campus. 
Your ONLY job is to analyze the command from the Supervisor and trigger the search_knowledge_base tool.

CRITICAL INSTRUCTIONS:
1. Do not attempt to answer the user yourself; just call the tool. Your raw tool output will be sent back to the Supervisor.
2. Extract any specific floors, rooms, document types, or people from the query and apply them as STRICT filters.
3. If the Supervisor is looking for a broad layout (e.g., 'all rooms on the third floor'), set limit to 'big'."""

def run_knowledge_agent(query: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query) 
    ]
    
    logger.info("Invoking LLM for knowledge base routing...")
    ai_msg = llm_with_tools.invoke(messages, config={"callbacks": []})
    
    if not ai_msg.tool_calls:
        logger.warning("LLM did not trigger the knowledge tool.")
        return f"Error: The LLM did not trigger any tools. Response: {ai_msg.content}"
    
    results = []
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        logger.info(f"Triggering {tool_name} with args: {tool_args}")
        
        try:
            raw_output = search_knowledge_base.invoke(tool_args)
            results.append(str(raw_output))
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}")
            results.append(f"Error executing {tool_name}: {e}")
            
    return "\n\n" + "="*50 + "\n\n".join(results)