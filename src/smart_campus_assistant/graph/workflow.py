import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# Import the existing LLM setup, tools, and prompt from your supervisor
from src.smart_campus_assistant.agents.supervisor import supervisor_llm, sub_systems, supervisor_prompt

logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
# The `add_messages` reducer ensures new messages are appended to the history, not overwritten.
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

# ==========================================
# 2. DEFINE THE NODES
# ==========================================
def call_supervisor(state: GraphState, config: RunnableConfig):
    """The brain of the graph. Evaluates the conversation history and makes decisions."""
    messages = state["messages"]
    
    # Prepend the system prompt dynamically so the LLM always has its instructions, 
    # but we don't bloat the saved state with duplicate SystemMessages.
    full_context = [SystemMessage(content=supervisor_prompt)] + messages
    
    logger.info("Supervisor LLM is evaluating the state...")
    response = supervisor_llm.invoke(full_context, config=config)
    
    # Return the new AI message to be appended to the state
    return {"messages": [response]}

# LangGraph's native ToolNode automatically handles executing your @tool functions
# and formatting their outputs as ToolMessages.
tool_node = ToolNode(sub_systems)

# ==========================================
# 3. DEFINE THE ROUTING LOGIC
# ==========================================
def should_continue(state: GraphState):
    """Checks if the LLM decided to use a tool or if it is done answering."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        logger.info(f"Routing to tools: {[t['name'] for t in last_message.tool_calls]}")
        return "tools"
    
    logger.info("No tools requested. Routing to END.")
    return END

# ==========================================
# 4. BUILD AND COMPILE THE GRAPH
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("supervisor", call_supervisor)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", should_continue, ["tools", END])
workflow.add_edge("tools", "supervisor")

# Add the Memory Checkpointer! This is what gives the agent a memory.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================
# 5. ENTRY POINT FOR TELEGRAM
# ==========================================
def run_graph_supervisor(user_query: str, thread_id: str, run_config: dict = None) -> str:
    """
    Executes the stateful graph. 
    The thread_id is used by MemorySaver to isolate different Telegram reply chains.
    """
    # 1. Prepare the standard LangGraph config with our thread ID
    config = {"configurable": {"thread_id": thread_id}}
    
    # 2. Merge in external configs (like your Langfuse callbacks) if they exist
    if run_config and "callbacks" in run_config:
        config["callbacks"] = run_config["callbacks"]
        
    logger.info(f"Invoking Graph for Thread ID: {thread_id}")
    
    # 3. Invoke the graph with the new user message
    final_state = app.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config=config
    )
    
    # 4. Extract and return the final text response from the last message in the state
    return final_state["messages"][-1].content