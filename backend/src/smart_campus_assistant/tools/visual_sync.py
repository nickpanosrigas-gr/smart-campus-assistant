from langchain_core.tools import tool

@tool
def verify_ui_state(rooms: list[str], domains: list[str]) -> str:
    """
    CRITICAL: You MUST call this tool immediately before delivering your final conversational answer to the user.
    Provide the exact rooms and domains you are about to discuss in your final response.
    This synchronizes the user's dashboard view with your answer.
    """
    # The actual Python execution does nothing except return a success string to the LLM.
    # The magic happens in FastAPI when it intercepts the "on_tool_start" event for this tool.
    return "UI State Verified. Proceed to generate final text response to the user."