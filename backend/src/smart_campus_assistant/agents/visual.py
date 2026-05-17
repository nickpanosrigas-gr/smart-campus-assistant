from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 
from src.smart_campus_assistant.config.settings import settings

# Your strict frontend constraints
Rooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant', 
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4', 
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

Domains = Literal[
    "Air Quality", "Climate", "Occupancy", "Lights", 
    "Doors/Windows", "Diagnostics", "Schedule"
]

ViewType = Literal["map", "graph", "chat"]

class UIResponsePackage(BaseModel):
    """Schema for determining the frontend visual layout based on user query."""
    view_type: ViewType = Field(
        description="The view mode. Use 'map' for broad spatial queries or current status, 'graph' for time-series/historical data, and 'chat' for general info."
    )
    rooms: list[Rooms] = Field(
        default_factory=list,
        description="Specific rooms mentioned or implied. Leave empty if the query is campus-wide."
    )
    domains: list[Domains] = Field(
        default_factory=list,
        description="Sensor domains relevant to the query. Leave empty if none apply."
    )

# The highly constrained prompt
visual_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the UI State Router for a Smart Campus Dashboard. 
    Your ONLY job is to extract the spatial and domain intent from the user's request and map it to the exact Literal values provided.
    Do NOT attempt to answer the user's question. Output only the requested JSON structure."""),
    ("human", "{query}")
])

# Initialize the LLM (Fast, low temperature)
# If using Gemini here instead of Ollama, swap to ChatGoogleGenAI
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    num_ctx=settings.OLLAMA_NUM_CTX, 
    temperature=0,
    think=False,
    disable_thinking=True
)
ui_router_chain = visual_prompt | llm.with_structured_output(UIResponsePackage)

async def get_ui_intent(user_query: str) -> UIResponsePackage:
    """Entry point for FastAPI to call before hitting the main graph."""
    return await ui_router_chain.ainvoke({"query": user_query})