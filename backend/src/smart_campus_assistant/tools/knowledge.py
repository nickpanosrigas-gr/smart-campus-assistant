import os
import re
import glob
import logging
import frontmatter
from enum import Enum
from typing import Optional, List, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.smart_campus_assistant.clients.qdrant_client import kb_client
from src.smart_campus_assistant.config.settings import settings

logger = logging.getLogger(__name__)

# [Keep your get_dynamic_options() and ENUM definitions exactly the same here...]
def get_dynamic_options():
    unique_doc_types = set()
    unique_rooms = set()
    unique_floors = set()
    unique_people = set()
    
    knowledge_dir = f"{settings.DATA_DIR}/knowledge"
    
    if os.path.exists(knowledge_dir):
        for filepath in glob.glob(os.path.join(knowledge_dir, "**", "*.md"), recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    meta = frontmatter.load(f).metadata
                    if "doc_type" in meta: unique_doc_types.add(meta["doc_type"])
                    if "room_id" in meta:
                        rooms = meta["room_id"]
                        unique_rooms.update(rooms) if isinstance(rooms, list) else unique_rooms.add(rooms)
                    if "floor" in meta:
                        floors = meta["floor"]
                        unique_floors.update(floors) if isinstance(floors, list) else unique_floors.add(floors)
                    if "people" in meta and meta["people"]:
                        people = meta["people"]
                        unique_people.update(people) if isinstance(people, list) else unique_people.add(people)
            except Exception:
                pass
                
    if not unique_doc_types: unique_doc_types = {"room_info"}
    if not unique_rooms: unique_rooms = {"unknown"}
    if not unique_floors: unique_floors = {0}
    if not unique_people: unique_people = {"None"}
    
    return list(unique_doc_types), list(unique_rooms), list(unique_floors), list(unique_people)

LIVE_DOC_TYPES, LIVE_ROOMS, LIVE_FLOORS, LIVE_PEOPLE = get_dynamic_options()

def sanitize_key(prefix: str, val) -> str:
    s = str(val)
    s = re.sub(r'[^a-zA-Z0-9]', '_', s).upper()
    return f"{prefix}_{s}"

DocTypeEnum = Enum("DocTypeEnum", {sanitize_key("DOC", t): t for t in LIVE_DOC_TYPES})
RoomEnum = Enum("RoomEnum", {sanitize_key("ROOM", r): r for r in LIVE_ROOMS})
FloorEnum = Enum("FloorEnum", {sanitize_key("FLOOR", f): f for f in LIVE_FLOORS})
PersonEnum = Enum("PersonEnum", {sanitize_key("PERSON", p): p for p in LIVE_PEOPLE})

# ==========================================
# 3. DEFINE THE PYDANTIC SCHEMA
# ==========================================
class KnowledgeQueryInput(BaseModel):
    query: Optional[str] = Field(
        default="", 
        description="The semantic search query. E.g., 'What are the office hours?' Leave empty if only using filters."
    )
    room_id: Optional[List[RoomEnum]] = Field(      # type: ignore
        None, 
        description="STRICT filter by exact room IDs. MUST BE FORMATTED AS A LIST. Example: ['2.1', '2.2']"
    )
    floor: Optional[List[FloorEnum]] = Field(       # type: ignore
        None, 
        description="STRICT filter by floor level. MUST BE FORMATTED AS A LIST. Example: [2] or [3]"
    )
    doc_type: Optional[DocTypeEnum] = Field(        # type: ignore
        None, 
        description="STRICT filter by document type. Must be a single value, not a list. Example: 'room_info'"
    )
    person: Optional[PersonEnum] = Field(           # type: ignore
        None,
        description="STRICT filter by a specific person's name. Must be a single value, not a list."
    )
    limit: Literal["normal", "big"] = Field(
        default="normal",
        description="Size of the result set. 'normal' returns 5 full documents. 'big' returns 10 full documents (best for mapping out broad room layouts or topology)."
    )

# ==========================================
# 4. DEFINE THE TOOL
# ==========================================
@tool("search_knowledge", args_schema=KnowledgeQueryInput)
def search_knowledge(
    query: str, 
    room_id: Optional[List[RoomEnum]] = None,   # type: ignore
    floor: Optional[List[FloorEnum]] = None,    # type: ignore
    doc_type: Optional[DocTypeEnum] = None,     # type: ignore
    person: Optional[PersonEnum] = None,        # type: ignore
    limit: Literal["normal", "big"] = "normal"
) -> str:
    """
    Queries the Smart Campus Vector Database. 
    Provides information on building topology, room layouts, faculty offices, and schedules.
    """
    
    filters = {}
    if room_id: filters["room_id"] = [r.value for r in room_id]
    if floor: filters["floor"] = [f.value for f in floor]
    if doc_type: filters["doc_type"] = doc_type.value
    if person: filters["people"] = [person.value]

    actual_limit = 10 if limit == "big" else 5
    points = kb_client.search_knowledge(query_text=query, filters=filters, limit=actual_limit)
    
    if not points:
        logger.info(f"[Knowledge Tool] Searched: '{query}' | Filters: {filters} | Returned: 0 files.")
        return f"Query_Context:\n  Query: '{query}'\n  Filters: {filters}\n\nError: No relevant documents found. Please retry without filters or with a broader query."

    retrieved_files = [p.payload.get("title", "Unknown") for p in points]
    logger.info(f"[Knowledge Tool] Searched: '{query}' | Limit: {limit} ({actual_limit}) | Returned Files: {retrieved_files}")

    output = [
        "Query_Context:",
        f"  Query: '{query}'",
        f"  Filters_Applied: {filters}",
        f"  Results_Found: {len(points)}",
        "\n--- Document Results ---\n"
    ]
    
    for idx, point in enumerate(points):
        meta = point.payload
        score = point.score
        title = meta.get("title", "Untitled Document")
        
        output.append(f"[Result {idx+1} | Relevance Score: {score:.2f}]")
        output.append(f"Title: {title}")
        
        if "room_id" in meta:
            output.append(f"Associated Room ID: {meta['room_id']}")
        
        content = meta.get("page_content", meta.get("text", "No text content available."))
        
        output.append(f"Content:\n{content}\n")
        output.append("-" * 40 + "\n")
        
    return "\n".join(output)