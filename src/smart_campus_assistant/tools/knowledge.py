import os
import re
import glob
import logging
import frontmatter
from enum import Enum
from typing import Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.smart_campus_assistant.clients.qdrant_client import kb_client

logger = logging.getLogger(__name__)

# ==========================================
# 1. DYNAMICALLY PULL OPTIONS ON MODULE LOAD
# ==========================================
def get_dynamic_options():
    """Scans local markdown files to extract unique values for all metadata fields."""
    unique_doc_types = set()
    unique_rooms = set()
    unique_floors = set()
    unique_people = set()
    
    knowledge_dir = "data/knowledge"
    if os.path.exists(knowledge_dir):
        for filepath in glob.glob(os.path.join(knowledge_dir, "**", "*.md"), recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    meta = frontmatter.load(f).metadata
                    
                    # doc_type
                    if "doc_type" in meta:
                        unique_doc_types.add(meta["doc_type"])
                    
                    # room_id
                    if "room_id" in meta:
                        rooms = meta["room_id"]
                        if isinstance(rooms, list):
                            unique_rooms.update(rooms)
                        else:
                            unique_rooms.add(rooms)
                            
                    # floor
                    if "floor" in meta:
                        floors = meta["floor"]
                        if isinstance(floors, list):
                            unique_floors.update(floors)
                        else:
                            unique_floors.add(floors)
                            
                    # people
                    if "people" in meta and meta["people"]:
                        people = meta["people"]
                        if isinstance(people, list):
                            unique_people.update(people)
                        else:
                            unique_people.add(people)
            except Exception:
                pass
                
    # Provide safe fallbacks if the DB is completely empty on first boot
    if not unique_doc_types: unique_doc_types = {"room_info"}
    if not unique_rooms: unique_rooms = {"unknown"}
    if not unique_floors: unique_floors = {0}
    if not unique_people: unique_people = {"None"}
    
    return list(unique_doc_types), list(unique_rooms), list(unique_floors), list(unique_people)

LIVE_DOC_TYPES, LIVE_ROOMS, LIVE_FLOORS, LIVE_PEOPLE = get_dynamic_options()

# ==========================================
# 2. CREATE SANITIZED DYNAMIC ENUMS
# ==========================================
def sanitize_key(prefix: str, val) -> str:
    """Converts a string like 'Dr. John' or '2.1' into a valid Python Enum key like 'PERSON_DR__JOHN'"""
    s = str(val)
    # Replace anything that isn't a letter or number with an underscore
    s = re.sub(r'[^a-zA-Z0-9]', '_', s).upper()
    return f"{prefix}_{s}"

# Create Enums where the KEY is sanitized, but the VALUE is exactly what we need to query Qdrant
DocTypeEnum = Enum("DocTypeEnum", {sanitize_key("DOC", t): t for t in LIVE_DOC_TYPES})
RoomEnum = Enum("RoomEnum", {sanitize_key("ROOM", r): r for r in LIVE_ROOMS})
FloorEnum = Enum("FloorEnum", {sanitize_key("FLOOR", f): f for f in LIVE_FLOORS})
PersonEnum = Enum("PersonEnum", {sanitize_key("PERSON", p): p for p in LIVE_PEOPLE})

# ==========================================
# 3. DEFINE THE PYDANTIC SCHEMA
# ==========================================
class KnowledgeQueryInput(BaseModel):
    query: str = Field(
        ..., 
        description="The semantic search query. E.g., 'What are the office hours?' or 'Where is the server room?'"
    )
    room_id: Optional[List[RoomEnum]] = Field(
        None, 
        description="STRICT filter by exact room IDs. Choose from the available options if mentioned."
    )
    floor: Optional[List[FloorEnum]] = Field(
        None, 
        description="STRICT filter by floor level. Choose from the available options."
    )
    doc_type: Optional[DocTypeEnum] = Field(
        None, 
        description="STRICT filter by document type. Choose from the available options."
    )
    person: Optional[PersonEnum] = Field(
        None,
        description="STRICT filter by a specific person's name. Choose from the available options."
    )

# ==========================================
# 4. DEFINE THE TOOL
# ==========================================
@tool("search_knowledge_base", args_schema=KnowledgeQueryInput)
def search_knowledge_base(
    query: str, 
    room_id: Optional[List[RoomEnum]] = None, 
    floor: Optional[List[FloorEnum]] = None, 
    doc_type: Optional[DocTypeEnum] = None, 
    person: Optional[PersonEnum] = None
) -> str:
    """
    Queries the Smart Campus Vector Database. 
    Provides information on building topology, room layouts, faculty offices, schedules, and sensor manuals.
    """
    
    # Package the filters. Because they are Enums, we MUST extract `.value` before passing to Qdrant.
    filters = {}
    
    # For lists of Enums, we extract the value of each Enum item
    if room_id: 
        filters["room_id"] = [r.value for r in room_id]
        
    if floor: 
        filters["floor"] = [f.value for f in floor]
        
    # For single Enums, we extract the direct value
    if doc_type: 
        filters["doc_type"] = doc_type.value
        
    if person: 
        filters["people"] = [person.value]

    points = kb_client.search_knowledge(query_text=query, filters=filters, limit=3)
    
    if not points:
        logger.info(f"[Knowledge Tool] LLM Searched: '{query}' | Filters: {filters} | Returned: 0 files.")
        return f"Query_Context:\n  Query: '{query}'\n  Filters: {filters}\n\nError: No relevant documents found in the database. Please retry without filters or with a broader query."

    retrieved_files = [p.payload.get("title", "Unknown") for p in points]
    logger.info(f"[Knowledge Tool] LLM Searched: '{query}' | Filters: {filters} | Returned Files: {retrieved_files}")

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
        content = meta.get("page_content", meta.get("text", "No text content available."))
        
        output.append(f"[Result {idx+1} | Relevance Score: {score:.2f}]")
        output.append(f"Title: {title}")
        output.append(f"Content:\n{content}\n")
        output.append("-" * 40 + "\n")
        
    return "\n".join(output)