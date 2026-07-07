import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Literal, Tuple, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.smart_campus_assistant.utils.schedule_registry import ScheduleRegistry
from src.smart_campus_assistant.utils.device_registry import registry as device_registry

# Initialize registry globally for the tools to share
registry = ScheduleRegistry()

# --- DEFINE HARDCODED TIME FRAME LITERAL ---
TimeframeLiteral = Literal["now", "today", "week", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# --- DEFINE ROOMS LITERAL ---
Rooms = Literal[
    'parkin.c', 'parkin.b', 'data_center', 'entrance', 'restaurant',
    '1.1', '1.2', 'kitchen', '2.1', '2.2', '2.3', '2.4',
    '3.7', '3.8', '3.9', '4.9', '5.6', '5.7'
]

# --- DYNAMIC ENUM HELPER ---
# LLMs and LangChain respect native Python Enums for strict function calling. 
# Because items like "Room 1.2" have spaces/dots, we must dynamically generate valid Enum keys.
def create_dynamic_enum(enum_name: str, values: List[str]) -> Enum:
    if not values:
        values = ["Unknown"]
    # Creates a safe dictionary for the Enum (e.g., {'ITEM_0': '1.2', 'ITEM_1': 'Auditorium'})
    enum_dict = {f"ITEM_{i}": str(v) for i, v in enumerate(values)}
    return Enum(enum_name, enum_dict)

CourseEnum = create_dynamic_enum("CourseEnum", registry.get_all_courses())
InstructorEnum = create_dynamic_enum("InstructorEnum", registry.get_all_instructors())
SemesterEnum = create_dynamic_enum("SemesterEnum", registry.get_all_semesters())

# --- DYNAMIC INPUT SCHEMAS FOR LANGGRAPH ---

class RoomScheduleInput(BaseModel):
    room: Rooms = Field(..., description="The exact room ID/name to query.") 
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

class CourseScheduleInput(BaseModel):
    course_name: CourseEnum = Field(..., description="The exact name of the course.")   # type: ignore
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

class InstructorScheduleInput(BaseModel):
    instructor_name: InstructorEnum = Field(..., description="The exact name of the instructor.")   # type: ignore
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

class SemesterScheduleInput(BaseModel):
    semester: SemesterEnum = Field(..., description="The semester number (e.g., '2', '4', '6', '8').")  # type: ignore
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

# --- FORMATTER ---

def _format_yaml_response(domain: str, tool_name: str, filters: str, results: List[dict], timeframe: str) -> Tuple[str, Any]:
    lines = [
        f"Query_Context:",
        f"  Domain: {domain}",
        f"  Tool: {tool_name}",
        f"  Filters: {filters}",
        f"  Timeframe: {timeframe}"
    ]
    
    # --- NEW: Academic Context Checks ---
    context_notes = []
    
    # 1. Check Semester Status
    is_active, semester_msg = registry.check_semester_active()
    if not is_active:
        context_notes.append(f"Semester Status: {semester_msg}")
        
    # 2. Check Holiday Status
    time_lower = timeframe.lower()
    target_day = None
    now = datetime.now(registry.tz)  # Use the timezone-aware datetime from the registry
    
    # Resolve timeframe to a day of the week
    if time_lower in ["today", "now"]:
        target_day = now.strftime("%A")
    elif time_lower == "tomorrow":
        target_day = (now + timedelta(days=1)).strftime("%A")
    elif time_lower in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        target_day = timeframe.capitalize()
        
    is_holiday = False
    if target_day:
        holiday_name = registry.check_holiday(target_day)
        if holiday_name:
            is_holiday = True
            context_notes.append(f"Holiday Alert: {target_day} is a holiday ({holiday_name}).")
            
    # Inject context notes into the LLM prompt if any exist
    if context_notes:
        lines.append("Academic_Context:")
        for note in context_notes:
            lines.append(f"  - {note}")
            
    # --- Evaluate Results ---
    if not results:
        if not is_active:
            lines.append("Status: No classes scheduled because the semester is inactive.")
        elif is_holiday:
            lines.append("Status: No classes scheduled because it is a holiday.")
        else:
            lines.append("Status: No classes found for this specific query and timeframe.")
        return "\n".join(lines), None
        
    # If we have results, list them, but explicitly warn the LLM if they aren't actually taking place
    lines.append("Scheduled_Classes:")
    if not is_active or is_holiday:
        lines.append("  Note_to_LLM: The classes below are technically on the schedule, BUT THEY ARE NOT TAKING PLACE because of the Academic_Context (holiday or inactive semester). Inform the user accordingly.")
        
    for res in results:
        lines.append(f"  - Course: {res.get('course_name')}")
        lines.append(f"    Type: {res.get('course_type')}")
        lines.append(f"    Instructor: {res.get('instructor_name')}")
        lines.append(f"    Day: {res.get('day_of_week')}")
        lines.append(f"    Time: {res.get('start_time')} - {res.get('end_time')}")
        lines.append(f"    Rooms: {', '.join(res.get('room_ids', []))}")
        
    yaml_str = "\n".join(lines)
    
    # --- Strict Artifact Generation ---
    artifact = None
    
    # Only generate an artifact if it's "now" AND the class is ACTUALLY taking place
    if time_lower == "now" and is_active and not is_holiday:
        room_ids = results[0].get("room_ids", [])
        if room_ids:
            first_room = room_ids[0]
            # Use device_registry to safely resolve underground floors
            floor_val = device_registry.get_floor_for_room(first_room) or (str(first_room)[0] if str(first_room)[0].isdigit() else "0")
            
            artifact = {
                "type": "map_update",
                "artifact": {
                    "view_type": "snapshot",
                    "domain": "Schedule",
                    "floor": floor_val,
                    "room_id": str(first_room),
                    "schedule_data": results[0]
                }
            }
            
    return yaml_str, artifact

# --- TOOLS ---

@tool("get_room_schedule", args_schema=RoomScheduleInput, response_format="content_and_artifact")
def get_room_schedule(room: Rooms, timeframe: str) -> Tuple[str, Any]: 
    """Get the academic schedule for a specific room."""
    room_val = room.value if hasattr(room, "value") else str(room)
    time_lower = timeframe.lower()
    
    # Resolve floor for the UI payload using device_registry
    floor_val = device_registry.get_floor_for_room(room_val) or (str(room_val)[0] if str(room_val)[0].isdigit() else "0")

    # 1. Non-Academic Room Check
    academic_rooms = registry.get_all_rooms()
    if room_val not in academic_rooms:
        msg = "No Lessons take place in this Room"
        llm_msg = f"Query_Context:\n  Domain: Campus_Schedule\n  Room: {room_val}\n  Timeframe: {timeframe}\nStatus: {msg}."
        
        artifact = None
        if time_lower == "now":
            artifact = {
                "type": "map_update",
                "artifact": {
                    "view_type": "info",
                    "domain": "Schedule",
                    "floor": floor_val,
                    "room_id": str(room_val),
                    "message": msg
                }
            }
        return llm_msg, artifact

    # 2. Semester Active Check
    is_active, status_message = registry.check_semester_active()
    if not is_active:
        msg = "The Semester Ended"
        llm_msg = f"Query_Context:\n  Domain: Campus_Schedule\n  Room: {room_val}\n  Timeframe: {timeframe}\nStatus: {status_message}. {msg}."
        
        artifact = None
        if time_lower == "now":
            artifact = {
                "type": "map_update",
                "artifact": {
                    "view_type": "info",
                    "domain": "Schedule",
                    "floor": floor_val,
                    "room_id": str(room_val),
                    "message": msg
                }
            }
        return llm_msg, artifact

    # 3. Fetch results for the academic room
    results = registry.get_by_room(room_val, timeframe)
    
    # 4. Class is Free Check (No classes in this timeframe)
    if not results:
        msg = "The Class is Free"
        llm_msg = f"Query_Context:\n  Domain: Campus_Schedule\n  Room: {room_val}\n  Timeframe: {timeframe}\nStatus: {msg}."
        
        artifact = None
        if time_lower == "now":
            artifact = {
                "type": "map_update",
                "artifact": {
                    "view_type": "info",
                    "domain": "Schedule",
                    "floor": floor_val,
                    "room_id": str(room_val),
                    "message": msg
                }
            }
        return llm_msg, artifact

    # 5. Normal Execution (Classes found)
    return _format_yaml_response("Campus_Schedule", "get_room_schedule", f"Room: {room_val}", results, timeframe)

@tool("get_course_schedule", args_schema=CourseScheduleInput, response_format="content_and_artifact")
def get_course_schedule(course_name: CourseEnum, timeframe: str) -> Tuple[str, Any]:    # type: ignore
    """Get the scheduled times and locations for a specific course."""
    course_val = course_name.value if hasattr(course_name, "value") else str(course_name)
    results = registry.get_by_course(course_val, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_course_schedule", f"Course: {course_val}", results, timeframe)

@tool("get_instructor_schedule", args_schema=InstructorScheduleInput, response_format="content_and_artifact")
def get_instructor_schedule(instructor_name: InstructorEnum, timeframe: str) -> Tuple[str, Any]:    # type: ignore
    """Get the teaching schedule and locations for a specific instructor."""
    instructor_val = instructor_name.value if hasattr(instructor_name, "value") else str(instructor_name)
    results = registry.get_by_instructor(instructor_val, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_instructor_schedule", f"Instructor: {instructor_val}", results, timeframe)

@tool("get_semester_schedule", args_schema=SemesterScheduleInput, response_format="content_and_artifact")
def get_semester_schedule(semester: SemesterEnum, timeframe: str) -> Tuple[str, Any]:   # type: ignore
    """Get the overall class schedule for an entire semester block."""
    semester_val = semester.value if hasattr(semester, "value") else str(semester)
    results = registry.get_by_semester(semester_val, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_semester_schedule", f"Semester: {semester_val}", results, timeframe)

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    print("Testing Schedule Tool Invocations...")
    print("-" * 50)
    
    try:
        print("\n[Testing...]")
        summary, raw_data = get_room_schedule.func(room="1.2", timeframe="now")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        
        print("\n" + "="*50)
        
        print("\n[Testing...]")
        summary2, raw_data2 = get_semester_schedule.func(semester="8", timeframe="now")
        print(summary2)
        print("\n[Artifact Payload]")
        print(raw_data2)
        
        print("\n" + "="*50)
        
        print("\n[Testing...]")
        summary3, raw_data3 = get_instructor_schedule.func(instructor_name="Eirini Liotou ", timeframe="week")
        print(summary3)
        print("\n[Artifact Payload]")
        print(raw_data3)

        print("\n" + "-"*50)
        print("All Schedule tool tests completed successfully.")
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)