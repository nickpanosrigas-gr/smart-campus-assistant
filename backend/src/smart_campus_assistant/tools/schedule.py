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

def _format_yaml_response(domain: str, tool_name: str, filters: str, results: List[dict], timeframe: str, room_id: str = None) -> Tuple[str, Any]:
    # 1. Global State Checks
    is_active, sem_msg = registry.check_semester_active()
    
    # Determine target day for holiday check
    target_day = registry._get_current_time_info()["day"]
    if timeframe.lower() not in ["now", "today", "week"]:
        target_day = timeframe.capitalize()
        
    holiday_name = registry.check_holiday(target_day)

    # 2. Build LLM Context String
    lines = [
        f"Query_Context:",
        f"  Domain: {domain}",
        f"  Tool: {tool_name}",
        f"  Filters: {filters}",
        f"  Timeframe: {timeframe}",
        f"  Current_Time: {datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')}"
    ]
    
    # Provide explicit state alerts so the LLM doesn't hallucinate active classes
    if not is_active:
        lines.append(f"  System_Alert: '{sem_msg}' - Note: The schedule below is for reference only. No classes are currently taking place.")
    elif holiday_name:
        lines.append(f"  System_Alert: 'Today is a holiday ({holiday_name})' - Note: The schedule below is for reference only. No classes are currently taking place.")

    # Status breakdown for the LLM
    if not results:
        lines.append("  Status: The Class is Free (No lessons scheduled for this timeframe).")
    else:
        lines.append("  Status: Schedule found (See results below).")
        lines.append("Results:")
        for res in results:
            lines.append(f"  - Course: {res.get('course_name')}")
            lines.append(f"    Type: {res.get('course_type')}")
            lines.append(f"    Instructor: {res.get('instructor_name')}")
            lines.append(f"    Room: {', '.join(res.get('room_ids', []))}")
            lines.append(f"    Time: {res.get('start_time')} - {res.get('end_time')}")

    yaml_str = "\n".join(lines)

    # 3. Artifact Generation (Strictly for the 'now' timeframe)
    artifact = None
    if timeframe.lower() == "now":
        # Try to resolve a room to target for the UI
        target_room = room_id
        if not target_room and results and results[0].get("room_ids"):
            # Fallback: Extract room from results (for instructor/course queries)
            target_room = results[0]["room_ids"][0] 
            
        if target_room:
            floor_val = device_registry.get_floor_for_room(target_room) or (str(target_room)[0] if str(target_room)[0].isdigit() else "0")
            
            # Map status conditions to your UI states
            if target_room not in registry.get_all_rooms():
                node_status = "unavailable"
                msg = "No Lessons take place in this Room"
            elif not is_active:
                node_status = "warning"
                msg = "Semester Inactive" 
            elif holiday_name:
                node_status = "warning"
                msg = f"Holiday: {holiday_name}"
            elif not results:
                node_status = "good"
                msg = "The Class is Free"
            else:
                node_status = "critical"
                msg = f"Class in progress: {results[0].get('course_name')}"
            
            # Construct standard snapshot artifact
            artifact = {
                "type": "map_update",
                "artifact": {
                    "view_type": "snapshot",
                    "domain": "Schedule",
                    "floor": floor_val,
                    "room_id": str(target_room),
                    "status": node_status,
                    "message": msg
                }
            }
            
            # Only attach full schedule data payload if a class is actually happening
            if node_status == "critical" and results:
                artifact["artifact"]["schedule_data"] = results[0]

    return yaml_str, artifact

# --- TOOLS ---

@tool("get_room_schedule", args_schema=RoomScheduleInput, response_format="content_and_artifact")
def get_room_schedule(room: Rooms, timeframe: str) -> Tuple[str, Any]: 
    """Get the academic schedule for a specific room."""
    room_val = room.value if hasattr(room, "value") else str(room)
    
    # 1. Immediate Non-Academic Check
    academic_rooms = registry.get_all_rooms()
    if room_val not in academic_rooms:
        floor_val = device_registry.get_floor_for_room(room_val) or (str(room_val)[0] if str(room_val)[0].isdigit() else "0")
        msg = "No Lessons Scheduled"
        llm_msg = f"Query_Context:\n  Domain: Campus_Schedule\n  Room: {room_val}\n  Timeframe: {timeframe}\nStatus: {msg}. Advise the user accordingly."
        
        artifact = None
        if timeframe.lower() == "now":
            artifact = {
                "type": "map_update",
                "artifact": {
                    "view_type": "snapshot",
                    "domain": "Schedule",
                    "floor": floor_val,
                    "room_id": str(room_val),
                    "status": "unavailable",
                    "message": msg
                }
            }
        return llm_msg, artifact

    # 2. Academic Room: Fetch results and defer to the global formatter
    results = registry.get_by_room(room_val, timeframe)
    
    # Pass room_id explicitly so the artifact generates even if results are empty (class is free)
    return _format_yaml_response("Campus_Schedule", "get_room_schedule", f"Room: {room_val}", results, timeframe, room_id=room_val)

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