import logging
from datetime import datetime
from enum import Enum
from typing import List, Dict, Literal, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.smart_campus_assistant.utils.schedule_registry import ScheduleRegistry

# Initialize registry globally for the tools to share
registry = ScheduleRegistry()

# --- DEFINE HARDCODED TIME FRAME LITERAL ---
TimeframeLiteral = Literal["now", "today", "week", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# --- DYNAMIC ENUM HELPER ---
# LLMs and LangChain respect native Python Enums for strict function calling. 
# Because items like "Room 1.2" have spaces/dots, we must dynamically generate valid Enum keys.
def create_dynamic_enum(enum_name: str, values: List[str]) -> Enum:
    if not values:
        values = ["Unknown"]
    # Creates a safe dictionary for the Enum (e.g., {'ITEM_0': '1.2', 'ITEM_1': 'Auditorium'})
    enum_dict = {f"ITEM_{i}": str(v) for i, v in enumerate(values)}
    return Enum(enum_name, enum_dict)

RoomEnum = create_dynamic_enum("RoomEnum", registry.get_all_rooms())
CourseEnum = create_dynamic_enum("CourseEnum", registry.get_all_courses())
InstructorEnum = create_dynamic_enum("InstructorEnum", registry.get_all_instructors())
SemesterEnum = create_dynamic_enum("SemesterEnum", registry.get_all_semesters())

# --- DYNAMIC INPUT SCHEMAS FOR LANGGRAPH ---

class RoomScheduleInput(BaseModel):
    room: RoomEnum = Field(..., description="The exact room ID/name to query.") # type: ignore
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

def _format_yaml_response(domain: str, tool_name: str, filters: str, results: List[Dict], timeframe: str) -> Tuple[str, dict]:
    # Capture and format the current time
    current_time_str = datetime.now().strftime("%A, %b %d, %Y at %I:%M %p")
    
    lines = []
    lines.append("Query_Context:")
    lines.append(f"  Current_Time: {current_time_str}")
    lines.append(f"  Domain: {domain}")
    lines.append(f"  Tool: {tool_name}")
    lines.append(f"  Filters: {filters} | Timeframe: {timeframe}")
    lines.append(f"  Total_Results: {len(results)}")
    lines.append("")
    
    artifact_results = []
    
    if not results:
        lines.append("Active_Schedule: []")
        return "\n".join(lines), {
            "view_type": "schedule_now" if timeframe.lower() == "now" else "schedule_list",
            "timeframe": timeframe,
            "results": []
        }

    lines.append("Active_Schedule:")
    for entry in results:
        # Prepare the clean dictionary for the UI Artifact
        entry_artifact = dict(entry)
        
        rooms_formatted = " or ".join(entry.get("room_ids", []))
        course_formatted = f"{entry.get('course_name')} (Sem {entry.get('semester')})"
        
        # Check for holidays based on the specific entry's day
        day_of_week = entry.get('day_of_week')
        holiday_name = registry.check_holiday(day_of_week)
        holiday_tag = f" [HOLIDAY: {holiday_name}]" if holiday_name else ""
        if holiday_name:
            entry_artifact["holiday"] = holiday_name
        
        if timeframe.lower() == "now":
            time_remaining = registry.calculate_time_remaining(entry.get("end_time"))
            entry_artifact["time_remaining"] = time_remaining # Inject into UI
            
            lines.append(f"  - Course: {course_formatted}")
            lines.append(f"    State: IN PROGRESS (Ends in {time_remaining})")
            lines.append(f"    Time: {entry.get('start_time')} - {entry.get('end_time')} EEST{holiday_tag}")
            lines.append(f"    Room: {rooms_formatted}")
            lines.append(f"    Instructor: {entry.get('instructor_name')}")
            lines.append(f"    Type: {entry.get('course_type')}")
        else:
            if timeframe.lower() == "week":
                lines.append(f"  - Time: {day_of_week}{holiday_tag} {entry.get('start_time')} - {entry.get('end_time')}")
            else:
                lines.append(f"  - Time: {entry.get('start_time')} - {entry.get('end_time')} EEST{holiday_tag}")
                
            lines.append(f"    Course: {course_formatted}")
            lines.append(f"    Room: {rooms_formatted}")
            lines.append(f"    Instructor: {entry.get('instructor_name')}")
            lines.append(f"    Type: {entry.get('course_type')}")
            
        artifact_results.append(entry_artifact)
        
    artifact = {
        "view_type": "schedule_now" if timeframe.lower() == "now" else "schedule_list",
        "timeframe": timeframe,
        "results": artifact_results
    }
            
    return "\n".join(lines), artifact

# --- TOOLS ---

@tool("get_room_schedule", args_schema=RoomScheduleInput, response_format="content_and_artifact")
def get_room_schedule(room: RoomEnum, timeframe: str) -> Tuple[str, dict]:   # type: ignore
    """Get the academic schedule for a specific room."""
    # Safely handle both Enum objects (from LLM) and raw strings (from local testing)
    room_val = room.value if hasattr(room, "value") else str(room)
    results = registry.get_by_room(room_val, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_room_schedule", f"Room: {room_val}", results, timeframe)

@tool("get_course_schedule", args_schema=CourseScheduleInput, response_format="content_and_artifact")
def get_course_schedule(course_name: CourseEnum, timeframe: str) -> Tuple[str, dict]:    # type: ignore
    """Get the scheduled times and locations for a specific course."""
    course_val = course_name.value if hasattr(course_name, "value") else str(course_name)
    results = registry.get_by_course(course_val, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_course_schedule", f"Course: {course_val}", results, timeframe)

@tool("get_instructor_schedule", args_schema=InstructorScheduleInput, response_format="content_and_artifact")
def get_instructor_schedule(instructor_name: InstructorEnum, timeframe: str) -> Tuple[str, dict]:    # type: ignore
    """Get the teaching schedule and locations for a specific instructor."""
    instructor_val = instructor_name.value if hasattr(instructor_name, "value") else str(instructor_name)
    results = registry.get_by_instructor(instructor_val, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_instructor_schedule", f"Instructor: {instructor_val}", results, timeframe)

@tool("get_semester_schedule", args_schema=SemesterScheduleInput, response_format="content_and_artifact")
def get_semester_schedule(semester: SemesterEnum, timeframe: str) -> Tuple[str, dict]:   # type: ignore
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
        print("\n[Testing get_room_schedule (Room 1.2, timeframe: today)]")
        summary, raw_data = get_room_schedule.func(room="1.2", timeframe="today")
        print(summary)
        print("\n[Artifact Payload]")
        print(raw_data)
        
        print("\n" + "="*50)
        
        print("\n[Testing get_semester_schedule (Semester 8, timeframe: now)]")
        summary2, raw_data2 = get_semester_schedule.func(semester="8", timeframe="now")
        print(summary2)
        print("\n[Artifact Payload]")
        print(raw_data2)

        print("\n" + "-"*50)
        print("All Schedule tool tests completed successfully.")
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)