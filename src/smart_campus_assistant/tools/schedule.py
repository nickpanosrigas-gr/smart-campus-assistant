import logging
from typing import List, Dict, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.smart_campus_assistant.utils.schedule_registry import ScheduleRegistry

# Initialize registry globally for the tools to share
registry = ScheduleRegistry()

# --- DEFINE HARDCODED TIME FRAME LITERAL ---
TimeframeLiteral = Literal["now", "today", "week", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# --- DYNAMIC INPUT SCHEMAS FOR LANGGRAPH ---

class RoomScheduleInput(BaseModel):
    room_id: str = Field(
        ..., 
        description="The exact room ID to query.", 
        json_schema_extra={"enum": registry.get_all_rooms() or ["Unknown"]}
    )
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

class CourseScheduleInput(BaseModel):
    course_name: str = Field(
        ..., 
        description="The exact name of the course.", 
        json_schema_extra={"enum": registry.get_all_courses() or ["Unknown"]}
    )
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

class InstructorScheduleInput(BaseModel):
    instructor_name: str = Field(
        ..., 
        description="The exact name of the instructor.", 
        json_schema_extra={"enum": registry.get_all_instructors() or ["Unknown"]}
    )
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

class SemesterScheduleInput(BaseModel):
    semester: str = Field(
        ..., 
        description="The semester number (e.g., '2', '4', '6', '8').", 
        json_schema_extra={"enum": registry.get_all_semesters() or ["Unknown"]}
    )
    timeframe: TimeframeLiteral = Field(..., description="The time window to query.")

# --- FORMATTER ---

def _format_yaml_response(domain: str, tool_name: str, filters: str, results: List[Dict], timeframe: str) -> str:
    lines = []
    lines.append("Query_Context:")
    lines.append(f"  Domain: {domain}")
    lines.append(f"  Tool: {tool_name}")
    lines.append(f"  Filters: {filters} | Timeframe: {timeframe}")
    lines.append(f"  Total_Results: {len(results)}")
    lines.append("")
    
    if not results:
        lines.append("Active_Schedule: []")
        return "\n".join(lines)

    lines.append("Active_Schedule:")
    for entry in results:
        rooms_formatted = " or ".join(entry.get("room_ids", []))
        course_formatted = f"{entry.get('course_name')} (Sem {entry.get('semester')})"
        
        # Check for holidays based on the specific entry's day
        day_of_week = entry.get('day_of_week')
        holiday_name = registry.check_holiday(day_of_week)
        holiday_tag = f" [HOLIDAY: {holiday_name}]" if holiday_name else ""
        
        if timeframe.lower() == "now":
            time_remaining = registry.calculate_time_remaining(entry.get("end_time"))
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
            
    return "\n".join(lines)

# --- TOOLS ---

@tool("get_room_schedule", args_schema=RoomScheduleInput)
def get_room_schedule(room_id: str, timeframe: str) -> str:
    """Get the academic schedule for a specific room."""
    results = registry.get_by_room(room_id, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_room_schedule", f"Room: {room_id}", results, timeframe)

@tool("get_course_schedule", args_schema=CourseScheduleInput)
def get_course_schedule(course_name: str, timeframe: str) -> str:
    """Get the scheduled times and locations for a specific course."""
    results = registry.get_by_course(course_name, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_course_schedule", f"Course: {course_name}", results, timeframe)

@tool("get_instructor_schedule", args_schema=InstructorScheduleInput)
def get_instructor_schedule(instructor_name: str, timeframe: str) -> str:
    """Get the teaching schedule and locations for a specific instructor."""
    results = registry.get_by_instructor(instructor_name, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_instructor_schedule", f"Instructor: {instructor_name}", results, timeframe)

@tool("get_semester_schedule", args_schema=SemesterScheduleInput)
def get_semester_schedule(semester: str, timeframe: str) -> str:
    """Get the overall class schedule for an entire semester block."""
    results = registry.get_by_semester(semester, timeframe)
    return _format_yaml_response("Campus_Schedule", "get_semester_schedule", f"Semester: {semester}", results, timeframe)

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
        print(get_room_schedule.invoke({"room_id": "1.2", "timeframe": "today"}))
        
        print("\n" + "="*50)
        
        print("\n[Testing get_semester_schedule (Semester 8, timeframe: week)]")
        print(get_semester_schedule.invoke({"semester": "8", "timeframe": "week"}))

        print("\n" + "-"*50)
        print("All Schedule tool tests completed successfully.")
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)