import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

from src.smart_campus_assistant.config.settings import settings

class ScheduleRegistry:
    """
    Registry for managing and querying the static campus schedule.
    Handles exact metadata matching, time-based filtering, and holiday checking.
    """
    def __init__(self, file_path: str = None):
        
        if file_path is None:
            file_path = f"{settings.DATA_DIR}/schedule.json"
        
        self.file_path = Path(file_path)
        self.tz = ZoneInfo("Europe/Athens")
        self._load_data()

    def _load_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data.get("metadata", {})
                self.schedule = data.get("schedule", [])
                self.holidays = data.get("holidays", [])
                self.start_end_of_classes = data.get("start_end_of_classes", [])
        except FileNotFoundError:
            self.metadata = {}
            self.schedule = []
            self.holidays = []
            self.start_end_of_classes = []
            print(f"Warning: Schedule file not found at {self.file_path}")

    def _get_current_time_info(self) -> dict:
        now = datetime.now(self.tz)
        return {
            "day": now.strftime("%A"),
            "time_str": now.strftime("%H:%M"),
            "datetime": now
        }

    def _filter_schedule(self, key: str, value: str, timeframe: str) -> List[Dict]:
        results = []
        current = self._get_current_time_info()
        
        target_days = []
        if timeframe.lower() == "week":
            target_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        elif timeframe.lower() in ["today", "now"]:
            target_days = [current["day"]]
        else:
            target_days = [timeframe.capitalize()]

        for entry in self.schedule:
            if key == "room_ids":
                if value not in entry.get("room_ids", []):
                    continue
            elif entry.get(key) != value:
                continue

            if entry.get("day_of_week") not in target_days:
                continue

            if timeframe.lower() == "now":
                start = entry.get("start_time")
                end = entry.get("end_time")
                if not (start <= current["time_str"] < end):
                    continue
            
            results.append(entry)

        return results

    def check_holiday(self, day_name: str) -> Optional[str]:
        """
        Calculates the exact date for the given day in the CURRENT week 
        and checks if it falls within any registered holiday periods.
        """
        now = datetime.now(self.tz)
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        try:
            target_idx = days_of_week.index(day_name.capitalize())
            current_idx = now.weekday()
            
            # Find the date for this day in the current week
            diff = target_idx - current_idx
            target_date = (now + timedelta(days=diff)).date()
            
            for holiday in self.holidays:
                start = datetime.strptime(holiday["start_date"], "%Y-%m-%d").date()
                end = datetime.strptime(holiday["end_date"], "%Y-%m-%d").date()
                
                if start <= target_date <= end:
                    return holiday["name"]
        except (ValueError, KeyError, TypeError):
            pass
            
        return None
    
    def check_semester_active(self) -> tuple[bool, str]:
        """
        Checks if the current date is within the active semester period.
        Returns a boolean indicating if active, and a message explaining the status.
        """
        if not self.start_end_of_classes:
            return True, "Active"
            
        now = datetime.now(self.tz).date()
        period = self.start_end_of_classes[0]
        
        start_date_str = period.get("start_date")
        end_date_str = period.get("end_date")
        
        if not start_date_str:
            return True, "Active"
            
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        
        if now < start_date:
            return False, f"The semester has not started yet. Classes begin on {start_date_str}."
            
        if end_date_str:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            if now > end_date:
                return False, f"The semester has ended. Classes ended on {end_date_str}."
                
        return True, "Active"
    
    # --- QUERY GETTERS ---

    def get_by_room(self, room_id: str, timeframe: str) -> List[Dict]:
        return self._filter_schedule("room_ids", room_id, timeframe)

    def get_by_course(self, course_name: str, timeframe: str) -> List[Dict]:
        return self._filter_schedule("course_name", course_name, timeframe)

    def get_by_instructor(self, instructor_name: str, timeframe: str) -> List[Dict]:
        return self._filter_schedule("instructor_name", instructor_name, timeframe)

    def get_by_semester(self, semester: str, timeframe: str) -> List[Dict]:
        return self._filter_schedule("semester", str(semester), timeframe)
        
    def calculate_time_remaining(self, end_time_str: str) -> str:
        now = datetime.now(self.tz)
        end_time = datetime.strptime(end_time_str, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day, tzinfo=self.tz
        )
        diff = end_time - now
        if diff.total_seconds() <= 0:
            return "Ending now"
        
        hours, remainder = divmod(int(diff.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}hr {minutes}mins"

    # --- UTILITY GETTERS (Global Extracts) ---

    def get_all_instructors(self) -> List[str]:
        instructors = {entry.get("instructor_name") for entry in self.schedule if entry.get("instructor_name")}
        return sorted(list(instructors))

    def get_all_rooms(self) -> List[str]:
        rooms = {room for entry in self.schedule for room in entry.get("room_ids", [])}
        return sorted(list(rooms))

    def get_all_courses(self) -> List[str]:
        courses = {entry.get("course_name") for entry in self.schedule if entry.get("course_name")}
        return sorted(list(courses))

    def get_all_semesters(self) -> List[str]:
        semesters = {entry.get("semester") for entry in self.schedule if entry.get("semester")}
        return sorted(list(semesters), key=lambda x: int(x) if x.isdigit() else x)


# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*40)
    print("RUNNING SCHEDULE REGISTRY TESTS")
    print("="*40)
    
    try:
        registry = ScheduleRegistry()
        
        print("\n[Holiday Check: Friday]")
        holiday = registry.check_holiday("Friday")
        print(f"Is Friday a holiday? {holiday if holiday else 'No'}")
        
        print("\n[Holiday Check: Wednesday]")
        holiday = registry.check_holiday("Wednesday")
        print(f"Is Wednesday a holiday? {holiday if holiday else 'No'}")

        print("\n" + "-"*50)
        print("All Registry tool tests completed successfully.\n")
        
    except Exception as e:
        logger.error(f"\nError during execution: {e}", exc_info=True)