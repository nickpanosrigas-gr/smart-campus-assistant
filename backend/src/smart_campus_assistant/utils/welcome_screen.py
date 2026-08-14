import random
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

# Comprehensive list of rooms for the "Initial Load" recommendation logic
ALL_ROOMS = [
    "parkin.c", "parkin.b", "data_center", "kitchen", "entrance", "restaurant",
    "1.1", "1.2", "2.1", "2.2", "2.3", "2.4", "3.7", "3.8", "3.9", "4.9", "5.6", "5.7"
]

# Rooms with explicit People Counters (PC) that return integer counts
PC_ROOMS = ["restaurant", "1.2", "2.2", "2.3", "2.4", "3.9", "4.9", "5.7", "building"]

# Rooms that officially support the Academic Schedule tool
SCHEDULE_ROOMS = ["1.2", "2.2", "2.3", "2.4", "3.7", "3.9", "4.9", "5.7"]

# Rooms and floors where Doors/Windows tools are unsupported
NO_DOORS_WINDOWS_ROOMS = [
    "parkin.c", "parkin.b", "kitchen", "restaurant", "entrance",
    "1.1", "2.2", "3.7"
]

# Friendly formatting mapped directly from the Omirou Building Topology
ROOM_NAMES = {
    "parkin.c": "Parking C",
    "parkin.b": "Parking B",
    "data_center": "the Main Data Center",
    "kitchen": "the Kitchen",
    "entrance": "the Entrance",
    "restaurant": "the Restaurant",
    "1.1": "the Conference Room (1.1)",
    "1.2": "the Main Amphitheater (1.2)",
    "2.1": "the Secretariat (2.1)",
    "2.2": "the Post Graduate Lab (2.2)",
    "2.3": "the Small Amphitheater (2.3)",
    "2.4": "the Under Graduate Computer Lab (2.4)",
    "3.7": "the Small Amphitheater (3.7)",
    "3.8": "the Small Server Room (3.8)",
    "3.9": "the Small Amphitheater (3.9)",
    "4.9": "the Under Graduate Lab (4.9)",
    "5.6": "the Small Server Room (5.6)",
    "5.7": "the Post Graduate Lab (5.7)",
    "building": "the building"
}

# Standard Sensor Timeframes
TIMEFRAME_STRINGS = {
    "now": "right now",
    "2h": "over the last 2 hours",
    "24h": "over the last 24 hours",
    "7d": "over the last 7 days",
    "30d": "over the last 30 days",
    "90d": "over the last 90 days"
}

# Schedule Specific Timeframes
SCHEDULE_TIMEFRAMES = ["now", "today", "week", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

SCHEDULE_TIMEFRAME_STRINGS = {
    "now": "right now",
    "today": "today",
    "week": "this week",
    "Monday": "on Monday",
    "Tuesday": "on Tuesday",
    "Wednesday": "on Wednesday",
    "Thursday": "on Thursday",
    "Friday": "on Friday"
}

WELCOME_PHRASES = [
    "What do you want to do today?",
    "How can I help you today?",
    "What would you like to check?",
    "Where should we start?",
    "How can I assist you with the campus?",
    "What do you want to explore?",
    "Ready to check the campus metrics?",
    "What information do you need?",
    "What are we looking at today?",
    "How can I help you manage the campus?"
]

TOOL_QUESTIONS = {
    "Occupancy_PC": [
        "How many people are in {target} {time_str}?",
        "What was the peak occupancy for {target} {time_str}?",
        "Show me the occupancy trends for {target} {time_str}.",
        "Calculate the average number of people in {target} {time_str}.",
        "Give me a detailed usage report for {target} {time_str}.",
        "How busy is {target} {time_str}?",
        "Check the occupancy density for {target} {time_str}.",
        "Are there available seats in {target} {time_str}?",
        "Is {target} crowded {time_str}?",
        "Did {target} reach maximum capacity {time_str}?"
    ],
    "Occupancy_Motion": [
        "Is there any activity in {target} {time_str}?",
        "Has there been movement in {target} {time_str}?",
        "Is {target} currently empty {time_str}?",
        "Show me the motion activity trends for {target} {time_str}.",
        "Check if {target} is in use {time_str}.",
        "Are people moving around in {target} {time_str}?",
        "Give me an activity profile for {target} {time_str}.",
        "Calculate the idle time for {target} {time_str}.",
        "Did anyone use {target} {time_str}?",
        "Verify the activity status of {target} {time_str}."
    ],
    "Climate": [
        "What is the temperature in {target} {time_str}?",
        "Is the humidity comfortable in {target} {time_str}?",
        "Compare indoor temperature to outdoor weather for {target} {time_str}.",
        "How does solar radiation affect {target} {time_str}?",
        "Have there been any temperature spikes in {target} {time_str}?",
        "Show me the temperature trends for {target} {time_str}.",
        "Are the climate conditions optimal in {target} {time_str}?",
        "Check the climate stability for {target} {time_str}.",
        "What is the average humidity in {target} {time_str}?",
        "Is {target} too hot or cold {time_str}?"
    ],
    "Air Quality": [
        "What is the current air quality in {target} {time_str}?",
        "Are the CO2 levels safe in {target} {time_str}?",
        "Check for any hazardous TVOC levels in {target} {time_str}.",
        "Compare indoor and outdoor PM levels for {target} {time_str}.",
        "Show me the PM2.5 and PM10 trends for {target} {time_str}.",
        "Is the ventilation adequate in {target} {time_str}?",
        "Has the air quality dropped in {target} {time_str}?",
        "What is the average CO2 concentration in {target} {time_str}?",
        "Are there any air quality health alerts for {target} {time_str}?",
        "Give me an air quality summary for {target} {time_str}."
    ],
    "Doors/Windows": [
        "Are the doors locked in {target} {time_str}?",
        "Check if any windows are left open in {target} {time_str}.",
        "Show me the physical access logs for {target} {time_str}.",
        "Are there any energy flags for open windows in {target} {time_str}?",
        "Verify the state of all magnetic contacts in {target} {time_str}.",
        "Were there any unauthorized entries in {target} {time_str}?",
        "Has anyone entered {target} {time_str}?",
        "Are the security perimeters secure in {target} {time_str}?",
        "Count the number of door transitions in {target} {time_str}.",
        "Is {target} fully sealed {time_str}?"
    ],
    "Lights": [
        "Are the lights on in {target} {time_str}?",
        "What is the ambient light level in {target} {time_str}?",
        "How is natural daylight affecting {target} {time_str}?",
        "Check if lights were left on overnight in {target} {time_str}.",
        "Show me the illumination trends for {target} {time_str}.",
        "Is {target} bright enough {time_str}?",
        "Give me an energy usage estimate for lighting in {target} {time_str}.",
        "Calculate the average light intensity in {target} {time_str}.",
        "Generate a lighting profile for {target} {time_str}.",
        "Has the lighting fluctuated heavily in {target} {time_str}?"
    ],
    "Diagnostics": [
        "Run a hardware health audit on {target} {time_str}.",
        "Check connectivity for offline sensors in {target} {time_str}.",
        "Show me the battery drain estimates for {target} {time_str}.",
        "Were any tamper alarms triggered in {target} {time_str}?",
        "Are there any dead batteries in {target} {time_str}?",
        "Have any devices dropped connection in {target} {time_str}?",
        "Are there any weak signal warnings in {target} {time_str}?",
        "Verify the hardware status of all gateways in {target} {time_str}.",
        "Generate a full system diagnostic for {target} {time_str}.",
        "Is the IoT infrastructure in {target} operating normally {time_str}?"
    ],
    "Schedule": [
        "What is the academic schedule for {target} {time_str}?",
        "Is {target} currently booked for a class {time_str}?",
        "Who is teaching in {target} {time_str}?",
        "Show me the upcoming events in {target} {time_str}.",
        "Are there any free slots in {target} {time_str}?",
        "When is the next lecture in {target} {time_str}?",
        "Has any class been canceled in {target} {time_str}?",
        "Give me the course list for {target} {time_str}.",
        "Generate a timetable overview for {target} {time_str}.",
        "Check the booking availability for {target} {time_str}."
    ]
}

def get_greeting_by_time() -> str:
    tz = pytz.timezone('Europe/Athens')
    hour = datetime.now(tz).hour
    
    if 5 <= hour < 12:
        return "Καλημέρα"
    elif 12 <= hour < 20:
        return "Καλησπέρα"
    else:
        return "Καληνύχτα"

def format_room_targets(rooms: list[str], floor: str) -> str:
    """Formats multiple rooms into grammatically correct text with 'and'."""
    clean_rooms = [r for r in rooms if r not in ["building", "ALL"] and str(r).strip()]
    
    if not clean_rooms:
        return f"Floor {floor}" if floor != "B" else "the building"
        
    formatted = [ROOM_NAMES.get(r, f"Room {r}") for r in clean_rooms]
    
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return f"{', '.join(formatted[:-1])} and {formatted[-1]}"

def generate_welcome_payload(
    name: str, 
    tool: str, 
    floor: str, 
    rooms: list[str] = None, 
    timeframe: str = "now",
    prev_msg: str = None,
    prev_templates: list[str] = None
) -> dict:
    if rooms is None:
        rooms = []
        
    clean_rooms = [r for r in rooms if r not in ["building", "ALL"] and str(r).strip()]
    is_initial_load = floor == "B" and timeframe == "now" and not clean_rooms
    
    # 1. Determine Target Category
    if tool == "Schedule":
        target_category = "Schedule"
    elif tool == "Occupancy":
        if is_initial_load:
            # Randomly pick PC or Motion category so we load valid templates
            target_category = random.choice(["Occupancy_PC", "Occupancy_Motion"])
        else:
            all_pc = clean_rooms and all(r in PC_ROOMS for r in clean_rooms)
            target_category = "Occupancy_PC" if all_pc else "Occupancy_Motion"
    else:
        target_category = tool

    # 2. Template Selection (Preserve state if valid across toggles)
    available_questions = TOOL_QUESTIONS.get(target_category, [])
    selected_templates = []
    
    if prev_templates:
        valid_prev = [t for t in prev_templates if t in available_questions]
        if len(valid_prev) == 3:
            selected_templates = valid_prev
            
    if not selected_templates:
        selected_templates = random.sample(available_questions, min(3, len(available_questions)))
        
    # 3. Format Questions (Randomizing per-question inside the loop for initial load)
    formatted_questions = []
    for template in selected_templates:
        if is_initial_load:
            # Generate a completely random Room & Timeframe FOR EACH QUESTION
            if tool == "Schedule":
                rand_room = random.choice(SCHEDULE_ROOMS)
                target = ROOM_NAMES.get(rand_room, f"Room {rand_room}")
                selected_timeframe = random.choice(SCHEDULE_TIMEFRAMES)
                time_str = SCHEDULE_TIMEFRAME_STRINGS[selected_timeframe]
            else:
                if target_category == "Occupancy_PC":
                    rand_room = random.choice([r for r in ALL_ROOMS if r in PC_ROOMS])
                elif target_category == "Occupancy_Motion":
                    rand_room = random.choice([r for r in ALL_ROOMS if r not in PC_ROOMS])
                elif tool == "Doors/Windows":
                    # Filter out the invalid rooms/floors for Doors & Windows
                    rand_room = random.choice([r for r in ALL_ROOMS if r not in NO_DOORS_WINDOWS_ROOMS])
                else:
                    rand_room = random.choice(ALL_ROOMS)
                
                target = ROOM_NAMES.get(rand_room, f"Room {rand_room}")
                selected_timeframe = random.choice(["now", "2h", "24h", "7d", "30d"])
                time_str = TIMEFRAME_STRINGS[selected_timeframe]
        else:
            # Follow the user's exact map context
            target = format_room_targets(clean_rooms, floor)
            if tool == "Schedule":
                ui_to_sched = {"now": "now", "2h": "today", "24h": "today", "7d": "week", "30d": "week", "90d": "week"}
                sched_tf = ui_to_sched.get(timeframe, "now")
                if sched_tf == "week" and random.random() > 0.5:
                    sched_tf = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                time_str = SCHEDULE_TIMEFRAME_STRINGS.get(sched_tf, "right now")
            else:
                time_str = TIMEFRAME_STRINGS.get(timeframe, "right now")

        formatted_q = template.format(target=target, time_str=time_str)
        formatted_q = formatted_q.replace(" right now", "").replace("  ", " ").replace(" ?", "?").replace(" .", ".").strip()
        formatted_questions.append(formatted_q)

    # 4. Maintain Welcome Message State
    welcome_message = prev_msg if prev_msg else random.choice(WELCOME_PHRASES)

    return {
        "greeting_time": get_greeting_by_time(),
        "name": name,
        "welcome_message": welcome_message,
        "questions": formatted_questions,
        "templates": selected_templates
    }