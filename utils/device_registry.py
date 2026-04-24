import json
import re
from pathlib import Path
from typing import Dict, List, Optional

class DeviceRegistry:
    def __init__(self, json_path: str = "data/campus_topology.json"):
        self.json_path = Path(json_path)
        
        # Data structures for our tools
        self.devices_by_type: Dict[str, Dict[str, str]] = {}
        self.devices_by_room: Dict[str, Dict[str, str]] = {}
        self.all_devices: Dict[str, str] = {}
        
        # Load and parse upon initialization
        self._load_and_parse()

    def _extract_sensor_type(self, device_name: str) -> str:
        """
        Extracts the base sensor type from the device name.
        Example: 'F0_Restaurant-IAQ-2' -> 'IAQ'
        """
        if '_' in device_name:
            suffix = device_name.split('_', 1)[1]
            if '-' in suffix:
                type_part = suffix.split('-', 1)[1]
                # Remove trailing dashes and numbers
                base_type = re.sub(r'-?\d+$', '', type_part)
                return base_type.upper()
        return "UNKNOWN"

    def _load_and_parse(self):
        """Reads the JSON and builds the queryable dictionaries."""
        if not self.json_path.exists():
            print(f"Warning: Topology file {self.json_path} not found.")
            return

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        buildings = data.get("campus", {}).get("buildings", {})
        
        for building_name, building_data in buildings.items():
            floors = building_data.get("floors", {})
            for floor_id, floor_data in floors.items():
                rooms = floor_data.get("rooms", {})
                for room_id, room_data in rooms.items():
                    devices = room_data.get("devices", {})
                    
                    room_key = f"{floor_id}_{room_id}"
                    if room_key not in self.devices_by_room:
                        self.devices_by_room[room_key] = {}
                        
                    for device_name, device_id in devices.items():
                        # 1. Save to flat dictionary
                        self.all_devices[device_name] = device_id
                        
                        # 2. Save by Room
                        self.devices_by_room[room_key][device_name] = device_id
                        
                        # 3. Save by Sensor Type
                        sensor_type = self._extract_sensor_type(device_name)
                        if sensor_type not in self.devices_by_type:
                            self.devices_by_type[sensor_type] = {}
                        self.devices_by_type[sensor_type][device_name] = device_id

    # --- Base Getter Methods ---

    def get_devices_by_type(self, sensor_type: str) -> Dict[str, str]:
        """Returns all devices of a specific type (e.g., 'IAQ', 'MC', 'DESK')."""
        return self.devices_by_type.get(sensor_type.upper(), {})

    def get_devices_in_room(self, floor_id: str, room_id: str) -> Dict[str, str]:
        """Returns all devices inside a specific room."""
        room_key = f"{floor_id}_{room_id}"
        return self.devices_by_room.get(room_key, {})

    def get_device_id(self, device_name: str) -> Optional[str]:
        """Gets a specific device ID by its exact name."""
        return self.all_devices.get(device_name)

    def get_available_types(self) -> List[str]:
        """Returns a list of all parsed sensor types."""
        return list(self.devices_by_type.keys())

    # --- NEW GETTER METHODS ---

    def get_total_device_count(self) -> int:
        """Returns the total number of registered devices across the campus."""
        return len(self.all_devices)

    def get_devices_by_type_and_room(self, sensor_type: str, floor_id: str, room_id: str) -> Dict[str, str]:
        """Returns devices of a specific type within a specific room."""
        room_devices = self.get_devices_in_room(floor_id, room_id)
        target_type = sensor_type.upper()
        
        # Filter the room's devices by the target type
        return {name: dev_id for name, dev_id in room_devices.items() 
                if self._extract_sensor_type(name) == target_type}

    def get_devices_by_type_and_floor(self, sensor_type: str, floor_id: str) -> Dict[str, str]:
        """Returns all devices of a specific type across an entire floor."""
        devices_of_type = self.get_devices_by_type(sensor_type)
        
        # Because device names are formatted like 'F1_1.1-IAQ-1', we can filter by the 'F1_' prefix
        prefix = f"{floor_id}_"
        return {name: dev_id for name, dev_id in devices_of_type.items() if name.startswith(prefix)}


# Instantiate a single global registry to be imported by other files
registry = DeviceRegistry()

if __name__ == "__main__":
    # --- Tests for the new methods ---
    
    # Total count and types
    total_count = registry.get_total_device_count()
    available_types = ", ".join(registry.get_available_types())
    print(f"Total Campus Devices: {total_count} (Types: {available_types})")
    
    # Desks in Room 4.9 (Name and ID)
    print("\nDesks in Room 4.9 (Floor F4):")
    desks_4_9 = registry.get_devices_by_type_and_room("Desk", "F4", "4.9-lab")
    print(f"Found {len(desks_4_9)} desks:")
    for name, device_id in desks_4_9.items():
        print(f" - {name}: {device_id}")
    
    # All IAQ sensors on Floor F2 (Now with IDs!)
    print("\nAll IAQ sensors on Floor F2:")
    f2_iaq = registry.get_devices_by_type_and_floor("IAQ", "F2")
    for name, device_id in f2_iaq.items(): # Changed .keys() to .items()
        print(f" - {name}: {device_id}")