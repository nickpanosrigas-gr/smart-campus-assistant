import json
import logging
import re
from pathlib import Path
from typing import Dict, List

# Configure logging for the module
logger = logging.getLogger(__name__)

class DeviceRegistry:
    """
    A registry to manage and query the campus device topology.
    Loads the topology JSON into memory to provide fast, case-insensitive 
    lookups for LangGraph tools.
    """
    
    def __init__(self, topology_path: str = "data/campus_topology.json"):
        """
        Initializes the registry and builds the in-memory cache.
        
        Args:
            topology_path (str): Relative or absolute path to the topology JSON file.
        """
        self.topology_path = Path(topology_path)
        self._topology: Dict = {}
        
        # Cache structure: { "room_name": { "device_name": "device_id" } }
        self._room_cache: Dict[str, Dict[str, str]] = {}
        
        self._load_topology()
        self._build_room_cache()

    def _load_topology(self) -> None:
        """Loads and parses the JSON topology file from disk."""
        if not self.topology_path.exists():
            logger.error(f"Topology file not found at {self.topology_path}")
            raise FileNotFoundError(f"Topology file missing: {self.topology_path}")
            
        try:
            with open(self.topology_path, 'r', encoding='utf-8') as f:
                self._topology = json.load(f)
            logger.info(f"Successfully loaded topology from {self.topology_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse topology JSON: {e}")
            raise

    def _build_room_cache(self) -> None:
        """
        Traverses the nested campus topology and flattens it into a 
        fast-lookup dictionary keyed by lowercase room names.
        """
        try:
            buildings = self._topology.get("campus", {}).get("buildings", {})
            for b_name, b_data in buildings.items():
                for f_name, f_data in b_data.get("floors", {}).items():
                    for r_name, r_data in f_data.get("rooms", {}).items():
                        
                        # Normalize room names to lowercase for robust tool querying
                        room_key = str(r_name).strip().lower()
                        devices = r_data.get("devices", {})
                        
                        # Handle potential duplicates if rooms have the same name
                        if room_key not in self._room_cache:
                            self._room_cache[room_key] = {}
                        
                        self._room_cache[room_key].update(devices)
                        
            logger.info(f"Successfully cached {len(self._room_cache)} rooms from topology.")
        except Exception as e:
            logger.error(f"Unexpected error while building room cache: {e}")

    def get_devices_by_room_and_type(self, room: str, sensor_type: str) -> Dict[str, str]:
        """
        Retrieves all devices of a specific sensor type within a given room.
        
        Args:
            room (str): The name of the room (e.g., "1.2", "restaurant", "DataCenter").
            sensor_type (str): The sensor prefix/suffix (e.g., "IAQ", "MC", "PC", "Desk").
            
        Returns:
            Dict[str, str]: A dictionary mapping device names to their ThingsBoard UUIDs.
        """
        room_key = str(room).strip().lower()
        room_devices = self._room_cache.get(room_key, {})
        
        matched_devices = {}
        
        # Look for the exact sensor type format used in the JSON (e.g., "-IAQ")
        target_marker = f"-{str(sensor_type).strip().upper()}"
        
        for device_name, device_id in room_devices.items():
            if target_marker in device_name.upper():
                matched_devices[device_name] = device_id
                
        if not matched_devices:
            logger.warning(f"No {sensor_type} sensors found in room '{room}'.")
            
        return matched_devices

    def get_all_devices_in_room(self, room: str) -> Dict[str, str]:
        """Returns all registered devices in a given room, regardless of type."""
        room_key = str(room).strip().lower()
        return self._room_cache.get(room_key, {})

    def get_available_rooms(self) -> List[str]:
        """Returns a list of all valid room names."""
        return list(self._room_cache.keys())

    def get_all_sensor_types(self) -> List[str]:
        """
        Dynamically extracts and returns a list of all unique sensor types 
        (e.g., IAQ, MC, PC, DESK) found across the entire campus.
        """
        types = set()
        # Regex to capture the alphabetical sensor type at the end of the string,
        # ignoring any trailing numbers (e.g., matches "IAQ" in "F1_1.2-IAQ-1" and "PC" in "F0_Entrance-PC")
        pattern = re.compile(r'-([A-Za-z]+)(?:-\d+)?$')
        
        for room_devices in self._room_cache.values():
            for device_name in room_devices.keys():
                match = pattern.search(device_name)
                if match:
                    types.add(match.group(1).upper())
                    
        return sorted(list(types))

    def get_total_sensor_count(self) -> int:
        """Returns the total number of sensors registered in the topology."""
        return sum(len(devices) for devices in self._room_cache.values())


# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Setup basic logging to see the initialization info
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # NOTE: Make sure your terminal is at the root of 'smart-campus-assistant' 
        # so 'data/campus_topology.json' resolves correctly.
        registry = DeviceRegistry(topology_path="data/campus_topology.json")
        
        print("\n" + "="*40)
        print("🧪 RUNNING DEVICE REGISTRY TESTS 🧪")
        print("="*40)

        # 1. Test get_total_sensor_count
        total_sensors = registry.get_total_sensor_count()
        print(f"\n[1] Total Sensors in Campus: {total_sensors}")

        # 2. Test get_all_sensor_types
        sensor_types = registry.get_all_sensor_types()
        print(f"\n[2] All Unique Sensor Types ({len(sensor_types)} total):")
        print(f"    {sensor_types}")

        # 3. Test get_available_rooms
        rooms = registry.get_available_rooms()
        print(f"\n[3] Available Rooms ({len(rooms)} total):")
        print(f"    Sample: {rooms}")

        # 4. Test get_devices_by_room_and_type
        test_room = "1.2"
        test_type = "IAQ"
        iaq_in_1_2 = registry.get_devices_by_room_and_type(test_room, test_type)
        print(f"\n[4] Querying '{test_type}' sensors in room '{test_room}':")
        for name, uid in iaq_in_1_2.items():
            print(f"    ✔️ {name} -> {uid}")

        # 5. Test get_all_devices_in_room
        test_room_2 = "restaurant"
        restaurant_devices = registry.get_all_devices_in_room(test_room_2)
        print(f"\n[5] Querying ALL sensors in room '{test_room_2}':")
        for name, uid in restaurant_devices.items():
            print(f"    ✔️ {name} -> {uid}")
            
        print("\n✅ All tests completed successfully.\n")

    except FileNotFoundError:
        print("\n❌ ERROR: Could not find 'data/campus_topology.json'.")
        print("Make sure you run this script from the root project directory.")