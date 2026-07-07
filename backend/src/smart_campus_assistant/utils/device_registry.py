import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

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
        """
        self.topology_path = Path(topology_path)
        self._topology: Dict = {}
        
        # Cache structure: { "room_name": { "device_name": {"id": "...", "zone": "...", "tag": "..."} } }
        self._room_cache: Dict[str, Dict[str, dict]] = {}
        
        self._load_topology()
        self._build_room_cache()

    def _load_topology(self) -> None:
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
        try:
            buildings = self._topology.get("campus", {}).get("buildings", {})
            for b_name, b_data in buildings.items():
                for f_name, f_data in b_data.get("floors", {}).items():
                    for r_name, r_data in f_data.get("rooms", {}).items():
                        room_key = str(r_name).strip().lower()
                        devices = r_data.get("devices", {})
                        
                        if room_key not in self._room_cache:
                            self._room_cache[room_key] = {}
                        
                        # Normalize devices so that string IDs become {"id": string}
                        for d_name, d_val in devices.items():
                            if isinstance(d_val, str):
                                self._room_cache[room_key][d_name] = {"id": d_val}
                            else:
                                self._room_cache[room_key][d_name] = d_val
                        
            logger.info(f"Successfully cached {len(self._room_cache)} rooms from topology.")
        except Exception as e:
            logger.error(f"Unexpected error while building room cache: {e}")

    def get_devices_by_room_and_type(self, room: str, sensor_type: str) -> Dict[str, dict]:
        room_key = str(room).strip().lower()
        room_devices = self._room_cache.get(room_key, {})
        matched_devices = {}
        
        target_marker = f"-{str(sensor_type).strip().upper()}"
        
        for device_name, device_data in room_devices.items():
            if target_marker in device_name.upper():
                matched_devices[device_name] = device_data
                
        if not matched_devices:
            logger.warning(f"No {sensor_type} sensors found in room '{room}'.")
            
        return matched_devices

    # ==========================================
    # NEW GLOBAL TYPE SEARCH METHOD
    # ==========================================
    def get_all_devices_by_type(self, sensor_type: str) -> Dict[str, dict]:
        """
        Sweeps all rooms in the campus to find devices of a specific type (e.g., 'WEATHER').
        """
        matched_devices = {}
        target_marker = str(sensor_type).strip().upper()
        
        for room_devices in self._room_cache.values():
            for device_name, device_data in room_devices.items():
                if target_marker in device_name.upper():
                    matched_devices[device_name] = device_data
                    
        if not matched_devices:
            logger.warning(f"No {sensor_type} sensors found across the campus topology.")
            
        return matched_devices

    def get_all_devices_in_room(self, room: str) -> Dict[str, dict]:
        room_key = str(room).strip().lower()
        return self._room_cache.get(room_key, {})

    def get_available_rooms(self) -> List[str]:
        return list(self._room_cache.keys())

    def get_all_sensor_types(self) -> List[str]:
        types = set()
        pattern = re.compile(r'-([A-Za-z]+)(?:-\d+)?$')
        
        for room_devices in self._room_cache.values():
            for device_name in room_devices.keys():
                match = pattern.search(device_name)
                if match:
                    types.add(match.group(1).upper())
                    
        return sorted(list(types))

    def get_total_sensor_count(self) -> int:
        return sum(len(devices) for devices in self._room_cache.values())

    # ==========================================
    # INFRASTRUCTURE MAPPING METHOD
    # ==========================================
    def get_energy_meters_for_target(self, target: str) -> Dict[str, dict]:
        """
        Maps a standard room name (e.g., '2.4') to its floor's energy meter, 
        or maps a special target (e.g., 'hvac', 'car_lift') to its specific meters.
        """
        target_lower = target.lower().strip()
        special_targets = ['hvac', 'car_lift', 'front_lift', 'back_lift']
        matched_meters = {}
        
        # 1. If it's a special target, sweep all infrastructure rooms for matching PPC/GEN meters
        if target_lower in special_targets:
            search_term = target_lower.replace("_", "")
            for b_name, b_data in self._topology.get("campus", {}).get("buildings", {}).items():
                for f_name, f_data in b_data.get("floors", {}).items():
                    infra_devices = f_data.get("rooms", {}).get("infrastructure", {}).get("devices", {})
                    for d_name, d_val in infra_devices.items():
                        if search_term in d_name.lower().replace("_", ""):
                            # Normalize string IDs to dicts to match new JSON format
                            matched_meters[d_name] = {"id": d_val} if isinstance(d_val, str) else d_val
            return matched_meters
            
        # 2. If it's a standard room, find its floor, then get the "FLOOR" meter from infrastructure
        for b_name, b_data in self._topology.get("campus", {}).get("buildings", {}).items():
            for f_name, f_data in b_data.get("floors", {}).items():
                if target_lower in [str(r).lower() for r in f_data.get("rooms", {}).keys()]:
                    
                    # We found the floor this room is on. Now pull its infrastructure meters
                    infra_devices = f_data.get("rooms", {}).get("infrastructure", {}).get("devices", {})
                    for d_name, d_val in infra_devices.items():
                        # Only grab the meter that tracks the whole floor, ignore lifts/hvacs in this sweep
                        if "FLOOR" in d_name.upper():
                            # Normalize string IDs to dicts to match new JSON format
                            matched_meters[d_name] = {"id": d_val} if isinstance(d_val, str) else d_val
                    return matched_meters
        
        return matched_meters

    def get_floor_for_room(self, room: str) -> Optional[str]:
        """
        Takes a room name as input and returns the floor ID that the room is located on,
        stripping the 'F' prefix so that values like 'F-1' return as '-1'.
        """
        room_lower = str(room).strip().lower()
        buildings = self._topology.get("campus", {}).get("buildings", {})
        
        for b_name, b_data in buildings.items():
            for f_name, f_data in b_data.get("floors", {}).items():
                rooms = f_data.get("rooms", {})
                if room_lower in [str(r).lower() for r in rooms.keys()]:
                    floor_id = str(f_name)
                    if floor_id.upper().startswith("F"):
                        return floor_id[1:]
                    return floor_id
                    
        logger.warning(f"Room '{room}' not found. Cannot determine floor.")
        return None

registry = DeviceRegistry()