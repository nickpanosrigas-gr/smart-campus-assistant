import logging
from typing import Optional, Dict, Any, List
import requests
from datetime import datetime, timedelta
from config.settings import settings

# Configure logging for the client
logger = logging.getLogger(__name__)

class ThingsBoardClient:
    """
    A centralized client for interacting with the ThingsBoard API.
    Used as a baseline by tools to fetch telemetry, attributes, and alarms.
    """
    
    def __init__(self):
        # Ensure there's no trailing slash to prevent double-slash in URLs
        self.base_url = settings.THINGSBOARD_BASE_URL.rstrip('/')
        self.username = settings.THINGSBOARD_USERNAME
        self.password = settings.THINGSBOARD_PASSWORD
        self.token: Optional[str] = None

    def login(self) -> bool:
        """Authenticates with ThingsBoard and stores the JWT token."""
        url = f"{self.base_url}/api/auth/login"
        payload = {
            "username": self.username,
            "password": self.password
        }
        
        try:
            # Added a timeout to prevent hanging
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status() 
            
            # Extract and store the token
            self.token = response.json().get("token")
            logger.info("Successfully authenticated with ThingsBoard API.")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to ThingsBoard: {e}")
            self.token = None
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Returns the necessary Authorization headers for API calls."""
        if not self.token:
            self.login()
            
        return {
            "X-Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Any]:
        """
        Internal helper to execute API calls. 
        Automatically handles 401 Unauthorized errors by re-authenticating and retrying.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_auth_headers()
        
        try:
            response = requests.request(method, url, headers=headers, timeout=15, **kwargs)
            
            # If the JWT token expired, try re-authenticating once
            if response.status_code == 401:
                logger.warning("ThingsBoard token might be expired. Re-authenticating...")
                if self.login():
                    headers = self._get_auth_headers()
                    response = requests.request(method, url, headers=headers, timeout=15, **kwargs)
            
            response.raise_for_status()
            
            # Return JSON if content exists, else None
            return response.json() if response.content else None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ThingsBoard API request failed ({method} {endpoint}): {e}")
            return None

    # ==========================================
    # TOOL-FACING METHODS
    # ==========================================

    def get_device_telemetry(self, device_id: str, keys: str) -> Optional[Dict[str, Any]]:
        """Fetch latest time-series data for a specific device."""
        endpoint = f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries?keys={keys}"
        return self._request("GET", endpoint)

    def get_historical_telemetry(self, device_id: str, keys: str, start_ts: int, end_ts: int, limit: int = 50000) -> Optional[Dict[str, Any]]:
        """Fetches historical time-series data for a specific device between two timestamps."""
        endpoint = (
            f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
            f"?keys={keys}&startTs={start_ts}&endTs={end_ts}&limit={limit}"
        )
        return self._request("GET", endpoint)

    def get_telemetry_for_timeframes(self, device_id: str, keys: str) -> Dict[str, Any]:
        """
        Fetches historical telemetry aggregated into specific timeframes and splits.
        Returns a dictionary containing the aggregated data for 1h, 24h, 7d, and 30d.
        """
        now = datetime.now()
        end_ts = int(now.timestamp() * 1000)
        
        # Define our desired timeframes, start times, and intervals (in milliseconds)
        # ThingsBoard aggregates data into these splits using the 'interval' parameter.
        configs = {
            "1h": {
                "start_ts": int((now - timedelta(hours=1)).timestamp() * 1000),
                "interval": 5 * 60 * 1000  # 5 minutes in ms
            },
            "24h": {
                "start_ts": int((now - timedelta(hours=24)).timestamp() * 1000),
                "interval": 2 * 60 * 60 * 1000  # 2 hours in ms
            },
            "7d": {
                "start_ts": int((now - timedelta(days=7)).timestamp() * 1000),
                "interval": 12 * 60 * 60 * 1000  # 12 hours in ms
            },
            "30d": {
                "start_ts": int((now - timedelta(days=30)).timestamp() * 1000),
                "interval": 2 * 24 * 60 * 60 * 1000  # 2 days in ms
            }
        }
        
        results = {}
        for timeframe, params in configs.items():
            # agg=AVG calculates the average of the data points within the 'interval' split.
            # You can change this to MAX, MIN, SUM, or NONE depending on your needs.
            endpoint = (
                f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
                f"?keys={keys}&startTs={params['start_ts']}&endTs={end_ts}"
                f"&interval={params['interval']}&agg=AVG&limit=50000"
            )
            
            data = self._request("GET", endpoint)
            results[timeframe] = data if data is not None else {}
            
        return results
    
    def get_device_alarms(self, device_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetches the recent active alarms/warnings for a specific device."""
        endpoint = f"/api/alarm/DEVICE/{device_id}?pageSize={limit}&page=0&fetchOriginator=true"
        response_data = self._request("GET", endpoint)
        
        if response_data and isinstance(response_data, dict):
            return response_data.get("data", [])
        return []
        
    def get_device_attributes(self, device_id: str) -> Optional[List[Dict[str, Any]]]:
        """Fetches all attributes (metadata, status, location) for a specific device."""
        endpoint = f"/api/plugins/telemetry/DEVICE/{device_id}/values/attributes"
        return self._request("GET", endpoint)

# ==========================================
# EXPORTED INSTANCE
# ==========================================
tb_client = ThingsBoardClient()

# Optional: Quick test execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Replace with an actual device ID and telemetry key from your ThingsBoard environment!
    TEST_DEVICE_ID = "76ec3ff0-0340-11f0-ab2a-1bdcb487461d" 
    TEST_TELEMETRY_KEY = "temperature" # Example key, change to yours
    
    print("\n--- 1. Fetching Device Attributes (Metadata/Status) ---")
    attributes = tb_client.get_device_attributes(device_id=TEST_DEVICE_ID)
    
    if attributes is not None:
        for attr in attributes:
            if attr.get("key") == "active":
                status = "ONLINE" if attr.get("value") else "OFFLINE"
                print(f"Device Status: {status}")
            
            if attr.get("key") == "lastActivityTime":
                timestamp = attr.get("value")
                if timestamp:
                    last_seen = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Last Seen: {last_seen}")
                
    print("\n--- 2. Fetching Device Alarms (Warnings) ---")
    alarms = tb_client.get_device_alarms(device_id=TEST_DEVICE_ID)
    if not alarms:
        print("Good news! No active alarms for this device.")
    else:
        print(f"Found {len(alarms)} alarms!")
        for alarm in alarms:
            print(f"- [{alarm.get('severity')}] {alarm.get('type')} (Status: {alarm.get('status')})")

    print("\n--- 3. Fetching Aggregated Timeframes Data ---")
    # Make sure TEST_TELEMETRY_KEY exists for your device otherwise it will return empty dicts
    multi_timeframe_data = tb_client.get_telemetry_for_timeframes(device_id=TEST_DEVICE_ID, keys=TEST_TELEMETRY_KEY)
    
    for timeframe, data in multi_timeframe_data.items():
        if not data:
             print(f"[{timeframe}]: No data returned (Device might be inactive or key is wrong).")
             continue
             
        # Find the specific key's data points
        points = data.get(TEST_TELEMETRY_KEY, [])
        print(f"[{timeframe}]: Retrieved {len(points)} aggregated data points (splits).")
        
        # Optional: print the latest point of each timeframe
        if points:
             latest = points[0] # Thingsboard usually returns newest first
             ts = datetime.fromtimestamp(latest['ts'] / 1000).strftime('%m-%d %H:%M')
             print(f"   -> Latest split point inside {timeframe}: {latest['value']} at {ts}")