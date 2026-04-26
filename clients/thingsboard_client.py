import time
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class ThingsBoardClient:
    """
    Client for interacting with the ThingsBoard REST API.
    Fetches RAW data. Aggregation and domain-logic are delegated to the specific LangGraph tools.
    """
    
    def __init__(self):
        # Ensure there's no trailing slash to prevent double-slash in URLs
        self.base_url = settings.THINGSBOARD_BASE_URL.rstrip('/')
        self.username = settings.THINGSBOARD_USERNAME
        self.password = settings.THINGSBOARD_PASSWORD
        self.token: Optional[str] = None
        
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticates with ThingsBoard using username/password and retrieves a new JWT token."""
        url = f"{self.base_url}/api/auth/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            self.token = response.json().get("token")
            logger.info("Successfully authenticated with ThingsBoard API.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to authenticate with ThingsBoard: {e}")
            raise

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Wrapper to handle automatic JWT token refreshing on 401 Unauthorized."""
        url = f"{self.base_url}{endpoint}"
        
        if not self.token:
            self._authenticate()
            
        headers = {"X-Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        response = requests.request(method, url, headers=headers, params=params, **kwargs)
        
        if response.status_code == 401:
            logger.warning("ThingsBoard JWT token expired. Re-authenticating...")
            self._authenticate()
            headers["X-Authorization"] = f"Bearer {self.token}"
            response = requests.request(method, url, headers=headers, params=params, **kwargs)
            
        response.raise_for_status()
        return response

    def _fetch_raw_telemetry(self, device_id: str, keys: List[str], start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Fetches ALL raw data points without server-side aggregation.
        Uses a massive limit to ensure no data is truncated.
        """
        endpoint = f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
        params = {
            "keys": ",".join(keys),
            "startTs": start_ts,
            "endTs": end_ts,
            "limit": 100000,  # Force TB to return up to 100k raw points
            "useStrictDataTypes": "false"
        }
        
        response = self._request("GET", endpoint, params=params)
        return response.json()

    # ==========================================
    # TIME-FRAME FUNCTIONS (Returning Raw Data)
    # ==========================================
    
    def get_now(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches the absolute latest telemetry point (No start/end constraints)."""
        endpoint = f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
        params = {
            "keys": ",".join(keys),
            "useStrictDataTypes": "false"
        }
        response = self._request("GET", endpoint, params=params)
        return response.json()

    def get_2h(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 2h of raw data."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (2 * 3600 * 1000)
        return self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)

    def get_24h(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 24h of raw data."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (24 * 3600 * 1000)
        return self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)

    def get_7d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 7 days of raw data."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (7 * 24 * 3600 * 1000)
        return self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)

    def get_30d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 30 days of raw data."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (30 * 24 * 3600 * 1000)
        return self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)

tb_client = ThingsBoardClient()

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    def print_chunk(title: str, data: Dict):
        print(f"\n{'-'*50}\n▶ {title}\n{'-'*50}")
        # Print a small preview of the raw data list
        preview = {}
        for key, points in data.items():
            preview[key] = points[:3] if isinstance(points, list) else points
            if isinstance(points, list) and len(points) > 3:
                preview[key].append({"...": f"({len(points) - 3} more raw points)"})
        print(json.dumps(preview, indent=2))

    print("\n" + "="*50)
    print("🧪 RUNNING RAW THINGSBOARD CLIENT TESTS 🧪")
    print("="*50)

    try:
        tb_client = ThingsBoardClient()

        # Testing with the Door/Window Sensor you provided
        TEST_DEVICE_ID = "34578b10-0343-11f0-ab2a-1bdcb487461d"
        TEST_KEYS = ["light_level"]

        print(f"\nTesting with Device ID: {TEST_DEVICE_ID}")
        print(f"Requesting Telemetry Keys: {TEST_KEYS}")

        now_data = tb_client.get_now(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_now() [Latest point]", now_data)

        h24_data = tb_client.get_30d(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_24h() [Raw points]", h24_data)

        print("\n✅ All ThingsBoard client tests completed successfully.\n")

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")