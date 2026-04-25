import time
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
from config.settings import settings

logger = logging.getLogger(__name__)

class ThingsBoardClient:
    """
    Client for interacting with the ThingsBoard REST API.
    Fetches RAW data to perform high-accuracy Min/Max/Avg calculations in Python.
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
            "limit": 100000,          # Force TB to return up to 100k raw points
            "useStrictDataTypes": "false"
        }
        
        response = self._request("GET", endpoint, params=params)
        return response.json()

    def _aggregate_raw_data(self, raw_data: Dict[str, List[Dict]], start_ts: int, end_ts: int, interval_ms: int) -> Dict[str, List[Dict]]:
        """
        Takes raw ThingsBoard telemetry and chunks it into specific timeframes.
        Calculates the true min, max, and avg for each chunk.
        """
        aggregated_result = defaultdict(list)

        for key, points in raw_data.items():
            bins = defaultdict(list)
            for pt in points:
                ts = int(pt["ts"])
                value = float(pt["value"])
                bin_index = (ts - start_ts) // interval_ms
                bins[bin_index].append(value)
            
            total_bins = (end_ts - start_ts) // interval_ms
            
            for i in range(total_bins):
                bin_values = bins.get(i)
                bin_start_time = start_ts + (i * interval_ms)
                
                if bin_values:
                    aggregated_result[key].append({
                        "ts_start": bin_start_time,
                        "min": round(min(bin_values), 2),
                        "max": round(max(bin_values), 2),
                        "avg": round(sum(bin_values) / len(bin_values), 2),
                        "data_points_count": len(bin_values)
                    })
                else:
                    aggregated_result[key].append({
                        "ts_start": bin_start_time,
                        "min": None, "max": None, "avg": None, "data_points_count": 0
                    })
                    
        return dict(aggregated_result)

    # ==========================================
    # TIME-FRAME FUNCTIONS (LangGraph Tools)
    # ==========================================
    
    def get_now(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches the absolute latest telemetry point for the requested keys."""
        endpoint = f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
        params = {
            "keys": ",".join(keys),
            "useStrictDataTypes": "false"
        }
        response = self._request("GET", endpoint, params=params)
        return response.json()

    def get_2h(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 2h raw data, chunks into 10-minute bins."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (2 * 3600 * 1000)
        interval = 10 * 60 * 1000
        raw_data = self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)
        return self._aggregate_raw_data(raw_data, start_ts, end_ts, interval)

    def get_24h(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 24h raw data, chunks into 2-hour bins."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (24 * 3600 * 1000)
        interval = 2 * 3600 * 1000
        raw_data = self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)
        return self._aggregate_raw_data(raw_data, start_ts, end_ts, interval)

    def get_7d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 7d raw data, chunks into 12-hour bins."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (7 * 24 * 3600 * 1000)
        interval = 12 * 3600 * 1000
        raw_data = self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)
        return self._aggregate_raw_data(raw_data, start_ts, end_ts, interval)

    def get_30d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches 30d raw data, chunks into 48-hour bins."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (30 * 24 * 3600 * 1000)
        interval = 48 * 3600 * 1000
        raw_data = self._fetch_raw_telemetry(device_id, keys, start_ts, end_ts)
        return self._aggregate_raw_data(raw_data, start_ts, end_ts, interval)


# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Setup basic logging to see the initialization info
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Format helper to print clean JSON chunks to the terminal
    def print_chunk(title: str, data: Dict):
        print(f"\n{'-'*50}\n▶ {title}\n{'-'*50}")
        # We only print the first two bins to avoid flooding the terminal
        preview = {}
        for key, bins in data.items():
            preview[key] = bins[:2] if isinstance(bins, list) else bins
        print(json.dumps(preview, indent=2))
        print(f"... (truncated for readability)")

    print("\n" + "="*50)
    print("🧪 RUNNING THINGSBOARD CLIENT TESTS 🧪")
    print("="*50)

    try:
        # Initialize client (This will automatically try to authenticate using your settings)
        tb_client = ThingsBoardClient()

        # Using an actual device ID from your campus_topology.json (F1_1.2-IAQ-1)
        TEST_DEVICE_ID = "8a993270-0353-11f0-ab2a-1bdcb487461d"
        TEST_KEYS = ["co2", "temperature"]

        print(f"\nTesting with Device ID: {TEST_DEVICE_ID}")
        print(f"Requesting Telemetry Keys: {TEST_KEYS}")

        # 1. Test "Now" (Absolute Latest Data)
        now_data = tb_client.get_now(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_now() [Absolute Latest via /latest endpoint]", now_data)

        # 2. Test 2 Hours (10-min bins)
        h2_data = tb_client.get_2h(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_2h() [Aggregated into 10-min bins]", h2_data)

        # 3. Test 24 Hours (2-hour bins)
        h24_data = tb_client.get_24h(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_24h() [Aggregated into 2-hour bins]", h24_data)

        # 4. Test 7 Days (12-hour bins)
        d7_data = tb_client.get_7d(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_7d() [Aggregated into 12-hour bins]", d7_data)

        # 5. Test 30 Days (48-hour bins)
        d30_data = tb_client.get_30d(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_30d() [Aggregated into 48-hour bins]", d30_data)

        print("\n✅ All ThingsBoard client tests completed successfully.\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to ThingsBoard.")
        print("Ensure your local/cloud ThingsBoard instance is running and the URL in config/settings.py is correct.")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP ERROR: {e}")
        print("Check your ThingsBoard credentials in config/settings.py.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")