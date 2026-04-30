import time
import logging
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from smart_campus_assistant.config.settings import settings

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
            
        # --- NEW ERROR CATCHER ---
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Print the actual text/JSON response from ThingsBoard to see why it rejected the parameters
            logger.error(f"HTTP {response.status_code} Error from ThingsBoard: {response.text}")
            raise
            
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
    
    def _fetch_aggregated_telemetry(self, device_id: str, keys: List[str], start_ts: int, end_ts: int, agg: str = "AVG") -> Dict[str, Any]:
        """
        Fetches server-side aggregated data (e.g., averages) over a specific time window.
        By setting the interval to the total time window, it returns a single summary point.
        """
        endpoint = f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
        
        interval = end_ts - start_ts
        if interval <= 0:
            interval = 86400000
            
        params = {
            "keys": ",".join(keys),
            "startTs": start_ts,
            "endTs": end_ts,
            "interval": interval,
            "agg": agg,
            "useStrictDataTypes": "false"
        }
        
        response = self._request("GET", endpoint, params=params)
        return response.json()

    # ==========================================
    # CONTEXTUAL DATA PROCESSING
    # ==========================================

    def _calculate_contextual_averages(self, hourly_data: Dict[str, Any], work_start_hr: int = 8, work_end_hr: int = 22) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Sorts hourly data points into 4 context buckets and averages them.
        Working hours updated to: 08:00 to 22:00.
        """
        results = {}
        
        for key, points in hourly_data.items():
            buckets = {
                "weekday_work": [],
                "weekday_nonwork": [],
                "weekend_work": [],
                "weekend_nonwork": []
            }
            
            for pt in points:
                dt = datetime.fromtimestamp(pt["ts"] / 1000.0)
                try:
                    val = float(pt["value"])
                except (ValueError, TypeError, KeyError):
                    continue 

                is_weekend = dt.weekday() >= 5 
                is_working_hour = work_start_hr <= dt.hour < work_end_hr
                
                if not is_weekend and is_working_hour:
                    buckets["weekday_work"].append(val)
                elif not is_weekend and not is_working_hour:
                    buckets["weekday_nonwork"].append(val)
                elif is_weekend and is_working_hour:
                    buckets["weekend_work"].append(val)
                else:
                    buckets["weekend_nonwork"].append(val)

            key_results = {}
            for bucket_name, values in buckets.items():
                if values:
                    key_results[bucket_name] = round(sum(values) / len(values), 2)
                else:
                    key_results[bucket_name] = None 
                    
            results[key] = key_results
            
        return results

    def _fetch_30d_context_baseline(self, device_id: str, keys: List[str], target_start_ts: int) -> Dict[str, Any]:
        """
        Fetches 30 days of RAW data prior to the target start, 
        then sorts them into weekday/weekend and work/non-work buckets.
        This bypasses ThingsBoard's strict server-side interval limits.
        """
        # 30 days before the target period
        start_ts = target_start_ts - (30 * 24 * 3600 * 1000)
        
        # Grab the raw telemetry directly!
        raw_data = self._fetch_raw_telemetry(device_id, keys, start_ts, target_start_ts)
        
        # Our context calculator doesn't care if it's hourly or raw, 
        # it just looks at the timestamp of every point and buckets it perfectly.
        return self._calculate_contextual_averages(raw_data)

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
    
    # ========================================================
    # CONTEXTUAL BASELINE FUNCTIONS (Prev 30d Contextual Avgs)
    # ========================================================

    def get_now_prev_30d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches the 30-day contextual average prior to 'now'."""
        target_start_ts = int(time.time() * 1000)
        return self._fetch_30d_context_baseline(device_id, keys, target_start_ts)

    def get_2h_prev_30d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches the 30-day contextual average prior to the last 2 hours."""
        target_start_ts = int(time.time() * 1000) - (2 * 3600 * 1000) 
        return self._fetch_30d_context_baseline(device_id, keys, target_start_ts)

    def get_24h_prev_30d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches the 30-day contextual average prior to the last 24 hours."""
        target_start_ts = int(time.time() * 1000) - (24 * 3600 * 1000) 
        return self._fetch_30d_context_baseline(device_id, keys, target_start_ts)

    def get_7d_prev_30d(self, device_id: str, keys: List[str]) -> Dict[str, Any]:
        """Fetches the 30-day contextual average prior to the last 7 days."""
        target_start_ts = int(time.time() * 1000) - (7 * 24 * 3600 * 1000) 
        return self._fetch_30d_context_baseline(device_id, keys, target_start_ts)

tb_client = ThingsBoardClient()

# ==========================================
# TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    def print_chunk(title: str, data: Dict):
        print(f"\n{'-'*50}\n▶ {title}\n{'-'*50}")
        preview = {}
        for key, points in data.items():
            if isinstance(points, dict):  
                preview[key] = points
            else:
                preview[key] = points[:50] if isinstance(points, list) else points
                if isinstance(points, list) and len(points) > 3:
                    preview[key].append({"...": f"({len(points) - 3} more raw points)"})
        print(json.dumps(preview, indent=2))

    print("\n" + "="*50)
    print("RUNNING RAW THINGSBOARD CLIENT TESTS")
    print("="*50)

    try:
        tb_client = ThingsBoardClient()

        # Testing with the Door/Window Sensor you provided
        TEST_DEVICE_ID = "36bd0be0-13b6-11f0-a6f6-451660aa424d"
        TEST_KEYS = ["air_temperature", "maximum_wind_speed"]

        print(f"\nTesting with Device ID: {TEST_DEVICE_ID}")
        print(f"Requesting Telemetry Keys: {TEST_KEYS}")

        now_data = tb_client.get_now(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_now() [Latest point]", now_data)

        d30_data = tb_client.get_30d(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_30d() [Raw points]", d30_data)
        
        d7_prev_30d_data = tb_client.get_7d_prev_30d(TEST_DEVICE_ID, TEST_KEYS)
        print_chunk("Testing get_7d_prev_30d [Contextual Averages]", d7_prev_30d_data)

        print("\nAll ThingsBoard client tests completed successfully.\n")

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")