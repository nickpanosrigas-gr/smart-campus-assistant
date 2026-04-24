import requests
from datetime import datetime, timedelta
from config.settings import settings

class ThingsBoardAPI:
    def __init__(self):
        self.base_url = settings.THINGSBOARD_BASE_URL
        self.username = settings.THINGSBOARD_USERNAME
        self.password = settings.THINGSBOARD_PASSWORD
        self.token = None

    def login(self) -> str | None:
        """Authenticates with ThingsBoard and stores the JWT token."""
        url = f"{self.base_url}/api/auth/login"
        payload = {
            "username": self.username,
            "password": self.password
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() 
            
            # Extract and store the token
            self.token = response.json().get("token")
            print("Successfully authenticated with ThingsBoard API!")
            return self.token
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to ThingsBoard: {e}")
            return None

    def get_auth_headers(self) -> dict:
        """Returns the necessary Authorization headers for subsequent API calls."""
        if not self.token:
            self.login()
            
        return {
            "X-Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def get_device_telemetry(self, device_id: str, keys: str):
        """
        Example method to fetch time-series data for a specific device.
        device_id: The UUID of the device in ThingsBoard.
        keys: Comma-separated string of telemetry keys (e.g., 'temperature,humidity,co2').
        """
        url = f"{self.base_url}/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries?keys={keys}"
        headers = self.get_auth_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching telemetry: {e}")
            return None

    def get_historical_telemetry(self, device_id: str, keys: str, days_back: int = 7):
        """
        Fetches historical time-series data for a specific device over a number of days.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        end_ts = int(end_time.timestamp() * 1000)
        start_ts = int(start_time.timestamp() * 1000)
        
        # ADDED: limit=50000 to override the default 100 limit
        url = f"{self.base_url}/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
        url += f"?keys={keys}&startTs={start_ts}&endTs={end_ts}&limit=50000"
        
        headers = self.get_auth_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical telemetry: {e}")
            return None
    
    def get_device_alarms(self, device_id: str, limit: int = 10):
        """
        Fetches the recent active alarms/warnings for a specific device.
        """
        # We search for alarms linked to this specific DEVICE
        url = f"{self.base_url}/api/alarm/DEVICE/{device_id}?pageSize={limit}&page=0&fetchOriginator=true"
        headers = self.get_auth_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get("data", []) # Returns a list of alarms
        except requests.exceptions.RequestException as e:
            print(f"Error fetching alarms: {e}")
            return None
        
    def get_device_attributes(self, device_id: str):
        """
        Fetches all attributes (metadata, status, location) for a specific device.
        """
        url = f"{self.base_url}/api/plugins/telemetry/DEVICE/{device_id}/values/attributes"
        headers = self.get_auth_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching attributes: {e}")
            return None
        
# Quick test execution
if __name__ == "__main__":
    tb_api = ThingsBoardAPI()
    token = tb_api.login()
    
    if token:
        TEST_DEVICE_ID = "76ec3ff0-0340-11f0-ab2a-1bdcb487461d" 
        
        print("\n--- 1. Fetching Device Attributes (Metadata/Status) ---")
        attributes = tb_api.get_device_attributes(device_id=TEST_DEVICE_ID)
        
        if attributes is not None:
            # We now know 'attributes' is a flat list, so we iterate through it directly
            for attr in attributes:
                # Use .get() on the dictionary inside the list
                if attr.get("key") == "active":
                    status = "ONLINE" if attr.get("value") else "OFFLINE"
                    print(f"Device Status: {status}")
                
                if attr.get("key") == "lastActivityTime":
                    # Convert the millisecond timestamp to a readable date
                    timestamp = attr.get("value")
                    if timestamp:
                        last_seen = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"Last Seen: {last_seen}")
                    
        print("\n--- 2. Fetching Device Alarms (Warnings) ---")
        alarms = tb_api.get_device_alarms(device_id=TEST_DEVICE_ID)
        if alarms is not None:
            if len(alarms) == 0:
                print("Good news! No active alarms for this device.")
            else:
                print(f"Found {len(alarms)} alarms!")
                for alarm in alarms:
                    print(f"- [{alarm.get('severity')}] {alarm.get('type')} (Status: {alarm.get('status')})")