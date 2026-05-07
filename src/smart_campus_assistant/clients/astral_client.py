import logging
from datetime import datetime, timedelta
import pytz
from astral import LocationInfo
from astral.sun import sun, elevation, azimuth

from src.smart_campus_assistant.config.settings import settings

logger = logging.getLogger(__name__)

class AstralClient:
    def __init__(self):
        # Pull from settings, with fallbacks
        lat = getattr(settings, 'LATITUDE', 35.3387)
        lon = getattr(settings, 'LONGITUDE', 25.1442)
        tz_str = getattr(settings, 'TIMEZONE', 'Europe/Athens')
        
        self.tz = pytz.timezone(tz_str)
        self.location = LocationInfo("Campus", "Greece", tz_str, lat, lon)
        
    def get_elevation_info(self, dt: datetime = None) -> tuple[str, str]:
        """Returns semantic label and description for solar elevation."""
        if dt is None:
            dt = datetime.now(self.tz)
            
        el = elevation(self.location.observer, dt)
        
        if el < 0:
            return "Night / Below Horizon", "No natural light available."
        elif 0 <= el < 12:
            return "Critical Glare Zone", "Sun is at eye level; blinds are likely needed."
        elif 12 <= el < 30:
            return "High Glare / Deep Penetration", "Light reaches far into the room; high solar heat gain."
        elif 30 <= el < 60:
            return "Optimal Daylight", "Sun is high enough that over-hangs/eaves might shade windows."
        else:
            return "Directly Overhead", "Sun hits the roof; minimal light through side windows."

    def get_average_elevation_info(self, hours_back: int) -> tuple[str, str]:
        """Returns semantic label and description for average solar elevation over the last X hours."""
        dt = datetime.now(self.tz) - timedelta(hours=hours_back / 2.0)
        return self.get_elevation_info(dt)
            
    def get_azimuth_info(self, dt: datetime = None) -> str:
        """Returns semantic cardinal direction of the sun."""
        if dt is None:
            dt = datetime.now(self.tz)
            
        az = azimuth(self.location.observer, dt)
        
        if az >= 337.5 or az < 22.5: return "North"
        elif 22.5 <= az < 67.5: return "North-East"
        elif 67.5 <= az < 112.5: return "East"
        elif 112.5 <= az < 157.5: return "South-East"
        elif 157.5 <= az < 202.5: return "South"
        elif 202.5 <= az < 247.5: return "South-West"
        elif 247.5 <= az < 292.5: return "West"
        else: return "North-West"
        
    def get_current_solar_context(self) -> dict:
        """Returns a snapshot of current sun position for real-time queries."""
        dt = datetime.now(self.tz)
        s_info = sun(self.location.observer, date=dt.date(), tzinfo=self.tz)
        
        el_label, el_desc = self.get_elevation_info(dt)
        az_label = self.get_azimuth_info(dt)
        
        return {
            "vertical": f"{el_label} ({el_desc})",
            "horizontal": f"Sun is facing {az_label}",
            "sunrise": s_info["sunrise"].strftime('%H:%M'),
            "sunset": s_info["sunset"].strftime('%H:%M')
        }
        
    def get_historical_solar_context(self, days_back: int) -> dict:
        """Returns the average sunrise, sunset, and trajectory for a historical period."""
        end_date = datetime.now(self.tz).date()
        # Find the middle day of the timeframe to represent the average daylight window
        mid_date = end_date - timedelta(days=max(1, days_back // 2))
        
        mid_s_info = sun(self.location.observer, date=mid_date, tzinfo=self.tz)
        
        az_sunrise = self.get_azimuth_info(mid_s_info["sunrise"])
        az_noon = self.get_azimuth_info(mid_s_info["noon"])
        az_sunset = self.get_azimuth_info(mid_s_info["sunset"])
        
        trajectory = f"Rises {az_sunrise} -> Transits {az_noon} -> Sets {az_sunset}"
        
        return {
            "avg_sunrise": mid_s_info["sunrise"].strftime('%H:%M'),
            "avg_sunset": mid_s_info["sunset"].strftime('%H:%M'),
            "trajectory": trajectory
        }

astral_client = AstralClient()

if __name__ == "__main__":
    print("Testing Astral Client...")
    print("\nCurrent Solar Context:")
    print(astral_client.get_current_solar_context())
    print("\nAverage Elevation (Last 2h):")
    print(astral_client.get_average_elevation_info(2))
    print("\nHistorical 30d Average Window:")
    print(astral_client.get_historical_solar_context(30))