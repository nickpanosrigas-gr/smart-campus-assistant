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
        
    # =======================================================
    # LIGHT & GLARE FOCUS (Used by lights.py)
    # =======================================================
    def get_elevation_info(self, dt: datetime = None) -> tuple[str, str]:
        """Returns semantic label and description for solar visual glare."""
        if dt is None: dt = datetime.now(self.tz)
        el = elevation(self.location.observer, dt)
        
        if el < 0: return "Night / Below Horizon", "No natural light available."
        elif 0 <= el < 12: return "Critical Glare Zone", "Sun is at eye level; blinds are likely needed."
        elif 12 <= el < 30: return "High Glare / Deep Penetration", "Light reaches far into the room."
        elif 30 <= el < 60: return "Optimal Daylight", "Sun is high enough that overhangs might shade windows."
        else: return "Directly Overhead", "Sun hits the roof; minimal light through side windows."

    def get_average_elevation_info(self, hours_back: int) -> tuple[str, str]:
        dt = datetime.now(self.tz) - timedelta(hours=hours_back / 2.0)
        return self.get_elevation_info(dt)

    # =======================================================
    # TEMPERATURE & HEAT FOCUS (Used by temp_humidity.py)
    # =======================================================
    def get_thermal_elevation_info(self, dt: datetime = None) -> tuple[str, str]:
        """Returns semantic label and description for solar heat gain potential."""
        if dt is None: dt = datetime.now(self.tz)
        el = elevation(self.location.observer, dt)
        
        if el < 0:
            return "Nighttime / Radiative Cooling", "No direct solar heat gain; building structure is cooling."
        elif 0 <= el < 20:
            return "Low Thermal Angle", "Glancing rays on East/West facades; mild ambient heat gain."
        elif 20 <= el < 50:
            return "Moderate Heat Load", "Increasing solar radiation intensity on walls and unshaded windows."
        elif 50 <= el < 65:
            return "High Heat Intensity", "Strong direct solar radiation causing rapid ambient temperature increases."
        else:
            return "Peak Overhead Radiation", "Sun is directly overhead; maximum thermal load on the roof (hottest part of the day)."

    def get_average_thermal_elevation_info(self, hours_back: int) -> tuple[str, str]:
        """Returns semantic thermal label over the last X hours."""
        dt = datetime.now(self.tz) - timedelta(hours=hours_back / 2.0)
        return self.get_thermal_elevation_info(dt)
            
    # =======================================================
    # SHARED METHODS
    # =======================================================
    def get_azimuth_info(self, dt: datetime = None) -> str:
        """Returns semantic cardinal direction of the sun."""
        if dt is None: dt = datetime.now(self.tz)
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
        
        # Fetch both contexts
        el_label, el_desc = self.get_elevation_info(dt)
        th_label, th_desc = self.get_thermal_elevation_info(dt)
        az_label = self.get_azimuth_info(dt)
        
        return {
            "vertical": f"{el_label} ({el_desc})",                 # For lights.py
            "thermal_vertical": f"{th_label} ({th_desc})",         # For temp_humidity.py
            "horizontal": f"Sun is facing {az_label}",
            "sunrise": s_info["sunrise"].strftime('%H:%M'),
            "sunset": s_info["sunset"].strftime('%H:%M')
        }
        
    def get_historical_solar_context(self, days_back: int) -> dict:
        """Returns the average sunrise, sunset, and trajectory for a historical period."""
        end_date = datetime.now(self.tz).date()
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