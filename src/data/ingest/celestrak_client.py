"""
Celestrak API client for fetching satellite data.
Simple implementation for testing purposes.
"""

import asyncio
import httpx
from typing import List, Optional
from datetime import datetime
from src.data.models import TLE

class CelestrakClient:
    """Simple Celestrak client for testing."""
    
    BASE_URL = "https://celestrak.org/NORAD/elements/"
    
    def __init__(self):
        self.session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_active_satellites(self, limit: int = 100) -> List[TLE]:
        """
        Fetch active satellites from Celestrak.
        This is a simplified implementation for testing.
        """
        if not self.session:
            async with self:
                return await self._fetch_active_impl(limit)
        return await self._fetch_active_impl(limit)
    
    async def _fetch_active_impl(self, limit: int) -> List[TLE]:
        """Implementation of active satellite fetch."""
        try:
            # For testing, we'll create some dummy TLEs
            # In a real implementation, we'd fetch from Celestrak's API
            dummy_tles = [
                TLE(
                    norad_id=25544,  # ISS
                    classification='U',
                    launch_year=98,
                    launch_number=67,
                    launch_piece='A',
                    epoch_datetime=datetime.utcnow(),
                    mean_motion_derivative=-0.001,
                    mean_motion_sec_derivative=0.0,
                    bstar_drag_term=0.0001,
                    element_set_number=999,
                    inclination_degrees=51.6,
                    raan_degrees=0.0,
                    eccentricity=0.001,
                    argument_of_perigee_degrees=0.0,
                    mean_anomaly_degrees=0.0,
                    mean_motion_orbits_per_day=15.5,
                    revolution_number_at_epoch=1000,
                    tle_line1="1 25544U 98067A   24020.60416667  .00016717  00000-0  10270-3 0  9991",
                    tle_line2="2 25544  51.6416 247.4627 0006703  95.3672 264.8598 15.50130001234567",
                    epoch_julian_date=2451545.0,
                    line1_checksum=1,
                    line2_checksum=7,
                    is_valid=True
                ),
                TLE(
                    norad_id=42982,  # Another satellite
                    classification='U',
                    launch_year=17,
                    launch_number=61,
                    launch_piece='H',
                    epoch_datetime=datetime.utcnow(),
                    mean_motion_derivative=0.0,
                    mean_motion_sec_derivative=0.0,
                    bstar_drag_term=0.0,
                    element_set_number=999,
                    inclination_degrees=97.4,
                    raan_degrees=123.5,
                    eccentricity=0.0012,
                    argument_of_perigee_degrees=300.1,
                    mean_anomaly_degrees=59.9,
                    mean_motion_orbits_per_day=14.8,
                    revolution_number_at_epoch=3456,
                    tle_line1="1 42982U 17061H   24020.60416667  .00000000  00000-0  00000-0 0  9998",
                    tle_line2="2 42982  97.3804 123.4567 0012345 300.1234  59.8765 14.81875000345678",
                    epoch_julian_date=2451545.0,
                    line1_checksum=8,
                    line2_checksum=8,
                    is_valid=True
                )
            ]
            
            return dummy_tles[:limit]
            
        except Exception as e:
            print(f"Error fetching from Celestrak: {e}")
            return []
    
    async def fetch_tle_by_norad_id(self, norad_id: int) -> Optional[TLE]:
        """
        Fetch TLE for specific NORAD ID.
        Simplified implementation for testing.
        """
        # Return dummy TLE for ISS
        if norad_id == 25544:
            return TLE(
                norad_id=25544,
                classification='U',
                launch_year=98,
                launch_number=67,
                launch_piece='A',
                epoch_datetime=datetime.utcnow(),
                mean_motion_derivative=-0.001,
                mean_motion_sec_derivative=0.0,
                bstar_drag_term=0.0001,
                element_set_number=999,
                inclination_degrees=51.6,
                raan_degrees=0.0,
                eccentricity=0.001,
                argument_of_perigee_degrees=0.0,
                mean_anomaly_degrees=0.0,
                mean_motion_orbits_per_day=15.5,
                revolution_number_at_epoch=1000,
                tle_line1="1 25544U 98067A   24020.60416667  .00016717  00000-0  10270-3 0  9991",
                tle_line2="2 25544  51.6416 247.4627 0006703  95.3672 264.8598 15.50130001234567",
                epoch_julian_date=2451545.0,
                line1_checksum=1,
                line2_checksum=7,
                is_valid=True
            )
        
        return None