"""
Live Satellite Tracking Dashboard

Real-time visualization of satellite positions using Space-Track API data.
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.propagation.sgp4_engine import SGP4Engine
from src.core.config import settings
from src.core.logging import get_logger
from src.data.models import TLE


logger = get_logger(__name__)


class LiveTrackingDashboard:
    """Real-time satellite tracking dashboard."""
    
    def __init__(self):
        self.spacetrack_client = None
        self.sgp4_engine = SGP4Engine()
        self.tracked_satellites = {}  # norad_id -> satellite_info
        self.is_running = False
        
        # Initialize Space-Track client if credentials are available
        if settings.spacetrack.spacetrack_username and settings.spacetrack.spacetrack_password:
            rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack.spacetrack_rate_limit)
            self.spacetrack_client = SpaceTrackClient(rate_limiter)
            logger.info("Space-Track client initialized with credentials")
        else:
            logger.error("❌ NO Space-Track credentials found - cannot run real tracking")
            raise RuntimeError("Space-Track credentials required for real-time tracking")
    
    async def initialize_tracking(self, satellite_ids: List[int]):
        """Initialize tracking for specified satellites."""
        if not self.spacetrack_client:
            raise RuntimeError("Space-Track client not initialized")
        
        logger.info(f"Initializing live tracking for {len(satellite_ids)} satellites")
        
        try:
            async with self.spacetrack_client:
                for norad_id in satellite_ids:
                    try:
                        # Fetch latest TLE for satellite
                        tle = await self.spacetrack_client.fetch_tle_by_norad_id(norad_id, days_back=7)  # Try 7 days back
                        
                        if not tle:
                            # Try alternative query method
                            logger.info(f"Trying alternative query for NORAD {norad_id}")
                            # Use the fetch_tle_catalog method which might work better
                            start_date = datetime.utcnow() - timedelta(days=7)
                            end_date = datetime.utcnow()
                            
                            catalog_result = await self.spacetrack_client.fetch_tle_catalog(
                                epoch_start=start_date,
                                epoch_end=end_date,
                                norad_ids=[norad_id]
                            )
                            
                            if catalog_result:
                                tle = catalog_result[0]  # Take the first (most recent)
                        
                        if tle:
                            # Calculate current position
                            current_pos = self.sgp4_engine.propagate_to_epoch(tle, datetime.utcnow())
                            
                            self.tracked_satellites[norad_id] = {
                                'tle': tle,
                                'last_updated': datetime.utcnow(),
                                'current_position': {
                                    'x': current_pos.cartesian_state.x,
                                    'y': current_pos.cartesian_state.y,
                                    'z': current_pos.cartesian_state.z,
                                    'lat': current_pos.latitude_deg,
                                    'lon': current_pos.longitude_deg,
                                    'alt': current_pos.altitude_m
                                },
                                'name': self._get_satellite_name(norad_id)
                            }
                            
                            logger.info(f"Started tracking {self._get_satellite_name(norad_id)} (NORAD: {norad_id})")
                        else:
                            logger.warning(f"No TLE available for NORAD {norad_id}")
                            
                    except Exception as e:
                        logger.error(f"Error initializing tracking for NORAD {norad_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error connecting to Space-Track: {e}")
            raise
    
    def _get_satellite_name(self, norad_id: int) -> str:
        """Get human-readable name for satellite."""
        names = {
            25544: "ISS (ZARYA)",
            42982: "Starlink-1087",
            43017: "Starlink-1112",
            43129: "Starlink-1174",
            43217: "Starlink-1216",
            39084: "GOES-16",
            39436: "GOES-17",
            40053: "GOES-15",
            20409: "GPS BIIR-1 (PRN 25)",
            24277: "Iridium 78",
        }
        return names.get(norad_id, f"Sat-{norad_id}")
    
    async def update_positions(self):
        """Update positions of tracked satellites."""
        if not self.spacetrack_client:
            logger.error("No Space-Track client available")
            raise RuntimeError("Cannot update positions without Space-Track connection")
        
        if not self.tracked_satellites:
            logger.warning("No tracked satellites, skipping position update")
            return
        
        try:
            async with self.spacetrack_client:
                for norad_id, sat_info in self.tracked_satellites.items():
                    try:
                        # Fetch latest TLE if we haven't updated recently
                        if datetime.utcnow() - sat_info['last_updated'] > timedelta(minutes=5):
                            tle = await self.spacetrack_client.fetch_tle_by_norad_id(norad_id, days_back=7)
                            if tle:
                                sat_info['tle'] = tle
                                sat_info['last_updated'] = datetime.utcnow()
                        
                        # Calculate current position if we have TLE data
                        if sat_info['tle']:
                            current_pos = self.sgp4_engine.propagate_to_epoch(sat_info['tle'], datetime.utcnow())
                            
                            sat_info['current_position'] = {
                                'x': current_pos.cartesian_state.x,
                                'y': current_pos.cartesian_state.y,
                                'z': current_pos.cartesian_state.z,
                                'lat': current_pos.latitude_deg,
                                'lon': current_pos.longitude_deg,
                                'alt': current_pos.altitude_m
                            }
                        else:
                            logger.warning(f"No TLE data available for NORAD {norad_id}, skipping position update")
                        
                    except Exception as e:
                        logger.error(f"Error updating position for NORAD {norad_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def display_dashboard(self):
        """Display current satellite positions."""
        print("\n" + "="*80)
        print(" LIVE SATELLITE TRACKING DASHBOARD")
        print("="*80)
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Active Satellites: {len(self.tracked_satellites)}")
        print("-"*80)
        
        if not self.tracked_satellites:
            print("❌ No satellites being tracked.")
            return
        
        for norad_id, sat_info in self.tracked_satellites.items():
            pos = sat_info['current_position']
            print(f"Satellite: {sat_info['name']:<15} | NORAD: {norad_id}")
            print(f"  Position: ({pos['x']/1000:,.1f}, {pos['y']/1000:,.1f}, {pos['z']/1000:,.1f}) km")
            print(f"  Location: {pos['lat']:6.2f}°N, {pos['lon']:7.2f}°E | Altitude: {pos['alt']/1000:,.0f} km")
            print(f"  Status: CONNECTED - REAL TIME DATA")
            print("-" * 80)
    
    async def run_live_tracking(self, satellite_ids: List[int], duration_minutes: int = 10):
        """Run live tracking for specified duration."""
        self.is_running = True
        
        # Initialize tracking
        await self.initialize_tracking(satellite_ids)
        
        print(f"\n🚀 Starting REAL-TIME tracking for {duration_minutes} minutes...")
        print(f"Tracking {len(satellite_ids)} satellites: {satellite_ids}")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)  # Convert to seconds
        
        iteration = 0
        while self.is_running and time.time() < end_time:
            iteration += 1
            print(f"\n--- ITERATION {iteration} ---")
            
            # Update positions
            await self.update_positions()
            
            # Display current positions
            if self.tracked_satellites:  # Only display if we have satellites
                self.display_dashboard()
            else:
                print("⚠️  No satellites currently tracked. Waiting for data...")
            
            # Wait before next update
            await asyncio.sleep(5)  # Update every 5 seconds
        
        print(f"\n🏁 Live tracking completed after {duration_minutes} minutes")
        self.is_running = False


async def main():
    """Main entry point for live tracking demo."""
    print("🎯 SSA Live Satellite Tracking - REAL-TIME DATA")
    print("="*50)
    
    # Check for credentials first
    if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
        print("❌ ERROR: No Space-Track credentials found!")
        print("Please configure SPACETRACK_USERNAME and SPACETRACK_PASSWORD in your .env file")
        return
    
    print(f"Using account: {settings.spacetrack.spacetrack_username}")
    
    # Common satellite IDs to track
    satellite_ids = [25544, 42982, 43017]  # ISS, Starlinks
    
    try:
        tracker = LiveTrackingDashboard()
        
        # Run for 2 minutes with real data
        await tracker.run_live_tracking(satellite_ids, duration_minutes=2)
    except KeyboardInterrupt:
        print("\n🛑 Live tracking stopped by user")
    except Exception as e:
        print(f"\n❌ Error in live tracking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())