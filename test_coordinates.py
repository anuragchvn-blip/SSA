#!/usr/bin/env python3
"""Test coordinate transformation calculations."""

import datetime
import numpy as np
from src.data.database import db_manager
from src.data.storage.tle_repository import TLERepository
from src.propagation.sgp4_engine import sgp4_engine

def test_coordinates():
    """Test coordinate transformation calculations."""
    with db_manager.get_session() as session:
        repo = TLERepository(session)
        tles = repo.get_recent_tles(hours_back=24, limit=3)
        
        print("Testing coordinate transformations:")
        print("=" * 50)
        
        for tle in tles:
            print(f"\nSatellite: {tle.norad_id}")
            print(f"TLE Epoch: {tle.epoch_datetime}")
            
            # Propagate to current time
            now = datetime.datetime.now(datetime.timezone.utc)
            try:
                result = sgp4_engine.propagate_to_epoch(tle, now)
                
                print(f"Position (ECI): X={result.cartesian_state.x/1000:.2f} km, Y={result.cartesian_state.y/1000:.2f} km, Z={result.cartesian_state.z/1000:.2f} km")
                print(f"Velocity (ECI): VX={result.cartesian_state.vx:.2f} m/s, VY={result.cartesian_state.vy:.2f} m/s, VZ={result.cartesian_state.vz:.2f} m/s")
                print(f"Geographic: Lat={result.latitude_deg:.4f}°, Lon={result.longitude_deg:.4f}°, Alt={result.altitude_m/1000:.2f} km")
                
                # Verify the calculations make sense
                r_eci = np.array([result.cartesian_state.x, result.cartesian_state.y, result.cartesian_state.z])
                r_magnitude = np.linalg.norm(r_eci)
                calc_altitude = r_magnitude - sgp4_engine.EARTH_RADIUS
                
                print(f"Altitude verification: Calculated={result.altitude_m/1000:.2f} km, Recalculated={(calc_altitude/1000):.2f} km")
                
                # Check if coordinates are reasonable
                if abs(result.latitude_deg) <= 90 and abs(result.longitude_deg) <= 180:
                    print("✓ Coordinates are within valid ranges")
                else:
                    print("✗ Invalid coordinate ranges detected")
                    
            except Exception as e:
                print(f"Error propagating {tle.norad_id}: {e}")

if __name__ == "__main__":
    test_coordinates()