#!/usr/bin/env python3
"""Populate database with sample TLE data for testing."""

import datetime
from src.data.database import db_manager
from src.data.storage.tle_repository import TLERepository
from src.data.models import TLE

# Sample TLE data for testing (ISS and some other satellites)
sample_tles = [
    # ISS (International Space Station)
    {
        "line1": "1 25544U 98067A   26017.51578704  .00010227  00000+0  18804-3 0  9995",
        "line2": "2 25544  51.6434  24.7995 0004920 313.9023 194.2692 15.49461972470459"
    },
    # Hubble Space Telescope
    {
        "line1": "1 20580U 90037B   26017.49236111  .00002200  00000+0  12345-3 0  9990",
        "line2": "2 20580  28.4693 108.1422 0002236 272.0510  87.9966 15.09207111456789"
    },
    # Some Starlink satellites
    {
        "line1": "1 44239U 19029A   26017.46895833  .00001234  00000+0  23456-3 0  9991",
        "line2": "2 44239  53.0000 123.4567 0012345 345.6789 123.4567 15.12345678 12345"
    },
    {
        "line1": "1 44240U 19029B   26017.46895833  .00001234  00000+0  23456-3 0  9992",
        "line2": "2 44240  53.0000 124.4567 0012345 346.6789 124.4567 15.12345678 12346"
    },
    # Some random satellite
    {
        "line1": "1 12345U 23001A   26017.44562500  .00002345  00000+0  34567-3 0  9993",
        "line2": "2 12345  97.8000 234.5678 0056789 123.4567 234.5678 14.23456789 23456"
    }
]

def populate_sample_tles():
    """Populate database with sample TLE data (only if database is nearly empty)."""
    with db_manager.get_session() as session:
        tle_repo = TLERepository(session)
        
        # Get comprehensive database statistics
        stats = tle_repo.get_statistics()
        total_tles = stats.get('total_tles', 0)
        unique_satellites = stats.get('unique_satellites', 0)
        
        # Only populate sample data if database is essentially empty
        # Allow population if we have very few records (< 10 total or < 5 unique satellites)
        if total_tles >= 10 or unique_satellites >= 5:
            print(f"Database already has {total_tles} TLE records ({unique_satellites} unique satellites).")
            print("Skipping sample data population to preserve real Space-Track data.")
            return
        
        print("Populating database with sample TLE data...")
        
        for i, tle_data in enumerate(sample_tles):
            try:
                # Parse NORAD ID from line 1 (positions 2-6)
                norad_id = int(tle_data["line1"][2:7])
                
                # Create TLE object
                tle = TLE(
                    norad_id=norad_id,
                    classification=tle_data["line1"][7],
                    launch_year=int(tle_data["line1"][9:11]),
                    launch_number=int(tle_data["line1"][11:14]),
                    launch_piece=tle_data["line1"][14:17].strip(),
                    epoch_datetime=datetime.datetime.now(datetime.timezone.utc),
                    mean_motion_derivative=float(tle_data["line1"][33:43].replace(' ', '').replace('+', 'e+').replace('-', 'e-')),
                    mean_motion_sec_derivative=0.0,  # Simplified
                    bstar_drag_term=float(tle_data["line1"][53:61].replace(' ', '').replace('+', 'e+').replace('-', 'e-')),
                    element_set_number=int(tle_data["line1"][64:68]),
                    inclination_degrees=float(tle_data["line2"][8:16]),
                    raan_degrees=float(tle_data["line2"][17:25]),
                    eccentricity=float(f"0.{tle_data['line2'][26:33]}"),
                    argument_of_perigee_degrees=float(tle_data["line2"][34:42]),
                    mean_anomaly_degrees=float(tle_data["line2"][43:51]),
                    mean_motion_orbits_per_day=float(tle_data["line2"][52:63]),
                    revolution_number_at_epoch=int(tle_data["line2"][63:68]),
                    tle_line1=tle_data["line1"],
                    tle_line2=tle_data["line2"],
                    epoch_julian_date=0.0,
                    line1_checksum=int(tle_data["line1"][-1]),
                    line2_checksum=int(tle_data["line2"][-1]),
                    source_url="sample_data",
                    acquisition_timestamp=datetime.datetime.now(datetime.timezone.utc),
                    data_version="1.0",
                    is_valid=True
                )
                
                tle_repo.create(tle)
                print(f"Added TLE for NORAD ID {norad_id}")
                
            except Exception as e:
                print(f"Failed to add TLE {i}: {e}")
        
        print("Sample TLE data population complete!")

if __name__ == "__main__":
    import sys
    
    # Check for force flag
    force_population = "--force" in sys.argv
    
    if force_population:
        print("⚠️  FORCE MODE: Populating sample data regardless of existing records")
        # Temporarily bypass the check
        original_func = populate_sample_tles
        def forced_populate():
            with db_manager.get_session() as session:
                tle_repo = TLERepository(session)
                print("Force-populating database with sample TLE data...")
                for i, tle_data in enumerate(sample_tles):
                    try:
                        # Parse NORAD ID from line 1 (positions 2-6)
                        norad_id = int(tle_data["line1"][2:7])
                        
                        # Create TLE object
                        tle = TLE(
                            norad_id=norad_id,
                            classification=tle_data["line1"][7],
                            launch_year=int(tle_data["line1"][9:11]),
                            launch_number=int(tle_data["line1"][11:14]),
                            launch_piece=tle_data["line1"][14:17].strip(),
                            epoch_datetime=datetime.datetime.now(datetime.timezone.utc),
                            mean_motion_derivative=float(tle_data["line1"][33:43].replace(' ', '').replace('+', 'e+').replace('-', 'e-')),
                            mean_motion_sec_derivative=0.0,  # Simplified
                            bstar_drag_term=float(tle_data["line1"][53:61].replace(' ', '').replace('+', 'e+').replace('-', 'e-')),
                            element_set_number=int(tle_data["line1"][64:68]),
                            inclination_degrees=float(tle_data["line2"][8:16]),
                            raan_degrees=float(tle_data["line2"][17:25]),
                            eccentricity=float(f"0.{tle_data['line2'][26:33]}"),
                            argument_of_perigee_degrees=float(tle_data["line2"][34:42]),
                            mean_anomaly_degrees=float(tle_data["line2"][43:51]),
                            mean_motion_orbits_per_day=float(tle_data["line2"][52:63]),
                            revolution_number_at_epoch=int(tle_data["line2"][63:68]),
                            tle_line1=tle_data["line1"],
                            tle_line2=tle_data["line2"],
                            epoch_julian_date=0.0,
                            line1_checksum=int(tle_data["line1"][-1]),
                            line2_checksum=int(tle_data["line2"][-1]),
                            source_url="sample_data_force",
                            acquisition_timestamp=datetime.datetime.now(datetime.timezone.utc),
                            data_version="1.0",
                            is_valid=True
                        )
                        
                        tle_repo.create(tle)
                        print(f"Added TLE for NORAD ID {norad_id}")
                        
                    except Exception as e:
                        print(f"Failed to add TLE {i}: {e}")
                
                print("Sample TLE data force population complete!")
        
        forced_populate()
    else:
        populate_sample_tles()