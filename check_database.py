#!/usr/bin/env python3
"""Check database contents for duplicates and satellite types."""

from src.data.database import db_manager
from src.data.storage.tle_repository import TLERepository

def check_database():
    with db_manager.get_session() as session:
        repo = TLERepository(session)
        tles = repo.get_recent_tles(hours_back=168, limit=100)
        
        print(f'Total TLE records: {len(tles)}')
        
        # Check for duplicates
        norad_ids = [t.norad_id for t in tles]
        unique_norad_ids = set(norad_ids)
        print(f'Unique NORAD IDs: {len(unique_norad_ids)}')
        
        # Count occurrences
        norad_counts = {}
        for tle in tles:
            norad_counts[tle.norad_id] = norad_counts.get(tle.norad_id, 0) + 1
        
        print('\nNORAD ID counts:')
        for norad_id, count in sorted(norad_counts.items()):
            print(f'  {norad_id}: {count} entries')
        
        # Check satellite types based on NORAD ranges
        print('\nSatellite type classification:')
        payload_count = 0
        rocket_body_count = 0
        debris_count = 0
        
        for norad_id in unique_norad_ids:
            if norad_id < 40000:
                payload_count += 1
            elif 40000 <= norad_id < 50000:
                rocket_body_count += 1
            else:
                debris_count += 1
        
        print(f'  PAYLOAD (<40000): {payload_count}')
        print(f'  ROCKET BODY (40000-49999): {rocket_body_count}')
        print(f'  DEBRIS (>=50000): {debris_count}')

if __name__ == "__main__":
    check_database()