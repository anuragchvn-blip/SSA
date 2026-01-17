#!/usr/bin/env python3
"""Check database contents."""

from src.data.database import db_manager
from src.data.storage.tle_repository import TLERepository

with db_manager.get_session() as session:
    repo = TLERepository(session)
    tles = repo.get_recent_tles(hours_back=168, limit=10)
    print(f'Found {len(tles)} TLE records')
    for t in tles[:5]:
        print(f'{t.norad_id}: {t.tle_line1[:50]}...')