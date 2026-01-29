#!/usr/bin/env python3
"""Generate real conjunction events from existing TLE data."""

from src.data.database import db_manager
from src.data.storage.tle_repository import TLERepository
from src.data.storage.conjunction_repository import ConjunctionEventRepository
from src.conjunction.screening import conjunction_screener

def generate_conjunctions():
    """Generate conjunction events from TLE catalog."""
    print("Generating conjunction events from TLE catalog...")
    
    with db_manager.get_session() as session:
        tle_repo = TLERepository(session)
        conj_repo = ConjunctionEventRepository(session)
        
        # Get TLE statistics
        stats = tle_repo.get_statistics()
        total_tles = stats.get('total_tles', 0)
        print(f"Found {total_tles} TLEs in database")
        
        if total_tles < 2:
            print("Not enough TLEs to generate conjunctions")
            return
        
        # Get recent TLEs for analysis
        all_tles = tle_repo.get_recent_tles(hours_back=72, limit=100)
        print(f"Analyzing {len(all_tles)} recent TLEs")
        
        if len(all_tles) < 2:
            print("Not enough recent TLEs")
            return
        
        # Use first TLE as primary, rest as catalog
        primary_tle = all_tles[0]
        catalog_tles = all_tles[1:]
        
        print(f"Primary: NORAD {primary_tle.norad_id}")
        print(f"Screening against {len(catalog_tles)} catalog objects")
        
        # Screen for conjunctions
        candidates = conjunction_screener.screen_catalog(
            primary_tle=primary_tle,
            catalog_tles=catalog_tles,
            screening_threshold_km=50.0,  # Larger threshold for demo
            time_window_hours=48.0
        )
        
        print(f"Found {len(candidates)} candidates")
        
        if not candidates:
            print("No candidates found, trying multiple primaries...")
            # Try multiple primaries
            events_created = 0
            for i, primary in enumerate(all_tles[:10]):
                catalog = [tle for tle in all_tles if tle.norad_id != primary.norad_id]
                cand = conjunction_screener.screen_catalog(
                    primary_tle=primary,
                    catalog_tles=catalog[:20],
                    screening_threshold_km=100.0,
                    time_window_hours=72.0
                )
                
                if cand:
                    print(f"Primary {primary.norad_id}: {len(cand)} candidates")
                    refined = conjunction_screener.refine_candidates(cand[:5])
                    events = conjunction_screener.create_conjunction_events(
                        primary_norad_id=primary.norad_id,
                        refined_results=refined,
                        probability_threshold=1e-10
                    )
                    
                    for event in events:
                        conj_repo.create(event)
                        events_created += 1
                    
                    session.commit()
                    
                    if events_created >= 5:
                        break
            
            print(f"Created {events_created} conjunction events")
            return
        
        # Refine candidates
        refined_results = conjunction_screener.refine_candidates(candidates[:10])
        print(f"Refined {len(refined_results)} results")
        
        # Create conjunction events
        events = conjunction_screener.create_conjunction_events(
            primary_norad_id=primary_tle.norad_id,
            refined_results=refined_results,
            probability_threshold=1e-10
        )
        
        print(f"Creating {len(events)} conjunction events...")
        
        for event in events:
            conj_repo.create(event)
        
        session.commit()
        print(f"Successfully created {len(events)} conjunction events")

if __name__ == "__main__":
    generate_conjunctions()
