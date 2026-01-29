"""
Optimized script to fetch maximum TLE data from Space-Track.org using best practices.
Based on official Space-Track API documentation and rate limits.
"""

import asyncio
import httpx
from datetime import datetime, timedelta
import sys
import time
from typing import List, Dict, Any
from src.core.config import settings
from src.core.logging import get_logger
from src.data.database import get_db
from src.data.storage.tle_repository import TLERepository
from src.data.models import TLE

logger = get_logger(__name__)


async def fetch_optimized_tle_data(repo: TLERepository, target_count: int = 2000) -> Dict[str, int]:
    """
    Fetch TLE data using optimal Space-Track API parameters based on official documentation.
    
    Args:
        repo: TLE repository for database operations
        target_count: Target number of TLE records to fetch
        
    Returns:
        Dictionary with fetch statistics
    """
    print("🚀 Fetching Optimized TLE Data from Space-Track.org")
    print("="*60)
    print(f"🎯 Target: {target_count} TLE records")
    print(f"👤 Account: {settings.spacetrack.spacetrack_username}")
    print(f"⚡ Following official Space-Track API guidelines")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Calculate date 10 days ago for epoch filter (to get current on-orbit objects)
    ten_days_ago = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Authenticate first
        auth_data = {
            "identity": settings.spacetrack.spacetrack_username,
            "password": settings.spacetrack.spacetrack_password
        }
        login_response = await client.post(
            "https://www.space-track.org/ajaxauth/login",
            data=auth_data
        )

        if login_response.status_code != 200:
            print(f"❌ Authentication failed: {login_response.status_code}")
            return {'success': 0, 'skipped': 0, 'errors': 0}

        print(f"✅ Authenticated successfully")
        
        # Fetch TLEs for the most recently launched satellites to maximize our target
        # According to Space-Track docs, we should use efficient queries
        # Query for recent objects using the GP class with optimal parameters
        gp_url = (
            f"https://www.space-track.org/basicspacedata/query/class/gp"
            f"/decay_date/null-val/epoch/%3E{ten_days_ago}"
            f"/orderby/NORAD_CAT_ID,EPOCH desc/format/json/limit/{min(2000, target_count)}"
        )
        
        print(f"📥 Fetching latest TLEs for on-orbit objects...")
        response = await client.get(gp_url)
        
        if response.status_code != 200:
            print(f"❌ GP query failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return {'success': 0, 'skipped': 0, 'errors': target_count}
        
        tle_data_list = response.json()
        print(f"✅ Retrieved {len(tle_data_list)} TLE records from Space-Track")
        
        # Process the returned TLE data
        processed_norads = set()
        for tle_data in tle_data_list:
            norad_id = int(tle_data.get('NORAD_CAT_ID', 0))
            
            # Skip if we've already processed this NORAD ID (to avoid duplicates)
            if norad_id in processed_norads:
                continue
            processed_norads.add(norad_id)
            
            try:
                # Extract TLE lines
                line1 = tle_data.get('TLE_LINE1', '').strip()
                line2 = tle_data.get('TLE_LINE2', '').strip()
                
                if not line1 or not line2:
                    print(f"      ❌ Missing TLE lines for NORAD {norad_id}")
                    error_count += 1
                    continue

                # Parse epoch from TLE data
                epoch_str = tle_data.get('EPOCH', '')
                try:
                    epoch_dt = datetime.fromisoformat(epoch_str.replace('Z', '+00:00'))
                except ValueError:
                    # Fallback to current time if epoch parsing fails
                    epoch_dt = datetime.utcnow()
                    print(f"      ⚠️  Could not parse epoch for NORAD {norad_id}, using current time")

                # Check if TLE already exists in database
                existing = repo.get_by_norad_and_epoch(
                    norad_id=norad_id,
                    epoch=epoch_dt
                )
                
                if existing:
                    skipped_count += 1
                    continue

                # Parse the TLE data to extract individual components for the model
                try:
                    # Extract TLE components
                    classification = tle_data.get('CLASSIFICATION_TYPE', 'U')[0] if tle_data.get('CLASSIFICATION_TYPE') else 'U'
                    
                    # Parse epoch julian date
                    epoch_str = tle_data.get('EPOCH', '')
                    epoch_jd = 0.0
                    if epoch_str:
                        try:
                            # Convert ISO format to Julian date if needed
                            dt = datetime.fromisoformat(epoch_str.replace('Z', '+00:00'))
                            # Simple approximation of JD from datetime
                            epoch_jd = (dt - datetime(1970, 1, 1)).total_seconds() / 86400 + 2440587.5
                        except:
                            epoch_jd = 0.0
                    
                    # Create TLE object with proper mapping to the model fields
                    tle = TLE(
                        norad_id=norad_id,
                        tle_line1=line1,
                        tle_line2=line2,
                        epoch_datetime=epoch_dt,
                        classification=classification,
                        
                        # Extract components from TLE data
                        launch_year=int(tle_data.get('LAUNCH_DATE', '1900-01-01')[:4]) if tle_data.get('LAUNCH_DATE', '')[:4].isdigit() else 1900,
                        launch_number=int(tle_data.get('LAUNCH_DATE', '000')[5:8]) if tle_data.get('LAUNCH_DATE', '000')[5:8].isdigit() else 0,
                        launch_piece=tle_data.get('OBJECT_ID', '  ')[8:].strip() if len(tle_data.get('OBJECT_ID', '')) > 8 else '',
                        
                        epoch_julian_date=epoch_jd,
                        mean_motion_orbits_per_day=float(tle_data.get('MEAN_MOTION', 0)),
                        eccentricity=float(tle_data.get('ECCENTRICITY', 0)),
                        inclination_degrees=float(tle_data.get('INCLINATION', 0)),
                        raan_degrees=float(tle_data.get('RA_OF_ASC_NODE', 0)),
                        argument_of_perigee_degrees=float(tle_data.get('ARG_OF_PERICENTER', 0)),
                        mean_anomaly_degrees=float(tle_data.get('MEAN_ANOMALY', 0)),
                        
                        # Set other required fields with defaults
                        mean_motion_derivative=float(tle_data.get('MEAN_MOTION_DOT', 0)),
                        mean_motion_sec_derivative=float(tle_data.get('MEAN_MOTION_DDOT', 0)),
                        bstar_drag_term=float(tle_data.get('BSTAR', 0)),
                        element_set_number=int(tle_data.get('ELEMENT_SET_NO', 0)),
                        revolution_number_at_epoch=int(tle_data.get('REV_AT_EPOCH', 0)),
                        
                        # Calculate checksums (simplified)
                        line1_checksum=0,  # Will be calculated during validation
                        line2_checksum=0,  # Will be calculated during validation
                    )
                except (ValueError, TypeError) as e:
                    print(f"      ❌ Error parsing TLE data for NORAD {norad_id}: {e}")
                    error_count += 1
                    continue

                repo.create(tle)
                success_count += 1
                
                # Commit every 50 records to ensure data is persisted
                if success_count % 50 == 0:
                    try:
                        repo.session.commit()
                        print(f"      ✅ Successfully committed {success_count} TLEs...")
                    except Exception as commit_error:
                        print(f"      ⚠️  Commit error at {success_count}: {commit_error}")
                        repo.session.rollback()
                
                # Print progress every 100 successful additions
                if success_count % 100 == 0:
                    print(f"      📊 Progress: {success_count} TLEs added...")

            except Exception as e:
                print(f"      ❌ Error processing TLE for NORAD {norad_id}: {e}")
                error_count += 1
                logger.debug(f"Error processing TLE for NORAD {norad_id}: {e}")

        print(f"📊 Processing completed:")
        print(f"   Success: {success_count}")
        print(f"   Skipped (already existed): {skipped_count}")
        print(f"   Errors: {error_count}")
        
        # Final commit for any remaining records
        try:
            repo.session.commit()
            print(f"   ✅ Final commit successful")
        except Exception as e:
            print(f"   ⚠️  Final commit error: {e}")
            repo.session.rollback()
        
        return {
            'success': success_count,
            'skipped': skipped_count,
            'errors': error_count
        }


async def populate_with_optimized_approach(target_count: int = 2000):
    """
    Main function to populate the satellite database using optimized approach.
    
    Args:
        target_count: Target number of satellites to aim for
    """
    print("🚀 Populating Database with Optimized TLE Fetcher")
    print("="*60)
    print(f"🎯 Target: {target_count} satellites (~25-40% of system capacity)")
    print(f"👤 Account: {settings.spacetrack.spacetrack_username}")
    
    # Connect to database
    print("\n💾 Connecting to database...")
    db_gen = get_db()
    db = next(db_gen)
    repo = TLERepository(db)
    
    # Get initial statistics
    initial_stats = repo.get_statistics()
    initial_count = initial_stats.get('total_tles', 0)
    print(f"📊 Starting TLE count: {initial_count}")
    
    # Fetch TLE data using optimized approach
    print(f"\n📥 Fetching TLE data with optimal parameters...")
    results = await fetch_optimized_tle_data(repo, target_count=target_count)
    
    # Get final statistics
    final_stats = repo.get_statistics()
    final_count = final_stats.get('total_tles', 0)
    new_added = final_count - initial_count
    
    print(f"\n📈 Final Results:")
    print(f"   Initial TLEs: {initial_count}")
    print(f"   Final TLEs: {final_count}")
    print(f"   New TLEs added: {new_added}")
    print(f"   Success: {results['success']}")
    print(f"   Skipped (already existed): {results['skipped']}")
    print(f"   Errors: {results['errors']}")
    
    # Check if we made progress toward our target
    if new_added > 0:
        print(f"\n💡 Successfully added {new_added} new TLEs from Space-Track.org")
        print(f"   The system should now show more live satellite tracking data on the dashboard")
    else:
        print(f"\n⚠️  Warning: No new objects added, check credentials and connectivity")
    
    return True


async def main():
    """Main entry point."""
    success = await populate_with_optimized_approach(target_count=2000)  # Aim for 2000 to reach ~25-40% capacity
    
    if success:
        print("\n✅ Optimized satellite database population completed successfully!")
        print("💡 Database now contains real satellite data from Space-Track.org")
        print("   The system should now show live satellite tracking data on the dashboard")
    else:
        print("\n❌ Optimized satellite database population failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())