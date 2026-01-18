"""Download and archive historical TLEs for known conjunction events."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
from pathlib import Path

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Directory structure for historical data
HISTORICAL_DATA_DIR = Path("data/historical")
HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


async def download_cosmos_iridium_tles() -> Dict:
    """
    Download TLEs for Cosmos-Iridium collision event.
    
    Returns JSON structure:
    {
        "event": "Cosmos 2251 / Iridium 33 Collision",
        "date": "2009-02-10T16:56:00Z",
        "primary": {
            "norad_id": 22675,
            "name": "COSMOS 2251",
            "tles": [...],  # List of TLEs from 2009-02-03 to 2009-02-10
            "tle_sources": [...]
        },
        "secondary": {
            "norad_id": 24946,
            "name": "IRIDIUM 33",
            "tles": [...]
        },
        "expected_outcome": {
            "collision_occurred": true,
            "should_flag_high_pc": true,
            "minimum_warning_hours": 24
        }
    }
    """
    logger.info("Downloading Cosmos-Iridium collision TLEs")
    
    # Event details
    collision_date = datetime(2009, 2, 10, 16, 56, 0)
    start_date = collision_date - timedelta(days=7)  # 1 week before
    end_date = collision_date + timedelta(days=1)    # 1 day after
    
    # Initialize Space-Track client
    rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack_rate_limit)
    client = SpaceTrackClient(rate_limiter)
    
    try:
        async with client:
            # Download TLEs for both objects
            cosmos_tles = await client.download_tles_for_satellite(
                norad_id=22675,
                start_date=start_date,
                end_date=end_date
            )
            
            iridium_tles = await client.download_tles_for_satellite(
                norad_id=24946,
                start_date=start_date,
                end_date=end_date
            )
            
            # Structure the data
            historical_data = {
                "event": "Cosmos 2251 / Iridium 33 Collision",
                "date": collision_date.isoformat() + "Z",
                "primary": {
                    "norad_id": 22675,
                    "name": "COSMOS 2251",
                    "tles": cosmos_tles,
                    "tle_count": len(cosmos_tles),
                    "first_tle_date": cosmos_tles[0]["EPOCH"] if cosmos_tles else None,
                    "last_tle_date": cosmos_tles[-1]["EPOCH"] if cosmos_tles else None
                },
                "secondary": {
                    "norad_id": 24946,
                    "name": "IRIDIUM 33",
                    "tles": iridium_tles,
                    "tle_count": len(iridium_tles),
                    "first_tle_date": iridium_tles[0]["EPOCH"] if iridium_tles else None,
                    "last_tle_date": iridium_tles[-1]["EPOCH"] if iridium_tles else None
                },
                "expected_outcome": {
                    "collision_occurred": True,
                    "should_flag_high_pc": True,
                    "minimum_warning_hours": 24,
                    "actual_miss_distance_meters": 0,  # They collided
                    "debris_created": 2000  # Estimated fragments
                },
                "download_metadata": {
                    "download_date": datetime.utcnow().isoformat() + "Z",
                    "source": "space-track.org",
                    "date_range": {
                        "start": start_date.isoformat() + "Z",
                        "end": end_date.isoformat() + "Z"
                    }
                }
            }
            
            # Save to file
            filename = HISTORICAL_DATA_DIR / "cosmos_iridium_2009.json"
            with open(filename, 'w') as f:
                json.dump(historical_data, f, indent=2)
            
            logger.info(f"Cosmos-Iridium TLEs saved to {filename}")
            logger.info(f"  COSMOS 2251: {len(cosmos_tles)} TLEs")
            logger.info(f"  IRIDIUM 33: {len(iridium_tles)} TLEs")
            
            return historical_data
            
    except Exception as e:
        logger.error(f"Failed to download Cosmos-Iridium TLEs: {e}")
        raise


async def download_iss_debris_avoidance_2020() -> Dict:
    """
    Download TLEs for ISS debris avoidance maneuver from 2020-09-22.
    
    This documented maneuver was performed to avoid Cosmos 2012 debris.
    """
    logger.info("Downloading ISS 2020 debris avoidance TLEs")
    
    # Maneuver date (approximate)
    maneuver_date = datetime(2020, 9, 22, 12, 0, 0)
    start_date = maneuver_date - timedelta(days=3)
    end_date = maneuver_date + timedelta(days=3)
    
    rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack_rate_limit)
    client = SpaceTrackClient(rate_limiter)
    
    try:
        async with client:
            # Download ISS TLEs
            iss_tles = await client.download_tles_for_satellite(
                norad_id=25544,  # ISS
                start_date=start_date,
                end_date=end_date
            )
            
            # Look for potential conjunction partners in the same timeframe
            # This would require querying the catalog for objects in similar orbits
            # For now, we'll just document the ISS data
            
            historical_data = {
                "event": "ISS Debris Avoidance Maneuver 2020",
                "date": maneuver_date.isoformat() + "Z",
                "primary": {
                    "norad_id": 25544,
                    "name": "ISS (ZARYA)",
                    "tles": iss_tles,
                    "tle_count": len(iss_tles)
                },
                "secondary": {
                    "norad_id": None,  # Specific debris object unknown
                    "name": "Cosmos 2012 Debris",
                    "tles": [],  # Would need specific debris tracking data
                    "tle_count": 0
                },
                "expected_outcome": {
                    "maneuver_performed": True,
                    "avoidance_successful": True,
                    "delta_v_mps": 1.2,  # Typical ISS reboost ΔV
                    "warning_time_hours": 48
                },
                "download_metadata": {
                    "download_date": datetime.utcnow().isoformat() + "Z",
                    "source": "space-track.org"
                }
            }
            
            filename = HISTORICAL_DATA_DIR / "iss_maneuver_2020.json"
            with open(filename, 'w') as f:
                json.dump(historical_data, f, indent=2)
            
            logger.info(f"ISS 2020 maneuver TLEs saved to {filename}")
            return historical_data
            
    except Exception as e:
        logger.error(f"Failed to download ISS 2020 TLEs: {e}")
        raise


async def download_fengyun_asat_2007() -> Dict:
    """
    Download TLEs for Fengyun-1C ASAT test debris cloud.
    
    China's 2007 ASAT test created thousands of trackable debris fragments.
    """
    logger.info("Downloading Fengyun-1C ASAT test TLEs")
    
    # ASAT test date
    asat_date = datetime(2007, 1, 11, 16, 0, 0)
    start_date = asat_date - timedelta(days=30)  # Month before (parent object)
    end_date = asat_date + timedelta(days=30)    # Month after (debris)
    
    rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack_rate_limit)
    client = SpaceTrackClient(rate_limiter)
    
    try:
        async with client:
            # Parent object TLEs
            parent_tles = await client.download_tles_for_satellite(
                norad_id=25730,  # Fengyun-1C
                start_date=start_date,
                end_date=asat_date  # Only before destruction
            )
            
            # Some major debris fragments (example NORAD IDs - would need actual tracking)
            debris_ids = [30000, 30001, 30002]  # Placeholder - real IDs needed
            debris_data = []
            
            for debris_id in debris_ids[:1]:  # Limit for testing
                try:
                    debris_tles = await client.download_tles_for_satellite(
                        norad_id=debris_id,
                        start_date=asat_date,
                        end_date=end_date
                    )
                    debris_data.append({
                        "norad_id": debris_id,
                        "tles": debris_tles,
                        "tle_count": len(debris_tles)
                    })
                except Exception as e:
                    logger.warning(f"Failed to get TLEs for debris {debris_id}: {e}")
            
            historical_data = {
                "event": "Fengyun-1C ASAT Test 2007",
                "date": asat_date.isoformat() + "Z",
                "primary": {
                    "norad_id": 25730,
                    "name": "FENGYUN 1C",
                    "tles": parent_tles,
                    "tle_count": len(parent_tles)
                },
                "secondary": {
                    "norad_id": "debris_cloud",
                    "name": "ASAT Debris Fragments",
                    "fragments": debris_data,
                    "total_tracked": len(debris_data)
                },
                "expected_outcome": {
                    "destruction_occurred": True,
                    "fragments_created": 3000,  # Estimated
                    "long_term_debris": True,
                    "tracking_start_delay_hours": 24
                },
                "download_metadata": {
                    "download_date": datetime.utcnow().isoformat() + "Z",
                    "source": "space-track.org"
                }
            }
            
            filename = HISTORICAL_DATA_DIR / "fengyun_asat_2007.json"
            with open(filename, 'w') as f:
                json.dump(historical_data, f, indent=2)
            
            logger.info(f"Fengyun ASAT TLEs saved to {filename}")
            return historical_data
            
    except Exception as e:
        logger.error(f"Failed to download Fengyun ASAT TLEs: {e}")
        raise


async def download_all_historical_events():
    """Download all historical conjunction events."""
    logger.info("Starting historical TLE archive download")
    
    try:
        # Download all events concurrently
        tasks = [
            download_cosmos_iridium_tles(),
            download_iss_debris_avoidance_2020(),
            download_fengyun_asat_2007()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_downloads = 0
        failed_downloads = 0
        
        for i, result in enumerate(results):
            event_names = ["Cosmos-Iridium", "ISS 2020", "Fengyun ASAT"]
            
            if isinstance(result, Exception):
                logger.error(f"{event_names[i]} download failed: {result}")
                failed_downloads += 1
            else:
                logger.info(f"{event_names[i]} download completed successfully")
                successful_downloads += 1
        
        logger.info(f"Historical archive download complete:")
        logger.info(f"  Successful: {successful_downloads}")
        logger.info(f"  Failed: {failed_downloads}")
        logger.info(f"  Data stored in: {HISTORICAL_DATA_DIR}")
        
        return {
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "total_events": len(tasks),
            "data_directory": str(HISTORICAL_DATA_DIR)
        }
        
    except Exception as e:
        logger.error(f"Historical archive download failed: {e}")
        raise


def validate_tle_checksums(tle_lines: List[str]) -> bool:
    """
    Validate TLE checksums according to standard algorithm.
    
    TLE checksum is modulo 10 sum of all digits plus 1 for each letter.
    Last character of each line should match computed checksum.
    """
    def compute_checksum(line: str) -> int:
        if len(line) < 69:
            return -1
            
        checksum = 0
        for char in line[:68]:  # First 68 characters
            if char.isdigit():
                checksum += int(char)
            elif char.isalpha():
                checksum += 1
            # Spaces and other characters contribute 0
        
        return checksum % 10
    
    for i in range(0, len(tle_lines), 2):
        if i + 1 >= len(tle_lines):
            continue
            
        line1 = tle_lines[i]
        line2 = tle_lines[i + 1]
        
        # Check lengths
        if len(line1) < 69 or len(line2) < 69:
            return False
            
        # Check checksums
        expected_checksum1 = int(line1[68]) if line1[68].isdigit() else -1
        expected_checksum2 = int(line2[68]) if line2[68].isdigit() else -1
        
        actual_checksum1 = compute_checksum(line1)
        actual_checksum2 = compute_checksum(line2)
        
        if actual_checksum1 != expected_checksum1 or actual_checksum2 != expected_checksum2:
            return False
    
    return True


if __name__ == "__main__":
    # Run the downloader
    asyncio.run(download_all_historical_events())