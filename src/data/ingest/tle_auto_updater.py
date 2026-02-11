"""
Automated TLE Update Service with 12-hour refresh cycle.

This service runs as a background task and ensures all active satellites
in the database have fresh TLEs by fetching from Space-Track.org every 12 hours.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Set, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.core.config import settings
from src.core.logging import get_logger
from src.data.database import db_manager
from src.data.storage.tle_repository import TLERepository
from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.data.models import TLE

logger = get_logger(__name__)


class TLEAutoUpdater:
    """
    Automated TLE updater that refreshes all tracked satellites every 12 hours.
    
    Features:
    - Discovers all unique NORAD IDs currently in database
    - Fetches latest TLEs from Space-Track.org for active satellites
    - Inserts new TLEs only if they have newer epochs
    - Respects API rate limits (300 requests/hour)
    - Handles errors gracefully with retry logic
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack.spacetrack_rate_limit)
        self.spacetrack_client = SpaceTrackClient(self.rate_limiter)
        self.is_running = False
        self._update_lock = asyncio.Lock()
        
    async def get_all_tracked_norad_ids(self) -> Set[int]:
        """
        Get all unique NORAD IDs from the database.
        
        Returns:
            Set of NORAD IDs currently being tracked
        """
        try:
            with db_manager.get_session() as session:
                repo = TLERepository(session)
                # Query distinct NORAD IDs
                norad_ids = session.query(TLE.norad_id).distinct().all()
                unique_ids = {norad_id[0] for norad_id in norad_ids}
                
                logger.info(f"Found {len(unique_ids)} unique satellites in database")
                return unique_ids
                
        except Exception as e:
            logger.error(f"Failed to get tracked NORAD IDs: {e}")
            return set()
    
    async def fetch_active_satellite_catalog(self) -> List[int]:
        """
        Fetch ALL active (non-decayed) satellites from Space-Track.org.
        This fetches ONLY satellites that have valid, recent TLE data.
        
        Returns:
            List of NORAD IDs for satellites with valid current TLEs
        """
        try:
            async with self.spacetrack_client:
                # Strategy: Fetch TLE data directly and extract NORAD IDs
                # This ensures we ONLY get satellites with actual TLE data
                logger.info("Fetching satellites with valid TLE data from Space-Track...")
                
                all_norad_ids = set()
                chunk_size = 1000
                offset = 0
                max_chunks = 20
                chunk_count = 0
                
                # Use gp class (current recommended) with epoch filter for recent TLEs
                base_url = f"{self.spacetrack_client.BASE_URL}{self.spacetrack_client.QUERY_ENDPOINT}"
                while chunk_count < max_chunks:
                    params = {
                        "class": "gp",
                        "epoch": ">now-7",
                        "orderby": "NORAD_CAT_ID",
                        "limit": chunk_size,
                        "offset": offset,
                        "format": "json"
                    }
                    
                    logger.info(f"Fetching chunk {chunk_count + 1}, offset {offset}...")
                    
                    await self.rate_limiter.acquire()
                    response = await self.spacetrack_client.session.get(base_url, params=params)
                    
                    if response.status_code != 200:
                        logger.error(f"Failed to fetch TLE catalog chunk: {response.status_code}")
                        logger.error(f"Response: {response.text[:500]}")
                        
                        # If first chunk fails, try without offset/pagination
                        if chunk_count == 0:
                            logger.info("Trying simpler query without pagination...")
                            alt_params = {
                                "class": "gp",
                                "epoch": ">now-7",
                                "orderby": "NORAD_CAT_ID",
                                "format": "json"
                            }
                            await self.rate_limiter.acquire()
                            response = await self.spacetrack_client.session.get(base_url, params=alt_params)
                            if response.status_code != 200:
                                logger.error(f"Alternative query also failed: {response.status_code}")
                                break
                        else:
                            break
                    
                    data = response.json()
                    if not isinstance(data, list) or len(data) == 0:
                        logger.info(f"No more data at offset {offset}")
                        break
                    
                    # Extract NORAD IDs - these are guaranteed to have valid TLEs
                    valid_count = 0
                    for item in data:
                        if 'NORAD_CAT_ID' in item:
                            try:
                                norad_id = int(item['NORAD_CAT_ID'])
                                all_norad_ids.add(norad_id)
                                valid_count += 1
                            except (ValueError, KeyError):
                                continue
                    
                    logger.info(f"Chunk {chunk_count + 1}: Found {valid_count} satellites with TLEs (total unique: {len(all_norad_ids)})")
                    
                    # If we got fewer results than chunk_size, we've reached the end
                    if len(data) < chunk_size:
                        logger.info(f"Received {len(data)} results (less than chunk size), catalog complete")
                        break
                    
                    # If no pagination support, break after first large fetch
                    if chunk_count == 0 and len(data) >= 1000:
                        logger.info(f"Single query returned {len(data)} objects, using as complete catalog")
                        break
                    
                    offset += chunk_size
                    chunk_count += 1
                    
                    # Small delay between chunks
                    await asyncio.sleep(2)
                
                norad_ids = sorted(list(all_norad_ids))
                logger.info(f"Found {len(norad_ids)} TOTAL satellites with valid current TLEs")
                
                return norad_ids
                
        except Exception as e:
            logger.error(f"Failed to fetch active catalog: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def update_tle_for_satellite(self, norad_id: int) -> bool:
        """
        Update TLE for a single satellite if a newer one is available.
        
        Args:
            norad_id: Satellite NORAD catalog ID
            
        Returns:
            True if TLE was updated, False otherwise
        """
        try:
            # Get current latest TLE from database
            with db_manager.get_session() as session:
                repo = TLERepository(session)
                existing_tle = repo.get_latest_tle(norad_id)
                # Ensure existing epoch is timezone-aware
                existing_epoch = None
                if existing_tle:
                    if existing_tle.epoch_datetime.tzinfo is None:
                        existing_epoch = existing_tle.epoch_datetime.replace(tzinfo=timezone.utc)
                    else:
                        existing_epoch = existing_tle.epoch_datetime
            
            # Fetch latest TLE from Space-Track
            async with self.spacetrack_client:
                new_tle = await self.spacetrack_client.fetch_tle_by_norad_id(norad_id, days_back=7)
                
                if not new_tle:
                    logger.debug(f"No TLE found for NORAD {norad_id} (likely decayed/inactive)")
                    return False
                
                # Ensure new epoch is timezone-aware
                if new_tle.epoch_datetime.tzinfo is None:
                    new_tle.epoch_datetime = new_tle.epoch_datetime.replace(tzinfo=timezone.utc)
                
                # Check if new TLE is actually newer
                if existing_epoch and new_tle.epoch_datetime <= existing_epoch:
                    logger.debug(f"NORAD {norad_id}: TLE is up to date (epoch: {existing_epoch})")
                    return False
                
                # Insert new TLE
                with db_manager.get_session() as session:
                    repo = TLERepository(session)
                    
                    # Check for duplicate by epoch (Space-Track may return same TLE)
                    duplicate = repo.get_by_norad_and_epoch(norad_id, new_tle.epoch_datetime)
                    if duplicate:
                        logger.debug(f"NORAD {norad_id}: TLE already exists for epoch {new_tle.epoch_datetime}")
                        return False
                    
                    repo.create(new_tle)
                    session.commit()
                    
                    logger.info(f"Updated TLE for NORAD {norad_id}: {existing_epoch} → {new_tle.epoch_datetime}")
                    return True
                    
        except Exception as e:
            # Log error but don't crash - satellite might be decayed or data incomplete
            error_msg = str(e)
            if "Incomplete TLE data" in error_msg or "No TLE found" in error_msg:
                logger.debug(f"NORAD {norad_id}: Skipping - {error_msg}")
            else:
                logger.warning(f"Failed to update TLE for NORAD {norad_id}: {error_msg}")
            return False
    
    async def update_all_tles(self):
        """
        Main update job: Refresh TLEs for all tracked satellites.
        
        Strategy:
        1. Fetch COMPLETE active satellite catalog from Space-Track (all on-orbit objects)
        2. Update ALL satellites with latest TLEs
        3. Insert new satellites not previously in database
        """
        async with self._update_lock:
            start_time = datetime.now(timezone.utc)
            logger.info("=" * 80)
            logger.info("Starting 12-hour TLE auto-update cycle - FULL CATALOG FETCH")
            logger.info("=" * 80)
            
            try:
                # Step 1: Fetch ALL active satellites from Space-Track
                logger.info("Fetching COMPLETE active satellite catalog from Space-Track.org...")
                active_ids = await self.fetch_active_satellite_catalog()
                
                if not active_ids:
                    logger.error("Failed to fetch active satellite catalog - aborting update")
                    return
                
                logger.info(f"Total satellites to update: {len(active_ids)}")
                logger.info(f"This includes ALL active on-orbit objects from Space-Track")
                
                # Step 2: Update TLEs in batches (respect rate limits)
                updated_count = 0
                skipped_count = 0
                error_count = 0
                new_satellites = 0
                
                # Track which satellites existed before
                with db_manager.get_session() as session:
                    repo = TLERepository(session)
                    existing_norads = {row[0] for row in session.query(TLE.norad_id).distinct().all()}
                
                # Process in batches to respect rate limits (300/hour = ~5/minute)
                batch_size = 50
                all_ids_list = list(active_ids)
                
                for i in range(0, len(all_ids_list), batch_size):
                    batch = all_ids_list[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(all_ids_list) + batch_size - 1) // batch_size
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} satellites)")
                    
                    for norad_id in batch:
                        try:
                            # Check if this is a new satellite
                            is_new = norad_id not in existing_norads
                            
                            success = await self.update_tle_for_satellite(norad_id)
                            if success:
                                updated_count += 1
                                if is_new:
                                    new_satellites += 1
                                    existing_norads.add(norad_id)
                            else:
                                skipped_count += 1
                                
                        except asyncio.CancelledError:
                            logger.warning(f"Update cancelled at NORAD {norad_id} (batch {batch_num}/{total_batches})")
                            logger.info(f"Partial results: Updated {updated_count}, New {new_satellites}, Skipped {skipped_count}, Errors {error_count}")
                            raise  # Re-raise to stop gracefully
                        except Exception as e:
                            logger.error(f"Error updating NORAD {norad_id}: {e}")
                            error_count += 1
                        
                        # Small delay to respect rate limits
                        try:
                            await asyncio.sleep(0.5)
                        except asyncio.CancelledError:
                            logger.warning("Update cancelled during sleep, stopping gracefully")
                            raise
                    
                    # Longer delay between batches
                    if i + batch_size < len(all_ids_list):
                        logger.info(f"Batch {batch_num} complete. Pausing before next batch...")
                        try:
                            await asyncio.sleep(5)
                        except asyncio.CancelledError:
                            logger.warning("Update cancelled between batches, stopping gracefully")
                            raise
                
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                logger.info("=" * 80)
                logger.info("TLE auto-update cycle complete")
                logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
                logger.info(f"  Updated: {updated_count}")
                logger.info(f"  New satellites added: {new_satellites}")
                logger.info(f"  Skipped (up-to-date): {skipped_count}")
                logger.info(f"  Errors: {error_count}")
                logger.info(f"  Total processed: {len(all_ids_list)}")
                logger.info(f"  Total unique satellites in DB: {len(existing_norads)}")
                logger.info("=" * 80)
                
            except asyncio.CancelledError:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.warning("=" * 80)
                logger.warning("TLE auto-update cycle CANCELLED (shutdown requested)")
                logger.warning(f"  Duration before cancellation: {duration:.1f} seconds")
                logger.warning(f"  Partial progress: Updated {updated_count}, New {new_satellites}, Skipped {skipped_count}")
                logger.warning("  Next scheduled update will continue from latest TLEs")
                logger.warning("=" * 80)
                # Don't re-raise - allow graceful shutdown
            except Exception as e:
                logger.error(f"TLE auto-update cycle failed: {e}")
    
    def start(self):
        """
        Start the automatic TLE update service.
        
        Schedules updates to run every 12 hours.
        """
        if self.is_running:
            logger.warning("TLE auto-updater already running")
            return
        
        logger.info("Starting TLE auto-updater service")
        logger.info(f"  Update interval: Every 12 hours")
        logger.info(f"  Space-Track account: {settings.spacetrack.spacetrack_username}")
        logger.info(f"  Rate limit: {settings.spacetrack.spacetrack_rate_limit} requests/hour")
        
        # Schedule the update job to run every 12 hours
        self.scheduler.add_job(
            self.update_all_tles,
            trigger=IntervalTrigger(hours=12),
            id='tle_auto_update',
            name='TLE Auto Update (12-hour cycle)',
            replace_existing=True,
            max_instances=1  # Prevent overlapping runs
        )
        
        # Run immediately on startup
        self.scheduler.add_job(
            self.update_all_tles,
            id='tle_initial_update',
            name='TLE Initial Update',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        
        logger.info("TLE auto-updater started successfully")
        logger.info("  Next update: 12 hours from now")
    
    def stop(self):
        """Stop the automatic TLE update service."""
        if not self.is_running:
            return
        
        logger.info("Stopping TLE auto-updater service")
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        logger.info("TLE auto-updater stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the auto-updater.
        
        Returns:
            Dictionary with status information
        """
        jobs = self.scheduler.get_jobs()
        
        return {
            "is_running": self.is_running,
            "scheduled_jobs": len(jobs),
            "next_run": jobs[0].next_run_time.isoformat() if jobs else None,
            "rate_limiter": {
                "requests_per_hour": self.rate_limiter.requests_per_hour,
                "requests_this_hour": self.rate_limiter.requests_this_hour
            }
        }


# Global instance
tle_auto_updater = TLEAutoUpdater()
