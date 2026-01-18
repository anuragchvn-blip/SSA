"""
TLE Update Pipeline for Space-Track Integration

Production-grade pipeline for regularly updating TLE data from Space-Track.
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.data.storage.tle_repository import TLERepository
from src.core.config import settings
from src.core.logging import get_logger
from src.data.models import TLE


logger = get_logger(__name__)


class UpdateStatus(Enum):
    """Status of TLE update operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class UpdateResult:
    """Result of a TLE update operation."""
    status: UpdateStatus
    updated_count: int
    error_count: int
    skipped_count: int
    errors: List[str]
    duration_seconds: float
    timestamp: datetime


class TLEUpdatePipeline:
    """Production-grade TLE update pipeline with error handling and monitoring."""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session
        self.rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack.spacetrack_rate_limit)
        self.spacetrack_client = SpaceTrackClient(self.rate_limiter)
        self.tle_repository = TLERepository(db_session) if db_session else None
        self.logger = logger
    
    async def update_latest_tles(self, 
                                satellite_filter: Optional[List[int]] = None,
                                days_back: int = 1,
                                batch_size: int = 100) -> UpdateResult:
        """
        Update latest TLEs for active satellites.
        
        Args:
            satellite_filter: Specific NORAD IDs to update (None for all)
            days_back: How many days of history to consider
            batch_size: Number of requests per batch to respect rate limits
            
        Returns:
            Update result with statistics
        """
        start_time = datetime.utcnow()
        errors = []
        updated_count = 0
        error_count = 0
        skipped_count = 0
        
        try:
            self.logger.info("Starting TLE update pipeline", 
                           satellite_filter=satellite_filter, 
                           days_back=days_back)
            
            async with self.spacetrack_client:
                # Determine which satellites to update
                if satellite_filter:
                    norad_ids = satellite_filter
                else:
                    # For production, this would come from a list of tracked satellites
                    # For now, we'll use a common set of active satellites
                    norad_ids = self._get_common_satellite_ids()
                
                self.logger.info(f"Updating TLEs for {len(norad_ids)} satellites")
                
                # Process in batches to respect rate limits
                for i in range(0, len(norad_ids), batch_size):
                    batch = norad_ids[i:i + batch_size]
                    
                    self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(norad_ids)-1)//batch_size + 1}")
                    
                    for norad_id in batch:
                        try:
                            # Fetch latest TLE for this satellite
                            tle = await self.spacetrack_client.fetch_tle_by_norad_id(
                                norad_id=norad_id, 
                                days_back=days_back
                            )
                            
                            if tle:
                                # Store in database if repository is available
                                if self.tle_repository:
                                    # Check if this TLE is newer than what we have
                                    existing_tle = self.tle_repository.get_latest_by_norad_id(norad_id)
                                    
                                    if not existing_tle or tle.epoch_datetime > existing_tle.epoch_datetime:
                                        self.tle_repository.update_or_create(tle)
                                        updated_count += 1
                                        self.logger.debug(f"Updated TLE for NORAD {norad_id}")
                                    else:
                                        skipped_count += 1
                                        self.logger.debug(f"Skipped older TLE for NORAD {norad_id}")
                                else:
                                    # If no DB session, just count it as updated
                                    updated_count += 1
                                    self.logger.debug(f"Fetched TLE for NORAD {norad_id} (no DB storage)")
                            else:
                                skipped_count += 1
                                self.logger.debug(f"No TLE available for NORAD {norad_id}")
                                
                        except Exception as e:
                            error_count += 1
                            error_msg = f"Error updating NORAD {norad_id}: {str(e)}"
                            errors.append(error_msg)
                            self.logger.error(error_msg, norad_id=norad_id, error=str(e))
                            
                        # Small delay between requests to be respectful
                        await asyncio.sleep(0.1)
            
            # Determine status based on results
            if error_count == 0 and skipped_count == 0:
                status = UpdateStatus.SUCCESS
            elif error_count == 0:
                status = UpdateStatus.PARTIAL_SUCCESS
            elif error_count > 0 and updated_count > 0:
                status = UpdateStatus.PARTIAL_SUCCESS
            else:
                status = UpdateStatus.FAILED
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            result = UpdateResult(
                status=status,
                updated_count=updated_count,
                error_count=error_count,
                skipped_count=skipped_count,
                errors=errors,
                duration_seconds=duration,
                timestamp=start_time
            )
            
            self.logger.info(
                "TLE update pipeline completed",
                status=result.status.value,
                updated_count=result.updated_count,
                error_count=result.error_count,
                skipped_count=result.skipped_count,
                duration_seconds=round(result.duration_seconds, 2)
            )
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"TLE update pipeline failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg, error=str(e))
            
            result = UpdateResult(
                status=UpdateStatus.FAILED,
                updated_count=0,
                error_count=1,
                skipped_count=0,
                errors=errors,
                duration_seconds=duration,
                timestamp=start_time
            )
            
            return result
    
    async def update_catalog_tles(self, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 object_types: Optional[List[str]] = None) -> UpdateResult:
        """
        Update TLEs for a catalog of satellites over a date range.
        
        Args:
            start_date: Start of date range for TLEs
            end_date: End of date range for TLEs
            object_types: Types of objects to include (DEBRIS, PAY, ROCKET, etc.)
            
        Returns:
            Update result with statistics
        """
        start_time = datetime.utcnow()
        errors = []
        updated_count = 0
        error_count = 0
        
        try:
            self.logger.info("Starting catalog TLE update",
                           start_date=start_date.isoformat(),
                           end_date=end_date.isoformat(),
                           object_types=object_types)
            
            async with self.spacetrack_client:
                # Fetch catalog TLEs
                tles = await self.spacetrack_client.fetch_tle_catalog(
                    epoch_start=start_date,
                    epoch_end=end_date,
                    object_class=object_types[0] if object_types else None,  # Limit to one type for now
                    include_debris=True
                )
                
                self.logger.info(f"Fetched {len(tles)} catalog TLEs from Space-Track")
                
                # Store TLEs in database
                if self.tle_repository:
                    for tle in tles:
                        try:
                            # Check if TLE already exists (based on epoch and NORAD ID)
                            existing = self.tle_repository.get_by_norad_and_epoch(
                                norad_id=tle.norad_id,
                                epoch=tle.epoch_datetime
                            )
                            
                            if not existing:
                                self.tle_repository.create(tle)
                                updated_count += 1
                            else:
                                # TLE already exists, skip
                                pass
                                
                        except Exception as e:
                            error_count += 1
                            error_msg = f"Error storing TLE for NORAD {tle.norad_id}: {str(e)}"
                            errors.append(error_msg)
                            self.logger.error(error_msg, norad_id=tle.norad_id, error=str(e))
                else:
                    # Count as updated if no DB
                    updated_count = len(tles)
                
                status = UpdateStatus.PARTIAL_SUCCESS if errors else UpdateStatus.SUCCESS
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                result = UpdateResult(
                    status=status,
                    updated_count=updated_count,
                    error_count=error_count,
                    skipped_count=0,  # For catalog updates, we don't really skip
                    errors=errors,
                    duration_seconds=duration,
                    timestamp=start_time
                )
                
                self.logger.info(
                    "Catalog TLE update completed",
                    status=result.status.value,
                    updated_count=result.updated_count,
                    error_count=result.error_count,
                    duration_seconds=round(result.duration_seconds, 2)
                )
                
                return result
                
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"Catalog TLE update failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg, error=str(e))
            
            result = UpdateResult(
                status=UpdateStatus.FAILED,
                updated_count=0,
                error_count=1,
                skipped_count=0,
                errors=errors,
                duration_seconds=duration,
                timestamp=start_time
            )
            
            return result
    
    async def monitor_stale_tles(self, stale_threshold_hours: int = 24) -> Dict[str, Any]:
        """
        Monitor and report on stale TLEs that need updating.
        
        Args:
            stale_threshold_hours: Age threshold in hours for considering TLEs stale
            
        Returns:
            Dictionary with stale TLE statistics
        """
        if not self.tle_repository:
            self.logger.warning("No database repository available for stale TLE monitoring")
            return {}
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=stale_threshold_hours)
            stale_stats = self.tle_repository.get_stale_tle_statistics(cutoff_time)
            
            self.logger.info(
                "Stale TLE monitoring completed",
                stale_count=stale_stats.get('stale_count', 0),
                total_count=stale_stats.get('total_count', 0),
                stale_threshold_hours=stale_threshold_hours
            )
            
            return stale_stats
            
        except Exception as e:
            self.logger.error(f"Stale TLE monitoring failed: {str(e)}")
            return {}
    
    def _get_common_satellite_ids(self) -> List[int]:
        """
        Get commonly tracked satellite IDs for regular updates.
        
        In production, this would come from a configuration or database of tracked objects.
        """
        # Commonly tracked satellites (ISS, Starlink, etc.)
        return [
            25544,  # ISS
            42982,  # Starlink-1087
            43017,  # Starlink-1112
            43129,  # Starlink-1174
            43217,  # Starlink-1216
            39084,  # GOES-16
            39436,  # GOES-17
            40053,  # GOES-15
            20409,  # GPS BIIR-1 (PRN 25)
            20624,  # GPS BIIR-2 (PRN 32)
            23450,  # Iridium 65
            24277,  # Iridium 78
            24600,  # Iridium 64
            24601,  # Iridium 63
            25396,  # Iridium 77
        ]
    
    async def run_health_check(self) -> Dict[str, Any]:
        """
        Run health check on the TLE update pipeline.
        
        Returns:
            Health check results
        """
        try:
            # Test Space-Track connection
            connection_ok = await self.spacetrack_client.test_connection()
            
            # Check rate limiter status
            rate_limiter_status = {
                "requests_per_hour": self.rate_limiter.requests_per_hour,
                "requests_this_hour": self.rate_limiter.requests_this_hour,
                "hour_start": self.rate_limiter.hour_start.isoformat()
            }
            
            # Check database connectivity if available
            db_status = None
            if self.tle_repository:
                try:
                    db_stats = self.tle_repository.get_statistics()
                    db_status = {
                        "connected": True,
                        "total_tles": db_stats.get('total_tles', 0),
                        "last_updated": db_stats.get('last_updated')
                    }
                except Exception as e:
                    db_status = {
                        "connected": False,
                        "error": str(e)
                    }
            
            health_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "connection_ok": connection_ok,
                "rate_limiter": rate_limiter_status,
                "database": db_status,
                "pipeline_ready": connection_ok  # Pipeline is ready if connection works
            }
            
            self.logger.info("Health check completed", health_ok=health_report["pipeline_ready"])
            return health_report
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "pipeline_ready": False
            }


# Global instance for easy access
tle_update_pipeline = TLEUpdatePipeline()


async def run_regular_updates():
    """
    Entry point for running regular TLE updates.
    
    This would typically be called from a scheduler or background service.
    """
    logger.info("Starting regular TLE update process")
    
    pipeline = TLEUpdatePipeline()
    result = await pipeline.update_latest_tles()
    
    logger.info(f"Regular update completed: {result.status.value}", 
               updated_count=result.updated_count,
               error_count=result.error_count)
    
    return result


if __name__ == "__main__":
    # For testing the pipeline
    async def test_pipeline():
        import os
        from src.core.config import settings
        
        # Check if credentials are available
        if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
            print("⚠️  Space-Track credentials not configured in environment")
            print("   Please set SPACETRACK_USERNAME and SPACETRACK_PASSWORD in .env")
            return
        
        print("Testing TLE Update Pipeline...")
        
        # Create pipeline instance
        pipeline = TLEUpdatePipeline()
        
        # Run health check
        print("\n1. Running health check...")
        health = await pipeline.run_health_check()
        print(f"   Health check: {'✅ OK' if health.get('pipeline_ready') else '❌ FAILED'}")
        
        # Test with a single satellite (ISS)
        print("\n2. Testing single satellite update (ISS)...")
        result = await pipeline.update_latest_tles(satellite_filter=[25544], days_back=1)
        print(f"   Result: {result.status.value}")
        print(f"   Updated: {result.updated_count}, Errors: {result.error_count}")
        
        # Show any errors
        if result.errors:
            print("   Errors:")
            for error in result.errors:
                print(f"     - {error}")
        
        print(f"\n3. Pipeline test completed in {result.duration_seconds:.2f}s")
    
    # Run the test
    asyncio.run(test_pipeline())