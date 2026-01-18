"""
Optimized multi-satellite tracking system with performance optimization.
"""
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import time
import threading
from dataclasses import dataclass

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.propagation.sgp4_engine import SGP4Engine, PropagationResult
from src.data.models import TLE
from src.core.logging import get_logger
from src.core.config import settings

logger = get_logger(__name__)


@dataclass
class SatelliteInfo:
    """Information about a tracked satellite."""
    norad_id: int
    tle: TLE
    last_updated: datetime
    current_position: Optional[Dict] = None
    propagation_result: Optional[PropagationResult] = None
    is_active: bool = True


class OptimizedSatelliteTracker:
    """High-performance multi-satellite tracking system."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.sgp4_engine = SGP4Engine()
        
        # Initialize Space-Track client with custom rate limiter for multiple satellites
        rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack.spacetrack_rate_limit)
        self.spacetrack_client = SpaceTrackClient(rate_limiter)
        
        self.tracked_satellites: Dict[int, SatelliteInfo] = {}
        self.is_running = False
        self.update_lock = threading.Lock()
        
    async def add_satellite(self, norad_id: int) -> bool:
        """Add a satellite to tracking."""
        try:
            async with self.spacetrack_client:
                tle = await self.spacetrack_client.fetch_tle_by_norad_id(norad_id, days_back=7)
                if tle:
                    current_pos = self.sgp4_engine.propagate_to_epoch(tle, datetime.utcnow())
                    self.tracked_satellites[norad_id] = SatelliteInfo(
                        norad_id=norad_id,
                        tle=tle,
                        last_updated=datetime.utcnow(),
                        current_position={
                            'x': current_pos.cartesian_state.x,
                            'y': current_pos.cartesian_state.y,
                            'z': current_pos.cartesian_state.z,
                            'lat': current_pos.latitude_deg,
                            'lon': current_pos.longitude_deg,
                            'alt': current_pos.altitude_m
                        },
                        propagation_result=current_pos
                    )
                    logger.info(f"Added satellite {norad_id} to tracking")
                    return True
                else:
                    logger.warning(f"Could not fetch TLE for satellite {norad_id}")
                    return False
        except Exception as e:
            logger.error(f"Error adding satellite {norad_id}: {e}")
            return False
    
    async def remove_satellite(self, norad_id: int):
        """Remove a satellite from tracking."""
        if norad_id in self.tracked_satellites:
            del self.tracked_satellites[norad_id]
            logger.info(f"Removed satellite {norad_id} from tracking")
    
    async def update_single_satellite(self, norad_id: int) -> bool:
        """Update position for a single satellite."""
        try:
            sat_info = self.tracked_satellites.get(norad_id)
            if not sat_info:
                return False
                
            # Refresh TLE if older than 5 minutes
            if datetime.utcnow() - sat_info.last_updated > timedelta(minutes=5):
                async with self.spacetrack_client:
                    new_tle = await self.spacetrack_client.fetch_tle_by_norad_id(norad_id, days_back=7)
                    if new_tle:
                        sat_info.tle = new_tle
                        sat_info.last_updated = datetime.utcnow()
            
            # Calculate current position
            current_pos = self.sgp4_engine.propagate_to_epoch(
                sat_info.tle, datetime.utcnow()
            )
            
            # Update satellite info
            with self.update_lock:
                sat_info.current_position = {
                    'x': current_pos.cartesian_state.x,
                    'y': current_pos.cartesian_state.y,
                    'z': current_pos.cartesian_state.z,
                    'lat': current_pos.latitude_deg,
                    'lon': current_pos.longitude_deg,
                    'alt': current_pos.altitude_m
                }
                sat_info.propagation_result = current_pos
            
            return True
        except Exception as e:
            logger.error(f"Error updating satellite {norad_id}: {e}")
            return False
    
    async def update_all_satellites(self):
        """Update positions for all tracked satellites concurrently."""
        if not self.tracked_satellites:
            logger.warning("No satellites to update")
            return
        
        start_time = time.time()
        tasks = [
            self.update_single_satellite(norad_id)
            for norad_id in self.tracked_satellites.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_updates = sum(1 for result in results if result is True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Updated {successful_updates}/{len(tasks)} satellites in {elapsed_time:.2f}s")
    
    async def batch_fetch_tles(self, norad_ids: List[int]) -> Dict[int, Optional[TLE]]:
        """Efficiently fetch TLEs for multiple satellites."""
        tles = {}
        
        async with self.spacetrack_client:
            # Use the fetch_tle_catalog method for batch operations
            try:
                tle_list = await self.spacetrack_client.fetch_tle_catalog(
                    epoch_start=datetime.utcnow() - timedelta(days=1),
                    epoch_end=datetime.utcnow(),
                    norad_ids=norad_ids
                )
                
                # Map TLEs by NORAD ID
                tle_map = {tle.norad_id: tle for tle in tle_list}
                
                for norad_id in norad_ids:
                    tles[norad_id] = tle_map.get(norad_id)
                    
            except Exception as e:
                logger.error(f"Batch TLE fetch failed: {e}")
                # Fall back to individual fetches
                for norad_id in norad_ids:
                    try:
                        tle = await self.spacetrack_client.fetch_tle_by_norad_id(norad_id, days_back=7)
                        tles[norad_id] = tle
                    except Exception as e:
                        logger.error(f"Failed to fetch TLE for {norad_id}: {e}")
                        tles[norad_id] = None
        
        return tles
    
    def get_tracked_satellites_status(self) -> Dict:
        """Get status information about tracked satellites."""
        active_count = sum(1 for sat in self.tracked_satellites.values() if sat.is_active)
        total_count = len(self.tracked_satellites)
        
        # Calculate average position age
        if self.tracked_satellites:
            avg_age = sum(
                (datetime.utcnow() - sat.last_updated).total_seconds()
                for sat in self.tracked_satellites.values()
            ) / len(self.tracked_satellites)
        else:
            avg_age = 0
        
        return {
            'active_count': active_count,
            'total_count': total_count,
            'average_position_age_seconds': avg_age,
            'last_update_time': datetime.utcnow().isoformat()
        }
    
    async def run_tracking_loop(self, update_interval: float = 5.0):
        """Run continuous tracking loop."""
        self.is_running = True
        iteration = 0
        
        while self.is_running:
            iteration += 1
            logger.info(f"Tracking loop iteration {iteration}")
            
            try:
                await self.update_all_satellites()
                
                # Log status periodically
                if iteration % 10 == 0:
                    status = self.get_tracked_satellites_status()
                    logger.info(f"Tracking status: {status}")
                
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
            
            await asyncio.sleep(update_interval)
    
    def stop_tracking(self):
        """Stop the tracking loop."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Tracking stopped")


class AdvancedTrackingDashboard:
    """Advanced dashboard for multi-satellite tracking with real-time updates."""
    
    def __init__(self):
        self.tracker = OptimizedSatelliteTracker()
        self.visualization = None  # Will be set when visualization module is available
    
    async def initialize_tracking(self, satellite_ids: List[int]):
        """Initialize tracking for multiple satellites."""
        logger.info(f"Initializing tracking for {len(satellite_ids)} satellites")
        
        # Add all satellites concurrently
        tasks = [self.tracker.add_satellite(norad_id) for norad_id in satellite_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_additions = sum(1 for result in results if result is True)
        logger.info(f"Successfully added {successful_additions}/{len(satellite_ids)} satellites")
        
        return successful_additions == len(satellite_ids)
    
    async def get_current_positions(self) -> Dict[int, Dict]:
        """Get current positions of all tracked satellites."""
        positions = {}
        
        with self.tracker.update_lock:
            for norad_id, sat_info in self.tracker.tracked_satellites.items():
                if sat_info.current_position:
                    positions[norad_id] = {
                        'position': sat_info.current_position,
                        'last_updated': sat_info.last_updated,
                        'norad_id': sat_info.norad_id
                    }
        
        return positions
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the tracking system."""
        return {
            'tracked_satellites_count': len(self.tracker.tracked_satellites),
            'executor_queue_size': self.tracker.executor._work_queue.qsize() if hasattr(self.tracker.executor._work_queue, 'qsize') else 0,
            'executor_max_workers': self.tracker.max_workers,
            'tracker_status': 'RUNNING' if self.tracker.is_running else 'STOPPED'
        }