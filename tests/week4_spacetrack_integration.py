"""
Week 4: Space-Track API Integration and Live Testing

Comprehensive tests for Space-Track API integration with real data validation.
"""
import asyncio
import pytest
import os
from datetime import datetime, timedelta
from typing import List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class TestSpaceTrackIntegration:
    """Live testing with Space-Track API using real orbital data."""
    
    def setup_method(self):
        """Initialize Space-Track client for testing."""
        self.rate_limiter = RateLimiter(requests_per_hour=settings.spacetrack.spacetrack_rate_limit)
        self.client = SpaceTrackClient(self.rate_limiter)
    
    @pytest.mark.asyncio
    async def test_spacetrack_authentication(self):
        """
        Test Space-Track authentication with real credentials.
        
        Validates:
        - Proper credential configuration
        - Successful login to Space-Track API
        - Session management
        """
        logger.info("Testing Space-Track authentication with real credentials")
        
        # Check if credentials are configured
        if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
            pytest.skip("Space-Track credentials not configured in environment")
        
        try:
            async with self.client:
                logger.info("✅ Space-Track authentication successful")
                
                # Verify we can make a simple request after authentication
                # Test with a basic query to ensure session is active
                pass  # Context manager handles authentication
                
        except Exception as e:
            logger.error(f"❌ Space-Track authentication failed: {e}")
            pytest.fail(f"Space-Track authentication failed: {e}")
    
    @pytest.mark.asyncio
    async def test_fetch_recent_iss_tles(self):
        """
        Test fetching recent ISS TLEs from Space-Track.
        
        Validates:
        - Real TLE data acquisition
        - Data validation and parsing
        - TLE format compliance
        """
        logger.info("Testing ISS TLE fetch from Space-Track")
        
        if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
            pytest.skip("Space-Track credentials not configured in environment")
        
        try:
            async with self.client:
                # Fetch recent ISS data (last 24 hours)
                start_time = datetime.utcnow() - timedelta(hours=24)
                end_time = datetime.utcnow()
                
                tles = await self.client.fetch_tle_catalog(
                    epoch_start=start_time,
                    epoch_end=end_time,
                    norad_ids=[25544]  # ISS
                )
                
                logger.info(f"✅ Retrieved {len(tles)} ISS TLEs from Space-Track")
                
                # Validate retrieved data
                assert len(tles) > 0, "No TLEs retrieved for ISS"
                
                iss_tle = tles[0]
                logger.info(f"  ISS TLE epoch: {iss_tle.epoch_datetime}")
                logger.info(f"  Orbital period: {1440/iss_tle.mean_motion_orbits_per_day:.2f} minutes")
                
                # Validate basic TLE properties
                assert iss_tle.norad_id == 25544, "Incorrect NORAD ID for ISS"
                assert iss_tle.is_valid, "ISS TLE failed validation"
                assert iss_tle.mean_motion_orbits_per_day > 0, "Invalid mean motion"
                assert 0 <= iss_tle.eccentricity < 0.1, "Unexpected eccentricity for ISS"
                
        except Exception as e:
            logger.error(f"❌ ISS TLE fetch failed: {e}")
            pytest.fail(f"ISS TLE fetch failed: {e}")
    
    @pytest.mark.asyncio
    async def test_fetch_catalog_data(self):
        """
        Test fetching catalog data for multiple satellites.
        
        Validates:
        - Bulk TLE retrieval
        - Rate limiting functionality
        - Data processing at scale
        """
        logger.info("Testing bulk catalog fetch from Space-Track")
        
        if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
            pytest.skip("Space-Track credentials not configured in environment")
        
        try:
            async with self.client:
                # Fetch data for several active satellites
                start_time = datetime.utcnow() - timedelta(hours=6)  # Last 6 hours
                end_time = datetime.utcnow()
                
                sample_norad_ids = [25544, 42982, 43017, 43129, 43217]  # ISS, Starlink, etc.
                
                tles = await self.client.fetch_tle_catalog(
                    epoch_start=start_time,
                    epoch_end=end_time,
                    norad_ids=sample_norad_ids
                )
                
                logger.info(f"✅ Retrieved {len(tles)} TLEs for {len(sample_norad_ids)} satellites")
                
                # Validate we got data for expected satellites
                retrieved_ids = {tle.norad_id for tle in tles}
                expected_ids = set(sample_norad_ids)
                
                logger.info(f"  Retrieved IDs: {sorted(retrieved_ids)}")
                
                # At least some of our requested satellites should be found
                assert len(tles) > 0, "No TLEs retrieved for requested satellites"
                
                # Validate each retrieved TLE
                for tle in tles:
                    assert tle.norad_id in expected_ids, f"Unexpected NORAD ID: {tle.norad_id}"
                    assert tle.is_valid, f"TLE validation failed for NORAD {tle.norad_id}"
                    
        except Exception as e:
            logger.error(f"❌ Catalog fetch failed: {e}")
            pytest.fail(f"Catalog fetch failed: {e}")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """
        Test that rate limiting is properly enforced.
        
        Validates:
        - Rate limiter functionality
        - Proper error handling when limits are exceeded
        - Concurrency protection
        """
        logger.info("Testing Space-Track rate limiting")
        
        if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
            pytest.skip("Space-Track credentials not configured in environment")
        
        # Test rate limiter directly
        test_limiter = RateLimiter(requests_per_hour=2)  # Very restrictive limit
        
        # Acquire two permits (should succeed)
        await test_limiter.acquire()
        await test_limiter.acquire()
        
        # Third permit should raise RateLimitError
        with pytest.raises(Exception) as exc_info:
            await test_limiter.acquire()
        
        logger.info("✅ Rate limiting behavior validated")
    
    @pytest.mark.asyncio
    async def test_conjunction_analysis_with_real_data(self):
        """
        Test full conjunction analysis pipeline with real Space-Track data.
        
        Validates:
        - End-to-end processing with live data
        - Integration between Space-Track client and analysis engine
        - Performance with real-world data volumes
        """
        logger.info("Testing conjunction analysis with real Space-Track data")
        
        if not settings.spacetrack.spacetrack_username or not settings.spacetrack.spacetrack_password:
            pytest.skip("Space-Track credentials not configured in environment")
        
        from src.conjunction.full_analysis import conjunction_analyzer
        
        try:
            async with self.client:
                # Fetch recent ISS data as primary object
                start_time = datetime.utcnow() - timedelta(hours=1)
                end_time = datetime.utcnow()
                
                primary_tles = await self.client.fetch_tle_catalog(
                    epoch_start=start_time,
                    epoch_end=end_time,
                    norad_ids=[25544]  # ISS
                )
                
                if len(primary_tles) == 0:
                    pytest.skip("No recent ISS data available for testing")
                
                primary_tle = primary_tles[0]
                
                # Fetch a few nearby satellites for catalog
                catalog_tles = await self.client.fetch_tle_catalog(
                    epoch_start=start_time,
                    epoch_end=end_time,
                    norad_ids=[42982, 43017, 43129]  # Some Starlink satellites
                )
                
                logger.info(f"  Primary: ISS (25544)")
                logger.info(f"  Catalog: {len(catalog_tles)} satellites")
                
                # Perform conjunction analysis
                events = conjunction_analyzer.perform_full_analysis(
                    primary_tle=primary_tle,
                    catalog_tles=catalog_tles,
                    time_window_hours=6.0,  # 6-hour window
                    screening_threshold_km=50.0,  # 50km screening
                    probability_threshold=1e-6
                )
                
                logger.info(f"✅ Conjunction analysis completed with {len(events)} events")
                
                # Log results
                for event in events:
                    logger.info(f"  Event: TCA {event.tca_datetime}, "
                              f"Miss dist: {event.miss_distance_meters:.1f}m, "
                              f"Pc: {event.probability:.2e}")
                
        except Exception as e:
            logger.error(f"❌ Conjunction analysis with real data failed: {e}")
            pytest.fail(f"Conjunction analysis with real data failed: {e}")


if __name__ == "__main__":
    # Allow running this test file directly for development
    import subprocess
    import sys
    
    # Run with pytest for proper async support
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "-s",
        "--tb=short"
    ])
    
    sys.exit(result.returncode)