"""
Optimized API handler with enhanced rate limiting and concurrent tracking capabilities.
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import threading
import httpx
from urllib.parse import urljoin

from src.core.logging import get_logger
from src.core.config import settings
from src.data.ingest.spacetrack_client import RateLimiter as BaseRateLimiter
from src.data.models import TLE

logger = get_logger(__name__)


@dataclass
class APICallRecord:
    """Record of an API call for monitoring purposes."""
    url: str
    timestamp: datetime
    response_code: int
    duration_ms: float
    rate_limit_remaining: int


class EnhancedRateLimiter:
    """Enhanced rate limiter with adaptive throttling and monitoring."""
    
    def __init__(self, requests_per_hour: int = 300, burst_capacity: int = 10):
        self.requests_per_hour = requests_per_hour
        self.burst_capacity = burst_capacity
        self.requests_this_hour = 0
        self.hour_start = datetime.now()
        self._lock = asyncio.Lock()
        
        # Token bucket implementation for smoother rate limiting
        self.tokens = burst_capacity
        self.max_tokens = burst_capacity
        self.token_refill_rate = requests_per_hour / 3600  # tokens per second
        
        # Track call history for analytics
        self.call_history = deque(maxlen=1000)
        self.last_refill_time = time.time()
        
        # Adaptive throttling based on server responses
        self.adaptive_factor = 1.0  # Multiplier for rate limit based on server response
        self.error_count_recent = 0  # Count of errors in recent window
    
    async def _refill_tokens(self):
        """Refill tokens based on time elapsed."""
        current_time = time.time()
        time_elapsed = current_time - self.last_refill_time
        
        # Add tokens based on elapsed time
        new_tokens = time_elapsed * self.token_refill_rate * self.adaptive_factor
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill_time = current_time
    
    async def acquire(self, weight: int = 1) -> float:
        """
        Acquire permission to make a request.
        
        Args:
            weight: Weight of the request (more expensive requests use more tokens)
            
        Returns:
            Wait time in seconds (0 if immediate access)
        """
        async with self._lock:
            await self._refill_tokens()
            
            # Check if we have enough tokens
            while self.tokens < weight:
                # Calculate wait time needed to refill sufficient tokens
                tokens_needed = weight - self.tokens
                wait_time = tokens_needed / (self.token_refill_rate * self.adaptive_factor)
                
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s for {weight} tokens", 
                           tokens_available=self.tokens, tokens_needed=tokens_needed)
                
                await asyncio.sleep(wait_time)
                await self._refill_tokens()
            
            # Deduct tokens
            self.tokens -= weight
            
            # Reset error count on successful acquisition
            if self.error_count_recent > 0:
                self.error_count_recent = max(0, self.error_count_recent - 1)
            
            # Adjust adaptive factor based on error rate
            error_rate = self.error_count_recent / 10  # Last 10 attempts
            if error_rate > 0.3:  # Too many errors, slow down
                self.adaptive_factor = max(0.5, self.adaptive_factor * 0.9)
            elif error_rate < 0.1:  # Few errors, can speed up slightly
                self.adaptive_factor = min(1.2, self.adaptive_factor * 1.01)
            
            return 0  # No explicit wait needed since we waited in the loop
    
    def record_response(self, response_code: int, duration_ms: float, url: str):
        """Record API response for monitoring and adaptive adjustments."""
        record = APICallRecord(
            url=url,
            timestamp=datetime.now(),
            response_code=response_code,
            duration_ms=duration_ms,
            rate_limit_remaining=self.tokens
        )
        self.call_history.append(record)
        
        # Adjust adaptive factor based on response codes
        if response_code == 429:  # Rate limited
            self.adaptive_factor *= 0.8  # Slow down
            self.error_count_recent += 2
        elif response_code >= 400:  # Other errors
            self.error_count_recent += 1
        else:  # Successful request
            self.error_count_recent = max(0, self.error_count_recent - 0.5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        return {
            'tokens_available': self.tokens,
            'max_tokens': self.max_tokens,
            'token_refill_rate_per_second': self.token_refill_rate,
            'adaptive_factor': self.adaptive_factor,
            'error_count_recent': self.error_count_recent,
            'total_calls_recorded': len(self.call_history),
            'recent_success_rate': sum(1 for r in list(self.call_history)[-50:] if r.response_code < 400) / min(50, len(self.call_history))
        }


class ConcurrentTLECache:
    """Cache for TLE data to reduce API calls."""
    
    def __init__(self, cache_ttl_minutes: int = 10):
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.cache: Dict[int, tuple] = {}  # norad_id -> (tle, timestamp, expiry)
        self._lock = threading.RLock()
    
    def get(self, norad_id: int) -> Optional[TLE]:
        """Get TLE from cache if available and not expired."""
        with self._lock:
            if norad_id in self.cache:
                tle, timestamp, expiry = self.cache[norad_id]
                if datetime.now() < expiry:
                    return tle
                else:
                    # Remove expired entry
                    del self.cache[norad_id]
            return None
    
    def put(self, norad_id: int, tle: TLE):
        """Put TLE in cache."""
        with self._lock:
            expiry = datetime.now() + self.cache_ttl
            self.cache[norad_id] = (tle, datetime.now(), expiry)
    
    def invalidate(self, norad_id: int):
        """Invalidate cache entry for specific satellite."""
        with self._lock:
            if norad_id in self.cache:
                del self.cache[norad_id]
    
    def clear_expired(self):
        """Clear all expired entries."""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                norad_id for norad_id, (_, _, expiry) in self.cache.items()
                if now >= expiry
            ]
            for key in expired_keys:
                del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total = len(self.cache)
            expired = sum(1 for (_, _, expiry) in self.cache.values() if datetime.now() >= expiry)
            valid = total - expired
            return {'total_entries': total, 'valid_entries': valid, 'expired_entries': expired}


class OptimizedSpaceTrackClient:
    """Space-Track client with optimized concurrent API handling."""
    
    def __init__(self, rate_limiter: Optional[EnhancedRateLimiter] = None):
        self.rate_limiter = rate_limiter or EnhancedRateLimiter()
        self.cache = ConcurrentTLECache()
        self.session: Optional[httpx.AsyncClient] = None
        self.authenticated = False
        self._auth_lock = asyncio.Lock()
        self.base_url = "https://www.space-track.org/"
        self.query_endpoint = "basicspacedata/query"
        
        # Semaphore to limit concurrent requests
        self.concurrent_request_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_authenticated()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_authenticated(self):
        """Ensure client is authenticated."""
        if self.authenticated and self.session:
            return
            
        async with self._auth_lock:
            # Double-check after acquiring lock
            if self.authenticated and self.session:
                return
                
            try:
                # Create session with appropriate timeouts
                self.session = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    follow_redirects=True,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                )
                
                # Authenticate
                auth_data = {
                    "identity": settings.spacetrack.spacetrack_username,
                    "password": settings.spacetrack.spacetrack_password
                }
                
                response = await self.session.post(
                    urljoin(self.base_url, "ajaxauth/login"),
                    data=auth_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"Space-Track authentication failed: {response.status_code}")
                
                self.authenticated = True
                logger.info("Space-Track authentication successful")
                
            except Exception as e:
                logger.error(f"Space-Track authentication failed: {e}")
                raise
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
            self.authenticated = False
    
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Any:
        """Make a rate-limited request to Space-Track."""
        async with self.concurrent_request_semaphore:
            # Acquire rate limit token (weight of 1 for typical requests)
            await self.rate_limiter.acquire(weight=1)
            
            start_time = time.time()
            try:
                if params:
                    response = await self.session.get(url, params=params)
                else:
                    response = await self.session.get(url)
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Record response for rate limiter analytics
                self.rate_limiter.record_response(response.status_code, duration_ms, str(response.url))
                
                if response.status_code != 200:
                    logger.warning(f"Space-Track request failed: {response.status_code} - {response.text[:200]}")
                    raise Exception(f"Request failed with status {response.status_code}")
                
                return response.json()
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.rate_limiter.record_response(500, duration_ms, url)
                raise
    
    async def fetch_tle_by_norad_id(self, norad_id: int, days_back: int = 1) -> Optional[TLE]:
        """Fetch latest TLE for a specific NORAD ID with caching."""
        # Check cache first
        cached_tle = self.cache.get(norad_id)
        if cached_tle:
            logger.debug(f"Cache hit for NORAD {norad_id}")
            return cached_tle
        
        logger.debug(f"Cache miss for NORAD {norad_id}, fetching from API")
        
        try:
            # Use the recommended 'gp' class
            url = f"{self.base_url}{self.query_endpoint}/class/gp/NORAD_CAT_ID/{norad_id}/format/json"
            data = await self._make_request(url)
            
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.info(f"No TLE data returned for NORAD {norad_id}")
                return None
            
            # Process the TLE data (this would need to create a TLE object)
            # Since we don't have the full TLE creation logic here, we'll simulate
            tle_data = data[0]
            line1 = tle_data.get("TLE_LINE1", "").strip()
            line2 = tle_data.get("TLE_LINE2", "").strip()
            
            if not line1 or not line2:
                logger.warning(f"Incomplete TLE data for NORAD {norad_id}")
                return None
            
            # For now, return a partial TLE-like object - in real implementation,
            # this would create a proper TLE object using the same logic as in spacetrack_client.py
            from datetime import timezone
            from src.data.models import TLE as TLEModel
            
            # This is a simplified representation - in practice, you'd want to use the
            # same TLE creation logic from the original spacetrack_client
            tle_obj = TLEModel(
                norad_id=norad_id,
                classification=tle_data.get("CLASSIFICATION_TYPE", "U"),
                launch_year=int(tle_data.get("LAUNCH_DATE", "1957").split("-")[0]),
                launch_number=1,
                launch_piece="",
                epoch_datetime=datetime.strptime(tle_data.get("EPOCH", datetime.utcnow().isoformat()), "%Y-%m-%dT%H:%M:%S.%f"),
                mean_motion_derivative=float(tle_data.get("MEAN_MOTION_DOT", 0)),
                mean_motion_sec_derivative=float(tle_data.get("MEAN_MOTION_DDOT", 0)),
                bstar_drag_term=float(tle_data.get("BSTAR", 0)),
                element_set_number=int(tle_data.get("ELEMENT_SET_NO", 0)),
                inclination_degrees=float(tle_data.get("INCLINATION", 0)),
                raan_degrees=float(tle_data.get("RA_OF_ASC_NODE", 0)),
                eccentricity=float(tle_data.get("ECCENTRICITY", 0)),
                argument_of_perigee_degrees=float(tle_data.get("ARG_OF_PERICENTER", 0)),
                mean_anomaly_degrees=float(tle_data.get("MEAN_ANOMALY", 0)),
                mean_motion_orbits_per_day=float(tle_data.get("MEAN_MOTION", 0)),
                revolution_number_at_epoch=int(tle_data.get("REV_AT_EPOCH", 0)),
                tle_line1=line1,
                tle_line2=line2,
                epoch_julian_date=float(tle_data.get("EPOCH", 0)),
                line1_checksum=0,  # Would need to calculate
                line2_checksum=0,  # Would need to calculate
                source_url=str(url),
                acquisition_timestamp=datetime.now(timezone.utc),
                is_valid=True
            )
            
            # Cache the result
            self.cache.put(norad_id, tle_obj)
            return tle_obj
            
        except Exception as e:
            logger.error(f"Error fetching TLE for NORAD {norad_id}: {e}")
            return None
    
    async def batch_fetch_tles(self, norad_ids: List[int]) -> Dict[int, Optional[TLE]]:
        """Efficiently fetch TLEs for multiple satellites concurrently."""
        # First check cache
        results = {}
        uncached_ids = []
        
        for norad_id in norad_ids:
            cached_tle = self.cache.get(norad_id)
            if cached_tle:
                results[norad_id] = cached_tle
            else:
                uncached_ids.append(norad_id)
        
        if not uncached_ids:
            return results
        
        # Fetch uncached TLEs concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent fetches to avoid overwhelming
        
        async def fetch_with_semaphore(norad_id):
            async with semaphore:
                return norad_id, await self.fetch_tle_by_norad_id(norad_id)
        
        fetch_tasks = [fetch_with_semaphore(norad_id) for norad_id in uncached_ids]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Process results
        for result in fetch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch fetch task failed: {result}")
                continue
            
            norad_id, tle = result
            results[norad_id] = tle
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'cache_stats': self.cache.get_cache_stats(),
            'authenticated': self.authenticated,
            'session_active': self.session is not None
        }


class APITracker:
    """Tracks and optimizes API usage across multiple satellite tracking operations."""
    
    def __init__(self):
        self.clients: Dict[str, OptimizedSpaceTrackClient] = {}
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        
    def get_client(self, identifier: str = "default") -> OptimizedSpaceTrackClient:
        """Get or create an optimized client for a specific purpose."""
        if identifier not in self.clients:
            rate_limiter = EnhancedRateLimiter(
                requests_per_hour=settings.spacetrack.spacetrack_rate_limit,
                burst_capacity=5
            )
            self.clients[identifier] = OptimizedSpaceTrackClient(rate_limiter)
        
        return self.clients[identifier]
    
    def record_request(self, success: bool = True):
        """Record an API request."""
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get overall API usage statistics."""
        uptime = datetime.utcnow() - self.start_time
        success_rate = (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 1.0
        
        # Aggregate stats from all clients
        all_client_stats = {}
        for name, client in self.clients.items():
            all_client_stats[name] = client.get_performance_stats()
        
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'success_rate': success_rate,
            'uptime_seconds': uptime.total_seconds(),
            'requests_per_minute': self.request_count / (uptime.total_seconds() / 60) if uptime.total_seconds() > 0 else 0,
            'client_stats': all_client_stats
        }


# Global API tracker instance
api_tracker = APITracker()