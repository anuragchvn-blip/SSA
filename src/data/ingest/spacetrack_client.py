"""Space-Track.org API client with rate limiting and robust error handling."""

import asyncio
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field, validator

from src.core.config import settings
from src.core.exceptions import (
    DataIngestionError, TLEValidationError, RateLimitError, 
    AuthenticationError
)
from src.core.logging import get_logger, log_execution_time
from src.data.models import TLE

logger = get_logger(__name__)


class TLEModel(BaseModel):
    """Pydantic model for TLE data validation."""
    
    line1: str = Field(..., min_length=69, max_length=69)
    line2: str = Field(..., min_length=69, max_length=69)
    norad_id: int = Field(..., ge=1, le=999999)
    epoch_datetime: datetime
    epoch_julian_date: float
    
    @validator('line1')
    def validate_line1(cls, v):
        """Validate TLE line 1 format and checksum."""
        if not v.startswith('1 '):
            raise ValueError("Line 1 must start with '1 '")
        
        # Validate checksum
        calculated_checksum = cls._calculate_checksum(v[:-1])
        provided_checksum = int(v[-1])
        if calculated_checksum != provided_checksum:
            raise ValueError(f"TLE line 1 checksum mismatch: calculated {calculated_checksum}, provided {provided_checksum}")
        return v
    
    @validator('line2')
    def validate_line2(cls, v):
        """Validate TLE line 2 format and checksum."""
        if not v.startswith('2 '):
            raise ValueError("Line 2 must start with '2 '")
        
        # Validate checksum
        calculated_checksum = cls._calculate_checksum(v[:-1])
        provided_checksum = int(v[-1])
        if calculated_checksum != provided_checksum:
            raise ValueError(f"TLE line 2 checksum mismatch: calculated {calculated_checksum}, provided {provided_checksum}")
        return v
    
    @staticmethod
    def _calculate_checksum(line: str) -> int:
        """Calculate TLE checksum according to standard."""
        checksum = 0
        for char in line:
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        return checksum % 10


class RateLimiter:
    """Rate limiter for Space-Track API requests."""
    
    def __init__(self, requests_per_hour: int = 300):
        self.requests_per_hour = requests_per_hour
        self.requests_this_hour = 0
        self.hour_start = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            
            # Reset counter if new hour
            if (now - self.hour_start).total_seconds() >= 3600:
                self.requests_this_hour = 0
                self.hour_start = now
            
            # Check rate limit
            if self.requests_this_hour >= self.requests_per_hour:
                wait_time = 3600 - (now - self.hour_start).total_seconds()
                raise RateLimitError(
                    message=f"Rate limit exceeded. Wait {wait_time:.0f} seconds",
                    error_code="RATE_LIMIT_EXCEEDED",
                    details={
                        "requests_this_hour": self.requests_this_hour,
                        "limit": self.requests_per_hour,
                        "reset_in_seconds": wait_time
                    }
                )
            
            self.requests_this_hour += 1
            logger.debug("Rate limit acquired", 
                        requests_remaining=self.requests_per_hour - self.requests_this_hour)


class SpaceTrackClient:
    """Space-Track.org API client with full production-grade features."""
    
    BASE_URL = "https://www.space-track.org/"
    LOGIN_ENDPOINT = "ajaxauth/login"
    QUERY_ENDPOINT = "basicspacedata/query"
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session: Optional[httpx.AsyncClient] = None
        self.authenticated = False
        self._auth_lock = asyncio.Lock()
    
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
                    follow_redirects=True
                )
                
                # Authenticate
                auth_data = {
                    "identity": settings.spacetrack.spacetrack_username,
                    "password": settings.spacetrack.spacetrack_password
                }
                
                response = await self.session.post(
                    urljoin(self.BASE_URL, self.LOGIN_ENDPOINT),
                    data=auth_data
                )
                
                if response.status_code != 200:
                    raise AuthenticationError(
                        message=f"Space-Track authentication failed: {response.status_code}",
                        error_code="SPACETRACK_AUTH_FAILED",
                        details={"status_code": response.status_code}
                    )
                
                # Parse response to confirm login
                try:
                    result = response.json()
                    if isinstance(result, dict) and result.get("Login") != "SUCCESS":
                        raise AuthenticationError(
                            message="Space-Track login response indicates failure",
                            error_code="SPACETRACK_AUTH_FAILED",
                            details={"response": result}
                        )
                except json.JSONDecodeError:
                    # Some successful responses aren't JSON
                    if "Space-Track" not in response.text:
                        raise AuthenticationError(
                            message="Space-Track authentication response unexpected",
                            error_code="SPACETRACK_AUTH_FAILED",
                            details={"response_text": response.text[:200]}
                        )
                
                self.authenticated = True
                logger.info("Space-Track authentication successful")
                
            except httpx.RequestError as e:
                raise DataIngestionError(
                    message=f"Network error during Space-Track authentication: {str(e)}",
                    error_code="SPACETRACK_AUTH_FAILED",
                    details={"error": str(e)}
                )
    
    @log_execution_time("spacetrack_fetch_tle_catalog")
    async def fetch_tle_catalog(
        self,
        epoch_start: datetime,
        epoch_end: datetime,
        object_class: Optional[str] = None,
        norad_ids: Optional[List[int]] = None,
        include_debris: bool = True
    ) -> List[TLE]:
        """
        Fetch TLE catalog from Space-Track with full validation.
        
        Args:
            epoch_start: Start datetime for TLE epochs
            epoch_end: End datetime for TLE epochs
            object_class: Filter by object class ('PAY', 'ROCKET', 'DEBRIS', etc.)
            norad_ids: Specific NORAD IDs to fetch
            include_debris: Whether to include space debris objects
            
        Returns:
            List of validated TLE objects ready for database storage
            
        Raises:
            RateLimitError: When API quota exceeded
            TLEValidationError: When TLE data is invalid
            DataIngestionError: For other failures
        """
        await self._ensure_authenticated()
        await self.rate_limiter.acquire()
        
        try:
            # Build query parameters
            # Use the recommended 'gp' class instead of 'tle_latest'
            query_params = {
                "class": "gp",
                "orderby": "EPOCH desc",
                "format": "json"
            }
            
            # Add filters using the where clause
            predicates = []
            
            # Object class filter
            if object_class:
                predicates.append(f"OBJECT_TYPE='{object_class}'")
            
            # NORAD ID filter
            if norad_ids:
                norad_list = ",".join(str(nid) for nid in norad_ids)
                predicates.append(f"NORAD_CAT_ID IN ({norad_list})")
            
            # Debris exclusion
            if not include_debris:
                predicates.append("OBJECT_TYPE!='DEBRIS'")
            
            if predicates:
                query_params["where"] = " AND ".join(predicates)
            
            # Execute query
            url = urljoin(self.BASE_URL, self.QUERY_ENDPOINT)
            response = await self.session.get(url, params=query_params)
            
            # Handle rate limiting response
            if response.status_code == 429:
                raise RateLimitError(
                    message="Space-Track rate limit exceeded",
                    error_code="RATE_LIMIT_EXCEEDED",
                    details={"retry_after": response.headers.get("Retry-After")}
                )
            
            if response.status_code != 200:
                raise DataIngestionError(
                    message=f"Space-Track query failed: {response.status_code}",
                    error_code="TLE_VALIDATION_FAILED",
                    details={
                        "status_code": response.status_code,
                        "url": str(response.url),
                        "response": response.text[:500]
                    }
                )
            
            # Parse response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise DataIngestionError(
                    message="Invalid JSON response from Space-Track",
                    error_code="TLE_VALIDATION_FAILED",
                    details={"error": str(e), "response_snippet": response.text[:200]}
                )
            
            if not isinstance(data, list):
                raise DataIngestionError(
                    message="Unexpected response format from Space-Track",
                    error_code="TLE_VALIDATION_FAILED",
                    details={"response_type": type(data)}
                )
            
            # Process and validate TLEs
            validated_tles = []
            validation_errors = []
            
            for item in data:
                try:
                    tle = self._parse_and_validate_tle_item(item, response.url)
                    validated_tles.append(tle)
                except TLEValidationError as e:
                    validation_errors.append({
                        "norad_id": item.get("NORAD_CAT_ID"),
                        "error": str(e)
                    })
                    logger.warning("TLE validation failed", 
                                 norad_id=item.get("NORAD_CAT_ID"), 
                                 error=str(e))
            
            # Log validation results
            logger.info(
                "TLE catalog fetch completed",
                total_received=len(data),
                valid_tles=len(validated_tles),
                validation_errors=len(validation_errors)
            )
            
            if validation_errors:
                logger.warning(
                    "Some TLEs failed validation",
                    error_count=len(validation_errors),
                    sample_errors=validation_errors[:5]
                )
            
            return validated_tles
            
        except httpx.RequestError as e:
            raise DataIngestionError(
                message=f"Network error during TLE fetch: {str(e)}",
                error_code="TLE_VALIDATION_FAILED",
                details={"error": str(e)}
            )
    
    def _parse_and_validate_tle_item(self, item: Dict[str, Any], source_url: str) -> TLE:
        """Parse and validate a single TLE item from API response."""
        
        # Extract required fields
        line1 = item.get("TLE_LINE1", "").strip()
        line2 = item.get("TLE_LINE2", "").strip()
        norad_id = int(item.get("NORAD_CAT_ID", 0))
        
        if not line1 or not line2 or not norad_id:
            raise TLEValidationError(
                message="Missing required TLE fields",
                error_code="TLE_VALIDATION_FAILED",
                details={
                    "missing_fields": [
                        k for k in ["TLE_LINE1", "TLE_LINE2", "NORAD_CAT_ID"]
                        if not item.get(k)
                    ]
                }
            )
        
        # Validate using Pydantic model
        try:
            tle_model = TLEModel(
                line1=line1,
                line2=line2,
                norad_id=norad_id,
                epoch_datetime=self._parse_epoch_from_tle(line1),
                epoch_julian_date=self._extract_julian_date(line1)
            )
        except ValueError as e:
            raise TLEValidationError(
                message=f"TLE validation failed: {str(e)}",
                error_code="TLE_VALIDATION_FAILED",
                details={"validation_error": str(e)}
            )
        
        # Parse additional TLE elements
        elements = self._parse_tle_elements(line1, line2)
        
        # Create database model
        tle = TLE(
            norad_id=tle_model.norad_id,
            classification=elements['classification'],
            launch_year=elements['launch_year'],
            launch_number=elements['launch_number'],
            launch_piece=elements['launch_piece'],
            epoch_datetime=tle_model.epoch_datetime,
            mean_motion_derivative=elements['mean_motion_derivative'],
            mean_motion_sec_derivative=elements['mean_motion_sec_derivative'],
            bstar_drag_term=elements['bstar_drag_term'],
            element_set_number=elements['element_set_number'],
            inclination_degrees=elements['inclination_degrees'],
            raan_degrees=elements['raan_degrees'],
            eccentricity=elements['eccentricity'],
            argument_of_perigee_degrees=elements['argument_of_perigee_degrees'],
            mean_anomaly_degrees=elements['mean_anomaly_degrees'],
            mean_motion_orbits_per_day=elements['mean_motion_orbits_per_day'],
            revolution_number_at_epoch=elements['revolution_number_at_epoch'],
            tle_line1=line1,
            tle_line2=line2,
            epoch_julian_date=tle_model.epoch_julian_date,
            line1_checksum=int(line1[-1]),
            line2_checksum=int(line2[-1]),
            source_url=source_url,
            acquisition_timestamp=datetime.now(timezone.utc),
            data_version="1.0"
        )
        
        return tle
    
    def _parse_epoch_from_tle(self, line1: str) -> datetime:
        """Parse epoch datetime from TLE line 1."""
        # Epoch is in columns 19-32: YYDDD.DDDDDDDD
        epoch_str = line1[18:32]
        year = int(epoch_str[:2])
        day_of_year = float(epoch_str[2:])
        
        # Handle 2-digit year (Space-Track uses 57-99 for 1957-1999, 00-56 for 2000-2056)
        if year < 57:
            full_year = 2000 + year
        else:
            full_year = 1900 + year
        
        # Convert day of year to datetime
        jan_1 = datetime(full_year, 1, 1, tzinfo=timezone.utc)
        epoch_dt = jan_1 + timedelta(days=day_of_year - 1)
        return epoch_dt
    
    def _extract_julian_date(self, line1: str) -> float:
        """Extract Julian date from TLE epoch."""
        epoch_str = line1[18:32]
        return float(f"20{epoch_str[:2]}{epoch_str[2:]}")
    
    def _parse_tle_elements(self, line1: str, line2: str) -> Dict[str, Any]:
        """Parse all orbital elements from TLE lines."""
        # Line 1 elements
        classification = line1[7]
        intl_desig = line1[9:17].strip()
        
        # Parse international designation
        if len(intl_desig) >= 7:
            launch_year = int(intl_desig[:2]) if intl_desig[:2].isdigit() else 0
            launch_number = int(intl_desig[2:5]) if intl_desig[2:5].isdigit() else 0
            launch_piece = intl_desig[5:7]
        else:
            launch_year = launch_number = 0
            launch_piece = ""
        
        mean_motion_derivative = float(line1[33:43].replace('-','E-').replace('+','E+'))
        mean_motion_sec_derivative = float(line1[44:50].replace('-','E-').replace('+','E+')) * 1e-5
        bstar_drag_term = float(line1[53:61].replace('-','E-').replace('+','E+')) * 1e-5
        element_set_number = int(line1[64:68])
        
        # Line 2 elements
        inclination_degrees = float(line2[8:16])
        raan_degrees = float(line2[17:25])
        eccentricity = float(f"0.{line2[26:33]}")
        argument_of_perigee_degrees = float(line2[34:42])
        mean_anomaly_degrees = float(line2[43:51])
        mean_motion_orbits_per_day = float(line2[52:63])
        revolution_number_at_epoch = int(line2[63:68])
        
        return {
            'classification': classification,
            'launch_year': launch_year,
            'launch_number': launch_number,
            'launch_piece': launch_piece,
            'mean_motion_derivative': mean_motion_derivative,
            'mean_motion_sec_derivative': mean_motion_sec_derivative,
            'bstar_drag_term': bstar_drag_term,
            'element_set_number': element_set_number,
            'inclination_degrees': inclination_degrees,
            'raan_degrees': raan_degrees,
            'eccentricity': eccentricity,
            'argument_of_perigee_degrees': argument_of_perigee_degrees,
            'mean_anomaly_degrees': mean_anomaly_degrees,
            'mean_motion_orbits_per_day': mean_motion_orbits_per_day,
            'revolution_number_at_epoch': revolution_number_at_epoch
        }
    
    async def fetch_tle_by_norad_id(self, norad_id: int, days_back: int = 1) -> Optional[TLE]:
        """
        Fetch latest TLE for a specific NORAD ID.
        
        Args:
            norad_id: Satellite NORAD catalog ID
            days_back: Number of days back to look for TLEs
            
        Returns:
            Latest TLE for the satellite or None if not found
        """
        await self._ensure_authenticated()
        await self.rate_limiter.acquire()
        
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            end_date = datetime.utcnow() + timedelta(days=1)  # Include today
            
            # Use the recommended 'gp' class instead of 'tle_latest'
            url = f"{self.BASE_URL}{self.QUERY_ENDPOINT}/class/gp/NORAD_CAT_ID/{norad_id}/format/json"
            response = await self.session.get(url)
            
            if response.status_code != 200:
                if response.status_code == 404:
                    logger.info(f"No TLE found for NORAD {norad_id}")
                    return None
                raise DataIngestionError(
                    message=f"Space-Track query failed: {response.status_code}",
                    error_code="TLE_FETCH_FAILED",
                    details={"status_code": response.status_code, "url": str(response.url)}
                )
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response for NORAD {norad_id}: {e}")
                logger.debug(f"Response text: {response.text[:500]}")
                return None
            
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.info(f"No TLE data returned for NORAD {norad_id}")
                return None
            
            # Parse the TLE data
            tle_data = data[0]
            line1 = tle_data.get("TLE_LINE1", "").strip()
            line2 = tle_data.get("TLE_LINE2", "").strip()
            
            if not line1 or not line2:
                raise TLEValidationError(
                    message=f"Incomplete TLE data for NORAD {norad_id}",
                    error_code="TLE_VALIDATION_FAILED",
                    details={"missing_fields": [k for k in ["TLE_LINE1", "TLE_LINE2"] if not tle_data.get(k)]}
                )
            
            # Create TLE object
            tle_model = TLEModel(
                line1=line1,
                line2=line2,
                norad_id=norad_id,
                epoch_datetime=self._parse_epoch_from_tle(line1),
                epoch_julian_date=self._extract_julian_date(line1)
            )
            
            elements = self._parse_tle_elements(line1, line2)
            
            tle = TLE(
                norad_id=tle_model.norad_id,
                classification=elements['classification'],
                launch_year=elements['launch_year'],
                launch_number=elements['launch_number'],
                launch_piece=elements['launch_piece'],
                epoch_datetime=tle_model.epoch_datetime,
                mean_motion_derivative=elements['mean_motion_derivative'],
                mean_motion_sec_derivative=elements['mean_motion_sec_derivative'],
                bstar_drag_term=elements['bstar_drag_term'],
                element_set_number=elements['element_set_number'],
                inclination_degrees=elements['inclination_degrees'],
                raan_degrees=elements['raan_degrees'],
                eccentricity=elements['eccentricity'],
                argument_of_perigee_degrees=elements['argument_of_perigee_degrees'],
                mean_anomaly_degrees=elements['mean_anomaly_degrees'],
                mean_motion_orbits_per_day=elements['mean_motion_orbits_per_day'],
                revolution_number_at_epoch=elements['revolution_number_at_epoch'],
                tle_line1=line1,
                tle_line2=line2,
                epoch_julian_date=tle_model.epoch_julian_date,
                line1_checksum=int(line1[-1]),
                line2_checksum=int(line2[-1]),
                source_url=str(response.url),
                acquisition_timestamp=datetime.now(timezone.utc),
                is_valid=True
            )
            
            logger.debug(f"Fetched TLE for NORAD {norad_id}, epoch: {tle.epoch_datetime}")
            return tle
            
        except Exception as e:
            logger.error(f"Error fetching TLE for NORAD {norad_id}: {e}")
            raise
    
    async def fetch_satellite_history(self, norad_id: int, start_date: datetime, end_date: datetime) -> List[TLE]:
        """
        Fetch historical TLEs for a satellite over a date range.
        
        Args:
            norad_id: Satellite NORAD catalog ID
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of TLEs ordered by epoch (newest first)
        """
        await self._ensure_authenticated()
        await self.rate_limiter.acquire()
        
        try:
            query_params = {
                "class": "tle",
                "norad_cat_id": str(norad_id),
                "orderby": "EPOCH desc",
                "format": "json",
                "startepoch": start_date.strftime("%Y-%m-%d"),
                "endepoch": end_date.strftime("%Y-%m-%d")
            }
            
            url = urljoin(self.BASE_URL, self.QUERY_ENDPOINT)
            response = await self.session.get(url, params=query_params)
            
            if response.status_code != 200:
                raise DataIngestionError(
                    message=f"Space-Track history query failed: {response.status_code}",
                    error_code="TLE_HISTORY_FETCH_FAILED",
                    details={"status_code": response.status_code}
                )
            
            data = response.json()
            if not data or not isinstance(data, list):
                return []
            
            tles = []
            for item in data:
                tle = self._parse_and_validate_tle_item(item, str(response.url))
                tles.append(tle)
            
            logger.info(f"Fetched {len(tles)} historical TLEs for NORAD {norad_id}")
            return tles
            
        except Exception as e:
            logger.error(f"Error fetching history for NORAD {norad_id}: {e}")
            raise
    
    async def get_satellite_metadata(self, norad_id: int) -> Dict[str, Any]:
        """
        Get satellite metadata from Space-Track.
        
        Args:
            norad_id: Satellite NORAD catalog ID
            
        Returns:
            Dictionary containing satellite metadata
        """
        await self._ensure_authenticated()
        await self.rate_limiter.acquire()
        
        try:
            query_params = {
                "class": "satcat",
                "NORAD_CAT_ID": str(norad_id),
                "format": "json"
            }
            
            url = urljoin(self.BASE_URL, self.QUERY_ENDPOINT)
            response = await self.session.get(url, params=query_params)
            
            if response.status_code != 200:
                raise DataIngestionError(
                    message=f"Space-Track metadata query failed: {response.status_code}",
                    error_code="METADATA_FETCH_FAILED",
                    details={"status_code": response.status_code}
                )
            
            data = response.json()
            if not data or not isinstance(data, list) or len(data) == 0:
                raise DataIngestionError(
                    message=f"No metadata found for NORAD {norad_id}",
                    error_code="METADATA_NOT_FOUND",
                    details={"norad_id": norad_id}
                )
            
            logger.debug(f"Retrieved metadata for NORAD {norad_id}")
            return data[0]
            
        except Exception as e:
            logger.error(f"Error fetching metadata for NORAD {norad_id}: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test connection to Space-Track API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            await self._ensure_authenticated()
            # Make a simple query to test the connection
            # Correct Space-Track query format for catalog data
            url = f"{self.BASE_URL}{self.QUERY_ENDPOINT}/class/satcat/format/json"
            response = await self.session.get(url)
            
            # If the above fails, try with a specific query
            if response.status_code != 200:
                # Use the traditional query format with proper parameters
                url = urljoin(self.BASE_URL, self.QUERY_ENDPOINT)
                response = await self.session.get(url, params={
                    "class": "tle_latest",
                    "NORAD_CAT_ID": "25544",
                    "format": "json"
                })
            
            success = response.status_code == 200
            logger.info(f"Space-Track connection test: {'SUCCESS' if success else 'FAILED'} - Status: {response.status_code}")
            if not success:
                logger.debug(f"Response content: {response.text[:200]}")
            return success
            
        except Exception as e:
            logger.error(f"Space-Track connection test failed: {e}")
            return False

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
            self.authenticated = False
            logger.debug("Space-Track client closed")