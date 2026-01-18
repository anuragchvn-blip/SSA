"""Automated pipeline for daily TLE updates from Space-Track."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, NamedTuple, Optional
import numpy as np

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.data.storage.tle_repository import TLERepository
from src.data.database import db_manager
from src.data.models import TLE
from src.core.config import settings
from src.core.logging import get_logger, log_execution_time
from src.core.exceptions import DataIngestionError

logger = get_logger(__name__)


class UpdateResult(NamedTuple):
    """Result of TLE update operation."""
    satellites_updated: int
    satellites_failed: int
    stale_tles_flagged: int
    errors: List[dict]
    update_duration: float
    total_processed: int


class TLEUpdatePipeline:
    """
    Automated pipeline for daily TLE updates from Space-Track.
    
    REQUIREMENTS:
    1. Fetch latest TLEs for tracked satellites daily
    2. Validate TLE checksums and orbital parameters
    3. Flag stale TLEs (>72 hours old)
    4. Store with full provenance (fetch time, source, API response)
    5. Handle rate limiting gracefully
    6. Retry failed requests with exponential backoff
    7. Log all operations for audit trail
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter(
            requests_per_hour=settings.spacetrack.spacetrack_rate_limit
        )
        self.client = SpaceTrackClient(self.rate_limiter)
        self.logger = get_logger(__name__)
    
    @log_execution_time("tle_update_pipeline")
    async def update_catalog(self, satellite_list: List[int]) -> UpdateResult:
        """
        Update TLEs for specified satellites.
        
        Args:
            satellite_list: List of NORAD IDs to update
            
        Returns:
            UpdateResult with:
            - satellites_updated: count
            - satellites_failed: count
            - stale_tles_flagged: count
            - errors: List[error details]
            - update_duration: seconds
            
        IMPLEMENTATION:
        1. Batch requests to respect rate limits
        2. Validate each TLE before storage
        3. Update database atomically (transaction per satellite)
        4. Flag conflicts (multiple TLEs for same epoch)
        5. Emit metrics for monitoring
        """
        start_time = datetime.utcnow()
        errors = []
        updated_count = 0
        failed_count = 0
        stale_count = 0
        
        # Batch processing to respect rate limits
        batch_size = 50  # Process in batches of 50 satellites
        semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        
        async def process_satellite_batch(batch: List[int]):
            nonlocal updated_count, failed_count, stale_count, errors
            
            async with semaphore:
                for norad_id in batch:
                    try:
                        # Check if existing TLE is stale
                        with db_manager.get_session() as session:
                            repo = TLERepository(session)
                            latest_tle = repo.get_latest_by_norad_id(norad_id)
                            
                            if latest_tle:
                                age_hours = (datetime.utcnow() - latest_tle.epoch_datetime).total_seconds() / 3600
                                if age_hours > settings.spacetrack.tle_staleness_threshold_hours:
                                    stale_count += 1
                                    self.logger.warning("Stale TLE detected",
                                                      norad_id=norad_id,
                                                      age_hours=age_hours)
                        
                        # Fetch latest TLE from Space-Track
                        async with self.client:
                            tle_data_list = await self.client.download_tles_for_satellite(
                                norad_id=norad_id,
                                start_date=datetime.utcnow() - timedelta(days=7),
                                end_date=datetime.utcnow()
                            )
                        
                        if not tle_data_list:
                            self.logger.debug("No TLE data found", norad_id=norad_id)
                            continue
                        
                        # Get most recent TLE
                        latest_tle_data = tle_data_list[-1]  # Most recent
                        
                        # Validate TLE
                        if not self._validate_tle_data(latest_tle_data):
                            error_detail = {
                                "norad_id": norad_id,
                                "error": "TLE validation failed",
                                "details": latest_tle_data
                            }
                            errors.append(error_detail)
                            failed_count += 1
                            continue
                        
                        # Convert to TLE model
                        tle_model = self._convert_tle_data_to_model(latest_tle_data)
                        
                        # Store in database
                        with db_manager.get_session() as session:
                            repo = TLERepository(session)
                            
                            # Check for duplicates/conflicts
                            existing = repo.get_by_norad_and_epoch(
                                norad_id=tle_model.norad_id,
                                epoch=tle_model.epoch_datetime
                            )
                            
                            if existing:
                                self.logger.debug("Duplicate TLE found, skipping",
                                                norad_id=norad_id,
                                                epoch=tle_model.epoch_datetime.isoformat())
                                continue
                            
                            # Store new TLE
                            created_tle = repo.create(tle_model)
                            session.commit()
                            
                            self.logger.info("TLE updated successfully",
                                           norad_id=norad_id,
                                           epoch=created_tle.epoch_datetime.isoformat())
                            updated_count += 1
                            
                    except Exception as e:
                        error_detail = {
                            "norad_id": norad_id,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        errors.append(error_detail)
                        failed_count += 1
                        self.logger.error("TLE update failed", norad_id=norad_id, error=str(e))
        
        # Process satellites in batches
        try:
            tasks = []
            for i in range(0, len(satellite_list), batch_size):
                batch = satellite_list[i:i + batch_size]
                task = process_satellite_batch(batch)
                tasks.append(task)
            
            # Execute all batch tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error("Batch processing failed", error=str(e))
            raise DataIngestionError(
                message=f"TLE update pipeline failed: {str(e)}",
                error_code="TLE_UPDATE_PIPELINE_FAILED",
                details={"satellite_count": len(satellite_list), "error": str(e)}
            )
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        result = UpdateResult(
            satellites_updated=updated_count,
            satellites_failed=failed_count,
            stale_tles_flagged=stale_count,
            errors=errors,
            update_duration=duration,
            total_processed=len(satellite_list)
        )
        
        self.logger.info("TLE update pipeline completed",
                        satellites_updated=updated_count,
                        satellites_failed=failed_count,
                        stale_tles=stale_count,
                        duration_seconds=duration)
        
        return result
    
    def _validate_tle_data(self, tle_data: dict) -> bool:
        """
        Validate TLE data from Space-Track API.
        
        Checks:
        - Required fields present
        - Epoch is reasonable (not future dated)
        - Orbital elements are physically plausible
        - Checksum validation
        """
        required_fields = [
            "NORAD_CAT_ID", "EPOCH", "MEAN_MOTION", "ECCENTRICITY",
            "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER",
            "MEAN_ANOMALY", "TLE_LINE1", "TLE_LINE2"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in tle_data or not tle_data[field]:
                self.logger.warning("Missing required TLE field", field=field)
                return False
        
        # Validate epoch
        try:
            epoch_str = tle_data["EPOCH"]
            epoch = datetime.fromisoformat(epoch_str.rstrip('Z'))
            
            # Check if epoch is reasonable (within last 30 days and not future)
            now = datetime.utcnow()
            if epoch > now or (now - epoch).days > 30:
                self.logger.warning("TLE epoch unreasonable", epoch=epoch_str)
                return False
        except Exception as e:
            self.logger.warning("Invalid epoch format", error=str(e))
            return False
        
        # Validate orbital elements
        try:
            ecc = float(tle_data["ECCENTRICITY"])
            inc = float(tle_data["INCLINATION"])
            mm = float(tle_data["MEAN_MOTION"])
            
            # Physical plausibility checks
            if not (0 <= ecc < 1.0):  # Eccentricity must be [0,1)
                return False
            if not (0 <= inc <= 180):  # Inclination must be [0,180] degrees
                return False
            if not (0 < mm < 20):     # Mean motion reasonable for Earth orbit
                return False
                
        except (ValueError, TypeError):
            return False
        
        # Validate TLE lines format and checksums
        line1 = tle_data["TLE_LINE1"]
        line2 = tle_data["TLE_LINE2"]
        
        if not self._validate_tle_checksums([line1, line2]):
            return False
        
        return True
    
    def _validate_tle_checksums(self, tle_lines: List[str]) -> bool:
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
        
        if len(tle_lines) < 2:
            return False
            
        line1 = tle_lines[0]
        line2 = tle_lines[1]
        
        # Check lengths
        if len(line1) < 69 or len(line2) < 69:
            return False
            
        # Check checksums
        try:
            expected_checksum1 = int(line1[68])
            expected_checksum2 = int(line2[68])
            
            actual_checksum1 = compute_checksum(line1)
            actual_checksum2 = compute_checksum(line2)
            
            return (actual_checksum1 == expected_checksum1 and 
                   actual_checksum2 == expected_checksum2)
        except (ValueError, IndexError):
            return False
    
    def _convert_tle_data_to_model(self, tle_data: dict) -> TLE:
        """
        Convert Space-Track API TLE data to internal TLE model.
        
        Handles field mapping and data type conversion.
        """
        # Parse international designator
        intl_des = tle_data.get("INTLDES", "00000A")
        launch_year = int(intl_des[:2]) if len(intl_des) >= 2 else 0
        launch_number = int(intl_des[2:5]) if len(intl_des) >= 5 else 0
        launch_piece = intl_des[5:] if len(intl_des) > 5 else "A"
        
        # Parse epoch
        epoch_str = tle_data["EPOCH"]
        epoch_dt = datetime.fromisoformat(epoch_str.rstrip('Z'))
        
        # Create TLE model
        tle = TLE(
            norad_id=int(tle_data["NORAD_CAT_ID"]),
            classification=tle_data.get("CLASSIFICATION_TYPE", "U"),
            launch_year=launch_year,
            launch_number=launch_number,
            launch_piece=launch_piece,
            epoch_datetime=epoch_dt,
            mean_motion_derivative=float(tle_data.get("MEAN_MOTION_DOT", 0.0)),
            mean_motion_sec_derivative=float(tle_data.get("MEAN_MOTION_DDOT", 0.0)),
            bstar_drag_term=float(tle_data.get("BSTAR", 0.0)),
            element_set_number=int(tle_data.get("ELEMENT_SET_NO", 1)),
            inclination_degrees=float(tle_data["INCLINATION"]),
            raan_degrees=float(tle_data["RA_OF_ASC_NODE"]),
            eccentricity=float(tle_data["ECCENTRICITY"]),
            argument_of_perigee_degrees=float(tle_data["ARG_OF_PERICENTER"]),
            mean_anomaly_degrees=float(tle_data["MEAN_ANOMALY"]),
            mean_motion_orbits_per_day=float(tle_data["MEAN_MOTION"]),
            revolution_number_at_epoch=int(tle_data.get("REV_AT_EPOCH", 1)),
            tle_line1=tle_data["TLE_LINE1"],
            tle_line2=tle_data["TLE_LINE2"],
            epoch_julian_date=2451545.0,  # Placeholder - would compute properly
            line1_checksum=0,  # Will compute from line
            line2_checksum=0,  # Will compute from line
            is_valid=True
        )
        
        # Compute checksums
        tle.line1_checksum = self._compute_single_checksum(tle.tle_line1)
        tle.line2_checksum = self._compute_single_checksum(tle.tle_line2)
        
        return tle
    
    def _compute_single_checksum(self, line: str) -> int:
        """Compute checksum for a single TLE line."""
        if len(line) < 69:
            return 0
            
        checksum = 0
        for char in line[:68]:
            if char.isdigit():
                checksum += int(char)
            elif char.isalpha():
                checksum += 1
            # Other characters contribute 0
        
        return checksum % 10
    
    async def update_active_satellite_catalog(self) -> UpdateResult:
        """
        Update TLEs for all actively tracked satellites.
        
        This would typically query a master catalog or use a predefined list
        of high-priority satellites for conjunction analysis.
        """
        # Example satellite list - in practice this would come from a catalog
        priority_satellites = [
            25544,  # ISS
            42982,  # Starlink example
            43800,  # OneWeb example
            44231,  # Iridium NEXT example
            # Add more based on operational requirements
        ]
        
        self.logger.info("Updating active satellite catalog",
                        satellite_count=len(priority_satellites))
        
        return await self.update_catalog(priority_satellites)


# Example usage
async def run_daily_update():
    """Run daily TLE update pipeline."""
    pipeline = TLEUpdatePipeline()
    
    # Update active catalog
    result = await pipeline.update_active_satellite_catalog()
    
    print(f"Daily TLE update completed:")
    print(f"  Satellites updated: {result.satellites_updated}")
    print(f"  Satellites failed: {result.satellites_failed}")
    print(f"  Stale TLEs flagged: {result.stale_tles_flagged}")
    print(f"  Duration: {result.update_duration:.2f} seconds")
    
    return result


if __name__ == "__main__":
    # Run the update pipeline
    asyncio.run(run_daily_update())