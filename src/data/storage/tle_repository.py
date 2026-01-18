"""Repository pattern for TLE data access with full CRUD operations."""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import Session

from src.core.exceptions import DatabaseError, RecordNotFoundError
from src.core.logging import get_logger, log_execution_time
from src.data.models import TLE, SatelliteState
from src.data.database import db_manager

logger = get_logger(__name__)


class TLERepository:
    """Repository for TLE data operations with audit trail."""
    
    def __init__(self, session: Session):
        self.session = session
    
    @log_execution_time("tle_create")
    def create(self, tle: TLE) -> TLE:
        """
        Create a new TLE record.
        
        Args:
            tle: TLE object to persist
            
        Returns:
            Created TLE object with assigned ID
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            self.session.add(tle)
            self.session.flush()  # Get ID without committing
            logger.info("TLE created", 
                       norad_id=tle.norad_id, 
                       epoch=tle.epoch_datetime.isoformat())
            return tle
        except Exception as e:
            logger.error("Failed to create TLE", error=str(e))
            raise DatabaseError(
                message="Failed to create TLE record",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "norad_id": getattr(tle, 'norad_id', None)}
            )
    
    @log_execution_time("tle_bulk_create")
    def bulk_create(self, tles: List[TLE]) -> List[TLE]:
        """
        Bulk create multiple TLE records efficiently.
        
        Args:
            tles: List of TLE objects to persist
            
        Returns:
            List of created TLE objects
            
        Raises:
            DatabaseError: If bulk creation fails
        """
        if not tles:
            return []
            
        try:
            self.session.bulk_save_objects(tles)
            self.session.flush()
            
            logger.info("Bulk TLE creation completed",
                       count=len(tles),
                       norad_ids=[tle.norad_id for tle in tles[:5]])  # Log first 5
            
            return tles
        except Exception as e:
            logger.error("Failed to bulk create TLEs", error=str(e))
            raise DatabaseError(
                message="Failed to bulk create TLE records",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "count": len(tles)}
            )
    
    def get_by_id(self, tle_id: int) -> Optional[TLE]:
        """
        Retrieve TLE by primary key ID.
        
        Args:
            tle_id: Primary key ID
            
        Returns:
            TLE object or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            tle = self.session.query(TLE).filter(TLE.id == tle_id).first()
            if tle:
                logger.debug("TLE retrieved by ID", tle_id=tle_id)
            else:
                logger.debug("TLE not found by ID", tle_id=tle_id)
            return tle
        except Exception as e:
            logger.error("Failed to retrieve TLE by ID", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve TLE by ID",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "tle_id": tle_id}
            )
    
    def get_by_norad_and_epoch(self, norad_id: int, epoch: datetime) -> Optional[TLE]:
        """
        Retrieve TLE by NORAD ID and exact epoch.
        
        Args:
            norad_id: NORAD catalog number
            epoch: Exact epoch datetime
            
        Returns:
            TLE object or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            tle = self.session.query(TLE).filter(
                and_(
                    TLE.norad_id == norad_id,
                    TLE.epoch_datetime == epoch
                )
            ).first()
            
            if tle:
                logger.debug("TLE retrieved by NORAD and epoch",
                           norad_id=norad_id,
                           epoch=epoch.isoformat())
            else:
                logger.debug("TLE not found by NORAD and epoch",
                           norad_id=norad_id,
                           epoch=epoch.isoformat())
            
            return tle
        except Exception as e:
            logger.error("Failed to retrieve TLE by NORAD and epoch", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve TLE by NORAD and epoch",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "norad_id": norad_id, "epoch": epoch.isoformat()}
            )
    
    def get_latest_tle(self, norad_id: int) -> Optional[TLE]:
        """
        Retrieve the latest TLE for a given NORAD ID.
        
        Args:
            norad_id: NORAD catalog number
            
        Returns:
            Latest TLE object or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            tle = self.session.query(TLE).filter(
                TLE.norad_id == norad_id
            ).order_by(desc(TLE.epoch_datetime)).first()
            
            if tle:
                logger.debug("Latest TLE retrieved",
                           norad_id=norad_id,
                           epoch=tle.epoch_datetime.isoformat())
            else:
                logger.debug("No TLE found for NORAD ID", norad_id=norad_id)
            
            return tle
        except Exception as e:
            logger.error("Failed to retrieve latest TLE", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve latest TLE",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "norad_id": norad_id}
            )
    
    def get_tles_in_time_range(
        self, 
        norad_id: int, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[TLE]:
        """
        Retrieve all TLEs for a satellite within a time range.
        
        Args:
            norad_id: NORAD catalog number
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            
        Returns:
            List of TLE objects ordered by epoch
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            tles = self.session.query(TLE).filter(
                and_(
                    TLE.norad_id == norad_id,
                    TLE.epoch_datetime >= start_time,
                    TLE.epoch_datetime <= end_time
                )
            ).order_by(TLE.epoch_datetime).all()
            
            logger.debug("TLEs retrieved in time range",
                        norad_id=norad_id,
                        count=len(tles),
                        start_time=start_time.isoformat(),
                        end_time=end_time.isoformat())
            
            return tles
        except Exception as e:
            logger.error("Failed to retrieve TLEs in time range", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve TLEs in time range",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "norad_id": norad_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
            )
    
    def get_recent_tles(
        self, 
        hours_back: int = 24, 
        limit: Optional[int] = None
    ) -> List[TLE]:
        """
        Retrieve recently acquired TLEs.
        
        Args:
            hours_back: How many hours back to look
            limit: Maximum number of results
            
        Returns:
            List of recent TLE objects
            
        Raises:
            DatabaseError: If query fails
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        try:
            query = self.session.query(TLE).filter(
                TLE.created_at >= cutoff_time
            ).order_by(desc(TLE.created_at))
            
            if limit:
                query = query.limit(limit)
            
            tles = query.all()
            
            logger.debug("Recent TLEs retrieved",
                        hours_back=hours_back,
                        count=len(tles),
                        limit=limit)
            
            return tles
        except Exception as e:
            logger.error("Failed to retrieve recent TLEs", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve recent TLEs",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "hours_back": hours_back, "limit": limit}
            )
    
    def get_invalid_tles(self, limit: Optional[int] = None) -> List[TLE]:
        """
        Retrieve TLEs marked as invalid.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of invalid TLE objects
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            query = self.session.query(TLE).filter(TLE.is_valid == False)
            
            if limit:
                query = query.limit(limit)
            
            tles = query.all()
            
            logger.debug("Invalid TLEs retrieved", count=len(tles), limit=limit)
            return tles
        except Exception as e:
            logger.error("Failed to retrieve invalid TLEs", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve invalid TLEs",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "limit": limit}
            )
    
    def update_validation_status(
        self, 
        tle_id: int, 
        is_valid: bool, 
        validation_errors: Optional[List[str]] = None
    ) -> bool:
        """
        Update TLE validation status.
        
        Args:
            tle_id: TLE primary key ID
            is_valid: Validation result
            validation_errors: List of validation error messages
            
        Returns:
            True if update successful, False if TLE not found
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            result = self.session.query(TLE).filter(TLE.id == tle_id).update({
                TLE.is_valid: is_valid,
                TLE.validation_errors: validation_errors,
                TLE.updated_at: datetime.now(timezone.utc)
            })
            
            if result > 0:
                logger.info("TLE validation status updated",
                           tle_id=tle_id,
                           is_valid=is_valid,
                           error_count=len(validation_errors) if validation_errors else 0)
                return True
            else:
                logger.warning("TLE not found for validation update", tle_id=tle_id)
                return False
                
        except Exception as e:
            logger.error("Failed to update TLE validation status", error=str(e))
            raise DatabaseError(
                message="Failed to update TLE validation status",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "tle_id": tle_id}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get TLE database statistics.
        
        Returns:
            Dictionary containing various statistics
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            total_count = self.session.query(func.count(TLE.id)).scalar()
            valid_count = self.session.query(func.count(TLE.id)).filter(TLE.is_valid == True).scalar()
            invalid_count = total_count - valid_count
            
            # Recent acquisitions
            day_ago = datetime.now(timezone.utc) - timedelta(days=1)
            recent_count = self.session.query(func.count(TLE.id)).filter(
                TLE.created_at >= day_ago
            ).scalar()
            
            # Unique satellites
            unique_satellites = self.session.query(func.count(func.distinct(TLE.norad_id))).scalar()
            
            stats = {
                "total_tles": total_count,
                "valid_tles": valid_count,
                "invalid_tles": invalid_count,
                "valid_percentage": round((valid_count / total_count * 100), 2) if total_count > 0 else 0,
                "recent_24h_count": recent_count,
                "unique_satellites": unique_satellites,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.debug("TLE statistics retrieved", **stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to retrieve TLE statistics", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve TLE statistics",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e)}
            )
    
    def delete_old_tles(self, older_than_days: int) -> int:
        """
        Delete TLEs older than specified days (maintenance operation).
        
        Args:
            older_than_days: Age threshold in days
            
        Returns:
            Number of TLEs deleted
            
        Raises:
            DatabaseError: If deletion fails
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        
        try:
            deleted_count = self.session.query(TLE).filter(
                TLE.epoch_datetime < cutoff_date
            ).delete(synchronize_session=False)
            
            logger.info("Old TLEs deleted",
                       deleted_count=deleted_count,
                       older_than_days=older_than_days)
            
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to delete old TLEs", error=str(e))
            raise DatabaseError(
                message="Failed to delete old TLEs",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "older_than_days": older_than_days}
            )


# Convenience functions for common operations
def get_tle_repository(session: Session) -> TLERepository:
    """Factory function to get TLE repository."""
    return TLERepository(session)


# Context manager for repository operations
from contextlib import contextmanager

@contextmanager
def tle_repository():
    """Context manager for TLE repository operations."""
    with db_manager.get_session() as session:
        repo = TLERepository(session)
        yield repo