"""Repository pattern for maneuver detection data access."""

from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from src.core.exceptions import DatabaseError
from src.core.logging import get_logger, log_execution_time
from src.data.models import ManeuverDetection

logger = get_logger(__name__)


class ManeuverDetectionRepository:
    """Repository for maneuver detection operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    @log_execution_time("maneuver_detection_create")
    def create(self, detection: ManeuverDetection) -> ManeuverDetection:
        """
        Create a new maneuver detection record.
        
        Args:
            detection: ManeuverDetection object to persist
            
        Returns:
            Created ManeuverDetection object with assigned ID
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            self.session.add(detection)
            self.session.flush()  # Get ID without committing
            
            logger.info(
                "Maneuver detection created",
                norad_id=detection.norad_id,
                detection_datetime=detection.detection_datetime.isoformat(),
                maneuver_detected=detection.maneuver_detected,
                confidence=detection.detection_confidence
            )
            return detection
        except Exception as e:
            logger.error(
                "Failed to create maneuver detection",
                error=str(e),
                norad_id=getattr(detection, 'norad_id', None)
            )
            raise DatabaseError(
                message="Failed to create maneuver detection record",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "norad_id": getattr(detection, 'norad_id', None)
                }
            )
    
    @log_execution_time("maneuver_detection_get_by_id")
    def get_by_id(self, detection_id: int) -> Optional[ManeuverDetection]:
        """
        Retrieve maneuver detection by primary key ID.
        
        Args:
            detection_id: Primary key ID
            
        Returns:
            ManeuverDetection object or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            detection = self.session.query(ManeuverDetection).filter(
                ManeuverDetection.id == detection_id
            ).first()
            
            if detection:
                logger.debug("Maneuver detection retrieved by ID", detection_id=detection_id)
            else:
                logger.debug("Maneuver detection not found by ID", detection_id=detection_id)
            
            return detection
        except Exception as e:
            logger.error("Failed to retrieve maneuver detection by ID", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve maneuver detection by ID",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "detection_id": detection_id}
            )
    
    @log_execution_time("maneuver_detection_get_recent")
    def get_recent_detections(
        self,
        norad_id: Optional[int] = None,
        hours_back: int = 24,
        detected_only: bool = True,
        limit: Optional[int] = 100
    ) -> List[ManeuverDetection]:
        """
        Retrieve recent maneuver detections.
        
        Args:
            norad_id: Specific satellite to query (None for all)
            hours_back: How many hours back to look
            detected_only: Only return positive detections
            limit: Maximum number of results
            
        Returns:
            List of ManeuverDetection objects ordered by detection time
            
        Raises:
            DatabaseError: If query fails
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            query = self.session.query(ManeuverDetection).filter(
                ManeuverDetection.detection_datetime >= cutoff_time
            )
            
            if norad_id:
                query = query.filter(ManeuverDetection.norad_id == norad_id)
            
            if detected_only:
                query = query.filter(ManeuverDetection.maneuver_detected == True)
            
            query = query.order_by(desc(ManeuverDetection.detection_datetime))
            
            if limit:
                query = query.limit(limit)
            
            detections = query.all()
            
            logger.debug(
                "Recent maneuver detections retrieved",
                norad_id=norad_id,
                hours_back=hours_back,
                detected_only=detected_only,
                count=len(detections),
                limit=limit
            )
            
            return detections
        except Exception as e:
            logger.error("Failed to retrieve recent maneuver detections", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve recent maneuver detections",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "norad_id": norad_id,
                    "hours_back": hours_back,
                    "detected_only": detected_only,
                    "limit": limit
                }
            )
    
    @log_execution_time("maneuver_detection_get_positive")
    def get_positive_detections(
        self,
        start_time: datetime,
        end_time: datetime,
        min_confidence: float = 0.5
    ) -> List[ManeuverDetection]:
        """
        Retrieve positive maneuver detections in time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of positive ManeuverDetection objects
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            detections = self.session.query(ManeuverDetection).filter(
                and_(
                    ManeuverDetection.detection_datetime >= start_time,
                    ManeuverDetection.detection_datetime <= end_time,
                    ManeuverDetection.maneuver_detected == True,
                    ManeuverDetection.detection_confidence >= min_confidence
                )
            ).order_by(desc(ManeuverDetection.detection_confidence)).all()
            
            logger.debug(
                "Positive maneuver detections retrieved",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                min_confidence=min_confidence,
                count=len(detections)
            )
            
            return detections
        except Exception as e:
            logger.error("Failed to retrieve positive maneuver detections", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve positive maneuver detections",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "min_confidence": min_confidence
                }
            )
    
    @log_execution_time("maneuver_detection_update_alert_status")
    def update_alert_status(
        self,
        detection_id: int,
        alert_generated: bool = None,
        alert_sent_at: datetime = None
    ) -> bool:
        """
        Update alert status for a maneuver detection.
        
        Args:
            detection_id: Detection ID to update
            alert_generated: New alert generated status
            alert_sent_at: Time alert was sent
            
        Returns:
            True if update successful, False if detection not found
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            update_data = {}
            if alert_generated is not None:
                update_data['alert_generated'] = alert_generated
            if alert_sent_at is not None:
                update_data['alert_sent_at'] = alert_sent_at
            
            if not update_data:
                return True  # Nothing to update
            
            result = self.session.query(ManeuverDetection).filter(
                ManeuverDetection.id == detection_id
            ).update(update_data)
            
            if result > 0:
                logger.info(
                    "Maneuver detection alert status updated",
                    detection_id=detection_id,
                    **update_data
                )
                return True
            else:
                logger.warning("Maneuver detection not found for alert update", detection_id=detection_id)
                return False
                
        except Exception as e:
            logger.error("Failed to update maneuver detection alert status", error=str(e))
            raise DatabaseError(
                message="Failed to update maneuver detection alert status",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "detection_id": detection_id}
            )
    
    @log_execution_time("maneuver_detection_get_statistics")
    def get_statistics(self, days_back: int = 7) -> dict:
        """
        Get maneuver detection statistics.
        
        Args:
            days_back: Number of days back to analyze
            
        Returns:
            Dictionary containing statistics
            
        Raises:
            DatabaseError: If query fails
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Total detections
            total_count = self.session.query(func.count(ManeuverDetection.id)).filter(
                ManeuverDetection.detection_datetime >= cutoff_date
            ).scalar()
            
            # Positive detections
            positive_count = self.session.query(func.count(ManeuverDetection.id)).filter(
                and_(
                    ManeuverDetection.detection_datetime >= cutoff_date,
                    ManeuverDetection.maneuver_detected == True
                )
            ).scalar()
            
            # Average confidence
            avg_confidence = self.session.query(
                func.avg(ManeuverDetection.detection_confidence)
            ).filter(
                ManeuverDetection.detection_datetime >= cutoff_date
            ).scalar()
            
            # Distinct satellites detected
            distinct_satellites = self.session.query(
                func.count(func.distinct(ManeuverDetection.norad_id))
            ).filter(
                and_(
                    ManeuverDetection.detection_datetime >= cutoff_date,
                    ManeuverDetection.maneuver_detected == True
                )
            ).scalar()
            
            stats = {
                "period_days": days_back,
                "total_detections": total_count,
                "positive_detections": positive_count,
                "positive_detection_rate": round((positive_count / total_count * 100) if total_count > 0 else 0, 2),
                "average_confidence": avg_confidence,
                "distinct_satellites_with_maneuvers": distinct_satellites,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.debug("Maneuver detection statistics retrieved", **stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to retrieve maneuver detection statistics", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve maneuver detection statistics",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e)}
            )


# Context manager for repository operations
from contextlib import contextmanager
from src.data.database import db_manager

@contextmanager
def maneuver_detection_repository():
    """Context manager for maneuver detection repository operations."""
    with db_manager.get_session() as session:
        repo = ManeuverDetectionRepository(session)
        yield repo