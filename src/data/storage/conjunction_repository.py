"""Repository pattern for conjunction event data access."""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from src.core.exceptions import DatabaseError, RecordNotFoundError
from src.core.logging import get_logger, log_execution_time
from src.data.models import ConjunctionEvent, TLE
from src.data.storage.tle_repository import TLERepository

logger = get_logger(__name__)


class ConjunctionEventRepository:
    """Repository for conjunction event operations."""
    
    def __init__(self, session: Session):
        self.session = session
        self.tle_repo = TLERepository(session)
    
    @log_execution_time("conjunction_event_create")
    def create(self, event: ConjunctionEvent) -> ConjunctionEvent:
        """
        Create a new conjunction event record.
        
        Args:
            event: ConjunctionEvent object to persist
            
        Returns:
            Created ConjunctionEvent object with assigned ID
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            self.session.add(event)
            self.session.flush()  # Get ID without committing
            
            logger.info(
                "Conjunction event created",
                primary_norad=event.primary_norad_id,
                secondary_norad=event.secondary_norad_id,
                tca=event.tca_datetime.isoformat(),
                pc=event.probability
            )
            return event
        except Exception as e:
            logger.error(
                "Failed to create conjunction event",
                error=str(e),
                primary_norad=getattr(event, 'primary_norad_id', None),
                secondary_norad=getattr(event, 'secondary_norad_id', None)
            )
            raise DatabaseError(
                message="Failed to create conjunction event record",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "primary_norad": getattr(event, 'primary_norad_id', None),
                    "secondary_norad": getattr(event, 'secondary_norad_id', None)
                }
            )
    
    @log_execution_time("conjunction_event_get_by_id")
    def get_by_id(self, event_id: int) -> Optional[ConjunctionEvent]:
        """
        Retrieve conjunction event by primary key ID.
        
        Args:
            event_id: Primary key ID
            
        Returns:
            ConjunctionEvent object or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            event = self.session.query(ConjunctionEvent).filter(
                ConjunctionEvent.id == event_id
            ).first()
            
            if event:
                logger.debug("Conjunction event retrieved by ID", event_id=event_id)
            else:
                logger.debug("Conjunction event not found by ID", event_id=event_id)
            
            return event
        except Exception as e:
            logger.error("Failed to retrieve conjunction event by ID", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve conjunction event by ID",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "event_id": event_id}
            )
    
    @log_execution_time("conjunction_event_get_by_tca")
    def get_by_tca(
        self, 
        primary_norad_id: int, 
        secondary_norad_id: int, 
        tca_datetime: datetime
    ) -> Optional[ConjunctionEvent]:
        """
        Retrieve conjunction event by primary/secondary IDs and TCA.
        
        Args:
            primary_norad_id: Primary object NORAD ID
            secondary_norad_id: Secondary object NORAD ID
            tca_datetime: Time of closest approach
            
        Returns:
            ConjunctionEvent object or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            event = self.session.query(ConjunctionEvent).filter(
                and_(
                    ConjunctionEvent.primary_norad_id == primary_norad_id,
                    ConjunctionEvent.secondary_norad_id == secondary_norad_id,
                    ConjunctionEvent.tca_datetime == tca_datetime
                )
            ).first()
            
            if event:
                logger.debug(
                    "Conjunction event retrieved by TCA",
                    primary_norad=primary_norad_id,
                    secondary_norad=secondary_norad_id,
                    tca=tca_datetime.isoformat()
                )
            else:
                logger.debug(
                    "Conjunction event not found by TCA",
                    primary_norad=primary_norad_id,
                    secondary_norad=secondary_norad_id,
                    tca=tca_datetime.isoformat()
                )
            
            return event
        except Exception as e:
            logger.error("Failed to retrieve conjunction event by TCA", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve conjunction event by TCA",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "primary_norad": primary_norad_id,
                    "secondary_norad": secondary_norad_id,
                    "tca": tca_datetime.isoformat()
                }
            )
    
    @log_execution_time("conjunction_event_get_recent")
    def get_recent_events(
        self, 
        hours_back: int = 24,
        min_probability: float = 1e-6,
        limit: Optional[int] = 100
    ) -> List[ConjunctionEvent]:
        """
        Retrieve recent conjunction events above probability threshold.
        
        Args:
            hours_back: How many hours back to look
            min_probability: Minimum probability threshold
            limit: Maximum number of results
            
        Returns:
            List of ConjunctionEvent objects ordered by TCA
            
        Raises:
            DatabaseError: If query fails
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            query = self.session.query(ConjunctionEvent).filter(
                and_(
                    ConjunctionEvent.tca_datetime >= cutoff_time,
                    ConjunctionEvent.probability >= min_probability
                )
            ).order_by(desc(ConjunctionEvent.tca_datetime))
            
            if limit:
                query = query.limit(limit)
            
            events = query.all()
            
            logger.debug(
                "Recent conjunction events retrieved",
                hours_back=hours_back,
                min_probability=min_probability,
                count=len(events),
                limit=limit
            )
            
            return events
        except Exception as e:
            logger.error("Failed to retrieve recent conjunction events", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve recent conjunction events",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "hours_back": hours_back,
                    "min_probability": min_probability,
                    "limit": limit
                }
            )
    
    @log_execution_time("conjunction_event_get_high_risk")
    def get_high_risk_events(
        self, 
        start_time: datetime, 
        end_time: datetime,
        probability_threshold: float = 1e-3
    ) -> List[ConjunctionEvent]:
        """
        Retrieve high-risk conjunction events in time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            probability_threshold: Probability threshold for "high risk"
            
        Returns:
            List of high-risk ConjunctionEvent objects
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            events = self.session.query(ConjunctionEvent).filter(
                and_(
                    ConjunctionEvent.tca_datetime >= start_time,
                    ConjunctionEvent.tca_datetime <= end_time,
                    ConjunctionEvent.probability >= probability_threshold
                )
            ).order_by(desc(ConjunctionEvent.probability)).all()
            
            logger.debug(
                "High-risk conjunction events retrieved",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                threshold=probability_threshold,
                count=len(events)
            )
            
            return events
        except Exception as e:
            logger.error("Failed to retrieve high-risk conjunction events", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve high-risk conjunction events",
                error_code="DATABASE_CONNECTION_FAILED",
                details={
                    "error": str(e),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "threshold": probability_threshold
                }
            )
    
    @log_execution_time("conjunction_event_update_alert_status")
    def update_alert_status(
        self, 
        event_id: int, 
        alert_generated: bool = None, 
        alert_threshold_exceeded: bool = None,
        alert_sent_at: datetime = None
    ) -> bool:
        """
        Update alert status for a conjunction event.
        
        Args:
            event_id: Event ID to update
            alert_generated: New alert generated status
            alert_threshold_exceeded: New threshold exceeded status
            alert_sent_at: Time alert was sent
            
        Returns:
            True if update successful, False if event not found
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            update_data = {}
            if alert_generated is not None:
                update_data['alert_generated'] = alert_generated
            if alert_threshold_exceeded is not None:
                update_data['alert_threshold_exceeded'] = alert_threshold_exceeded
            if alert_sent_at is not None:
                update_data['alert_sent_at'] = alert_sent_at
            
            if not update_data:
                return True  # Nothing to update
            
            result = self.session.query(ConjunctionEvent).filter(
                ConjunctionEvent.id == event_id
            ).update(update_data)
            
            if result > 0:
                logger.info(
                    "Conjunction event alert status updated",
                    event_id=event_id,
                    **update_data
                )
                return True
            else:
                logger.warning("Conjunction event not found for alert update", event_id=event_id)
                return False
                
        except Exception as e:
            logger.error("Failed to update conjunction event alert status", error=str(e))
            raise DatabaseError(
                message="Failed to update conjunction event alert status",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e), "event_id": event_id}
            )
    
    @log_execution_time("conjunction_event_get_statistics")
    def get_statistics(self, days_back: int = 7) -> dict:
        """
        Get conjunction event statistics.
        
        Args:
            days_back: Number of days back to analyze
            
        Returns:
            Dictionary containing statistics
            
        Raises:
            DatabaseError: If query fails
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Total events
            total_count = self.session.query(func.count(ConjunctionEvent.id)).filter(
                ConjunctionEvent.tca_datetime >= cutoff_date
            ).scalar()
            
            # High-risk events
            high_risk_count = self.session.query(func.count(ConjunctionEvent.id)).filter(
                and_(
                    ConjunctionEvent.tca_datetime >= cutoff_date,
                    ConjunctionEvent.probability >= 1e-3
                )
            ).scalar()
            
            # Medium-risk events
            medium_risk_count = self.session.query(func.count(ConjunctionEvent.id)).filter(
                and_(
                    ConjunctionEvent.tca_datetime >= cutoff_date,
                    ConjunctionEvent.probability >= 1e-6,
                    ConjunctionEvent.probability < 1e-3
                )
            ).scalar()
            
            # Low-risk events
            low_risk_count = self.session.query(func.count(ConjunctionEvent.id)).filter(
                and_(
                    ConjunctionEvent.tca_datetime >= cutoff_date,
                    ConjunctionEvent.probability < 1e-6
                )
            ).scalar()
            
            # Average miss distance
            avg_miss_distance = self.session.query(
                func.avg(ConjunctionEvent.miss_distance_meters)
            ).filter(
                ConjunctionEvent.tca_datetime >= cutoff_date
            ).scalar()
            
            # Average relative velocity
            avg_relative_velocity = self.session.query(
                func.avg(ConjunctionEvent.relative_velocity_mps)
            ).filter(
                ConjunctionEvent.tca_datetime >= cutoff_date
            ).scalar()
            
            stats = {
                "period_days": days_back,
                "total_events": total_count,
                "high_risk_events": high_risk_count,
                "medium_risk_events": medium_risk_count,
                "low_risk_events": low_risk_count,
                "high_risk_percentage": round((high_risk_count / total_count * 100) if total_count > 0 else 0, 2),
                "average_miss_distance_meters": avg_miss_distance,
                "average_relative_velocity_mps": avg_relative_velocity,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.debug("Conjunction event statistics retrieved", **stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to retrieve conjunction event statistics", error=str(e))
            raise DatabaseError(
                message="Failed to retrieve conjunction event statistics",
                error_code="DATABASE_CONNECTION_FAILED",
                details={"error": str(e)}
            )


# Context manager for repository operations
from contextlib import contextmanager
from src.data.database import db_manager

@contextmanager
def conjunction_event_repository():
    """Context manager for conjunction event repository operations."""
    with db_manager.get_session() as session:
        repo = ConjunctionEventRepository(session)
        yield repo