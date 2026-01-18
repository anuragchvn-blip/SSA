"""
Data persistence for historical satellite tracking analysis.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import json

from src.core.logging import get_logger
from src.core.config import settings
from src.data.models import TLE, SatelliteState, ConjunctionEvent
from src.propagation.sgp4_engine import PropagationResult
from src.tracking.multi_satellite_tracker import SatelliteInfo

logger = get_logger(__name__)
Base = declarative_base()


class HistoricalTrackingRecord(Base):
    """Historical record of satellite tracking data."""
    
    __tablename__ = "historical_tracking"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    norad_id: Mapped[int] = mapped_column(Integer, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    
    # Position in ECI frame (meters)
    x_eci: Mapped[float] = mapped_column(Float)
    y_eci: Mapped[float] = mapped_column(Float)
    z_eci: Mapped[float] = mapped_column(Float)
    
    # Velocity in ECI frame (m/s)
    vx_eci: Mapped[float] = mapped_column(Float)
    vy_eci: Mapped[float] = mapped_column(Float)
    vz_eci: Mapped[float] = mapped_column(Float)
    
    # Geographic coordinates
    latitude_deg: Mapped[float] = mapped_column(Float)
    longitude_deg: Mapped[float] = mapped_column(Float)
    altitude_m: Mapped[float] = mapped_column(Float)
    
    # Keplerian elements
    semi_major_axis_m: Mapped[float] = mapped_column(Float)
    eccentricity: Mapped[float] = mapped_column(Float)
    inclination_deg: Mapped[float] = mapped_column(Float)
    raan_deg: Mapped[float] = mapped_column(Float)
    argument_of_perigee_deg: Mapped[float] = mapped_column(Float)
    true_anomaly_deg: Mapped[float] = mapped_column(Float)
    
    # Source information
    tle_id: Mapped[Optional[int]] = mapped_column(Integer)  # Foreign key reference
    propagated_from_epoch: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Quality metrics
    position_uncertainty_m: Mapped[Optional[float]] = mapped_column(Float)
    velocity_uncertainty_mps: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    tracking_session_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    source_system: Mapped[str] = mapped_column(String(50), default="live_tracking")
    
    __table_args__ = (
        Index('idx_hist_tracking_norad_ts', 'norad_id', 'timestamp'),
        Index('idx_hist_tracking_session', 'tracking_session_id'),
    )


class TrackingSession(Base):
    """Record of a tracking session."""
    
    __tablename__ = "tracking_sessions"
    
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    satellites_tracked: Mapped[List[int]] = mapped_column(String)  # JSON encoded list
    status: Mapped[str] = mapped_column(String(20), default="active")  # active, completed, failed
    
    # Statistics
    records_stored: Mapped[int] = mapped_column(Integer, default=0)
    average_update_frequency_hz: Mapped[Optional[float]] = mapped_column(Float)
    total_tracking_duration_minutes: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    created_by: Mapped[Optional[str]] = mapped_column(String(100))
    notes: Mapped[Optional[str]] = mapped_column(String(500))


class HistoricalTrackingRepository:
    """Repository for historical tracking data."""
    
    def __init__(self):
        self.engine = create_engine(
            settings.database.sqlalchemy_database_uri,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def save_tracking_record(self, sat_info: SatelliteInfo, timestamp: datetime, session_id: str = None):
        """Save a single tracking record to history."""
        session = self.SessionLocal()
        try:
            record = HistoricalTrackingRecord(
                norad_id=sat_info.norad_id,
                timestamp=timestamp,
                x_eci=sat_info.current_position['x'] if sat_info.current_position else 0,
                y_eci=sat_info.current_position['y'] if sat_info.current_position else 0,
                z_eci=sat_info.current_position['z'] if sat_info.current_position else 0,
                vx_eci=0,  # Would come from propagation result
                vy_eci=0,
                vz_eci=0,
                latitude_deg=sat_info.current_position['lat'] if sat_info.current_position else 0,
                longitude_deg=sat_info.current_position['lon'] if sat_info.current_position else 0,
                altitude_m=sat_info.current_position['alt'] if sat_info.current_position else 0,
                semi_major_axis_m=0,  # Would come from propagation result
                eccentricity=0,
                inclination_deg=0,
                raan_deg=0,
                argument_of_perigee_deg=0,
                true_anomaly_deg=0,
                tle_id=getattr(sat_info.tle, 'id', None) if sat_info.tle else None,
                propagated_from_epoch=sat_info.tle.epoch_datetime if sat_info.tle else None,
                position_uncertainty_m=None,
                velocity_uncertainty_mps=None,
                tracking_session_id=session_id,
                source_system="live_tracking"
            )
            
            # If we have a propagation result, use more detailed data
            if sat_info.propagation_result:
                prop_result = sat_info.propagation_result
                record.vx_eci = prop_result.cartesian_state.vx
                record.vy_eci = prop_result.cartesian_state.vy
                record.vz_eci = prop_result.cartesian_state.vz
                
                record.semi_major_axis_m = prop_result.keplerian_elements.semi_major_axis
                record.eccentricity = prop_result.keplerian_elements.eccentricity
                record.inclination_deg = prop_result.keplerian_elements.inclination
                record.raan_deg = prop_result.keplerian_elements.raan
                record.argument_of_perigee_deg = prop_result.keplerian_elements.argument_of_perigee
                record.true_anomaly_deg = prop_result.keplerian_elements.true_anomaly
            
            session.add(record)
            session.commit()
            
            logger.debug(f"Saved tracking record for NORAD {sat_info.norad_id} at {timestamp}")
            return record.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving tracking record: {e}")
            raise
        finally:
            session.close()
    
    def save_tracking_records_batch(self, records: List[Tuple[SatelliteInfo, datetime, str]]):
        """Save multiple tracking records in a batch."""
        session = self.SessionLocal()
        try:
            db_records = []
            for sat_info, timestamp, session_id in records:
                record = HistoricalTrackingRecord(
                    norad_id=sat_info.norad_id,
                    timestamp=timestamp,
                    x_eci=sat_info.current_position['x'] if sat_info.current_position else 0,
                    y_eci=sat_info.current_position['y'] if sat_info.current_position else 0,
                    z_eci=sat_info.current_position['z'] if sat_info.current_position else 0,
                    vx_eci=0,
                    vy_eci=0,
                    vz_eci=0,
                    latitude_deg=sat_info.current_position['lat'] if sat_info.current_position else 0,
                    longitude_deg=sat_info.current_position['lon'] if sat_info.current_position else 0,
                    altitude_m=sat_info.current_position['alt'] if sat_info.current_position else 0,
                    semi_major_axis_m=0,
                    eccentricity=0,
                    inclination_deg=0,
                    raan_deg=0,
                    argument_of_perigee_deg=0,
                    true_anomaly_deg=0,
                    tle_id=getattr(sat_info.tle, 'id', None) if sat_info.tle else None,
                    propagated_from_epoch=sat_info.tle.epoch_datetime if sat_info.tle else None,
                    position_uncertainty_m=None,
                    velocity_uncertainty_mps=None,
                    tracking_session_id=session_id,
                    source_system="live_tracking"
                )
                
                # Add detailed data if available
                if sat_info.propagation_result:
                    prop_result = sat_info.propagation_result
                    record.vx_eci = prop_result.cartesian_state.vx
                    record.vy_eci = prop_result.cartesian_state.vy
                    record.vz_eci = prop_result.cartesian_state.vz
                    
                    record.semi_major_axis_m = prop_result.keplerian_elements.semi_major_axis
                    record.eccentricity = prop_result.keplerian_elements.eccentricity
                    record.inclination_deg = prop_result.keplerian_elements.inclination
                    record.raan_deg = prop_result.keplerian_elements.raan
                    record.argument_of_perigee_deg = prop_result.keplerian_elements.argument_of_perigee
                    record.true_anomaly_deg = prop_result.keplerian_elements.true_anomaly
                
                db_records.append(record)
            
            session.add_all(db_records)
            session.commit()
            
            logger.info(f"Saved batch of {len(records)} tracking records")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving batch tracking records: {e}")
            raise
        finally:
            session.close()
    
    def create_tracking_session(self, session_id: str, satellite_ids: List[int], created_by: str = None):
        """Create a new tracking session record."""
        session = self.SessionLocal()
        try:
            tracking_session = TrackingSession(
                id=session_id,
                start_time=datetime.utcnow(),
                satellites_tracked=json.dumps(satellite_ids),
                status="active",
                created_by=created_by
            )
            
            session.add(tracking_session)
            session.commit()
            
            logger.info(f"Created tracking session {session_id} for satellites {satellite_ids}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating tracking session: {e}")
            raise
        finally:
            session.close()
    
    def update_tracking_session(self, session_id: str, **kwargs):
        """Update a tracking session with new information."""
        session = self.SessionLocal()
        try:
            session_record = session.query(TrackingSession).filter(
                TrackingSession.id == session_id
            ).first()
            
            if session_record:
                for key, value in kwargs.items():
                    if hasattr(session_record, key):
                        setattr(session_record, key, value)
                
                session.commit()
                logger.debug(f"Updated tracking session {session_id}")
            else:
                logger.warning(f"Tracking session {session_id} not found")
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating tracking session: {e}")
            raise
        finally:
            session.close()
    
    def get_historical_positions(
        self,
        norad_id: int,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[HistoricalTrackingRecord]:
        """Retrieve historical positions for a satellite within a time range."""
        session = self.SessionLocal()
        try:
            records = session.query(HistoricalTrackingRecord)\
                .filter(
                    HistoricalTrackingRecord.norad_id == norad_id,
                    HistoricalTrackingRecord.timestamp >= start_time,
                    HistoricalTrackingRecord.timestamp <= end_time
                )\
                .order_by(HistoricalTrackingRecord.timestamp)\
                .limit(limit)\
                .all()
            
            logger.info(f"Retrieved {len(records)} historical records for NORAD {norad_id}")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving historical positions: {e}")
            raise
        finally:
            session.close()
    
    def get_tracking_statistics(
        self,
        start_time: datetime,
        end_time: datetime,
        satellite_ids: Optional[List[int]] = None
    ) -> Dict:
        """Get statistics about tracking data in a time range."""
        session = self.SessionLocal()
        try:
            query = session.query(HistoricalTrackingRecord)
            query = query.filter(
                HistoricalTrackingRecord.timestamp >= start_time,
                HistoricalTrackingRecord.timestamp <= end_time
            )
            
            if satellite_ids:
                query = query.filter(HistoricalTrackingRecord.norad_id.in_(satellite_ids))
            
            # Get counts and statistics
            total_records = query.count()
            norad_counts = session.query(
                HistoricalTrackingRecord.norad_id,
                func.count(HistoricalTrackingRecord.id).label('count')
            ).filter(
                HistoricalTrackingRecord.timestamp >= start_time,
                HistoricalTrackingRecord.timestamp <= end_time
            )
            
            if satellite_ids:
                norad_counts = norad_counts.filter(HistoricalTrackingRecord.norad_id.in_(satellite_ids))
            
            norad_counts = norad_counts.group_by(HistoricalTrackingRecord.norad_id).all()
            
            # Calculate average update frequency
            if total_records > 1:
                time_span = (end_time - start_time).total_seconds()
                avg_frequency = total_records / time_span if time_span > 0 else 0
            else:
                avg_frequency = 0
            
            stats = {
                'total_records': total_records,
                'time_span_seconds': (end_time - start_time).total_seconds(),
                'average_frequency_hz': avg_frequency,
                'satellites_tracked': [row.norad_id for row in norad_counts],
                'records_per_satellite': {row.norad_id: row.count for row in norad_counts},
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting tracking statistics: {e}")
            raise
        finally:
            session.close()
    
    def cleanup_old_records(self, retention_days: int = 30):
        """Clean up historical records older than the retention period."""
        session = self.SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            deleted_count = session.query(HistoricalTrackingRecord)\
                .filter(HistoricalTrackingRecord.timestamp < cutoff_date)\
                .delete()
            
            session.commit()
            
            logger.info(f"Cleaned up {deleted_count} historical records older than {retention_days} days")
            return deleted_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old records: {e}")
            raise
        finally:
            session.close()


class HistoricalTrackingService:
    """Service layer for historical tracking data management."""
    
    def __init__(self):
        self.repository = HistoricalTrackingRepository()
        self.active_sessions: Dict[str, datetime] = {}
    
    async def start_tracking_session(
        self,
        session_id: str,
        satellite_ids: List[int],
        created_by: str = None
    ) -> bool:
        """Start a new tracking session."""
        try:
            self.repository.create_tracking_session(session_id, satellite_ids, created_by)
            self.active_sessions[session_id] = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(f"Failed to start tracking session {session_id}: {e}")
            return False
    
    async def save_current_positions(
        self,
        satellite_infos: List[SatelliteInfo],
        session_id: str = None
    ):
        """Save current positions of multiple satellites."""
        if not satellite_infos:
            return
        
        timestamp = datetime.utcnow()
        records = [(info, timestamp, session_id) for info in satellite_infos]
        
        try:
            self.repository.save_tracking_records_batch(records)
        except Exception as e:
            logger.error(f"Failed to save current positions: {e}")
    
    async def save_single_position(
        self,
        sat_info: SatelliteInfo,
        session_id: str = None
    ):
        """Save position of a single satellite."""
        timestamp = datetime.utcnow()
        try:
            self.repository.save_tracking_record(sat_info, timestamp, session_id)
        except Exception as e:
            logger.error(f"Failed to save position for satellite {sat_info.norad_id}: {e}")
    
    async def end_tracking_session(self, session_id: str):
        """End a tracking session and update statistics."""
        if session_id in self.active_sessions:
            start_time = self.active_sessions[session_id]
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            
            # Get record count for this session
            stats = self.repository.get_tracking_statistics(
                start_time=start_time,
                end_time=end_time,
                satellite_ids=None  # Get all for this session
            )
            
            # Update session record
            self.repository.update_tracking_session(
                session_id,
                end_time=end_time,
                status="completed",
                total_tracking_duration_minutes=duration,
                records_stored=stats['total_records'],
                average_update_frequency_hz=stats['average_frequency_hz']
            )
            
            del self.active_sessions[session_id]
            logger.info(f"Ended tracking session {session_id}")
    
    async def get_historical_analysis(
        self,
        satellite_ids: List[int],
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Get historical analysis for specified satellites and time range."""
        # Get statistics
        stats = self.repository.get_tracking_statistics(start_time, end_time, satellite_ids)
        
        # Get sample positions for each satellite
        sample_positions = {}
        for norad_id in satellite_ids:
            records = self.repository.get_historical_positions(
                norad_id, start_time, end_time, limit=100
            )
            
            if records:
                # Get recent positions for trend analysis
                recent_positions = [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'position': {
                            'x': record.x_eci,
                            'y': record.y_eci,
                            'z': record.z_eci,
                            'lat': record.latitude_deg,
                            'lon': record.longitude_deg,
                            'alt': record.altitude_m
                        }
                    }
                    for record in records[-10:]  # Last 10 positions
                ]
                sample_positions[norad_id] = recent_positions
        
        analysis = {
            'period_start': start_time.isoformat(),
            'period_end': end_time.isoformat(),
            'satellite_ids': satellite_ids,
            'tracking_statistics': stats,
            'sample_positions': sample_positions,
            'coverage_percentage': self._calculate_coverage(stats, start_time, end_time),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return analysis
    
    def _calculate_coverage(self, stats: Dict, start_time: datetime, end_time: datetime) -> float:
        """Calculate tracking coverage percentage."""
        if not stats['satellites_tracked']:
            return 0.0
        
        time_span_seconds = (end_time - start_time).total_seconds()
        if time_span_seconds <= 0:
            return 0.0
        
        # Assume ideal tracking would be once per second
        ideal_records = time_span_seconds * len(stats['satellites_tracked'])
        actual_records = stats['total_records']
        
        return min(100.0, (actual_records / ideal_records) * 100) if ideal_records > 0 else 0.0


# Global historical tracking service instance
historical_service = HistoricalTrackingService()