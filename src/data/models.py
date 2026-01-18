"""Database models for SSA conjunction analysis engine."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Boolean, 
    ForeignKey, Index, UniqueConstraint, Text, JSON
)
from sqlalchemy.orm import relationship, mapped_column, Mapped
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


class ProvenanceMixin:
    """Mixin for data provenance tracking."""
    
    source_url: Mapped[Optional[str]] = mapped_column(String(500))
    acquisition_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    data_version: Mapped[Optional[str]] = mapped_column(String(50))


class TLE(Base, TimestampMixin, ProvenanceMixin):
    """Two-Line Element set storage with full provenance."""
    
    __tablename__ = "tle"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    norad_id: Mapped[int] = mapped_column(Integer, index=True)
    classification: Mapped[str] = mapped_column(String(1))  # U, C, S
    launch_year: Mapped[int] = mapped_column(Integer)
    launch_number: Mapped[int] = mapped_column(Integer)
    launch_piece: Mapped[str] = mapped_column(String(3))
    epoch_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    mean_motion_derivative: Mapped[float] = mapped_column(Float)
    mean_motion_sec_derivative: Mapped[float] = mapped_column(Float)
    bstar_drag_term: Mapped[float] = mapped_column(Float)
    element_set_number: Mapped[int] = mapped_column(Integer)
    inclination_degrees: Mapped[float] = mapped_column(Float)
    raan_degrees: Mapped[float] = mapped_column(Float)
    eccentricity: Mapped[float] = mapped_column(Float)
    argument_of_perigee_degrees: Mapped[float] = mapped_column(Float)
    mean_anomaly_degrees: Mapped[float] = mapped_column(Float)
    mean_motion_orbits_per_day: Mapped[float] = mapped_column(Float)
    revolution_number_at_epoch: Mapped[int] = mapped_column(Integer)
    tle_line1: Mapped[str] = mapped_column(Text)
    tle_line2: Mapped[str] = mapped_column(Text)
    epoch_julian_date: Mapped[float] = mapped_column(Float)
    
    # Checksums for validation
    line1_checksum: Mapped[int] = mapped_column(Integer)
    line2_checksum: Mapped[int] = mapped_column(Integer)
    
    # Processing metadata
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    validation_errors: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    __table_args__ = (
        UniqueConstraint('norad_id', 'epoch_datetime', name='unique_satellite_epoch'),
        Index('idx_tle_norad_epoch', 'norad_id', 'epoch_datetime'),
        Index('idx_tle_epoch_datetime', 'epoch_datetime'),
    )
    
    def __repr__(self):
        return f"<TLE(norad_id={self.norad_id}, epoch={self.epoch_datetime})>"


class SatelliteState(Base, TimestampMixin):
    """Propagated satellite state with covariance at specific epoch."""
    
    __tablename__ = "satellite_state"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tle_id: Mapped[int] = mapped_column(ForeignKey("tle.id"), index=True)
    epoch_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    
    # Cartesian state vector (ECI frame) - meters, m/s
    x_eci: Mapped[float] = mapped_column(Float)
    y_eci: Mapped[float] = mapped_column(Float)
    z_eci: Mapped[float] = mapped_column(Float)
    vx_eci: Mapped[float] = mapped_column(Float)
    vy_eci: Mapped[float] = mapped_column(Float)
    vz_eci: Mapped[float] = mapped_column(Float)
    
    # Covariance matrix (6x6 in RTN frame) - stored as flattened array
    covariance_rtn_flat: Mapped[List[float]] = mapped_column(JSON)
    
    # Keplerian elements (derived)
    semi_major_axis_meters: Mapped[float] = mapped_column(Float)
    eccentricity: Mapped[float] = mapped_column(Float)
    inclination_degrees: Mapped[float] = mapped_column(Float)
    raan_degrees: Mapped[float] = mapped_column(Float)
    argument_of_perigee_degrees: Mapped[float] = mapped_column(Float)
    true_anomaly_degrees: Mapped[float] = mapped_column(Float)
    
    # Propagation metadata
    propagator_used: Mapped[str] = mapped_column(String(50))  # SGP4, Orekit, etc.
    force_models: Mapped[Optional[List[str]]] = mapped_column(JSON)
    step_size_seconds: Mapped[Optional[float]] = mapped_column(Float)
    convergence_tolerance: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    tle: Mapped["TLE"] = relationship("TLE", back_populates="states")
    
    __table_args__ = (
        Index('idx_satellite_state_tle_epoch', 'tle_id', 'epoch_datetime'),
        Index('idx_satellite_state_epoch', 'epoch_datetime'),
    )
    
    def __repr__(self):
        return f"<SatelliteState(tle_id={self.tle_id}, epoch={self.epoch_datetime})>"


# Add back-populates to TLE
TLE.states = relationship("SatelliteState", back_populates="tle", cascade="all, delete-orphan")


class ConjunctionEvent(Base, TimestampMixin):
    """Detected conjunction event with full analysis results."""
    
    __tablename__ = "conjunction_event"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    primary_norad_id: Mapped[int] = mapped_column(Integer, index=True)
    secondary_norad_id: Mapped[int] = mapped_column(Integer, index=True)
    tca_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    
    # TCA state vectors
    primary_x_eci: Mapped[float] = mapped_column(Float)
    primary_y_eci: Mapped[float] = mapped_column(Float)
    primary_z_eci: Mapped[float] = mapped_column(Float)
    primary_vx_eci: Mapped[float] = mapped_column(Float)
    primary_vy_eci: Mapped[float] = mapped_column(Float)
    primary_vz_eci: Mapped[float] = mapped_column(Float)
    secondary_x_eci: Mapped[float] = mapped_column(Float)
    secondary_y_eci: Mapped[float] = mapped_column(Float)
    secondary_z_eci: Mapped[float] = mapped_column(Float)
    secondary_vx_eci: Mapped[float] = mapped_column(Float)
    secondary_vy_eci: Mapped[float] = mapped_column(Float)
    secondary_vz_eci: Mapped[float] = mapped_column(Float)
    
    # Relative motion at TCA
    miss_distance_meters: Mapped[float] = mapped_column(Float, index=True)
    relative_velocity_mps: Mapped[float] = mapped_column(Float)
    
    # Probability of collision
    probability: Mapped[float] = mapped_column(Float, index=True)
    probability_method: Mapped[str] = mapped_column(String(50))  # monte_carlo, foster, akella
    probability_confidence_lower: Mapped[Optional[float]] = mapped_column(Float)
    probability_confidence_upper: Mapped[Optional[float]] = mapped_column(Float)
    probability_samples: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Screening parameters
    screening_threshold_km: Mapped[float] = mapped_column(Float)
    time_window_hours: Mapped[float] = mapped_column(Float)
    
    # Object metadata
    primary_object_name: Mapped[Optional[str]] = mapped_column(String(100))
    secondary_object_name: Mapped[Optional[str]] = mapped_column(String(100))
    primary_object_type: Mapped[Optional[str]] = mapped_column(String(50))  # payload, rocket_body, debris
    secondary_object_type: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Hardbody radii (meters)
    primary_radius_meters: Mapped[float] = mapped_column(Float)
    secondary_radius_meters: Mapped[float] = mapped_column(Float)
    
    # Alerting
    alert_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_threshold_exceeded: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Analysis metadata
    analysis_version: Mapped[str] = mapped_column(String(50))
    algorithm_parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    
    __table_args__ = (
        UniqueConstraint('primary_norad_id', 'secondary_norad_id', 'tca_datetime', 
                        name='unique_conjunction_tca'),
        Index('idx_conjunction_tca_datetime', 'tca_datetime'),
        Index('idx_conjunction_probability', 'probability'),
        Index('idx_conjunction_miss_distance', 'miss_distance_meters'),
    )
    
    def __repr__(self):
        return (f"<ConjunctionEvent(primary={self.primary_norad_id}, "
                f"secondary={self.secondary_norad_id}, tca={self.tca_datetime}, "
                f"Pc={self.probability:.2e})>")


class ManeuverDetection(Base, TimestampMixin):
    """Maneuver detection results from ML analysis."""
    
    __tablename__ = "maneuver_detection"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    norad_id: Mapped[int] = mapped_column(Integer, index=True)
    detection_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    
    # Detection results
    maneuver_detected: Mapped[bool] = mapped_column(Boolean)
    detection_confidence: Mapped[float] = mapped_column(Float)
    detection_method: Mapped[str] = mapped_column(String(50))  # random_forest, threshold_based
    
    # Feature values that triggered detection
    position_residual_magnitude: Mapped[Optional[float]] = mapped_column(Float)
    velocity_residual_magnitude: Mapped[Optional[float]] = mapped_column(Float)
    along_track_residual: Mapped[Optional[float]] = mapped_column(Float)
    cross_track_residual: Mapped[Optional[float]] = mapped_column(Float)
    radial_residual: Mapped[Optional[float]] = mapped_column(Float)
    days_since_last_tle: Mapped[Optional[float]] = mapped_column(Float)
    
    # Orbital element changes
    sma_change_meters: Mapped[Optional[float]] = mapped_column(Float)
    eccentricity_change: Mapped[Optional[float]] = mapped_column(Float)
    inclination_change_degrees: Mapped[Optional[float]] = mapped_column(Float)
    
    # Model metadata
    model_version: Mapped[str] = mapped_column(String(50))
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON)
    shap_values: Mapped[Optional[List[float]]] = mapped_column(JSON)
    
    # Alerting
    alert_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    __table_args__ = (
        Index('idx_maneuver_detection_norad_datetime', 'norad_id', 'detection_datetime'),
        Index('idx_maneuver_detection_datetime', 'detection_datetime'),
    )
    
    def __repr__(self):
        return (f"<ManeuverDetection(norad_id={self.norad_id}, "
                f"detection_datetime={self.detection_datetime}, "
                f"detected={self.maneuver_detected})>")


class AlertHistory(Base, TimestampMixin):
    """History of all alerts generated by the system."""
    
    __tablename__ = "alert_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_type: Mapped[str] = mapped_column(String(50), index=True)  # conjunction, maneuver, system
    severity: Mapped[str] = mapped_column(String(20))  # critical, high, medium, low
    recipient: Mapped[str] = mapped_column(String(100))  # email, webhook, etc.
    
    # Alert content
    subject: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    payload: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Delivery tracking
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    delivery_confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=1)
    
    # Reference to triggering event
    conjunction_event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("conjunction_event.id"))
    maneuver_detection_id: Mapped[Optional[int]] = mapped_column(ForeignKey("maneuver_detection.id"))
    
    __table_args__ = (
        Index('idx_alert_history_sent_at', 'sent_at'),
        Index('idx_alert_history_alert_type', 'alert_type'),
    )
    
    def __repr__(self):
        return (f"<AlertHistory(alert_type={self.alert_type}, "
                f"severity={self.severity}, sent_at={self.sent_at})>")