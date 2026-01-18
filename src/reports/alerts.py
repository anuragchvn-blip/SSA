"""Alert generation and management system for conjunction analysis."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

from src.core.logging import get_logger, log_execution_time
from src.core.exceptions import BaseSSAException
from src.data.models import ConjunctionEvent, ManeuverDetection, AlertHistory
from src.data.storage.conjunction_repository import ConjunctionEventRepository
from src.data.storage.maneuver_repository import ManeuverDetectionRepository

logger = get_logger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alert generation."""
    pc_threshold_high: float = 1e-3  # High risk
    pc_threshold_medium: float = 1e-4  # Medium risk
    pc_threshold_low: float = 1e-6    # Low risk
    miss_distance_threshold: float = 1000.0  # meters
    relative_velocity_threshold: float = 1000.0  # m/s
    maneuver_confidence_threshold: float = 0.7
    recipients: List[str] = None
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""


class AlertGenerationError(BaseSSAException):
    """Exception for alert generation errors."""
    pass


class AlertGenerator:
    """System for generating and sending alerts based on conjunction and maneuver events."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    @log_execution_time("alert_generator_process_conjunction")
    def process_conjunction_event(
        self,
        event: ConjunctionEvent,
        repo: ConjunctionEventRepository
    ) -> List[AlertHistory]:
        """
        Process a conjunction event and generate alerts if thresholds are exceeded.
        
        Args:
            event: Conjunction event to process
            repo: Repository for database operations
            
        Returns:
            List of alert history records created
        """
        alerts_created = []
        
        # Determine alert severity based on probability and other factors
        severity = self._determine_conjunction_severity(event)
        
        if severity == "LOW" and event.probability < self.config.pc_threshold_low:
            # No alert needed for very low probability events
            logger.debug(
                "Conjunction event below alert threshold",
                event_id=event.id,
                pc=event.probability
            )
            return []
        
        # Generate alert content
        alert_subject, alert_body = self._generate_conjunction_alert_content(event, severity)
        
        # Create alert history record
        alert_history = AlertHistory(
            alert_type="conjunction",
            severity=severity,
            recipient="default_recipient@example.com",  # Will be configurable
            subject=alert_subject,
            body=alert_body,
            payload={
                "event_id": event.id,
                "primary_norad_id": event.primary_norad_id,
                "secondary_norad_id": event.secondary_norad_id,
                "tca_datetime": event.tca_datetime.isoformat(),
                "miss_distance_meters": event.miss_distance_meters,
                "probability": event.probability,
                "relative_velocity_mps": event.relative_velocity_mps,
                "severity": severity
            },
            sent_at=datetime.utcnow(),
            conjunction_event_id=event.id
        )
        
        # Save alert to database
        try:
            # We'll need a session to save this, so we'll need to pass that in
            # For now, just return the alert object to be saved by caller
            alerts_created.append(alert_history)
            
            # Update event to mark that alert was generated
            repo.update_alert_status(
                event_id=event.id,
                alert_generated=True,
                alert_threshold_exceeded=True
            )
            
            logger.info(
                "Conjunction alert generated",
                event_id=event.id,
                severity=severity,
                pc=event.probability
            )
            
        except Exception as e:
            logger.error(
                "Failed to create conjunction alert",
                event_id=event.id,
                error=str(e)
            )
            raise AlertGenerationError(
                message=f"Failed to create conjunction alert: {str(e)}",
                error_code="ALERT_GENERATION_FAILED"
            )
        
        return alerts_created
    
    @log_execution_time("alert_generator_process_maneuver")
    def process_maneuver_detection(
        self,
        detection: ManeuverDetection,
        repo: ManeuverDetectionRepository
    ) -> List[AlertHistory]:
        """
        Process a maneuver detection and generate alerts if confidence is high enough.
        
        Args:
            detection: Maneuver detection to process
            repo: Repository for database operations
            
        Returns:
            List of alert history records created
        """
        alerts_created = []
        
        # Check if confidence exceeds threshold
        if detection.detection_confidence < self.config.maneuver_confidence_threshold:
            logger.debug(
                "Maneuver detection below confidence threshold",
                detection_id=detection.id,
                confidence=detection.detection_confidence
            )
            return []
        
        # Generate alert content
        alert_subject, alert_body = self._generate_maneuver_alert_content(detection)
        
        # Create alert history record
        alert_history = AlertHistory(
            alert_type="maneuver",
            severity="MEDIUM",  # Maneuvers are typically medium priority
            recipient="default_recipient@example.com",
            subject=alert_subject,
            body=alert_body,
            payload={
                "detection_id": detection.id,
                "norad_id": detection.norad_id,
                "detection_datetime": detection.detection_datetime.isoformat(),
                "maneuver_detected": detection.maneuver_detected,
                "confidence": detection.detection_confidence,
                "detection_method": detection.detection_method
            },
            sent_at=datetime.utcnow(),
            maneuver_detection_id=detection.id
        )
        
        # Add to results - actual saving done by caller
        alerts_created.append(alert_history)
        
        # Update detection to mark that alert was generated
        repo.update_alert_status(
            detection_id=detection.id,
            alert_generated=True
        )
        
        logger.info(
            "Maneuver alert generated",
            detection_id=detection.id,
            confidence=detection.detection_confidence
        )
        
        return alerts_created
    
    def _determine_conjunction_severity(self, event: ConjunctionEvent) -> str:
        """Determine alert severity based on conjunction event parameters."""
        if (event.probability >= self.config.pc_threshold_high or 
            event.miss_distance_meters <= 100 or  # Very close approach
            event.relative_velocity_mps <= 100):   # Very slow relative velocity
            return "CRITICAL"
        elif (event.probability >= self.config.pc_threshold_medium or
              event.miss_distance_meters <= 500):
            return "HIGH"
        elif (event.probability >= self.config.pc_threshold_low or
              event.miss_distance_meters <= 1000):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_conjunction_alert_content(self, event: ConjunctionEvent, severity: str) -> tuple:
        """Generate alert subject and body for conjunction event."""
        subject = f"[{severity}] CONJUNCTION ALERT: {event.primary_norad_id} vs {event.secondary_norad_id}"
        
        body = f"""
CONJUNCTION ALERT GENERATED

Event Details:
- Primary Object: {event.primary_norad_id}
- Secondary Object: {event.secondary_norad_id}
- Time of Closest Approach: {event.tca_datetime.isoformat()}
- Miss Distance: {event.miss_distance_meters:,.2f} meters
- Relative Velocity: {event.relative_velocity_mps:,.2f} m/s
- Collision Probability: {event.probability:.2e}
- Probability Method: {event.probability_method}
- Alert Severity: {severity}

Object Information:
- Primary Type: {event.primary_object_type or 'Unknown'}
- Secondary Type: {event.secondary_object_type or 'Unknown'}
- Primary Radius: {event.primary_radius_meters} meters
- Secondary Radius: {event.secondary_radius_meters} meters

Analysis Metadata:
- Analysis Version: {event.analysis_version}
- Screening Threshold: {event.screening_threshold_km} km
- Time Window: {event.time_window_hours} hours

Recommended Actions:
- Assess collision risk
- Consider debris avoidance maneuver
- Notify relevant operators
"""
        
        return subject, body.strip()
    
    def _generate_maneuver_alert_content(self, detection: ManeuverDetection) -> tuple:
        """Generate alert subject and body for maneuver detection."""
        subject = f"[MEDIUM] MANEUVER DETECTED: Satellite {detection.norad_id}"
        
        body = f"""
MANEUVER DETECTION ALERT

Event Details:
- Satellite NORAD ID: {detection.norad_id}
- Detection Time: {detection.detection_datetime.isoformat()}
- Maneuver Detected: {detection.maneuver_detected}
- Detection Confidence: {detection.detection_confidence:.2f}
- Detection Method: {detection.detection_method}

Feature Values:
- Position Residual Magnitude: {detection.position_residual_magnitude or 'N/A'}
- Velocity Residual Magnitude: {detection.velocity_residual_magnitude or 'N/A'}
- Days Since Last TLE: {detection.days_since_last_tle or 'N/A'}
- SMA Change: {detection.sma_change_meters or 'N/A'} meters
- Eccentricity Change: {detection.eccentricity_change or 'N/A'}
- Inclination Change: {detection.inclination_change_degrees or 'N/A'} degrees

Recommended Actions:
- Verify with operator if possible
- Update orbital prediction models
- Monitor for continued anomalies
"""
        
        return subject, body.strip()
    
    @log_execution_time("alert_generator_send_email")
    def send_email_alert(self, alert: AlertHistory) -> bool:
        """
        Send email alert using SMTP configuration.
        
        Args:
            alert: Alert history record to send
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.config.smtp_server:
            logger.warning("SMTP server not configured, skipping email alert")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = alert.recipient
            msg['Subject'] = alert.subject
            
            # Add body
            msg.attach(MIMEText(alert.body, 'plain'))
            
            # Connect to server and send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.config.smtp_username, alert.recipient, text)
            server.quit()
            
            logger.info("Email alert sent successfully", alert_id=alert.id)
            return True
            
        except Exception as e:
            logger.error("Failed to send email alert", alert_id=alert.id, error=str(e))
            return False


# Default configuration
default_alert_config = AlertConfig()
alert_generator = AlertGenerator(config=default_alert_config)