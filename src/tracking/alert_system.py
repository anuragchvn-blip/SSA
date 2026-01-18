"""
Real-time alert system for satellite tracking and conjunction detection.
"""
import asyncio
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
import redis.asyncio as redis
import aiohttp
from fastapi import WebSocket

from src.core.logging import get_logger
from src.core.config import settings
from src.data.models import ConjunctionEvent, AlertHistory

logger = get_logger(__name__)


@dataclass
class Alert:
    """Represents a real-time alert."""
    id: str
    alert_type: str  # 'conjunction', 'proximity', 'region_entry', 'maneuver'
    severity: str    # 'critical', 'high', 'medium', 'low'
    timestamp: datetime
    message: str
    satellite_ids: List[int]
    probability: Optional[float] = None
    distance_km: Optional[float] = None
    tca: Optional[datetime] = None
    details: Optional[Dict] = None


class AlertNotifier:
    """Handles sending alerts through various channels."""
    
    def __init__(self):
        self.email_enabled = bool(getattr(settings, 'email', None) and getattr(settings.email, 'smtp_server', None))
        self.webhook_enabled = bool(getattr(settings, 'alerts', None) and getattr(settings.alerts, 'webhook_url', None))
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Redis for alert deduplication and rate limiting
        self.redis_client = redis.Redis(
            host=getattr(settings.redis, 'host', 'localhost'),
            port=getattr(settings.redis, 'port', 6379),
            db=0,
            password=getattr(settings.redis, 'password', None),
            decode_responses=True
        )
    
    async def send_email_alert(self, alert: Alert, recipients: List[str]):
        """Send alert via email."""
        if not self.email_enabled:
            logger.warning("Email notifications disabled - no SMTP settings configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.alert_type.title()} Alert"
            msg['From'] = settings.email.sender_address
            msg['To'] = ", ".join(recipients)
            
            body = f"""
Satellite Tracking Alert

Type: {alert.alert_type}
Severity: {alert.severity.upper()}
Timestamp: {alert.timestamp.isoformat()}
Message: {alert.message}

Satellite IDs: {', '.join(map(str, alert.satellite_ids))}
Probability: {alert.probability or 'N/A'}
Distance: {alert.distance_km or 'N/A'} km
TCA: {alert.tca.isoformat() if alert.tca else 'N/A'}

Details: {json.dumps(alert.details or {}, indent=2)}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(settings.email.smtp_server, settings.email.smtp_port) as server:
                if settings.email.use_tls:
                    server.starttls()
                if settings.email.username and settings.email.password:
                    server.login(settings.email.username, settings.email.password)
                
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    async def send_webhook_alert(self, alert: Alert):
        """Send alert via webhook."""
        if not self.webhook_enabled:
            logger.warning("Webhook notifications disabled - no webhook URL configured")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'id': alert.id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat(),
                    'message': alert.message,
                    'satellite_ids': alert.satellite_ids,
                    'probability': alert.probability,
                    'distance_km': alert.distance_km,
                    'tca': alert.tca.isoformat() if alert.tca else None,
                    'details': alert.details
                }
                
                async with session.post(
                    settings.alerts.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logger.info("Webhook alert sent successfully")
                        return True
                    else:
                        logger.error(f"Webhook alert failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    async def broadcast_websocket_alert(self, alert: Alert):
        """Broadcast alert to connected WebSocket clients."""
        disconnected_clients = []
        
        for client_id, websocket in self.websocket_connections.items():
            try:
                await websocket.send_text(json.dumps(asdict(alert)))
            except Exception as e:
                logger.warning(f"Failed to send alert to WebSocket client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            del self.websocket_connections[client_id]
        
        logger.info(f"Broadcast alert to {len(self.websocket_connections)} WebSocket clients")
        return len(self.websocket_connections)
    
    async def send_alert(self, alert: Alert, recipients: Optional[List[str]] = None):
        """Send alert through all configured channels."""
        results = {}
        
        # Email notification
        if recipients:
            results['email'] = await self.send_email_alert(alert, recipients)
        
        # Webhook notification
        results['webhook'] = await self.send_webhook_alert(alert)
        
        # WebSocket broadcast
        results['websocket'] = await self.broadcast_websocket_alert(alert)
        
        return results


class AlertManager:
    """Manages real-time alerts for satellite tracking events."""
    
    def __init__(self):
        self.notifier = AlertNotifier()
        self.active_alerts: Dict[str, Alert] = {}
        self.suppressed_alerts: Dict[str, datetime] = {}  # For deduplication
        self.alert_rules: List[Dict] = []
        self.alert_history: List[Alert] = []
        
        # Load alert rules from configuration
        self.load_alert_rules()
    
    def load_alert_rules(self):
        """Load alert rules from configuration."""
        # Default rules
        self.alert_rules = [
            {
                'name': 'high_risk_conjunction',
                'condition': lambda event: event.probability >= 1e-4,
                'severity': 'critical',
                'message_template': 'HIGH RISK CONJUNCTION: Satellites {primary_id} and {secondary_id} have Pc >= 1e-4'
            },
            {
                'name': 'medium_risk_conjunction',
                'condition': lambda event: 1e-5 <= event.probability < 1e-4,
                'severity': 'high',
                'message_template': 'MEDIUM RISK CONJUNCTION: Satellites {primary_id} and {secondary_id} have 1e-5 <= Pc < 1e-4'
            },
            {
                'name': 'close_approach',
                'condition': lambda event: event.miss_distance_meters <= 1000,  # Within 1km
                'severity': 'high',
                'message_template': 'CLOSE APPROACH: Satellites {primary_id} and {secondary_id} within 1km'
            },
            {
                'name': 'very_close_approach',
                'condition': lambda event: event.miss_distance_meters <= 100,  # Within 100m
                'severity': 'critical',
                'message_template': 'VERY CLOSE APPROACH: Satellites {primary_id} and {secondary_id} within 100m'
            }
        ]
    
    async def check_and_trigger_alerts(self, conjunction_event: ConjunctionEvent) -> List[Alert]:
        """Check if a conjunction event triggers any alerts."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if rule['condition'](conjunction_event):
                # Generate unique alert ID based on the event
                alert_id = f"alert_{conjunction_event.id}_{rule['name']}"
                
                # Check if this alert has been suppressed recently
                if alert_id in self.suppressed_alerts:
                    suppression_duration = datetime.utcnow() - self.suppressed_alerts[alert_id]
                    if suppression_duration < timedelta(minutes=5):  # Suppress for 5 minutes
                        continue
                    else:
                        # Remove expired suppression
                        del self.suppressed_alerts[alert_id]
                
                # Create alert message
                message = rule['message_template'].format(
                    primary_id=conjunction_event.primary_norad_id,
                    secondary_id=conjunction_event.secondary_norad_id,
                    probability=conjunction_event.probability,
                    distance_km=conjunction_event.miss_distance_meters / 1000
                )
                
                # Create alert
                alert = Alert(
                    id=alert_id,
                    alert_type='conjunction',
                    severity=rule['severity'],
                    timestamp=datetime.utcnow(),
                    message=message,
                    satellite_ids=[conjunction_event.primary_norad_id, conjunction_event.secondary_norad_id],
                    probability=conjunction_event.probability,
                    distance_km=conjunction_event.miss_distance_meters / 1000,
                    tca=conjunction_event.tca_datetime,
                    details={
                        'conjunction_event_id': conjunction_event.id,
                        'relative_velocity_mps': conjunction_event.relative_velocity_mps
                    }
                )
                
                # Add to active alerts
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Suppress duplicate alerts for a period
                self.suppressed_alerts[alert_id] = datetime.utcnow()
                
                triggered_alerts.append(alert)
                
                logger.info(f"Triggered {rule['severity']} alert: {message}")
        
        return triggered_alerts
    
    async def trigger_proximity_alert(
        self,
        satellite_ids: List[int],
        distance_km: float,
        threshold_km: float
    ) -> Optional[Alert]:
        """Trigger an alert for satellite proximity."""
        severity = 'high' if distance_km < threshold_km / 2 else 'medium'
        
        alert_id = f"proximity_{'_'.join(map(str, satellite_ids))}_{datetime.utcnow().timestamp()}"
        
        if alert_id in self.suppressed_alerts:
            suppression_duration = datetime.utcnow() - self.suppressed_alerts[alert_id]
            if suppression_duration < timedelta(minutes=1):
                return None
        
        message = f"PROXIMITY ALERT: Satellites {', '.join(map(str, satellite_ids))} within {distance_km:.2f}km (threshold: {threshold_km}km)"
        
        alert = Alert(
            id=alert_id,
            alert_type='proximity',
            severity=severity,
            timestamp=datetime.utcnow(),
            message=message,
            satellite_ids=satellite_ids,
            distance_km=distance_km
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.suppressed_alerts[alert_id] = datetime.utcnow()
        
        logger.warning(message)
        return alert
    
    async def trigger_region_entry_alert(
        self,
        satellite_id: int,
        region_name: str,
        coordinates: Dict[str, float]
    ) -> Optional[Alert]:
        """Trigger an alert when a satellite enters a restricted region."""
        alert_id = f"region_{satellite_id}_{region_name}_{datetime.utcnow().timestamp()}"
        
        if alert_id in self.suppressed_alerts:
            suppression_duration = datetime.utcnow() - self.suppressed_alerts[alert_id]
            if suppression_duration < timedelta(minutes=5):
                return None
        
        message = f"REGION ENTRY: Satellite {satellite_id} entered restricted region '{region_name}' at {coordinates}"
        
        alert = Alert(
            id=alert_id,
            alert_type='region_entry',
            severity='medium',
            timestamp=datetime.utcnow(),
            message=message,
            satellite_ids=[satellite_id],
            details={'region_name': region_name, 'coordinates': coordinates}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.suppressed_alerts[alert_id] = datetime.utcnow()
        
        logger.warning(message)
        return alert
    
    async def send_alerts(self, alerts: List[Alert], recipients: Optional[List[str]] = None):
        """Send multiple alerts."""
        for alert in alerts:
            await self.notifier.send_alert(alert, recipients)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:] if len(self.alert_history) > limit else self.alert_history
    
    def clear_alert(self, alert_id: str):
        """Clear a specific alert."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Cleared alert {alert_id}")


class RealTimeAlertSystem:
    """Main real-time alert system integrating with tracking and conjunction analysis."""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.alert_recipients = getattr(getattr(settings, 'alerts', None), 'recipients', '').split(',') if getattr(getattr(settings, 'alerts', None), 'recipients', None) else []
        self.is_running = False
    
    async def process_conjunction_event(self, conjunction_event: ConjunctionEvent):
        """Process a conjunction event and trigger alerts if needed."""
        alerts = await self.alert_manager.check_and_trigger_alerts(conjunction_event)
        if alerts:
            await self.alert_manager.send_alerts(alerts, self.alert_recipients)
    
    async def process_proximity_event(
        self,
        satellite_ids: List[int],
        distance_km: float,
        threshold_km: float = 5.0
    ):
        """Process a satellite proximity event."""
        if distance_km <= threshold_km:
            alert = await self.alert_manager.trigger_proximity_alert(
                satellite_ids, distance_km, threshold_km
            )
            if alert:
                await self.alert_manager.notifier.send_alert(alert, self.alert_recipients)
    
    async def process_region_entry(
        self,
        satellite_id: int,
        region_name: str,
        coordinates: Dict[str, float]
    ):
        """Process a satellite region entry event."""
        alert = await self.alert_manager.trigger_region_entry_alert(
            satellite_id, region_name, coordinates
        )
        if alert:
            await self.alert_manager.notifier.send_alert(alert, self.alert_recipients)
    
    def get_system_status(self) -> Dict:
        """Get alert system status."""
        return {
            'active_alerts_count': len(self.alert_manager.active_alerts),
            'total_alerts_sent': len(self.alert_manager.alert_history),
            'alert_recipients': self.alert_recipients,
            'system_running': self.is_running,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def add_websocket_client(self, client_id: str, websocket: WebSocket):
        """Add a WebSocket client for real-time alerts."""
        self.alert_manager.notifier.websocket_connections[client_id] = websocket
        logger.info(f"Added WebSocket client {client_id}")
    
    async def remove_websocket_client(self, client_id: str):
        """Remove a WebSocket client."""
        if client_id in self.alert_manager.notifier.websocket_connections:
            del self.alert_manager.notifier.websocket_connections[client_id]
            logger.info(f"Removed WebSocket client {client_id}")