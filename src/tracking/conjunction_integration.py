"""
Integration module connecting live tracking with conjunction analysis system.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.tracking.multi_satellite_tracker import OptimizedSatelliteTracker
from src.conjunction.screening import ConjunctionScreener
from src.conjunction.full_analysis import ConjunctionAnalyzer
from src.data.models import TLE, ConjunctionEvent
from src.propagation.sgp4_engine import SGP4Engine
from src.core.logging import get_logger
from src.data.storage.conjunction_repository import ConjunctionEventRepository

logger = get_logger(__name__)


class LiveConjunctionAnalyzer:
    """Analyzer that connects live tracking with conjunction analysis."""
    
    def __init__(self):
        self.tracker = OptimizedSatelliteTracker()
        self.sgp4_engine = SGP4Engine()
        self.screening_engine = ConjunctionScreener(sgp4_engine=self.sgp4_engine)
        self.analysis_engine = ConjunctionAnalyzer()
        from src.data.storage.conjunction_repository import conjunction_event_repository
        self.repository = conjunction_event_repository
        
        # Configuration for live analysis
        self.default_screening_threshold_km = 5.0  # 5 km screening threshold
        self.time_window_hours = 24  # Analyze next 24 hours
        self.time_step_minutes = 5  # Time step for analysis
        
    async def analyze_conjunctions_for_tracked_satellites(
        self,
        min_probability_threshold: float = 1e-6
    ) -> List[ConjunctionEvent]:
        """
        Analyze conjunctions for all currently tracked satellites.
        
        Args:
            min_probability_threshold: Minimum probability to report conjunctions
            
        Returns:
            List of detected conjunction events
        """
        tracked_positions = await self.tracker.get_current_positions()
        norad_ids = list(tracked_positions.keys())
        
        if len(norad_ids) < 2:
            logger.info("Need at least 2 satellites to analyze conjunctions")
            return []
        
        conjunction_events = []
        
        # Get TLEs for all tracked satellites
        tles = {}
        for norad_id in norad_ids:
            sat_info = self.tracker.tracked_satellites.get(norad_id)
            if sat_info:
                tles[norad_id] = sat_info.tle
        
        # Perform pairwise analysis between all tracked satellites
        for i in range(len(norad_ids)):
            for j in range(i + 1, len(norad_ids)):
                primary_id = norad_ids[i]
                secondary_id = norad_ids[j]
                
                primary_tle = tles.get(primary_id)
                secondary_tle = tles.get(secondary_id)
                
                if not primary_tle or not secondary_tle:
                    continue
                
                try:
                    # Perform screening analysis
                    screening_result = await self.screening_engine.perform_screening(
                        primary_tle=primary_tle,
                        secondary_tle=secondary_tle,
                        screening_threshold_km=self.default_screening_threshold_km,
                        time_window_hours=self.time_window_hours,
                        time_step_minutes=self.time_step_minutes
                    )
                    
                    if screening_result.min_distance_meters < (self.default_screening_threshold_km * 1000):
                        # Perform full analysis for close approaches
                        analysis_result = await self.analysis_engine.analyze_conjunction(
                            primary_tle=primary_tle,
                            secondary_tle=secondary_tle,
                            tca_estimate=screening_result.tca_datetime,
                            refinement_window_minutes=60
                        )
                        
                        if analysis_result.probability_collision >= min_probability_threshold:
                            # Create conjunction event
                            conjunction_event = ConjunctionEvent(
                                primary_norad_id=primary_id,
                                secondary_norad_id=secondary_id,
                                tca_datetime=analysis_result.tca_datetime,
                                miss_distance_meters=screening_result.min_distance_meters,
                                relative_velocity_mps=analysis_result.relative_velocity_mps,
                                probability=analysis_result.probability_collision,
                                probability_method=analysis_result.method_used,
                                screening_threshold_km=self.default_screening_threshold_km,
                                time_window_hours=self.time_window_hours,
                                primary_object_name=f"Sat-{primary_id}",
                                secondary_object_name=f"Sat-{secondary_id}",
                                primary_radius_meters=1.0,  # Default radius
                                secondary_radius_meters=1.0,  # Default radius
                                analysis_version="live-tracking-v1",
                                algorithm_parameters={}
                            )
                            
                            conjunction_events.append(conjunction_event)
                            
                            logger.info(
                                f"Conjunction detected: {primary_id} vs {secondary_id} "
                                f"at TCA {analysis_result.tca_datetime}, "
                                f"distance: {screening_result.min_distance_meters/1000:.2f} km, "
                                f"Pc: {analysis_result.probability_collision:.2e}"
                            )
                
                except Exception as e:
                    logger.error(
                        f"Error analyzing conjunction between {primary_id} and {secondary_id}: {e}"
                    )
                    continue
        
        return conjunction_events
    
    async def get_real_time_conjunction_risks(
        self,
        risk_level_threshold: float = 1e-4
    ) -> Dict:
        """
        Get real-time conjunction risk assessment for tracked satellites.
        
        Args:
            risk_level_threshold: Threshold for high-risk conjunctions
            
        Returns:
            Risk assessment dictionary
        """
        conjunctions = await self.analyze_conjunctions_for_tracked_satellites()
        
        high_risk = []
        medium_risk = []
        low_risk = []
        
        for event in conjunctions:
            if event.probability >= risk_level_threshold:
                high_risk.append(event)
            elif event.probability >= risk_level_threshold / 10:
                medium_risk.append(event)
            else:
                low_risk.append(event)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'high_risk_conjunctions': len(high_risk),
            'medium_risk_conjunctions': len(medium_risk),
            'low_risk_conjunctions': len(low_risk),
            'total_conjunctions': len(conjunctions),
            'high_risk_details': [
                {
                    'primary_id': event.primary_norad_id,
                    'secondary_id': event.secondary_norad_id,
                    'tca': event.tca_datetime.isoformat(),
                    'distance_km': event.miss_distance_meters / 1000,
                    'probability': event.probability
                }
                for event in high_risk
            ],
            'medium_risk_details': [
                {
                    'primary_id': event.primary_norad_id,
                    'secondary_id': event.secondary_norad_id,
                    'tca': event.tca_datetime.isoformat(),
                    'distance_km': event.miss_distance_meters / 1000,
                    'probability': event.probability
                }
                for event in medium_risk
            ]
        }
    
    async def monitor_and_store_conjunctions(self):
        """Monitor conjunctions and store significant events in the database."""
        conjunctions = await self.analyze_conjunctions_for_tracked_satellites(min_probability_threshold=1e-7)
        
        stored_count = 0
        for event in conjunctions:
            try:
                # Check if this conjunction event already exists in the database
                existing = await self.repository.get_conjunction_by_norad_ids_and_tca(
                    primary_norad_id=event.primary_norad_id,
                    secondary_norad_id=event.secondary_norad_id,
                    tca_datetime=event.tca_datetime
                )
                
                if not existing:
                    # Store new conjunction event
                    await self.repository.create_conjunction_event(event)
                    stored_count += 1
                    logger.info(f"Stored new conjunction event: {event.primary_norad_id} vs {event.secondary_norad_id}")
                else:
                    logger.debug(f"Conjunction event already exists: {event.primary_norad_id} vs {event.secondary_norad_id}")
                    
            except Exception as e:
                logger.error(f"Error storing conjunction event: {e}")
        
        logger.info(f"Stored {stored_count} new conjunction events")
        return stored_count
    
    async def get_conjunction_alerts(self) -> List[Dict]:
        """
        Get real-time conjunction alerts for tracked satellites.
        
        Returns:
            List of conjunction alerts
        """
        risk_assessment = await self.get_real_time_conjunction_risks()
        
        alerts = []
        
        # High-risk alerts
        for event in risk_assessment['high_risk_details']:
            alerts.append({
                'level': 'HIGH',
                'message': f'HIGH RISK CONJUNCTION: Satellite {event["primary_id"]} and {event["secondary_id"]} '
                          f'predicted to pass within {event["distance_km"]:.2f} km at {event["tca"]}',
                'probability': event['probability'],
                'timestamp': datetime.utcnow().isoformat(),
                'satellites': [event['primary_id'], event['secondary_id']],
                'tca': event['tca']
            })
        
        # Medium-risk alerts
        for event in risk_assessment['medium_risk_details']:
            alerts.append({
                'level': 'MEDIUM',
                'message': f'MEDIUM RISK CONJUNCTION: Satellite {event["primary_id"]} and {event["secondary_id"]} '
                          f'predicted to pass within {event["distance_km"]:.2f} km at {event["tca"]}',
                'probability': event['probability'],
                'timestamp': datetime.utcnow().isoformat(),
                'satellites': [event['primary_id'], event['secondary_id']],
                'tca': event['tca']
            })
        
        return alerts


class IntegratedTrackingConjunctionSystem:
    """Integrated system combining live tracking and conjunction analysis."""
    
    def __init__(self):
        self.live_analyzer = LiveConjunctionAnalyzer()
        self.tracker = self.live_analyzer.tracker
    
    async def initialize_system(self, satellite_ids: List[int]) -> bool:
        """Initialize the integrated system with specified satellites."""
        success = await self.tracker.initialize_tracking(satellite_ids)
        if success:
            logger.info(f"Initialized integrated system with {len(satellite_ids)} satellites")
        return success
    
    async def run_continuous_monitoring(
        self,
        monitoring_interval: float = 300.0,  # 5 minutes
        conjunction_check_interval: float = 600.0  # 10 minutes
    ):
        """Run continuous monitoring with periodic conjunction checks."""
        last_conjunction_check = datetime.utcnow()
        
        while self.tracker.is_running:
            try:
                # Update satellite positions
                await self.tracker.update_all_satellites()
                
                # Periodically check for conjunctions
                current_time = datetime.utcnow()
                if (current_time - last_conjunction_check).total_seconds() >= conjunction_check_interval:
                    logger.info("Performing conjunction analysis...")
                    
                    # Monitor and store conjunctions
                    stored_count = await self.live_analyzer.monitor_and_store_conjunctions()
                    
                    # Get alerts
                    alerts = await self.live_analyzer.get_conjunction_alerts()
                    
                    if alerts:
                        for alert in alerts:
                            logger.warning(f"CONJUNCTION ALERT [{alert['level']}]: {alert['message']}")
                    
                    last_conjunction_check = current_time
                
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(10)  # Brief pause before continuing
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        tracking_status = self.tracker.get_tracked_satellites_status()
        risk_assessment = await self.live_analyzer.get_real_time_conjunction_risks()
        
        return {
            'tracking_status': tracking_status,
            'conjunction_risk_assessment': risk_assessment,
            'system_timestamp': datetime.utcnow().isoformat(),
            'performance_metrics': self.tracker.get_performance_metrics()
        }