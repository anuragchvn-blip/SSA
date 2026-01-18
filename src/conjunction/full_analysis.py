"""Implementation of conjunction screening using all system components."""

from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np

from src.core.exceptions import ConjunctionAnalysisError
from src.core.logging import get_logger, log_execution_time
from src.data.models import TLE, ConjunctionEvent
from src.data.storage.tle_repository import TLERepository
from src.data.storage.conjunction_repository import ConjunctionEventRepository
from src.propagation.sgp4_engine import SGP4Engine, CartesianState
from src.conjunction.probability import ProbabilityCalculator
from src.conjunction.screening import ConjunctionScreener, RefinementResult
from src.reports.alerts import alert_generator

logger = get_logger(__name__)


class ConjunctionAnalyzer:
    """Full conjunction analysis system integrating all components."""
    
    def __init__(self):
        self.sgp4_engine = SGP4Engine()
        self.probability_calculator = ProbabilityCalculator()
        self.conjunction_screener = ConjunctionScreener(self.sgp4_engine)
        self.logger = get_logger(__name__)
    
    @log_execution_time("conjunction_full_analysis")
    def perform_full_analysis(
        self,
        primary_tle: TLE,
        catalog_tles: List[TLE],
        time_window_hours: float = 24.0,
        screening_threshold_km: float = 5.0,
        probability_threshold: float = 1e-6
    ) -> List[ConjunctionEvent]:
        """
        Perform complete conjunction analysis: screening -> refinement -> probability -> alerts.
        
        Args:
            primary_tle: Primary object TLE
            catalog_tles: List of catalog TLEs to screen against
            time_window_hours: Time window for analysis
            screening_threshold_km: Initial screening distance threshold
            probability_threshold: Minimum Pc for event creation
            
        Returns:
            List of ConjunctionEvent objects for events exceeding probability threshold
        """
        try:
            logger.info(
                "Starting full conjunction analysis",
                primary_norad=primary_tle.norad_id,
                catalog_size=len(catalog_tles),
                time_window_hours=time_window_hours
            )
            
            # Step 1: Screen catalog for potential conjunctions
            candidates = self.conjunction_screener.screen_catalog(
                primary_tle=primary_tle,
                catalog_tles=catalog_tles,
                screening_threshold_km=screening_threshold_km,
                time_window_hours=time_window_hours
            )
            
            logger.info(
                "Catalog screening completed",
                candidates_identified=len(candidates)
            )
            
            if not candidates:
                logger.info("No conjunction candidates found, returning empty list")
                return []
            
            # Step 2: Refine candidates to get precise TCA and miss distance
            refined_results = self.conjunction_screener.refine_candidates(candidates)
            
            logger.info(
                "Candidate refinement completed",
                refined_results=len(refined_results)
            )
            
            # Step 3: Calculate collision probability for each refined result
            events = []
            high_risk_count = 0
            
            for i, result in enumerate(refined_results):
                try:
                    # Calculate probability using the best method for this scenario
                    method = self.probability_calculator.select_best_method(
                        miss_distance=result.miss_distance,
                        covariance_info={},  # Placeholder - would use real covariance in production
                        combined_radius=10.0  # Placeholder: 10m combined radius
                    )
                    
                    if method == "foster_2d":
                        # For demo purposes, use Foster method with simplified parameters
                        pc_result = self.probability_calculator.compute_pc_foster_method(
                            miss_distance=result.miss_distance,
                            sigma_x=100.0,  # Placeholder: 100m uncertainty
                            sigma_y=100.0,  # Placeholder: 100m uncertainty
                            combined_radius=10.0  # 10m combined radius
                        )
                    else:
                        # For now, default to Foster method
                        pc_result = self.probability_calculator.compute_pc_foster_method(
                            miss_distance=result.miss_distance,
                            sigma_x=100.0,
                            sigma_y=100.0,
                            combined_radius=10.0
                        )
                    
                    # Create conjunction event if probability exceeds threshold
                    if pc_result.probability >= probability_threshold:
                        event = ConjunctionEvent(
                            primary_norad_id=primary_tle.norad_id,
                            secondary_norad_id=0,  # Would be determined from the TLE
                            tca_datetime=result.tca_datetime,
                            primary_x_eci=result.primary_state.x,
                            primary_y_eci=result.primary_state.y,
                            primary_z_eci=result.primary_state.z,
                            secondary_x_eci=result.secondary_state.x,
                            secondary_y_eci=result.secondary_state.y,
                            secondary_z_eci=result.secondary_state.z,
                            miss_distance_meters=result.miss_distance,
                            relative_velocity_mps=result.relative_velocity,
                            probability=pc_result.probability,
                            probability_method=pc_result.method,
                            probability_confidence_lower=pc_result.confidence_interval[0],
                            probability_confidence_upper=pc_result.confidence_interval[1],
                            screening_threshold_km=screening_threshold_km,
                            time_window_hours=time_window_hours,
                            primary_radius_meters=5.0,  # 5m radius for primary
                            secondary_radius_meters=0.5,  # 0.5m radius for secondary
                            analysis_version="1.0.0",
                            alert_threshold_exceeded=(pc_result.probability >= 1e-3)
                        )
                        
                        events.append(event)
                        
                        if pc_result.probability >= 1e-3:
                            high_risk_count += 1
                            
                        logger.debug(
                            "Conjunction event created",
                            primary_norad=primary_tle.norad_id,
                            secondary_norad=0,  # Placeholder
                            tca=result.tca_datetime.isoformat(),
                            pc=pc_result.probability,
                            miss_distance_km=result.miss_distance/1000
                        )
                    else:
                        logger.debug(
                            "Conjunction event below probability threshold",
                            miss_distance_km=result.miss_distance/1000,
                            pc=pc_result.probability,
                            threshold=probability_threshold
                        )
                        
                except Exception as e:
                    logger.warning(
                        "Failed to calculate probability for candidate",
                        candidate_index=i,
                        error=str(e)
                    )
                    continue
            
            logger.info(
                "Full conjunction analysis completed",
                primary_norad=primary_tle.norad_id,
                events_created=len(events),
                high_risk_events=high_risk_count,
                total_candidates=len(refined_results)
            )
            
            return events
            
        except Exception as e:
            logger.error("Full conjunction analysis failed", error=str(e))
            raise ConjunctionAnalysisError(
                message=f"Full conjunction analysis failed: {str(e)}",
                error_code="CONJUNCTION_ANALYSIS_FAILED",
                details={
                    "primary_norad": getattr(primary_tle, 'norad_id', None),
                    "catalog_size": len(catalog_tles),
                    "error": str(e)
                }
            )
    
    @log_execution_time("conjunction_batch_analysis")
    def perform_batch_analysis(
        self,
        primary_tles: List[TLE],
        catalog_tles: List[TLE],
        time_window_hours: float = 24.0,
        screening_threshold_km: float = 5.0
    ) -> List[ConjunctionEvent]:
        """
        Perform conjunction analysis for multiple primary objects against catalog.
        
        Args:
            primary_tles: List of primary objects to analyze
            catalog_tles: Catalog of objects to screen against
            time_window_hours: Time window for analysis
            screening_threshold_km: Screening distance threshold
            
        Returns:
            List of all conjunction events found
        """
        all_events = []
        
        for i, primary_tle in enumerate(primary_tles):
            logger.info(
                "Processing batch analysis",
                primary_index=i,
                total_primaries=len(primary_tles),
                primary_norad=primary_tle.norad_id
            )
            
            try:
                events = self.perform_full_analysis(
                    primary_tle=primary_tle,
                    catalog_tles=catalog_tles,
                    time_window_hours=time_window_hours,
                    screening_threshold_km=screening_threshold_km
                )
                
                all_events.extend(events)
                
                logger.info(
                    "Batch analysis for primary completed",
                    primary_norad=primary_tle.norad_id,
                    events_found=len(events)
                )
                
            except Exception as e:
                logger.error(
                    "Failed to analyze primary object",
                    primary_norad=primary_tle.norad_id,
                    error=str(e)
                )
                continue
        
        logger.info(
            "Batch conjunction analysis completed",
            total_primaries=len(primary_tles),
            total_events=len(all_events)
        )
        
        return all_events


# Global analyzer instance
conjunction_analyzer = ConjunctionAnalyzer()