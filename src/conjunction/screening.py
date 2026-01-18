"""Conjunction screening and detection system."""

import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, NamedTuple
from scipy.spatial.distance import cdist

from src.core.exceptions import ConjunctionAnalysisError
from src.core.logging import get_logger, log_execution_time
from src.data.models import TLE, ConjunctionEvent
from src.propagation.sgp4_engine import SGP4Engine, CartesianState, PropagationResult

logger = get_logger(__name__)


class ConjunctionCandidate(NamedTuple):
    """Potential conjunction candidate with screening metrics."""
    primary_tle: TLE
    secondary_tle: TLE
    screening_distance: float  # meters
    time_to_closest_approach: timedelta
    epoch: datetime


class RefinementResult(NamedTuple):
    """Detailed conjunction analysis results."""
    miss_distance: float  # meters
    relative_velocity: float  # m/s
    tca_datetime: datetime
    primary_state: CartesianState
    secondary_state: CartesianState


class ConjunctionScreener:
    """Screen for potential conjunctions with multi-stage filtering."""
    
    def __init__(self, sgp4_engine: SGP4Engine):
        self.sgp4_engine = sgp4_engine
        self.logger = get_logger(__name__)
    
    @log_execution_time("conjunction_screen_catalog")
    def screen_catalog(
        self,
        primary_tle: TLE,
        catalog_tles: List[TLE],
        screening_threshold_km: float = 5.0,
        time_window_hours: float = 24.0,
        time_step_minutes: int = 10
    ) -> List[ConjunctionCandidate]:
        """
        Screen entire catalog for potential conjunctions with primary object.
        
        Args:
            primary_tle: Primary object TLE
            catalog_tles: List of TLEs to screen against
            screening_threshold_km: Initial screening distance threshold (km)
            time_window_hours: Time window for screening (hours)
            time_step_minutes: Time step for propagation grid (minutes)
            
        Returns:
            List of conjunction candidates requiring refinement
            
        Raises:
            ConjunctionAnalysisError: If screening fails
        """
        if not catalog_tles:
            return []
        
        try:
            # Generate time grid
            time_points = self._generate_time_grid(
                primary_tle.epoch_datetime,
                time_window_hours,
                time_step_minutes
            )
            
            # Propagate primary object to all time points
            primary_states = self.sgp4_engine.batch_propagate(primary_tle, time_points)
            primary_positions = np.array([
                [state.cartesian_state.x, state.cartesian_state.y, state.cartesian_state.z]
                for state in primary_states
            ])
            
            candidates = []
            screening_threshold_m = screening_threshold_km * 1000.0
            
            # Screen each catalog object
            for secondary_tle in catalog_tles:
                if secondary_tle.norad_id == primary_tle.norad_id:
                    continue  # Skip self-conjunctions
                
                try:
                    # Propagate secondary object
                    secondary_states = self.sgp4_engine.batch_propagate(secondary_tle, time_points)
                    secondary_positions = np.array([
                        [state.cartesian_state.x, state.cartesian_state.y, state.cartesian_state.z]
                        for state in secondary_states
                    ])
                    
                    # Calculate distances at all time points
                    distances = np.linalg.norm(primary_positions - secondary_positions, axis=1)
                    
                    # Check if any point is within screening threshold
                    min_distance_idx = np.argmin(distances)
                    min_distance = distances[min_distance_idx]
                    
                    if min_distance <= screening_threshold_m:
                        candidate = ConjunctionCandidate(
                            primary_tle=primary_tle,
                            secondary_tle=secondary_tle,
                            screening_distance=min_distance,
                            time_to_closest_approach=time_points[min_distance_idx] - primary_tle.epoch_datetime,
                            epoch=time_points[min_distance_idx]
                        )
                        candidates.append(candidate)
                        
                        self.logger.debug(
                            "Conjunction candidate identified",
                            primary_norad=primary_tle.norad_id,
                            secondary_norad=secondary_tle.norad_id,
                            min_distance_km=min_distance/1000.0,
                            tca_epoch=time_points[min_distance_idx].isoformat()
                        )
                
                except Exception as e:
                    self.logger.warning(
                        "Failed to screen object",
                        norad_id=secondary_tle.norad_id,
                        error=str(e)
                    )
                    continue  # Continue with other objects
            
            self.logger.info(
                "Catalog screening completed",
                primary_norad=primary_tle.norad_id,
                catalog_size=len(catalog_tles),
                candidates_found=len(candidates),
                screening_threshold_km=screening_threshold_km
            )
            
            return candidates
            
        except Exception as e:
            raise ConjunctionAnalysisError(
                message=f"Catalog screening failed: {str(e)}",
                error_code="SCREENING_FAILED",
                details={
                    "primary_norad": getattr(primary_tle, 'norad_id', None),
                    "catalog_size": len(catalog_tles),
                    "error": str(e)
                }
            )
    
    @log_execution_time("conjunction_refine_candidates")
    def refine_candidates(
        self,
        candidates: List[ConjunctionCandidate],
        refinement_window_minutes: int = 60
    ) -> List[RefinementResult]:
        """
        Refine candidates using higher time resolution around TCA.
        
        Args:
            candidates: List of conjunction candidates from screening
            refinement_window_minutes: Window around estimated TCA for refinement
            
        Returns:
            List of refined conjunction results
        """
        if not candidates:
            return []
        
        refined_results = []
        
        for candidate in candidates:
            try:
                # Create fine time grid around estimated TCA
                tca_estimate = candidate.epoch
                start_time = tca_estimate - timedelta(minutes=refinement_window_minutes/2)
                end_time = tca_estimate + timedelta(minutes=refinement_window_minutes/2)
                
                # High-resolution time steps (1 minute)
                fine_time_points = []
                current_time = start_time
                while current_time <= end_time:
                    fine_time_points.append(current_time)
                    current_time += timedelta(minutes=1)
                
                # Propagate both objects at high resolution
                primary_states = self.sgp4_engine.batch_propagate(
                    candidate.primary_tle, fine_time_points
                )
                secondary_states = self.sgp4_engine.batch_propagate(
                    candidate.secondary_tle, fine_time_points
                )
                
                # Find actual TCA (minimum distance)
                min_distance = float('inf')
                tca_idx = 0
                
                for i, (p_state, s_state) in enumerate(zip(primary_states, secondary_states)):
                    p_pos = np.array([
                        p_state.cartesian_state.x,
                        p_state.cartesian_state.y,
                        p_state.cartesian_state.z
                    ])
                    s_pos = np.array([
                        s_state.cartesian_state.x,
                        s_state.cartesian_state.y,
                        s_state.cartesian_state.z
                    ])
                    
                    distance = np.linalg.norm(p_pos - s_pos)
                    if distance < min_distance:
                        min_distance = distance
                        tca_idx = i
                
                # Calculate relative velocity at TCA
                primary_vel = np.array([
                    primary_states[tca_idx].cartesian_state.vx,
                    primary_states[tca_idx].cartesian_state.vy,
                    primary_states[tca_idx].cartesian_state.vz
                ])
                secondary_vel = np.array([
                    secondary_states[tca_idx].cartesian_state.vx,
                    secondary_states[tca_idx].cartesian_state.vy,
                    secondary_states[tca_idx].cartesian_state.vz
                ])
                relative_vel = np.linalg.norm(primary_vel - secondary_vel)
                
                refinement_result = RefinementResult(
                    miss_distance=min_distance,
                    relative_velocity=relative_vel,
                    tca_datetime=fine_time_points[tca_idx],
                    primary_state=primary_states[tca_idx].cartesian_state,
                    secondary_state=secondary_states[tca_idx].cartesian_state
                )
                
                refined_results.append(refinement_result)
                
                self.logger.debug(
                    "Candidate refined",
                    primary_norad=candidate.primary_tle.norad_id,
                    secondary_norad=candidate.secondary_tle.norad_id,
                    miss_distance_km=min_distance/1000.0,
                    relative_velocity_kms=relative_vel/1000.0
                )
                
            except Exception as e:
                self.logger.warning(
                    "Failed to refine candidate",
                    primary_norad=getattr(candidate.primary_tle, 'norad_id', None),
                    secondary_norad=getattr(candidate.secondary_tle, 'norad_id', None),
                    error=str(e)
                )
                continue
        
        self.logger.info(
            "Candidate refinement completed",
            candidates_processed=len(candidates),
            results_produced=len(refined_results)
        )
        
        return refined_results
    
    def _generate_time_grid(
        self, 
        start_epoch: datetime, 
        time_window_hours: float, 
        time_step_minutes: int
    ) -> List[datetime]:
        """Generate evenly spaced time points for propagation."""
        num_steps = int((time_window_hours * 60) / time_step_minutes) + 1
        time_points = []
        
        for i in range(num_steps):
            time_offset = timedelta(minutes=i * time_step_minutes)
            time_points.append(start_epoch + time_offset)
        
        return time_points
    
    @log_execution_time("conjunction_create_events")
    def create_conjunction_events(
        self,
        primary_norad_id: int,
        refined_results: List[RefinementResult],
        probability_threshold: float = 1e-6,
        primary_radius_meters: float = 5.0,
        secondary_default_radius_meters: float = 0.5
    ) -> List[ConjunctionEvent]:
        """
        Convert refined results to database conjunction events.
        
        Args:
            primary_norad_id: NORAD ID of primary object
            refined_results: List of refined conjunction results
            probability_threshold: Minimum Pc for event creation
            primary_radius_meters: Hardbody radius of primary object
            secondary_default_radius_meters: Default radius for secondary objects
            
        Returns:
            List of ConjunctionEvent objects ready for database storage
        """
        events = []
        
        for result in refined_results:
            try:
                # Create basic conjunction event
                event = ConjunctionEvent(
                    primary_norad_id=primary_norad_id,
                    secondary_norad_id=result.secondary_state.norad_id,  # This needs to be stored in state
                    tca_datetime=result.tca_datetime,
                    primary_x_eci=result.primary_state.x,
                    primary_y_eci=result.primary_state.y,
                    primary_z_eci=result.primary_state.z,
                    secondary_x_eci=result.secondary_state.x,
                    secondary_y_eci=result.secondary_state.y,
                    secondary_z_eci=result.secondary_state.z,
                    miss_distance_meters=result.miss_distance,
                    relative_velocity_mps=result.relative_velocity,
                    probability=0.0,  # Will be calculated separately
                    probability_method="pending",
                    screening_threshold_km=5.0,  # Default value
                    time_window_hours=24.0,  # Default value
                    primary_radius_meters=primary_radius_meters,
                    secondary_radius_meters=secondary_default_radius_meters,
                    analysis_version="1.0.0",
                    alert_threshold_exceeded=False
                )
                
                # Set alert flag based on thresholds
                if (result.miss_distance <= 1000.0 or  # 1km miss distance
                    result.relative_velocity <= 1000.0):  # 1km/s relative velocity
                    event.alert_threshold_exceeded = True
                
                events.append(event)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to create conjunction event",
                    error=str(e),
                    tca=result.tca_datetime.isoformat() if result.tca_datetime else "unknown"
                )
                continue
        
        self.logger.info(
            "Conjunction events created",
            result_count=len(refined_results),
            events_created=len(events)
        )
        
        return events


# Global screener instance
conjunction_screener = ConjunctionScreener(sgp4_engine=SGP4Engine())