"""Collect and label TLE sequences for maneuver detection training."""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import NamedTuple, List, Dict, Any, Optional
from dataclasses import dataclass

from src.data.ingest.spacetrack_client import SpaceTrackClient, RateLimiter
from src.propagation.sgp4_engine import SGP4Engine, CartesianState
from src.data.models import TLE
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LabeledExample:
    """Single labeled training example."""
    features: np.ndarray
    label: int  # 0 = no maneuver, 1 = maneuver
    metadata: Dict[str, Any]


class LabeledDataset(NamedTuple):
    """Complete labeled dataset for ML training."""
    features: pd.DataFrame
    labels: np.ndarray
    metadata: List[Dict[str, Any]]
    feature_names: List[str]


class ManeuverDataCollector:
    """
    Collect and label TLE sequences for maneuver detection training.
    
    DATA SOURCES:
    1. Starlink satellites (frequent station-keeping maneuvers)
    2. ISS (documented debris avoidance and reboost maneuvers)
    3. GEO satellites (known station-keeping windows)
    
    LABELING STRATEGY:
    - Positive examples: TLE sequences with documented maneuvers
    - Negative examples: TLE sequences with no maneuvers (natural drift only)
    - Features: Residual position/velocity, orbital element changes
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter(
            requests_per_hour=settings.spacetrack.spacetrack_rate_limit
        )
        self.client = SpaceTrackClient(self.rate_limiter)
        self.sgp4_engine = SGP4Engine()
        self.logger = get_logger(__name__)
    
    async def collect_starlink_maneuvers(
        self,
        num_satellites: int = 100,
        time_window_days: int = 90
    ) -> LabeledDataset:
        """
        Collect Starlink TLE sequences with maneuver labels.
        
        Starlink satellites maneuver frequently for orbit raising and
        station-keeping. We can identify these from TLE discontinuities.
        
        ALGORITHM:
        1. Fetch TLE history for N Starlink satellites
        2. Propagate each TLE forward to next TLE epoch
        3. Compare predicted vs. actual position
        4. Residual > threshold → label as maneuver
        5. Create feature vectors for ML training
        
        Returns:
            LabeledDataset with:
            - features: DataFrame (residuals, element changes, etc.)
            - labels: array (0 = no maneuver, 1 = maneuver detected)
            - metadata: provenance for each example
        """
        logger.info("Collecting Starlink maneuver data",
                   num_satellites=num_satellites,
                   time_window_days=time_window_days)
        
        # Get Starlink satellite IDs (example range - would query catalog in practice)
        starlink_ids = list(range(44231, 44231 + min(num_satellites, 100)))
        
        # Collect data for each satellite
        all_examples = []
        
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def process_satellite(norad_id: int):
            async with semaphore:
                try:
                    examples = await self._collect_satellite_maneuvers(
                        norad_id, time_window_days
                    )
                    return examples
                except Exception as e:
                    logger.warning("Failed to collect data for satellite",
                                 norad_id=norad_id, error=str(e))
                    return []
        
        # Process satellites concurrently
        tasks = [process_satellite(sid) for sid in starlink_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for result in results:
            if isinstance(result, Exception):
                continue
            all_examples.extend(result)
        
        logger.info(f"Collected {len(all_examples)} training examples")
        
        # Convert to dataset format
        return self._examples_to_dataset(all_examples)
    
    async def _collect_satellite_maneuvers(
        self, 
        norad_id: int, 
        time_window_days: int
    ) -> List[LabeledExample]:
        """
        Collect maneuver examples for a single satellite.
        
        Uses residual analysis to detect maneuvers:
        - Propagate TLE_i to TLE_{i+1} epoch
        - Compare prediction with actual TLE_{i+1}
        - Large residuals indicate maneuvers
        """
        # Download TLE history
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_window_days)
        
        async with self.client:
            tle_history = await self.client.download_tles_for_satellite(
                norad_id=norad_id,
                start_date=start_date,
                end_date=end_date
            )
        
        if len(tle_history) < 3:  # Need at least 3 TLEs for meaningful analysis
            return []
        
        examples = []
        
        # Process consecutive TLE pairs
        for i in range(len(tle_history) - 2):
            try:
                # Get three consecutive TLEs: prev -> current -> next
                prev_tle_data = tle_history[i]
                current_tle_data = tle_history[i + 1]
                next_tle_data = tle_history[i + 2]
                
                # Convert to TLE objects
                prev_tle = self._tle_dict_to_model(prev_tle_data)
                current_tle = self._tle_dict_to_model(current_tle_data)
                next_tle = self._tle_dict_to_model(next_tle_data)
                
                # Create example from current->next transition
                example = self._create_maneuver_example(
                    prev_tle, current_tle, next_tle
                )
                
                if example is not None:
                    examples.append(example)
                    
            except Exception as e:
                logger.debug("Failed to process TLE sequence",
                           norad_id=norad_id, index=i, error=str(e))
                continue
        
        return examples
    
    def _create_maneuver_example(
        self, 
        prev_tle: TLE, 
        current_tle: TLE, 
        next_tle: TLE
    ) -> Optional[LabeledExample]:
        """
        Create labeled example from TLE sequence.
        
        Computes features based on orbital element differences and
        propagation residuals to detect maneuvers.
        """
        try:
            # Propagate current TLE to next TLE epoch
            propagation_result = self.sgp4_engine.propagate_to_epoch(
                current_tle, next_tle.epoch_datetime
            )
            
            # Compute residuals (difference between predicted and actual)
            predicted_state = propagation_result.cartesian_state
            actual_state = self._tle_to_cartesian_state(next_tle)
            
            # Position and velocity residuals
            pos_residual = np.array([
                actual_state.x - predicted_state.x,
                actual_state.y - predicted_state.y,
                actual_state.z - predicted_state.z
            ])
            
            vel_residual = np.array([
                actual_state.vx - predicted_state.vx,
                actual_state.vy - predicted_state.vy,
                actual_state.vz - predicted_state.vz
            ])
            
            pos_residual_norm = np.linalg.norm(pos_residual)
            vel_residual_norm = np.linalg.norm(vel_residual)
            
            # Orbital element differences
            current_elements = self.sgp4_engine._cartesian_to_keplerian(
                self._tle_to_cartesian_state(current_tle)
            )
            next_elements = self.sgp4_engine._cartesian_to_keplerian(actual_state)
            
            element_differences = np.array([
                next_elements.semi_major_axis - current_elements.semi_major_axis,
                next_elements.eccentricity - current_elements.eccentricity,
                next_elements.inclination - current_elements.inclination,
                next_elements.raan - current_elements.raan,
                next_elements.argument_of_perigee - current_elements.argument_of_perigee,
                next_elements.true_anomaly - current_elements.true_anomaly
            ])
            
            # Time interval between TLEs
            time_delta_hours = (next_tle.epoch_datetime - current_tle.epoch_datetime).total_seconds() / 3600
            
            # Label determination (heuristic thresholds)
            # Large residuals or significant orbital changes indicate maneuvers
            maneuver_detected = (
                pos_residual_norm > 1000.0 or  # 1km position jump
                vel_residual_norm > 10.0 or    # 10m/s velocity jump
                abs(element_differences[0]) > 10000.0 or  # 10km SMA change
                abs(element_differences[1]) > 0.01 or     # Eccentricity change
                abs(element_differences[2]) > 0.1         # 0.1 rad inclination change
            )
            
            # Create feature vector
            features = np.concatenate([
                pos_residual,           # 3 elements
                vel_residual,           # 3 elements
                element_differences,    # 6 elements
                [time_delta_hours,      # 1 element
                 pos_residual_norm,     # 1 element
                 vel_residual_norm]     # 1 element
            ])
            
            # Create metadata
            metadata = {
                "norad_id": current_tle.norad_id,
                "prev_epoch": prev_tle.epoch_datetime.isoformat(),
                "current_epoch": current_tle.epoch_datetime.isoformat(),
                "next_epoch": next_tle.epoch_datetime.isoformat(),
                "time_delta_hours": time_delta_hours,
                "pos_residual_norm": pos_residual_norm,
                "vel_residual_norm": vel_residual_norm,
                "maneuver_detected": maneuver_detected,
                "element_changes": element_differences.tolist()
            }
            
            return LabeledExample(
                features=features,
                label=1 if maneuver_detected else 0,
                metadata=metadata
            )
            
        except Exception as e:
            logger.debug("Failed to create maneuver example", error=str(e))
            return None
    
    def _tle_dict_to_model(self, tle_data: Dict) -> TLE:
        """Convert Space-Track TLE dictionary to TLE model."""
        # Parse epoch
        epoch_str = tle_data["EPOCH"]
        epoch_dt = datetime.fromisoformat(epoch_str.rstrip('Z'))
        
        # Parse international designator
        intl_des = tle_data.get("INTLDES", "00000A")
        launch_year = int(intl_des[:2]) if len(intl_des) >= 2 else 0
        launch_number = int(intl_des[2:5]) if len(intl_des) >= 5 else 0
        launch_piece = intl_des[5:] if len(intl_des) > 5 else "A"
        
        return TLE(
            norad_id=int(tle_data["NORAD_CAT_ID"]),
            classification=tle_data.get("CLASSIFICATION_TYPE", "U"),
            launch_year=launch_year,
            launch_number=launch_number,
            launch_piece=launch_piece,
            epoch_datetime=epoch_dt,
            mean_motion_derivative=float(tle_data.get("MEAN_MOTION_DOT", 0.0)),
            mean_motion_sec_derivative=float(tle_data.get("MEAN_MOTION_DDOT", 0.0)),
            bstar_drag_term=float(tle_data.get("BSTAR", 0.0)),
            element_set_number=int(tle_data.get("ELEMENT_SET_NO", 1)),
            inclination_degrees=float(tle_data["INCLINATION"]),
            raan_degrees=float(tle_data["RA_OF_ASC_NODE"]),
            eccentricity=float(tle_data["ECCENTRICITY"]),
            argument_of_perigee_degrees=float(tle_data["ARG_OF_PERICENTER"]),
            mean_anomaly_degrees=float(tle_data["MEAN_ANOMALY"]),
            mean_motion_orbits_per_day=float(tle_data["MEAN_MOTION"]),
            revolution_number_at_epoch=int(tle_data.get("REV_AT_EPOCH", 1)),
            tle_line1=tle_data["TLE_LINE1"],
            tle_line2=tle_data["TLE_LINE2"],
            epoch_julian_date=2451545.0,  # Placeholder
            line1_checksum=0,
            line2_checksum=0,
            is_valid=True
        )
    
    def _tle_to_cartesian_state(self, tle: TLE) -> CartesianState:
        """Convert TLE to Cartesian state at its epoch."""
        result = self.sgp4_engine.propagate_to_epoch(tle, tle.epoch_datetime)
        return result.cartesian_state
    
    def _examples_to_dataset(self, examples: List[LabeledExample]) -> LabeledDataset:
        """Convert list of examples to structured dataset."""
        if not examples:
            return LabeledDataset(
                features=pd.DataFrame(),
                labels=np.array([]),
                metadata=[],
                feature_names=[]
            )
        
        # Define feature names
        feature_names = [
            "pos_residual_x", "pos_residual_y", "pos_residual_z",
            "vel_residual_x", "vel_residual_y", "vel_residual_z",
            "sma_change", "ecc_change", "inc_change", 
            "raan_change", "arg_peri_change", "true_anom_change",
            "time_delta_hours", "pos_residual_norm", "vel_residual_norm"
        ]
        
        # Extract features and labels
        features_list = [example.features for example in examples]
        labels = np.array([example.label for example in examples])
        metadata = [example.metadata for example in examples]
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list, columns=feature_names)
        
        return LabeledDataset(
            features=features_df,
            labels=labels,
            metadata=metadata,
            feature_names=feature_names
        )
    
    async def collect_balanced_dataset(
        self,
        positive_samples: int = 500,
        negative_samples: int = 500,
        time_window_days: int = 180
    ) -> LabeledDataset:
        """
        Collect balanced dataset with specified number of positive/negative examples.
        
        Strategy:
        - Collect more data than needed
        - Filter/select to achieve desired balance
        - Mix different satellite types for diversity
        """
        logger.info("Collecting balanced maneuver dataset",
                   positive_target=positive_samples,
                   negative_target=negative_samples)
        
        # Collect data from multiple satellite constellations
        starlink_dataset = await self.collect_starlink_maneuvers(
            num_satellites=150,  # More satellites for better sampling
            time_window_days=time_window_days
        )
        
        # Balance the dataset
        balanced_dataset = self._balance_dataset(
            starlink_dataset, positive_samples, negative_samples
        )
        
        logger.info(f"Balanced dataset created:")
        logger.info(f"  Positive examples: {np.sum(balanced_dataset.labels)}")
        logger.info(f"  Negative examples: {len(balanced_dataset.labels) - np.sum(balanced_dataset.labels)}")
        logger.info(f"  Total features: {len(balanced_dataset.feature_names)}")
        
        return balanced_dataset
    
    def _balance_dataset(
        self, 
        dataset: LabeledDataset, 
        positive_target: int, 
        negative_target: int
    ) -> LabeledDataset:
        """Create balanced dataset by sampling."""
        # Separate positive and negative examples
        positive_indices = np.where(dataset.labels == 1)[0]
        negative_indices = np.where(dataset.labels == 0)[0]
        
        # Sample to achieve target balance
        positive_sample = np.random.choice(
            positive_indices, 
            size=min(positive_target, len(positive_indices)), 
            replace=False
        )
        negative_sample = np.random.choice(
            negative_indices,
            size=min(negative_target, len(negative_indices)),
            replace=False
        )
        
        # Combine indices
        selected_indices = np.concatenate([positive_sample, negative_sample])
        selected_indices.sort()
        
        # Create balanced dataset
        balanced_features = dataset.features.iloc[selected_indices]
        balanced_labels = dataset.labels[selected_indices]
        balanced_metadata = [dataset.metadata[i] for i in selected_indices]
        
        return LabeledDataset(
            features=balanced_features,
            labels=balanced_labels,
            metadata=balanced_metadata,
            feature_names=dataset.feature_names
        )


# Example usage
async def collect_training_data():
    """Collect training data for maneuver detection."""
    collector = ManeuverDataCollector()
    
    # Collect balanced dataset
    dataset = await collector.collect_balanced_dataset(
        positive_samples=300,
        negative_samples=300,
        time_window_days=120
    )
    
    print(f"Training dataset collected:")
    print(f"  Samples: {len(dataset.labels)}")
    print(f"  Features: {len(dataset.feature_names)}")
    print(f"  Positive ratio: {np.mean(dataset.labels):.3f}")
    print(f"  Feature names: {dataset.feature_names}")
    
    # Save dataset
    dataset.features.to_csv("maneuver_training_features.csv", index=False)
    pd.Series(dataset.labels).to_csv("maneuver_training_labels.csv", index=False, header=False)
    
    print("Dataset saved to CSV files")
    
    return dataset


if __name__ == "__main__":
    # Run data collection
    asyncio.run(collect_training_data())