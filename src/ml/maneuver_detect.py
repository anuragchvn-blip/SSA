"""Machine Learning module for maneuver detection using interpretable models."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import pickle
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import shap

from src.core.logging import get_logger, log_execution_time
from src.core.exceptions import BaseSSAException
from src.data.models import TLE, SatelliteState
from src.propagation.sgp4_engine import SGP4Engine, CartesianState

logger = get_logger(__name__)


@dataclass
class ManeuverDetectionResult:
    """Result of maneuver detection analysis."""
    norad_id: int
    detection_datetime: datetime
    maneuver_detected: bool
    confidence: float
    detection_method: str
    feature_importance: Dict[str, float]
    shap_values: Optional[List[float]]
    features_used: Dict[str, float]


class ManeuverDetectionError(BaseSSAException):
    """Exception for maneuver detection errors."""
    pass


class ManeuverDetector:
    """
    ML-based maneuver detector using interpretable models.
    
    Uses Random Forest classifier on engineered features from TLE residuals.
    Features include position/velocity residuals, orbital element changes,
    and temporal patterns that indicate potential maneuvers.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.explainer = None  # SHAP explainer
        self.feature_names = [
            'position_residual_magnitude',
            'velocity_residual_magnitude',
            'along_track_residual',
            'cross_track_residual',
            'radial_residual',
            'days_since_last_tle',
            'sma_change_meters',
            'eccentricity_change',
            'inclination_change_degrees',
            'raan_change_degrees',
            'argument_perigee_change_degrees',
            'mean_anomaly_change_degrees',
            'mean_motion_change',
            'rate_of_position_residual_change',
            'rate_of_velocity_residual_change'
        ]
        self.is_trained = False
        
    @log_execution_time("maneuver_detector_extract_features")
    def extract_features(
        self,
        tle_history: List[TLE],
        propagated_states: List[SatelliteState],
        current_tle: TLE
    ) -> pd.DataFrame:
        """
        Extract features for maneuver detection from TLE history.
        
        Args:
            tle_history: Historical TLEs for the satellite
            propagated_states: Propagated states from previous TLEs
            current_tle: Current TLE to compare against predictions
            
        Returns:
            DataFrame with extracted features for ML model
        """
        if len(tle_history) < 2:
            raise ManeuverDetectionError(
                message="Need at least 2 TLEs for maneuver detection",
                error_code="MANEUVER_DETECTION_FAILED"
            )
        
        # Use SGP4 to propagate previous TLE to current epoch
        sgp4_engine = SGP4Engine()
        prev_tle = tle_history[-2]  # Previous TLE
        
        # Propagate previous TLE to current epoch
        propagation_result = sgp4_engine.propagate_to_epoch(prev_tle, current_tle.epoch_datetime)
        predicted_state = propagation_result.cartesian_state
        
        # Get actual state from current TLE
        actual_prop_result = sgp4_engine.propagate_to_epoch(current_tle, current_tle.epoch_datetime)
        actual_state = actual_prop_result.cartesian_state
        
        # Calculate residuals
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
        
        # Calculate orbital element differences
        prev_keplerian = sgp4_engine._cartesian_to_keplerian(propagation_result.cartesian_state)
        curr_keplerian = sgp4_engine._cartesian_to_keplerian(actual_prop_result.cartesian_state)
        
        # Calculate time since last TLE
        time_diff = (current_tle.epoch_datetime - prev_tle.epoch_datetime).total_seconds()
        days_since_last = time_diff / (24 * 3600)
        
        # Calculate rate of change features (if we have more than 2 TLEs)
        rate_pos_residual_change = 0.0
        rate_vel_residual_change = 0.0
        
        if len(tle_history) >= 3:
            # Get the TLE before the previous one
            prev_prev_tle = tle_history[-3]
            
            # Propagate to middle epoch
            mid_result = sgp4_engine.propagate_to_epoch(prev_prev_tle, prev_tle.epoch_datetime)
            mid_actual = sgp4_engine.propagate_to_epoch(prev_tle, prev_tle.epoch_datetime)
            
            prev_pos_residual = np.array([
                mid_actual.cartesian_state.x - mid_result.cartesian_state.x,
                mid_actual.cartesian_state.y - mid_result.cartesian_state.y,
                mid_actual.cartesian_state.z - mid_result.cartesian_state.z
            ])
            
            prev_vel_residual = np.array([
                mid_actual.cartesian_state.vx - mid_result.cartesian_state.vx,
                mid_actual.cartesian_state.vy - mid_result.cartesian_state.vy,
                mid_actual.cartesian_state.vz - mid_result.cartesian_state.vz
            ])
            
            # Calculate rates of change
            time_between_prev = (prev_tle.epoch_datetime - prev_prev_tle.epoch_datetime).total_seconds() / (24 * 3600)
            if time_between_prev > 0:
                rate_pos_residual_change = np.linalg.norm(pos_residual) - np.linalg.norm(prev_pos_residual) / time_between_prev
                rate_vel_residual_change = np.linalg.norm(vel_residual) - np.linalg.norm(prev_vel_residual) / time_between_prev
        
        # Create feature dictionary
        features = {
            'position_residual_magnitude': float(np.linalg.norm(pos_residual)),
            'velocity_residual_magnitude': float(np.linalg.norm(vel_residual)),
            'along_track_residual': float(pos_residual[1]),  # Assuming Y is along-track
            'cross_track_residual': float(pos_residual[0]),  # Assuming X is cross-track
            'radial_residual': float(pos_residual[2]),  # Assuming Z is radial
            'days_since_last_tle': float(days_since_last),
            'sma_change_meters': float(curr_keplerian.semi_major_axis - prev_keplerian.semi_major_axis),
            'eccentricity_change': float(curr_keplerian.eccentricity - prev_keplerian.eccentricity),
            'inclination_change_degrees': float(np.degrees(curr_keplerian.inclination - prev_keplerian.inclination)),
            'raan_change_degrees': float(np.degrees(curr_keplerian.raan - prev_keplerian.raan)),
            'argument_perigee_change_degrees': float(np.degrees(curr_keplerian.argument_of_perigee - prev_keplerian.argument_of_perigee)),
            'mean_anomaly_change_degrees': float(np.degrees(curr_keplerian.true_anomaly - prev_keplerian.true_anomaly)),
            'mean_motion_change': float(actual_prop_result.keplerian_elements.semi_major_axis - propagation_result.keplerian_elements.semi_major_axis),  # Approx
            'rate_of_position_residual_change': float(rate_pos_residual_change),
            'rate_of_velocity_residual_change': float(rate_vel_residual_change)
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        logger.debug(
            "Features extracted for maneuver detection",
            norad_id=current_tle.norad_id,
            feature_vector=features
        )
        
        return df
    
    @log_execution_time("maneuver_detector_train")
    def train(
        self,
        feature_data: pd.DataFrame,
        labels: List[bool],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the maneuver detection model.
        
        Args:
            feature_data: DataFrame with features for training
            labels: True/false labels indicating maneuver occurrence
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics and results
        """
        if len(feature_data) != len(labels):
            raise ManeuverDetectionError(
                message="Feature data and labels must have same length",
                error_code="MANEUVER_DETECTION_FAILED"
            )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_data, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info("Starting model training", training_samples=len(X_train))
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        try:
            test_auc = roc_auc_score(y_test, test_pred)
        except ValueError:
            test_auc = 0.0  # Handle edge case
        
        # Generate classification report
        report = classification_report(y_test, test_pred, output_dict=True)
        
        # Train SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Mark as trained
        self.is_trained = True
        
        metrics = {
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'classification_report': report,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        logger.info("Model training completed", **metrics)
        
        return metrics
    
    @log_execution_time("maneuver_detector_predict")
    def predict(
        self,
        features: pd.DataFrame,
        include_shap: bool = True
    ) -> List[ManeuverDetectionResult]:
        """
        Predict maneuver detection for given features.
        
        Args:
            features: DataFrame with features to predict on
            include_shap: Whether to include SHAP explanations
            
        Returns:
            List of maneuver detection results
        """
        if not self.is_trained:
            raise ManeuverDetectionError(
                message="Model must be trained before prediction",
                error_code="MANEUVER_DETECTION_FAILED"
            )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions and probabilities
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        results = []
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Get confidence as max probability
            confidence = float(max(prob))
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Calculate SHAP values if requested
            shap_values = None
            if include_shap and self.explainer:
                try:
                    shap_vals = self.explainer.shap_values(features_scaled[i:i+1])
                    # For binary classification, take the positive class
                    shap_values = shap_vals[1][0].tolist() if len(shap_vals) > 1 else shap_vals[0].tolist()
                except Exception as e:
                    logger.warning("SHAP calculation failed", error=str(e))
                    shap_values = None
            
            result = ManeuverDetectionResult(
                norad_id=int(features.iloc[i].get('norad_id', 0)),
                detection_datetime=datetime.now(),
                maneuver_detected=bool(pred),
                confidence=confidence,
                detection_method="random_forest",
                feature_importance=feature_importance,
                shap_values=shap_values,
                features_used=features.iloc[i].to_dict()
            )
            
            results.append(result)
        
        logger.info(
            "Maneuver detection predictions completed",
            prediction_count=len(results),
            positive_detections=sum(1 for r in results if r.maneuver_detected)
        )
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ManeuverDetectionError(
                message="Cannot save untrained model",
                error_code="MANEUVER_DETECTION_FAILED"
            )
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'explainer': self.explainer,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully", filepath=filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.explainer = model_data['explainer']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info("Model loaded successfully", filepath=filepath)


# Global maneuver detector instance
maneuver_detector = ManeuverDetector()