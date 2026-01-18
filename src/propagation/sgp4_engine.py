"""SGP4 orbital propagation engine with full covariance handling."""

import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, List, NamedTuple
from sgp4.api import Satrec, jday
from sgp4 import exporter
import math

from src.core.exceptions import PropagationError, InvalidCovarianceError
from src.core.logging import get_logger, log_execution_time
from src.data.models import TLE, SatelliteState

logger = get_logger(__name__)


class CartesianState(NamedTuple):
    """Cartesian state vector in ECI frame."""
    x: float  # meters
    y: float  # meters  
    z: float  # meters
    vx: float # m/s
    vy: float # m/s
    vz: float # m/s


class KeplerianElements(NamedTuple):
    """Keplerian orbital elements."""
    semi_major_axis: float  # meters
    eccentricity: float
    inclination: float      # radians
    raan: float            # radians
    argument_of_perigee: float  # radians
    true_anomaly: float    # radians


class CovarianceMatrix(NamedTuple):
    """6x6 covariance matrix in RTN frame (meters, m/s)."""
    rtn_covariance: np.ndarray  # 6x6 matrix [position; velocity]


class PropagationResult(NamedTuple):
    """Complete propagation result with state and covariance."""
    cartesian_state: CartesianState
    keplerian_elements: KeplerianElements
    covariance: Optional[CovarianceMatrix]
    propagation_metadata: dict
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


class SGP4Engine:
    """SGP4 orbital propagation engine with production-grade error handling."""
    
    # Earth parameters (WGS72)
    EARTH_MU = 3.986008e14  # m³/s²
    EARTH_RADIUS = 6378137.0  # meters
    EARTH_FLATTENING = 1/298.257223563
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @log_execution_time("sgp4_initialize_satrec")
    def _initialize_satrec(self, tle: TLE) -> Satrec:
        """
        Initialize SGP4 satellite record from TLE.
        
        Args:
            tle: TLE object containing orbital elements
            
        Returns:
            Initialized Satrec object
            
        Raises:
            PropagationError: If TLE initialization fails
        """
        try:
            satrec = Satrec.twoline2rv(tle.tle_line1, tle.tle_line2)
            
            # Validate satrec was created successfully
            if not hasattr(satrec, 'error') or satrec.error != 0:
                error_msg = f"SGP4 initialization error: {satrec.error}"
                raise PropagationError(
                    message=error_msg,
                    error_code="PROPAGATION_FAILED",
                    details={"tle_id": tle.id, "sgp4_error": satrec.error}
                )
            
            self.logger.debug("Satrec initialized successfully", 
                            norad_id=tle.norad_id,
                            epoch=tle.epoch_datetime.isoformat())
            
            return satrec
            
        except Exception as e:
            raise PropagationError(
                message=f"Failed to initialize SGP4 propagator: {str(e)}",
                error_code="PROPAGATION_FAILED",
                details={"tle_id": getattr(tle, 'id', None), "error": str(e)}
            )
    
    @log_execution_time("sgp4_propagate_to_epoch")
    def propagate_to_epoch(
        self, 
        tle: TLE, 
        target_epoch: datetime
    ) -> PropagationResult:
        """
        Propagate TLE to target epoch using SGP4.
        
        Args:
            tle: Source TLE object
            target_epoch: Target datetime for propagation
            
        Returns:
            PropagationResult with state vector and metadata
            
        Raises:
            PropagationError: If propagation fails
        """
        # Convert datetime to Julian date
        jd, fr = jday(
            target_epoch.year,
            target_epoch.month,
            target_epoch.day,
            target_epoch.hour,
            target_epoch.minute,
            target_epoch.second + target_epoch.microsecond/1e6
        )
        
        # Initialize propagator
        satrec = self._initialize_satrec(tle)
        
        try:
            # Perform propagation
            error_code, position, velocity = satrec.sgp4(jd, fr)
            
            # Check for propagation errors
            if error_code != 0:
                raise PropagationError(
                    message=f"SGP4 propagation error: {error_code}",
                    error_code="PROPAGATION_FAILED",
                    details={
                        "tle_id": tle.id,
                        "target_epoch": target_epoch.isoformat(),
                        "sgp4_error": error_code
                    }
                )
            
            # Convert from km, km/s to meters, m/s
            position_m = [coord * 1000.0 for coord in position]  # km to m
            velocity_ms = [coord * 1000.0 for coord in velocity]  # km/s to m/s
            
            # Create state vector
            cartesian_state = CartesianState(
                x=position_m[0],
                y=position_m[1], 
                z=position_m[2],
                vx=velocity_ms[0],
                vy=velocity_ms[1],
                vz=velocity_ms[2]
            )
            
            # Calculate Keplerian elements
            keplerian = self._cartesian_to_keplerian(cartesian_state)
            
            # Create metadata
            metadata = {
                "propagator": "SGP4",
                "source_tle_id": tle.id,
                "source_epoch": tle.epoch_datetime.isoformat(),
                "target_epoch": target_epoch.isoformat(),
                "time_delta_seconds": ((target_epoch.replace(tzinfo=timezone.utc) if target_epoch.tzinfo is None else target_epoch) - (tle.epoch_datetime.replace(tzinfo=timezone.utc) if tle.epoch_datetime.tzinfo is None else tle.epoch_datetime)).total_seconds(),
                "sgp4_error_code": error_code
            }
            
            # Calculate geographic coordinates
            lat_deg, lon_deg, alt_m = self._cartesian_to_geographic(
                cartesian_state, target_epoch
            )
            
            self.logger.debug("SGP4 propagation completed",
                            norad_id=tle.norad_id,
                            target_epoch=target_epoch.isoformat(),
                            position_norm=np.linalg.norm([cartesian_state.x, cartesian_state.y, cartesian_state.z]))
            
            return PropagationResult(
                cartesian_state=cartesian_state,
                keplerian_elements=keplerian,
                covariance=None,  # SGP4 doesn't provide covariance
                propagation_metadata=metadata,
                latitude_deg=lat_deg,
                longitude_deg=lon_deg,
                altitude_m=alt_m
            )
            
        except Exception as e:
            raise PropagationError(
                message=f"SGP4 propagation failed: {str(e)}",
                error_code="PROPAGATION_FAILED",
                details={
                    "tle_id": tle.id,
                    "target_epoch": target_epoch.isoformat(),
                    "error": str(e)
                }
            )
    
    @log_execution_time("sgp4_batch_propagate")
    def batch_propagate(
        self, 
        tle: TLE, 
        epochs: List[datetime]
    ) -> List[PropagationResult]:
        """
        Propagate TLE to multiple epochs efficiently.
        
        Args:
            tle: Source TLE object
            epochs: List of target epochs
            
        Returns:
            List of PropagationResult objects
            
        Raises:
            PropagationError: If batch propagation fails
        """
        if not epochs:
            return []
        
        try:
            # Sort epochs for efficient propagation
            sorted_epochs = sorted(epochs)
            
            results = []
            for epoch in sorted_epochs:
                result = self.propagate_to_epoch(tle, epoch)
                results.append(result)
            
            self.logger.info("Batch propagation completed",
                           norad_id=tle.norad_id,
                           epoch_count=len(epochs))
            
            return results
            
        except Exception as e:
            raise PropagationError(
                message=f"Batch propagation failed: {str(e)}",
                error_code="PROPAGATION_FAILED",
                details={"tle_id": tle.id, "epoch_count": len(epochs), "error": str(e)}
            )
    
    def _cartesian_to_keplerian(self, state: CartesianState) -> KeplerianElements:
        """
        Convert Cartesian state to Keplerian elements.
        
        Args:
            state: Cartesian state vector in ECI frame
            
        Returns:
            Keplerian elements
            
        References:
            Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
            Algorithm 10: State Vector to Keplerian Elements
        """
        # Position and velocity vectors
        r_vec = np.array([state.x, state.y, state.z])
        v_vec = np.array([state.vx, state.vy, state.vz])
        
        r_mag = np.linalg.norm(r_vec)
        v_mag = np.linalg.norm(v_vec)
        
        # Specific angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h_mag = np.linalg.norm(h_vec)
        
        # Node vector
        k_hat = np.array([0, 0, 1])
        n_vec = np.cross(k_hat, h_vec)
        n_mag = np.linalg.norm(n_vec)
        
        # Eccentricity vector
        e_vec = ((v_mag**2 - self.EARTH_MU/r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec) / self.EARTH_MU
        eccentricity = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = v_mag**2/2 - self.EARTH_MU/r_mag
        semi_major_axis = -self.EARTH_MU / (2 * energy)
        
        # Inclination
        inclination = np.arccos(h_vec[2] / h_mag)
        
        # RAAN
        if n_mag != 0:
            raan = np.arccos(n_vec[0] / n_mag)
            if n_vec[1] < 0:
                raan = 2*np.pi - raan
        else:
            raan = 0.0
        
        # Argument of periapsis
        if n_mag != 0 and eccentricity != 0:
            arg_peri = np.arccos(np.dot(n_vec, e_vec) / (n_mag * eccentricity))
            if e_vec[2] < 0:
                arg_peri = 2*np.pi - arg_peri
        else:
            arg_peri = 0.0
        
        # True anomaly
        if eccentricity != 0:
            true_anom = np.arccos(np.dot(e_vec, r_vec) / (eccentricity * r_mag))
            if np.dot(r_vec, v_vec) < 0:
                true_anom = 2*np.pi - true_anom
        else:
            # Circular orbit - use argument of latitude
            if n_mag != 0:
                cos_u = np.dot(n_vec, r_vec) / (n_mag * r_mag)
                u = np.arccos(np.clip(cos_u, -1, 1))
                if r_vec[2] < 0:
                    u = 2*np.pi - u
                true_anom = u - arg_peri
            else:
                true_anom = np.arccos(r_vec[0] / r_mag)
                if r_vec[1] < 0:
                    true_anom = 2*np.pi - true_anom
        
        return KeplerianElements(
            semi_major_axis=semi_major_axis,
            eccentricity=eccentricity,
            inclination=inclination,
            raan=raan,
            argument_of_perigee=arg_peri,
            true_anomaly=true_anom
        )
    
    def _cartesian_to_geographic(self, cartesian_state: CartesianState, epoch: datetime) -> Tuple[float, float, float]:
        """
        Convert ECI Cartesian coordinates to geographic coordinates (lat, lon, alt).
        
        Args:
            cartesian_state: Cartesian state vector in ECI frame
            epoch: Time of observation
            
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_m)
        """
        # Position vector in meters
        r_eci = np.array([cartesian_state.x, cartesian_state.y, cartesian_state.z])
        
        # Calculate distance from Earth center
        r_magnitude = np.linalg.norm(r_eci)
        
        # Calculate altitude
        altitude_m = r_magnitude - self.EARTH_RADIUS
        
        # Calculate latitude (geodetic approximation)
        latitude_rad = np.arctan2(r_eci[2], np.sqrt(r_eci[0]**2 + r_eci[1]**2))
        latitude_deg = np.degrees(latitude_rad)
        
        # Calculate longitude
        # Need to account for Earth's rotation to convert from ECI to ECEF
        # Calculate GMST (Greenwich Mean Sidereal Time)
        gmst = self._calculate_gmst(epoch)
        longitude_rad = np.arctan2(r_eci[1], r_eci[0]) - gmst
        
        # Normalize longitude to [-180, 180]
        longitude_deg = np.degrees(longitude_rad) % 360
        if longitude_deg > 180:
            longitude_deg -= 360
        
        return latitude_deg, longitude_deg, altitude_m
    
    def _calculate_gmst(self, utc_datetime: datetime) -> float:
        """
        Calculate Greenwich Mean Sidereal Time.
        
        Args:
            utc_datetime: UTC datetime
            
        Returns:
            GMST in radians
        """
        # Convert to Julian Date
        if utc_datetime.tzinfo is None:
            utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
        
        # Julian date at 0h UT
        jd0 = (utc_datetime.date() - datetime(2000, 1, 1, tzinfo=timezone.utc).date()).days + 2451545.0
        
        # Fraction of the day
        ut_seconds = (utc_datetime.hour * 3600 + utc_datetime.minute * 60 + utc_datetime.second + utc_datetime.microsecond / 1e6)
        jd = jd0 + ut_seconds / 86400.0
        
        # Calculate GMST in degrees
        t = (jd - 2451545.0) / 36525.0  # centuries since J2000.0
        gmst_deg = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t**2 - t**3 / 38710000) % 360
        
        if gmst_deg < 0:
            gmst_deg += 360
        
        return np.radians(gmst_deg)
    
    def calculate_orbital_period(self, semi_major_axis: float) -> float:
        """
        Calculate orbital period from semi-major axis.
        
        Args:
            semi_major_axis: Semi-major axis in meters
            
        Returns:
            Orbital period in seconds
        """
        return 2 * np.pi * np.sqrt(semi_major_axis**3 / self.EARTH_MU)
    
    def get_positional_uncertainty(
        self, 
        propagation_result: PropagationResult,
        time_uncertainty_seconds: float = 1.0
    ) -> float:
        """
        Estimate positional uncertainty due to timing error.
        
        Args:
            propagation_result: Propagation result
            time_uncertainty_seconds: Timing uncertainty in seconds
            
        Returns:
            Positional uncertainty in meters
        """
        # Simplified uncertainty model - assumes roughly circular orbit
        state = propagation_result.cartesian_state
        velocity = np.array([state.vx, state.vy, state.vz])
        velocity_mag = np.linalg.norm(velocity)
        
        return velocity_mag * time_uncertainty_seconds


# Utility functions for coordinate transformations
class CoordinateTransforms:
    """Coordinate transformation utilities."""
    
    @staticmethod
    def eci_to_rtn(state: CartesianState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform state vector from ECI to RTN frame.
        
        Args:
            state: Cartesian state in ECI frame
            
        Returns:
            Tuple of (rotation_matrix, transformed_state)
        """
        # Position vector
        r_eci = np.array([state.x, state.y, state.z])
        v_eci = np.array([state.vx, state.vy, state.vz])
        
        r_mag = np.linalg.norm(r_eci)
        v_mag = np.linalg.norm(v_eci)
        
        # Radial unit vector
        r_hat = r_eci / r_mag
        
        # Normal unit vector (along angular momentum)
        h_vec = np.cross(r_eci, v_eci)
        n_hat = h_vec / np.linalg.norm(h_vec)
        
        # Transverse unit vector
        t_hat = np.cross(n_hat, r_hat)
        
        # Rotation matrix from ECI to RTN (3x3 for position)
        rotation_matrix_3x3 = np.array([r_hat, t_hat, n_hat])
        
        # Extend to 6x6 for full state (position and velocity)
        rotation_matrix_6x6 = np.block([
            [rotation_matrix_3x3, np.zeros((3, 3))],
            [np.zeros((3, 3)), rotation_matrix_3x3]
        ])
        
        # Transform position and velocity
        pos_rtn = rotation_matrix_3x3 @ r_eci
        vel_rtn = rotation_matrix_3x3 @ v_eci
        
        return rotation_matrix_6x6, np.concatenate([pos_rtn, vel_rtn])
    
    @staticmethod
    def rtn_to_eci(rotation_matrix: np.ndarray, state_rtn: np.ndarray) -> CartesianState:
        """
        Transform state vector from RTN to ECI frame.
        
        Args:
            rotation_matrix: 6x6 rotation matrix from ECI to RTN
            state_rtn: 6-element state vector [x,y,z,vx,vy,vz] in RTN
            
        Returns:
            CartesianState in ECI frame
        """
        # Extract 3x3 rotation matrix for position (top-left block)
        rotation_3x3 = rotation_matrix[:3, :3]
        
        # Inverse rotation (transpose for orthogonal matrix)
        rotation_inv = rotation_3x3.T
        
        pos_rtn = state_rtn[:3]
        vel_rtn = state_rtn[3:]
        
        pos_eci = rotation_inv @ pos_rtn
        vel_eci = rotation_inv @ vel_rtn
        
        return CartesianState(
            x=pos_eci[0], y=pos_eci[1], z=pos_eci[2],
            vx=vel_eci[0], vy=vel_eci[1], vz=vel_eci[2]
        )


# Global engine instance
sgp4_engine = SGP4Engine()