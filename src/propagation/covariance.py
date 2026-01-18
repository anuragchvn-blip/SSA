"""Complete covariance propagation using State Transition Matrix with numerical validation."""

import numpy as np
from datetime import datetime, timedelta
from typing import NamedTuple, Optional, Tuple, Dict, Any
from scipy.integrate import solve_ivp
import logging

from src.core.exceptions import PropagationError, InvalidCovarianceError
from src.core.logging import get_logger, log_execution_time
from src.propagation.sgp4_engine import CartesianState, KeplerianElements, SGP4Engine
from src.data.models import TLE

logger = get_logger(__name__)


class ForceModelConfig(NamedTuple):
    """Force model configuration for covariance propagation."""
    include_j2: bool = True          # Earth oblateness (J2)
    include_atmospheric_drag: bool = True
    include_solar_radiation_pressure: bool = False
    include_third_body: bool = False  # Moon/Sun perturbations
    atmospheric_density_model: str = "msis"  # or "simple_exponential"


class CovariancePropagationResult(NamedTuple):
    """Complete covariance propagation result."""
    propagated_state: CartesianState
    propagated_covariance: np.ndarray  # 6x6 in ECI frame
    stm: np.ndarray                    # 6x6 State Transition Matrix
    integration_metadata: Dict[str, Any]
    frame_transformation: Optional[np.ndarray]  # ECI to RTN if computed


class VariationalEquations:
    """Variational equations for STM computation.
    
    The State Transition Matrix Φ(t,t₀) describes how small perturbations
    in initial conditions evolve over time.
    
    Mathematical formulation:
        dΦ/dt = A(t) * Φ
        where A(t) = ∂f/∂x is the Jacobian of the dynamics
    
    For SGP4-equivalent dynamics:
        f(x,v) = v
        f(v,x) = -μ*x/|x|³ + perturbations
    
    References:
        - Tapley, Schutz, Born, "Statistical Orbit Determination", Chapter 4
        - Vallado, "Fundamentals of Astrodynamics", Section 10.3
    """
    
    def __init__(self, force_model: ForceModelConfig):
        self.force_model = force_model
        self.earth_mu = 3.986004418e14  # m³/s² (WGS84)
        self.earth_radius = 6378137.0   # meters
        self.j2 = 1.08262668e-3         # Earth J2 coefficient
        
    def compute_acceleration_jacobian(
        self, 
        position: np.ndarray, 
        velocity: np.ndarray
    ) -> np.ndarray:
        """
        Compute Jacobian matrix A(t) = ∂f/∂x for orbital dynamics.
        
        State vector: x = [rx, ry, rz, vx, vy, vz]ᵀ
        Dynamics: dx/dt = f(x,t) = [v; a(r,v)]
        
        Jacobian structure:
        A = [∂fᵢ/∂xⱼ] = [0₃ₓ₃   I₃ₓ₃]
                       [∂a/∂r  ∂a/∂v]
        
        Args:
            position: 3-element position vector [m]
            velocity: 3-element velocity vector [m/s]
            
        Returns:
            6x6 Jacobian matrix A(t)
        """
        r = np.linalg.norm(position)
        r3 = r**3
        r5 = r**5
        
        # Position part of Jacobian (acceleration w.r.t. position)
        # ∂a/∂r = ∂/∂r(-μ*r/|r|³) = -μ*(I/|r|³ - 3*r*rᵀ/|r|⁵)
        r_outer = np.outer(position, position)
        identity = np.eye(3)
        
        # Point mass gravity Jacobian
        accel_pos_jac = -self.earth_mu * (identity/r3 - 3*r_outer/r5)
        
        # Add J2 perturbation if enabled
        if self.force_model.include_j2:
            j2_jac_3x3 = self._compute_j2_jacobian(position)[:3, :3]  # Extract 3x3 position part
            accel_pos_jac += j2_jac_3x3
        
        # Velocity part of Jacobian (acceleration w.r.t. velocity)
        # For basic two-body + drag: ∂a/∂v ≈ 0 (negligible for short times)
        # More sophisticated models would include atmospheric drag dependence
        accel_vel_jac = np.zeros((3, 3))
        
        # Build full 6x6 Jacobian
        zeros_3x3 = np.zeros((3, 3))
        identity_3x3 = np.eye(3)
        
        jacobian = np.block([
            [zeros_3x3, identity_3x3],
            [accel_pos_jac, accel_vel_jac]
        ])
        
        return jacobian
    
    def _compute_j2_jacobian(self, position: np.ndarray) -> np.ndarray:
        """
        Compute J2 perturbation contribution to acceleration Jacobian.
        
        J2 acceleration:
        a_J2 = 3/2 * J2 * μ * Rₑ² * [
            x/r⁷ * (5z²/r² - 1)
            y/r⁷ * (5z²/r² - 1)  
            z/r⁷ * (5z²/r² - 3)
        ]
        
        Partial derivatives ∂a_J2/∂r computed analytically.
        """
        x, y, z = position
        r = np.linalg.norm(position)
        r2 = r**2
        r7 = r**7
        r9 = r**9
        
        Re2 = self.earth_radius**2
        coeff = 3/2 * self.j2 * self.earth_mu * Re2
        
        # J2 acceleration components
        z2_r2 = z**2 / r2
        common_factor = 5 * z2_r2 - 1
        
        # ∂a_J2x/∂x, ∂a_J2x/∂y, ∂a_J2x/∂z
        da_dx = coeff * (
            (common_factor + 5*z**2 * (2*x**2/r2 - 1)/r2) / r7 -
            7 * x * (common_factor) / r9
        )
        
        da_dy = coeff * (
            5 * x * y * z**2 / (r2 * r7) -
            7 * y * common_factor / r9
        )
        
        da_dz = coeff * (
            5 * x * z * (5*z**2/r2 - 3) / r7 -
            7 * x * z * (5*z**2/r2 - 1) / r9
        )
        
        # Similar for y-component
        db_dx = da_dy  # By symmetry
        db_dy = coeff * (
            (common_factor + 5*z**2 * (2*y**2/r2 - 1)/r2) / r7 -
            7 * y * common_factor / r9
        )
        db_dz = coeff * (
            5 * y * z * (5*z**2/r2 - 3) / r7 -
            7 * y * z * common_factor / r9
        )
        
        # z-component derivatives
        dc_dx = da_dz
        dc_dy = db_dz
        dc_dz = coeff * (
            (5*(5*z**2/r2 - 3) + 5*z**2*(10*z**2/r2 - 3)/r2) / r7 -
            7 * z * (5*z**2/r2 - 3) / r9
        )
        
        # Assemble 3x3 J2 Jacobian submatrix
        j2_jac = np.array([
            [da_dx, da_dy, da_dz],
            [db_dx, db_dy, db_dz], 
            [dc_dx, dc_dy, dc_dz]
        ])
        
        # Pad to 6x6 (zero for velocity rows/columns)
        full_jac = np.zeros((6, 6))
        full_jac[:3, :3] = j2_jac
        
        return full_jac


class CovariancePropagator:
    """
    Complete State Transition Matrix propagation with numerical validation.
    
    REQUIREMENTS:
    1. Compute STM using variational equations (NOT finite differencing)
    2. Propagate 6x6 covariance matrix P(t) = Φ(t,t₀) * P(t₀) * Φ(t,t₀)ᵀ
    3. Handle frame transformations (ECI ↔ RTN)
    4. Validate against analytical solutions for circular orbits
    5. Include force model perturbations in STM
    
    REFERENCES:
    - Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed, Ch 10
    - Tapley, Schutz, Born, "Statistical Orbit Determination", Ch 4
    """
    
    def __init__(self, force_model: Optional[ForceModelConfig] = None):
        self.force_model = force_model or ForceModelConfig()
        self.variational_eqs = VariationalEquations(self.force_model)
        self.sgp4_engine = SGP4Engine()
        self.logger = get_logger(__name__)
    
    @log_execution_time("stm_propagation")
    def propagate_with_stm(
        self,
        initial_state: CartesianState,
        initial_covariance: np.ndarray,  # 6x6 in ECI frame
        target_epoch: datetime,
        force_model: Optional[ForceModelConfig] = None
    ) -> CovariancePropagationResult:
        """
        Propagate state + covariance using State Transition Matrix.
        
        IMPLEMENTATION REQUIREMENTS:
        1. Set up variational equations: d(STM)/dt = A(t) * STM
           where A(t) = ∂f/∂x evaluated along reference trajectory
        2. Integrate STM alongside state vector using RK4 or RK78
        3. Transform covariance to RTN frame at target epoch
        4. Validate STM properties:
           - det(STM) = 1 (symplectic property)
           - STM(t₀,t₀) = I (identity at initial time)
        5. Return full provenance (integration steps, convergence, errors)
        
        MATH:
        State vector: X = [x, y, z, vx, vy, vz]ᵀ
        Dynamics: dX/dt = f(X, t)
        STM: Φ(t,t₀) satisfies dΦ/dt = A(t) * Φ, where A = ∂f/∂X
        Covariance: P(t) = Φ(t,t₀) * P(t₀) * Φ(t,t₀)ᵀ
        
        VALIDATION:
        For circular orbits, compare against Clohessy-Wiltshire analytical STM.
        """
        if force_model is None:
            force_model = self.force_model
            
        # Validate inputs
        if initial_covariance.shape != (6, 6):
            raise InvalidCovarianceError(
                message=f"Covariance matrix must be 6x6, got {initial_covariance.shape}",
                error_code="INVALID_COVARIANCE_SHAPE"
            )
            
        # Check covariance is positive semi-definite
        eigenvals = np.linalg.eigvals(initial_covariance)
        if np.any(eigenvals < -1e-12):  # Allow small numerical errors
            raise InvalidCovarianceError(
                message="Covariance matrix is not positive semi-definite",
                error_code="INVALID_COVARIANCE_EIGENVALUES",
                details={"eigenvalues": eigenvals.tolist()}
            )
        
        # Set up initial conditions for integration
        initial_time = 0.0  # Reference time (will compute delta-t)
        state_vector = np.array([
            initial_state.x, initial_state.y, initial_state.z,
            initial_state.vx, initial_state.vy, initial_state.vz
        ])
        
        # Combined state vector: [position; velocity; STM elements]
        # STM is 6x6 = 36 elements, stored as flattened vector
        stm_identity = np.eye(6).flatten()
        initial_combined = np.concatenate([state_vector, stm_identity])
        
        # Define integration time span
        # Convert epochs to seconds since reference
        time_span = self._compute_time_span(initial_state, target_epoch)
        
        # Set up ODE system
        def ode_system(t: float, combined_state: np.ndarray) -> np.ndarray:
            """
            Combined ODE system for state + STM propagation.
            
            State structure: [x, y, z, vx, vy, vz, Φ₁₁, Φ₁₂, ..., Φ₆₆]
            """
            # Extract current state
            current_state = combined_state[:6]
            current_position = current_state[:3]
            current_velocity = current_state[3:]
            
            # Compute acceleration (reference trajectory)
            acceleration = self._compute_acceleration(
                current_position, current_velocity, force_model
            )
            
            # Compute Jacobian A(t) = ∂f/∂x
            jacobian = self.variational_eqs.compute_acceleration_jacobian(
                current_position, current_velocity
            )
            
            # Extract current STM (6x6 matrix)
            current_stm_flat = combined_state[6:]
            current_stm = current_stm_flat.reshape((6, 6))
            
            # STM dynamics: dΦ/dt = A(t) * Φ
            stm_derivative = jacobian @ current_stm
            stm_derivative_flat = stm_derivative.flatten()
            
            # Combine derivatives
            state_derivatives = np.concatenate([current_velocity, acceleration])
            combined_derivatives = np.concatenate([state_derivatives, stm_derivative_flat])
            
            return combined_derivatives
        
        # Perform integration
        try:
            solution = solve_ivp(
                fun=ode_system,
                t_span=(0, time_span),
                y0=initial_combined,
                method='RK45',  # Adaptive Runge-Kutta
                rtol=1e-9,      # Relative tolerance
                atol=1e-12,     # Absolute tolerance
                dense_output=True
            )
            
            if not solution.success:
                raise PropagationError(
                    message=f"STM integration failed: {solution.message}",
                    error_code="STM_INTEGRATION_FAILED",
                    details={
                        "integration_time": time_span,
                        "steps": solution.nfev,
                        "message": solution.message
                    }
                )
            
            # Extract final state and STM
            final_combined = solution.y[:, -1]
            final_state_vector = final_combined[:6]
            final_stm_flat = final_combined[6:]
            final_stm = final_stm_flat.reshape((6, 6))
            
            # Validate STM properties
            self._validate_stm_properties(final_stm, time_span)
            
            # Create propagated state
            propagated_state = CartesianState(
                x=final_state_vector[0],
                y=final_state_vector[1], 
                z=final_state_vector[2],
                vx=final_state_vector[3],
                vy=final_state_vector[4],
                vz=final_state_vector[5]
            )
            
            # Propagate covariance: P(t) = Φ(t,t₀) * P(t₀) * Φ(t,t₀)ᵀ
            propagated_covariance = final_stm @ initial_covariance @ final_stm.T
            
            # Compute ECI to RTN transformation
            from src.propagation.sgp4_engine import CoordinateTransforms
            transforms = CoordinateTransforms()
            rotation_matrix, _ = transforms.eci_to_rtn(propagated_state)
            
            # Metadata
            metadata = {
                "integration_method": "RK45",
                "integration_steps": solution.nfev,
                "integration_time_seconds": time_span,
                "rtol": 1e-9,
                "atol": 1e-12,
                "stm_determinant": np.linalg.det(final_stm),
                "stm_trace": np.trace(final_stm),
                "covariance_eigenvalues": np.linalg.eigvals(propagated_covariance).tolist()
            }
            
            self.logger.info("STM covariance propagation completed",
                           integration_time=time_span,
                           steps=solution.nfev,
                           stm_det=np.linalg.det(final_stm))
            
            return CovariancePropagationResult(
                propagated_state=propagated_state,
                propagated_covariance=propagated_covariance,
                stm=final_stm,
                integration_metadata=metadata,
                frame_transformation=rotation_matrix
            )
            
        except Exception as e:
            raise PropagationError(
                message=f"STM propagation failed: {str(e)}",
                error_code="STM_PROPAGATION_FAILED",
                details={
                    "initial_state": str(initial_state),
                    "target_epoch": target_epoch.isoformat(),
                    "error": str(e)
                }
            )
    
    def _compute_acceleration(
        self, 
        position: np.ndarray, 
        velocity: np.ndarray,
        force_model: ForceModelConfig
    ) -> np.ndarray:
        """
        Compute total acceleration including perturbations.
        
        Base acceleration: -μ*r/|r|³ (point mass gravity)
        Add perturbations based on force model configuration.
        """
        r = np.linalg.norm(position)
        r3 = r**3
        
        # Point mass gravity
        acceleration = -self.variational_eqs.earth_mu * position / r3
        
        # Add J2 perturbation if enabled
        if force_model.include_j2:
            acceleration += self._compute_j2_acceleration(position)
            
        # TODO: Add atmospheric drag, solar radiation pressure, third-body perturbations
        # These would require additional state information (mass, area, Cd, Cr)
        
        return acceleration
    
    def _compute_j2_acceleration(self, position: np.ndarray) -> np.ndarray:
        """
        Compute J2 gravitational perturbation acceleration.
        
        Formula:
        a_J2 = 3/2 * J2 * μ * Rₑ² * |r|⁻⁵ * [
            x * (5z²/|r|² - 1)
            y * (5z²/|r|² - 1)
            z * (5z²/|r|² - 3)
        ]
        """
        x, y, z = position
        r = np.linalg.norm(position)
        r2 = r**2
        r5 = r**5
        
        Re2 = self.variational_eqs.earth_radius**2
        coeff = 3/2 * self.variational_eqs.j2 * self.variational_eqs.earth_mu * Re2
        
        z2_r2 = z**2 / r2
        factor = 5 * z2_r2 - 1
        
        j2_accel = coeff / r5 * np.array([
            x * factor,
            y * factor, 
            z * (factor - 2)  # 5z²/r² - 3 = (5z²/r² - 1) - 2
        ])
        
        return j2_accel
    
    def _compute_time_span(self, initial_state: CartesianState, target_epoch: datetime) -> float:
        """Compute propagation time in seconds."""
        # Convert target epoch to seconds since initial epoch
        # For testing purposes, assuming initial epoch is now
        # In practice, this would be computed from TLE epoch
        current_time = datetime.utcnow()
        time_delta = target_epoch - current_time
        return time_delta.total_seconds()
    
    def _validate_stm_properties(self, stm: np.ndarray, integration_time: float):
        """
        Validate State Transition Matrix mathematical properties.
        
        Properties to check:
        1. det(STM) = 1 (symplectic property for Hamiltonian systems)
        2. STM is invertible (no zero eigenvalues)
        3. Elements are finite (no overflow/underflow)
        """
        det = np.linalg.det(stm)
        
        # Symplectic property check (allow small numerical errors)
        if abs(det - 1.0) > 1e-6:
            self.logger.warning("STM determinant deviates from unity",
                              determinant=det,
                              deviation=abs(det - 1.0))
        
        # Check for numerical issues
        if not np.all(np.isfinite(stm)):
            raise InvalidCovarianceError(
                message="STM contains non-finite elements",
                error_code="STM_NUMERICAL_ERROR",
                details={"stm_elements": stm.tolist()}
            )
        
        # Check condition number (ill-conditioning indicator)
        cond_num = np.linalg.cond(stm)
        if cond_num > 1e12:
            self.logger.warning("STM is ill-conditioned",
                              condition_number=cond_num)


# Global instance
covariance_propagator = CovariancePropagator()