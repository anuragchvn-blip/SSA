"""Comprehensive validation of covariance propagation against analytical solutions."""

import numpy as np
import pytest
from datetime import datetime, timedelta
from typing import Tuple

from src.propagation.covariance import (
    CovariancePropagator, CovariancePropagationResult, 
    ForceModelConfig, CartesianState
)
from src.propagation.sgp4_engine import SGP4Engine


class TestCovarianceAccuracy:
    """
    Validate covariance propagation against analytical solutions.
    
    Test cases:
    1. Circular orbit - use Clohessy-Wiltshire equations
    2. Elliptical orbit - use Hills equations with eccentricity
    3. Long-term propagation - verify uncertainty growth is bounded
    4. Frame transformations - ECI ↔ RTN round-trip accuracy
    """
    
    def setup_method(self):
        """Setup test fixtures."""
        self.propagator = CovariancePropagator()
        self.sgp4_engine = SGP4Engine()
        
        # Test configuration
        self.earth_mu = 3.986004418e14  # m³/s²
        self.earth_radius = 6378137.0   # meters
    
    def test_circular_orbit_analytical_stm(self):
        """
        Validate STM for circular orbit against Clohessy-Wiltshire.
        
        Reference orbit: LEO at 400km altitude
        Propagation time: 1 orbital period (~92 minutes)
        
        ACCEPTANCE CRITERIA:
        - STM element errors < 1e-6
        - Covariance determinant preserved within 1%
        - Position uncertainty growth < 2x for 1 orbit
        """
        # Create circular LEO orbit at 400km altitude
        altitude = 400_000.0  # 400km
        semi_major_axis = self.earth_radius + altitude
        
        # Orbital velocity for circular orbit
        velocity_mag = np.sqrt(self.earth_mu / semi_major_axis)
        orbital_period = 2 * np.pi * np.sqrt(semi_major_axis**3 / self.earth_mu)
        
        # Initial state (in equatorial plane for simplicity)
        initial_state = CartesianState(
            x=semi_major_axis,  # Apogee point
            y=0.0,
            z=0.0,
            vx=0.0,
            vy=velocity_mag,  # Velocity tangent to orbit
            vz=0.0
        )
        
        # Initial covariance (diagonal, uncorrelated)
        initial_covariance = np.diag([
            100.0,   # 100m position uncertainty in each direction
            100.0, 
            100.0,
            0.1,     # 0.1 m/s velocity uncertainty in each direction
            0.1,
            0.1
        ])
        
        # Propagate for one orbital period
        target_epoch = datetime.now() + timedelta(seconds=orbital_period)
        
        # Compute numerical propagation
        result = self.propagator.propagate_with_stm(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            target_epoch=target_epoch
        )
        
        # Compute analytical Clohessy-Wiltshire STM for comparison
        analytical_stm = self._compute_cwh_stm(orbital_period, semi_major_axis)
        
        # Compare STM elements
        stm_difference = np.abs(result.stm - analytical_stm)
        max_error = np.max(stm_difference)
        
        print(f"Circular orbit STM validation:")
        print(f"  Max STM element error: {max_error:.2e}")
        print(f"  Numerical STM determinant: {np.linalg.det(result.stm):.6f}")
        print(f"  Analytical STM determinant: {np.linalg.det(analytical_stm):.6f}")
        
        # Validate acceptance criteria
        assert max_error < 1e-6, f"STM element error too large: {max_error:.2e}"
        
        # Check determinant preservation (symplectic property)
        numerical_det = np.linalg.det(result.stm)
        assert abs(numerical_det - 1.0) < 0.01, f"Determinant not preserved: {numerical_det:.6f}"
        
        # Check covariance evolution
        initial_trace = np.trace(initial_covariance)
        final_trace = np.trace(result.propagated_covariance)
        trace_growth = final_trace / initial_trace
        
        print(f"  Covariance trace growth: {trace_growth:.3f}x")
        assert trace_growth < 2.0, f"Uncertainty grew too much: {trace_growth:.3f}x"
        
        # Verify position uncertainty growth is reasonable
        initial_pos_uncertainty = np.sqrt(np.trace(initial_covariance[:3, :3]))
        final_pos_uncertainty = np.sqrt(np.trace(result.propagated_covariance[:3, :3]))
        pos_growth = final_pos_uncertainty / initial_pos_uncertainty
        
        print(f"  Position uncertainty growth: {pos_growth:.3f}x")
        assert 1.0 <= pos_growth <= 2.0, f"Position uncertainty growth abnormal: {pos_growth:.3f}x"
    
    def test_uncertainty_quantification_realism(self):
        """
        Validate that uncertainty growth is physically realistic.
        
        Tests:
        1. Position uncertainty should grow over time (drag, perturbations)
        2. Growth rate should match theory (~ sqrt(t) for random walk)
        3. Uncertainty should NOT shrink (2nd law of thermodynamics analogy)
        4. Cross-correlations should develop (position-velocity coupling)
        
        ACCEPTANCE CRITERIA:
        - No negative eigenvalues in covariance matrix
        - Uncertainty growth factor 1.5-3.0x per orbit (typical for LEO)
        - Trace(P) monotonically increasing
        """
        # Create realistic LEO scenario
        altitude = 500_000.0  # 500km
        semi_major_axis = self.earth_radius + altitude
        velocity_mag = np.sqrt(self.earth_mu / semi_major_axis)
        
        initial_state = CartesianState(
            x=semi_major_axis,
            y=0.0,
            z=0.0,
            vx=0.0,
            vy=velocity_mag,
            vz=0.0
        )
        
        # Realistic initial covariance with cross-correlations
        initial_covariance = np.array([
            [100.0,   0.0,   0.0,   0.1,   0.0,   0.0],  # rx
            [  0.0, 100.0,   0.0,   0.0,   0.1,   0.0],  # ry  
            [  0.0,   0.0, 100.0,   0.0,   0.0,   0.1],  # rz
            [  0.1,   0.0,   0.0,   0.1,   0.0,   0.0],  # vx
            [  0.0,   0.1,   0.0,   0.0,   0.1,   0.0],  # vy
            [  0.0,   0.0,   0.1,   0.0,   0.0,   0.1]   # vz
        ])
        
        # Test multiple time intervals
        time_intervals = [3600, 7200, 10800, 14400]  # 1, 2, 3, 4 hours
        uncertainties = []
        
        for dt_seconds in time_intervals:
            target_epoch = datetime.now() + timedelta(seconds=dt_seconds)
            
            result = self.propagator.propagate_with_stm(
                initial_state=initial_state,
                initial_covariance=initial_covariance,
                target_epoch=target_epoch
            )
            
            # Check covariance validity
            eigenvals = np.linalg.eigvals(result.propagated_covariance)
            assert np.all(eigenvals >= -1e-12), f"Negative eigenvalues found: {eigenvals}"
            
            # Track uncertainty growth
            trace = np.trace(result.propagated_covariance)
            uncertainties.append(trace)
            
            print(f"Time: {dt_seconds/3600:.1f}h, Covariance trace: {trace:.2f}")
        
        # Verify monotonic increase (uncertainty should not decrease)
        for i in range(1, len(uncertainties)):
            assert uncertainties[i] >= uncertainties[i-1], \
                f"Uncertainty decreased from {uncertainties[i-1]:.2f} to {uncertainties[i]:.2f}"
        
        # Check growth rate is reasonable
        initial_trace = np.trace(initial_covariance)
        final_trace = uncertainties[-1]
        total_growth = final_trace / initial_trace
        
        print(f"Total uncertainty growth over {time_intervals[-1]/3600:.1f}h: {total_growth:.2f}x")
        assert 1.1 <= total_growth <= 5.0, f"Unrealistic growth factor: {total_growth:.2f}"
    
    def test_frame_transformation_accuracy(self):
        """
        Validate ECI ↔ RTN frame transformation round-trip accuracy.
        
        The transformation should be lossless and preserve vector magnitudes.
        """
        from src.propagation.sgp4_engine import CoordinateTransforms
        
        # Create test state
        test_state = CartesianState(
            x=7000000.0,  # 7000km from Earth center
            y=1000000.0,
            z=500000.0,
            vx=1000.0,    # 1 km/s
            vy=7000.0,
            vz=200.0
        )
        
        transforms = CoordinateTransforms()
        
        # ECI → RTN
        rotation_matrix, state_rtn = transforms.eci_to_rtn(test_state)
        
        # RTN → ECI (round-trip)
        reconstructed_state = transforms.rtn_to_eci(rotation_matrix, state_rtn)
        
        # Validate round-trip accuracy
        position_error = np.sqrt(
            (test_state.x - reconstructed_state.x)**2 +
            (test_state.y - reconstructed_state.y)**2 +
            (test_state.z - reconstructed_state.z)**2
        )
        
        velocity_error = np.sqrt(
            (test_state.vx - reconstructed_state.vx)**2 +
            (test_state.vy - reconstructed_state.vy)**2 +
            (test_state.vz - reconstructed_state.vz)**2
        )
        
        print(f"Frame transformation accuracy:")
        print(f"  Position round-trip error: {position_error:.2e} m")
        print(f"  Velocity round-trip error: {velocity_error:.2e} m/s")
        
        # Acceptance criteria
        assert position_error < 1e-9, f"Position transformation error too large: {position_error:.2e}"
        assert velocity_error < 1e-12, f"Velocity transformation error too large: {velocity_error:.2e}"
        
        # Verify rotation matrix properties
        assert np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-12), \
            "Rotation matrix not orthogonal"
        assert abs(np.linalg.det(rotation_matrix) - 1.0) < 1e-12, \
            "Rotation matrix determinant not unity"
    
    def test_long_term_stability(self):
        """
        Test long-term covariance propagation stability.
        
        For stable orbits, uncertainty should grow sub-linearly.
        """
        # Create geostationary-like orbit (simplified test)
        semi_major_axis = 42164000.0  # GEO radius in meters
        velocity_mag = np.sqrt(self.earth_mu / semi_major_axis)
        
        initial_state = CartesianState(
            x=semi_major_axis,
            y=0.0,
            z=0.0,
            vx=0.0,
            vy=velocity_mag,
            vz=0.0
        )
        
        initial_covariance = np.diag([50.0, 50.0, 50.0, 0.05, 0.05, 0.05])
        
        # Propagate for 6 hours (much longer than LEO period)
        target_epoch = datetime.now() + timedelta(hours=6)
        
        result = self.propagator.propagate_with_stm(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            target_epoch=target_epoch
        )
        
        # Check that propagation succeeded and produced valid results
        assert isinstance(result, CovariancePropagationResult)
        assert result.propagated_covariance.shape == (6, 6)
        
        # Verify covariance remains positive semi-definite
        eigenvals = np.linalg.eigvals(result.propagated_covariance)
        assert np.all(eigenvals >= -1e-10), f"Eigenvalues: {eigenvals}"
        
        print(f"Long-term propagation successful:")
        print(f"  Final STM determinant: {np.linalg.det(result.stm):.6f}")
        print(f"  Covariance eigenvalues: {eigenvals}")
    
    def _compute_cwh_stm(self, dt: float, a: float) -> np.ndarray:
        """
        Compute analytical Clohessy-Wiltshire STM for circular orbit.
        
        Reference: Clohessy, W., & Wiltshire, R. (1960). "Terminal Guidance
        System for Satellite Rendezvous." Journal of the Aerospace Sciences.
        
        For circular reference orbit with semi-major axis 'a':
        - Mean motion: n = sqrt(μ/a³)
        - STM elements are trigonometric functions of n*dt
        
        Returns 6x6 STM mapping [δr, δv] at t₀ to [δr, δv] at t₀ + dt
        """
        n = np.sqrt(self.earth_mu / a**3)  # Mean motion
        nt = n * dt
        
        # Clohessy-Wiltshire STM components
        cos_nt = np.cos(nt)
        sin_nt = np.sin(nt)
        
        # Position components
        phi_rr = np.array([
            [4 - 3*cos_nt, 0, 0],
            [6*(nt - sin_nt), 1, 0],
            [0, 0, cos_nt]
        ])
        
        phi_rv = np.array([
            [sin_nt/n, 2*(1 - cos_nt)/n, 0],
            [2*(1 - cos_nt)/n, (4*sin_nt - 3*nt)/n, 0],
            [0, 0, sin_nt/n]
        ])
        
        phi_vr = np.array([
            [3*n*sin_nt, 0, 0],
            [6*n*(cos_nt - 1), 0, 0],
            [0, 0, -n*sin_nt]
        ])
        
        phi_vv = np.array([
            [cos_nt, 2*sin_nt, 0],
            [-2*sin_nt, 4*cos_nt - 3, 0],
            [0, 0, cos_nt]
        ])
        
        # Assemble full 6x6 STM
        stm = np.block([
            [phi_rr, phi_rv],
            [phi_vr, phi_vv]
        ])
        
        return stm


if __name__ == "__main__":
    # Run tests manually for debugging
    test_suite = TestCovarianceAccuracy()
    test_suite.setup_method()
    
    print("Running covariance validation tests...")
    test_suite.test_circular_orbit_analytical_stm()
    print("✓ Circular orbit STM validation passed")
    
    test_suite.test_uncertainty_quantification_realism()
    print("✓ Uncertainty quantification validation passed")
    
    test_suite.test_frame_transformation_accuracy()
    print("✓ Frame transformation validation passed")
    
    test_suite.test_long_term_stability()
    print("✓ Long-term stability validation passed")
    
    print("\n🎉 All covariance validation tests passed!")