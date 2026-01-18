"""Unit tests for core SSA engine components."""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.core.exceptions import TLEValidationError, PropagationError
from src.data.models import TLE
from src.propagation.sgp4_engine import SGP4Engine, CartesianState
from src.conjunction.screening import ConjunctionScreener
from src.conjunction.probability import ProbabilityCalculator


class TestTLEValidation:
    """Test TLE validation and parsing."""
    
    def test_valid_tle_creation(self):
        """Test creation of valid TLE object."""
        # Valid ISS TLE (simplified)
        tle = TLE(
            norad_id=25544,
            classification='U',
            launch_year=98,
            launch_number=67,
            launch_piece='A',
            epoch_datetime=datetime(2026, 1, 17, 12, 0, 0, tzinfo=timezone.utc),
            mean_motion_derivative=-0.0001,
            mean_motion_sec_derivative=0.0,
            bstar_drag_term=0.0001,
            element_set_number=123,
            inclination_degrees=51.6416,
            raan_degrees=123.4567,
            eccentricity=0.0001234,
            argument_of_perigee_degrees=45.6789,
            mean_anomaly_degrees=123.4567,
            mean_motion_orbits_per_day=15.5,
            revolution_number_at_epoch=12345,
            tle_line1="1 25544U 98067A   26017.50000000  .00010000  00000-0  10000-3 0  1234",
            tle_line2="2 25544  51.6416 123.4567 0001234  45.6789 123.4567 15.50000000123450",
            epoch_julian_date=2459215.0,
            line1_checksum=4,
            line2_checksum=0,
            is_valid=True
        )
        
        assert tle.norad_id == 25544
        assert tle.is_valid == True
        assert tle.epoch_datetime.year == 2026
    
    def test_tle_validation_error_handling(self):
        """Test TLE validation error handling."""
        # Test missing required fields
        with pytest.raises(Exception):
            TLE()  # Should fail due to missing required fields


class TestSGP4Engine:
    """Test SGP4 orbital propagation engine."""
    
    def test_cartesian_to_keplerian_conversion(self):
        """Test conversion from Cartesian to Keplerian elements."""
        engine = SGP4Engine()
        
        # Test state vector (ISS-like orbit)
        state = CartesianState(
            x=6778000.0,    # meters
            y=0.0,
            z=0.0,
            vx=0.0,
            vy=7667.0,      # m/s (orbital velocity)
            vz=0.0
        )
        
        keplerian = engine._cartesian_to_keplerian(state)
        
        # Verify semi-major axis is approximately correct
        # For ISS altitude ~400km, SMA should be ~6778km
        expected_sma = 6778000.0
        assert abs(keplerian.semi_major_axis - expected_sma) < 100000.0  # Within 100km
        
        # Verify eccentricity is reasonable (nearly circular)
        assert keplerian.eccentricity < 0.1
    
    def test_orbital_period_calculation(self):
        """Test orbital period calculation."""
        engine = SGP4Engine()
        
        # ISS-like orbit: SMA = 6778 km
        sma = 6778000.0
        period = engine.calculate_orbital_period(sma)
        
        # Expected period ~92 minutes = 5520 seconds
        expected_period = 5520.0
        assert abs(period - expected_period) < 300.0  # Within 5 minutes


class TestProbabilityCalculator:
    """Test collision probability calculations."""
    
    def test_foster_method_basic_case(self):
        """Test Foster's method with basic parameters."""
        calc = ProbabilityCalculator()
        
        # Test case: 500m miss distance, 100m 1-sigma uncertainties, 10m radius
        result = calc.compute_pc_foster_method(
            miss_distance=500.0,
            sigma_x=100.0,
            sigma_y=100.0,
            combined_radius=10.0
        )
        
        assert 0.0 <= result.probability <= 1.0
        assert result.method == "foster_2d"
        assert result.convergence_achieved == True
        
        # Probability should be very small for large miss distance
        assert result.probability < 1e-6
    
    def test_foster_method_zero_miss_distance(self):
        """Test Foster's method with zero miss distance."""
        calc = ProbabilityCalculator()
        
        result = calc.compute_pc_foster_method(
            miss_distance=0.0,
            sigma_x=100.0,
            sigma_y=100.0,
            combined_radius=10.0
        )
        
        # Should be high probability when miss distance is zero
        assert result.probability > 0.9
    
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo method convergence."""
        calc = ProbabilityCalculator()
        
        # Simple test case
        miss_vector = np.array([100.0, 50.0, 0.0])  # meters
        covariance = np.diag([10000.0, 10000.0, 10000.0])  # 100m sigma each direction
        radius = 20.0  # meters
        
        result = calc.compute_pc_monte_carlo(
            miss_distance_vector=miss_vector,
            covariance_matrix=covariance,
            combined_radius=radius,
            n_samples=10000,
            convergence_threshold=0.05
        )
        
        assert 0.0 <= result.probability <= 1.0
        assert result.method == "monte_carlo"
        assert isinstance(result.n_samples, int)
        assert result.n_samples >= 10000


class TestConjunctionScreener:
    """Test conjunction screening logic."""
    
    def test_time_grid_generation(self):
        """Test generation of time grid for propagation."""
        screener = ConjunctionScreener(sgp4_engine=Mock())
        
        start_time = datetime(2026, 1, 17, 12, 0, 0, tzinfo=timezone.utc)
        time_points = screener._generate_time_grid(
            start_epoch=start_time,
            time_window_hours=2.0,
            time_step_minutes=30
        )
        
        # Should generate 5 points: 0, 30, 60, 90, 120 minutes
        assert len(time_points) == 5
        assert time_points[0] == start_time
        assert time_points[-1] == start_time + timedelta(minutes=120)


# Integration test examples (would require actual TLE data)
@pytest.mark.asyncio
async def test_spacetrack_client_initialization():
    """Test Space-Track client initialization."""
    with patch('src.data.ingest.spacetrack_client.httpx.AsyncClient') as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        
        # Import here to avoid initialization issues
        from src.data.ingest.spacetrack_client import SpaceTrackClient
        
        client = SpaceTrackClient()
        assert client is not None
        # Client initialization should not fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])