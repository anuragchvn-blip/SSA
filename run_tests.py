"""Simple test runner for core components."""

import sys
import traceback
from datetime import datetime, timezone

# Import our components
try:
    from src.core.config import settings
    from src.data.models import TLE
    from src.propagation.sgp4_engine import SGP4Engine, CartesianState
    from src.conjunction.probability import ProbabilityCalculator
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_configuration():
    """Test configuration loading."""
    try:
        # This should work without environment variables for basic test
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_tle_creation():
    """Test TLE object creation."""
    try:
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
        print("✓ TLE creation test passed")
        return True
    except Exception as e:
        print(f"✗ TLE creation test failed: {e}")
        traceback.print_exc()
        return False

def test_sgp4_engine():
    """Test SGP4 engine functionality."""
    try:
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
        expected_sma = 6778000.0
        assert abs(keplerian.semi_major_axis - expected_sma) < 100000.0  # Within 100km
        
        # Test orbital period calculation
        period = engine.calculate_orbital_period(6778000.0)
        expected_period = 5520.0  # ~92 minutes
        assert abs(period - expected_period) < 300.0  # Within 5 minutes
        
        print("✓ SGP4 engine test passed")
        return True
    except Exception as e:
        print(f"✗ SGP4 engine test failed: {e}")
        traceback.print_exc()
        return False

def test_probability_calculator():
    """Test probability calculator functionality."""
    try:
        calc = ProbabilityCalculator()
        
        # Test Foster's method
        result = calc.compute_pc_foster_method(
            miss_distance=500.0,
            sigma_x=100.0,
            sigma_y=100.0,
            combined_radius=10.0
        )
        
        assert 0.0 <= result.probability <= 1.0
        assert result.method == "foster_2d"
        # For 500m miss distance with 100m uncertainties, expect moderate probability
        assert result.probability > 1e-6  # Should be detectable
        
        print("✓ Probability calculator test passed")
        return True
    except Exception as e:
        print(f"✗ Probability calculator test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running SSA Engine Core Component Tests")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_tle_creation,
        test_sgp4_engine,
        test_probability_calculator
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())