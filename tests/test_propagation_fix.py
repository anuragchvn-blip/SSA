#!/usr/bin/env python3
"""Test full covariance propagation after matrix fix."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime, timedelta
from src.propagation.covariance import CovariancePropagator, CartesianState

def test_full_propagation():
    print("Testing full covariance propagation...")
    
    # Create propagator
    propagator = CovariancePropagator()
    print("✓ Covariance propagator created")
    
    # Create test scenario (ISS-like orbit)
    altitude = 400_000.0  # 400km
    earth_radius = 6378137.0
    semi_major_axis = earth_radius + altitude
    velocity_mag = np.sqrt(3.986004418e14 / semi_major_axis)
    
    initial_state = CartesianState(
        x=semi_major_axis,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=velocity_mag,
        vz=0.0
    )
    
    # Simple covariance matrix
    initial_covariance = np.diag([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])
    
    target_epoch = datetime.utcnow() + timedelta(hours=1)
    
    print(f"Initial state: [{initial_state.x/1000:.0f}km, {initial_state.vy/1000:.3f}km/s]")
    print(f"Target epoch: {target_epoch}")
    
    try:
        # This should work now with the matrix fix
        result = propagator.propagate_with_stm(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            target_epoch=target_epoch
        )
        
        print("✓ Full propagation completed successfully!")
        print(f"  Final STM shape: {result.stm.shape}")
        print(f"  Propagated covariance shape: {result.propagated_covariance.shape}")
        print(f"  STM determinant: {np.linalg.det(result.stm):.6f}")
        
        # Quick validation
        assert result.stm.shape == (6, 6)
        assert result.propagated_covariance.shape == (6, 6)
        assert abs(np.linalg.det(result.stm) - 1.0) < 0.1  # Symplectic property
        
        print("✓ All validations passed!")
        return True
        
    except Exception as e:
        print(f"✗ Propagation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_propagation()
    if success:
        print("\n🎉 Covariance propagation is now working correctly!")
    else:
        print("\n❌ Still having issues with covariance propagation")