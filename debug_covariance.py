#!/usr/bin/env python3
"""Debug test for covariance propagation matrix shapes."""

import numpy as np
from datetime import datetime, timedelta
from src.propagation.covariance import CovariancePropagator, ForceModelConfig, CartesianState

def debug_covariance_shapes():
    """Debug the matrix shapes in covariance propagation."""
    print("Debugging covariance propagation matrix shapes...")
    
    # Create simple test case
    propagator = CovariancePropagator()
    
    # Create test state (ISS-like orbit)
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
    initial_covariance = np.eye(6) * 100.0  # 100m and 0.1m/s uncertainties
    
    target_epoch = datetime.utcnow() + timedelta(hours=1)
    
    print(f"Initial state: {initial_state}")
    print(f"Initial covariance shape: {initial_covariance.shape}")
    print(f"Target epoch: {target_epoch}")
    
    try:
        result = propagator.propagate_with_stm(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            target_epoch=target_epoch
        )
        print("✓ Propagation completed successfully")
        print(f"Final STM shape: {result.stm.shape}")
        print(f"Propagated covariance shape: {result.propagated_covariance.shape}")
        return True
    except Exception as e:
        print(f"✗ Propagation failed: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_covariance_shapes()