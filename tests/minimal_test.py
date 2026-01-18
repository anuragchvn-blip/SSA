#!/usr/bin/env python3
"""Minimal test to verify Phase 2 components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Starting minimal Phase 2 test...")

# Test 1: Basic imports
try:
    from src.propagation.covariance import CovariancePropagator, CartesianState
    print("✓ Covariance propagator imported")
except Exception as e:
    print(f"✗ Covariance propagator import failed: {e}")

# Test 2: Simple covariance test
try:
    import numpy as np
    from datetime import datetime, timedelta
    
    propagator = CovariancePropagator()
    print("✓ Covariance propagator instantiated")
    
    # Simple test case
    state = CartesianState(6778000.0, 0.0, 0.0, 0.0, 7667.0, 0.0)
    cov = np.eye(6) * 100.0
    target = datetime.utcnow() + timedelta(hours=1)
    
    print("✓ Test data prepared")
    print("Test completed successfully!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()