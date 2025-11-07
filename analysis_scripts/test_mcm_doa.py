"""
Test MCM Integration with DOA Estimation Module
================================================

Quick test to verify mutual coupling matrix (MCM) support works correctly.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geometry_processors.z5_processor import Z5ArrayProcessor
from doa_estimation.music import MUSICEstimator
from doa_estimation.metrics import DOAMetrics

print("="*70)
print("MCM INTEGRATION TEST")
print("="*70)

# Create Z5 array
print("\n1. Creating Z5 array (N=7)...")
processor = Z5ArrayProcessor(N=7, d=1.0)
results = processor.run_full_analysis()
print(f"   ✓ {results.num_sensors} sensors")
print(f"   ✓ Positions: {results.sensors_positions}")

# Test 1: WITHOUT MCM
print("\n2. Testing WITHOUT MCM...")
estimator_no_mcm = MUSICEstimator(
    sensor_positions=results.sensors_positions,
    wavelength=2.0,
    enable_mcm=False
)
print(f"   ✓ MCM enabled: {estimator_no_mcm.enable_mcm}")
print(f"   ✓ Coupling matrix: {estimator_no_mcm.coupling_matrix}")

# Simulate and estimate
true_angles = [-30, 0, 30]
signals = estimator_no_mcm.simulate_signals(
    true_angles=true_angles, SNR_dB=20, snapshots=500
)
est_no_mcm, _ = estimator_no_mcm.estimate(signals, K_sources=3, return_spectrum=True)

print(f"\n   True angles:        {[f'{a:.1f}°' for a in true_angles]}")
print(f"   Estimated (no MCM): {[f'{a:.1f}°' for a in est_no_mcm]}")

rmse_no_mcm = DOAMetrics.compute_rmse(true_angles, est_no_mcm)
print(f"   RMSE: {rmse_no_mcm:.3f}°")

# Test 2: WITH MCM (exponential model)
print("\n3. Testing WITH MCM (exponential model)...")
estimator_mcm = MUSICEstimator(
    sensor_positions=results.sensors_positions,
    wavelength=2.0,
    enable_mcm=True,
    mcm_model='exponential',
    mcm_params={'c1': 0.3, 'alpha': 0.5}
)
print(f"   ✓ MCM enabled: {estimator_mcm.enable_mcm}")
print(f"   ✓ Coupling matrix shape: {estimator_mcm.coupling_matrix.shape}")
print(f"   ✓ Coupling matrix type: {type(estimator_mcm.coupling_matrix)}")

# Simulate and estimate (with coupling effects)
true_angles_mcm = [-30, 0, 30]
signals_mcm = estimator_mcm.simulate_signals(
    true_angles=true_angles_mcm, SNR_dB=20, snapshots=500
)
est_mcm, _ = estimator_mcm.estimate(signals_mcm, K_sources=3, return_spectrum=True)

print(f"\n   True angles:      {[f'{a:.1f}°' for a in true_angles_mcm]}")
print(f"   Estimated (MCM):  {[f'{a:.1f}°' for a in est_mcm]}")

rmse_mcm = DOAMetrics.compute_rmse(true_angles_mcm, est_mcm)
print(f"   RMSE: {rmse_mcm:.3f}°")

# Test 3: WITH MCM (toeplitz model)
print("\n4. Testing WITH MCM (toeplitz model)...")
estimator_toep = MUSICEstimator(
    sensor_positions=results.sensors_positions,
    wavelength=2.0,
    enable_mcm=True,
    mcm_model='toeplitz',
    mcm_params={'coupling_values': [1.0, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]}
)
print(f"   ✓ MCM enabled: {estimator_toep.enable_mcm}")
print(f"   ✓ Coupling matrix shape: {estimator_toep.coupling_matrix.shape}")

# Simulate and estimate
true_angles_toep = [-30, 0, 30]
signals_toep = estimator_toep.simulate_signals(
    true_angles=true_angles_toep, SNR_dB=20, snapshots=500
)
est_toep, _ = estimator_toep.estimate(signals_toep, K_sources=3, return_spectrum=True)

print(f"\n   True angles:          {[f'{a:.1f}°' for a in true_angles_toep]}")
print(f"   Estimated (toeplitz): {[f'{a:.1f}°' for a in est_toep]}")

rmse_toep = DOAMetrics.compute_rmse(true_angles_toep, est_toep)
print(f"   RMSE: {rmse_toep:.3f}°")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  No MCM:    RMSE = {rmse_no_mcm:.3f}°")
print(f"  MCM (exp): RMSE = {rmse_mcm:.3f}°")
print(f"  MCM (toep): RMSE = {rmse_toep:.3f}°")

# Check if MCM causes degradation (expected with coupling)
if rmse_mcm > rmse_no_mcm * 1.1:
    print("\n  ✓ MCM causes performance degradation (expected behavior)")
else:
    print("\n  Note: MCM performance similar to no-MCM case")

print("\n✓ MCM integration test COMPLETE!")
print("="*70)
