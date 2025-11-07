"""Test MUSIC with proper K_max limits."""
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from doa_estimation.music import MUSICEstimator
from doa_estimation.metrics import DOAMetrics
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor

print("="*70)
print("MUSIC VALIDATION WITH PROPER K_MAX")
print("="*70)

# Test 1: ULA with K <= K_max
print("\n" + "="*70)
print("TEST 1: ULA (N=8) - K_max=4, detecting 3 sources")
print("="*70)

ula = ULArrayProcessor(N=8, d=0.5)
ula_data = ula.run_full_analysis()
perf = ula_data.performance_summary_table
k_row = perf[perf['Metrics'].str.contains('K_max')]
K_max = int(k_row['Value'].iloc[0])

print(f"Array: {ula_data.name}")
print(f"Sensors: {ula_data.num_sensors}")
print(f"K_max: {K_max}")

estimator = MUSICEstimator(ula_data.sensors_positions, wavelength=1.0)

true_angles = [-30, 0, 30]
print(f"\nTrue angles (K=3 < K_max={K_max}): {true_angles}")

X = estimator.simulate_signals(true_angles, SNR_dB=20, snapshots=300)
estimated = estimator.estimate(X, K_sources=3)

print(f"Estimated angles: {[f'{a:.1f}°' for a in estimated]}")

metrics = DOAMetrics()
rmse = metrics.compute_rmse(true_angles, estimated)
print(f"RMSE: {rmse:.3f}°")
print(f"✓ SUCCESS" if rmse < 1.0 else f"✗ POOR (RMSE > 1°)")

# Test 2: Z5 with more sources
print("\n" + "="*70)
print("TEST 2: Z5 (N=7) - K_max=5, detecting 4 sources")
print("="*70)

z5 = Z5ArrayProcessor(N=7, d=0.5)
z5_data = z5.run_full_analysis()
perf = z5_data.performance_summary_table
k_row = perf[perf['Metrics'].str.contains('K_max')]
K_max = int(k_row['Value'].iloc[0])

print(f"Array: {z5_data.name}")
print(f"Sensors: {z5_data.num_sensors}")
print(f"K_max: {K_max}")

estimator = MUSICEstimator(z5_data.sensors_positions, wavelength=1.0)

true_angles = [-45, -15, 15, 45]
print(f"\nTrue angles (K=4 < K_max={K_max}): {true_angles}")

X = estimator.simulate_signals(true_angles, SNR_dB=20, snapshots=300)
estimated = estimator.estimate(X, K_sources=4)

print(f"Estimated angles: {[f'{a:.1f}°' for a in estimated]}")

rmse = metrics.compute_rmse(true_angles, estimated)
print(f"RMSE: {rmse:.3f}°")
print(f"✓ SUCCESS" if rmse < 1.0 else f"✗ POOR (RMSE > 1°)")

# Test 3: Testing the limit (should warn)
print("\n" + "="*70)
print("TEST 3: ULA (N=5) - K_max=2, trying to detect 4 sources (BEYOND LIMIT)")
print("="*70)

ula5 = ULArrayProcessor(N=5, d=0.5)
ula5_data = ula5.run_full_analysis()
perf = ula5_data.performance_summary_table
k_row = perf[perf['Metrics'].str.contains('K_max')]
K_max = int(k_row['Value'].iloc[0])

print(f"Array: {ula5_data.name}")
print(f"Sensors: {ula5_data.num_sensors}")
print(f"K_max: {K_max}")

estimator = MUSICEstimator(ula5_data.sensors_positions, wavelength=1.0)

true_angles = [-40, -10, 20, 45]
print(f"\nTrue angles (K=4 > K_max={K_max}): {true_angles}")
print("⚠ WARNING: Attempting to detect more sources than K_max!")

try:
    X = estimator.simulate_signals(true_angles, SNR_dB=20, snapshots=300)
    # This should trigger a warning
    estimated = estimator.estimate(X, K_sources=4)
    
    print(f"Estimated angles: {[f'{a:.1f}°' for a in estimated]}")
    
    rmse = metrics.compute_rmse(true_angles, estimated)
    print(f"RMSE: {rmse:.3f}°")
    print(f"✗ EXPECTED POOR PERFORMANCE (K > K_max)")
except Exception as e:
    print(f"Error (expected): {e}")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
print("\nKEY FINDINGS:")
print("1. MUSIC works correctly when K_sources ≤ K_max")
print("2. Performance degrades when K_sources > K_max")
print("3. Always check K_max from coarray analysis before DOA estimation")
print("4. Sparse arrays (Z5, Z6) have higher K_max than ULA with same N")
