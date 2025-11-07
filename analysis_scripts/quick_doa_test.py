"""Quick test of DOA estimation module."""
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from doa_estimation.music import MUSICEstimator
from doa_estimation.metrics import DOAMetrics
from geometry_processors.z5_processor import Z5ArrayProcessor

print("="*70)
print("QUICK DOA ESTIMATION TEST")
print("="*70)

# Create Z5 array
z5 = Z5ArrayProcessor(N=7, d=0.5)
results = z5.run_full_analysis()

print(f"\nArray: {results.name}")
print(f"Sensors: {results.num_sensors}")
print(f"Positions: {results.sensors_positions}")

# Create MUSIC estimator
estimator = MUSICEstimator(
    sensor_positions=results.sensors_positions,
    wavelength=1.0,
    angle_range=(-90, 90),
    angle_resolution=0.5
)

# Test with well-separated sources
true_angles = [-40, 0, 40]
print(f"\nTrue angles: {true_angles}")

# Simulate with high SNR
X = estimator.simulate_signals(
    true_angles=true_angles,
    SNR_dB=20,
    snapshots=300,
    signal_type='random'
)

print(f"Signal shape: {X.shape}")

# Estimate
estimated_angles = estimator.estimate(X, K_sources=3)
print(f"Estimated angles: {[f'{a:.1f}°' for a in estimated_angles]}")

# Metrics
metrics = DOAMetrics()
rmse = metrics.compute_rmse(true_angles, estimated_angles)
print(f"RMSE: {rmse:.3f}°")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
