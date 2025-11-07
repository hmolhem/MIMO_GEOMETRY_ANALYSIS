"""Debug MUSIC algorithm to identify estimation issues."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from doa_estimation.music import MUSICEstimator

print("="*70)
print("MUSIC ALGORITHM DEBUG")
print("="*70)

# Test with simple ULA
positions = [0, 1, 2, 3, 4]
print(f"\nArray positions: {positions}")
print(f"Number of sensors: {len(positions)}")

estimator = MUSICEstimator(
    sensor_positions=positions,
    wavelength=1.0,
    angle_range=(-90, 90),
    angle_resolution=0.5
)

# Test Case 1: Two well-separated sources
print("\n" + "="*70)
print("TEST 1: Two well-separated sources")
print("="*70)

true_angles = [-30, 30]
print(f"True angles: {true_angles}")

# Simulate with very high SNR
X = estimator.simulate_signals(
    true_angles=true_angles,
    SNR_dB=30,
    snapshots=500,
    signal_type='random'
)

print(f"Signal matrix shape: {X.shape}")
print(f"Signal power: {np.mean(np.abs(X)**2):.4f}")

# Get spectrum
estimated_angles, spectrum = estimator.estimate(
    X, K_sources=2, return_spectrum=True
)

print(f"Estimated angles: {estimated_angles}")
print(f"Errors: {[true - est for true, est in zip(true_angles, sorted(estimated_angles))]}")

# Plot spectrum
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(estimator.angle_grid, spectrum, 'b-', linewidth=2, label='MUSIC Spectrum')
for angle in true_angles:
    plt.axvline(angle, color='g', linestyle='--', linewidth=2, label='True' if angle == true_angles[0] else '')
for angle in estimated_angles:
    plt.axvline(angle, color='r', linestyle=':', linewidth=2, label='Est' if angle == estimated_angles[0] else '')
plt.xlabel('Angle (degrees)')
plt.ylabel('MUSIC Spectrum (dB)')
plt.title('Test 1: Two Well-Separated Sources')
plt.legend()
plt.grid(True, alpha=0.3)

# Test Case 2: Three sources
print("\n" + "="*70)
print("TEST 2: Three sources")
print("="*70)

true_angles2 = [-40, 0, 40]
print(f"True angles: {true_angles2}")

X2 = estimator.simulate_signals(
    true_angles=true_angles2,
    SNR_dB=25,
    snapshots=500,
    signal_type='random'
)

estimated_angles2, spectrum2 = estimator.estimate(
    X2, K_sources=3, return_spectrum=True
)

print(f"Estimated angles: {estimated_angles2}")

plt.subplot(1, 2, 2)
plt.plot(estimator.angle_grid, spectrum2, 'b-', linewidth=2, label='MUSIC Spectrum')
for angle in true_angles2:
    plt.axvline(angle, color='g', linestyle='--', linewidth=2, label='True' if angle == true_angles2[0] else '')
for angle in estimated_angles2:
    plt.axvline(angle, color='r', linestyle=':', linewidth=2, label='Est' if angle == estimated_angles2[0] else '')
plt.xlabel('Angle (degrees)')
plt.ylabel('MUSIC Spectrum (dB)')
plt.title('Test 2: Three Sources')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/music_debug.png', dpi=150)
print(f"\nSpectrum plot saved to: results/plots/music_debug.png")

# Check covariance and eigenvalues
print("\n" + "="*70)
print("COVARIANCE ANALYSIS")
print("="*70)

R = estimator.estimate_covariance(X2)
eigenvalues, eigenvectors = np.linalg.eigh(R)
eigenvalues = eigenvalues[::-1]  # Sort descending

print(f"Eigenvalues: {eigenvalues}")
print(f"Signal/Noise gap: {eigenvalues[2] / eigenvalues[3]:.2f}x")

plt.figure(figsize=(8, 5))
plt.semilogy(range(len(eigenvalues)), eigenvalues, 'bo-', markersize=8, linewidth=2)
plt.axvline(2.5, color='r', linestyle='--', linewidth=2, label='K_sources=3')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue (log scale)')
plt.title('Eigenvalue Spectrum (Should show clear signal/noise separation)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/plots/eigenvalues.png', dpi=150)
print(f"Eigenvalue plot saved to: results/plots/eigenvalues.png")

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70)
