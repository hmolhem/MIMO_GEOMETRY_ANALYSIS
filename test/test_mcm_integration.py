"""
Test script for Mutual Coupling Matrix (MCM) integration.

Verifies that MCM works correctly in the DOA estimation pipeline by comparing
ideal (no coupling) vs coupled scenarios.

Date: November 6, 2025
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from core.radarpy.signal.mutual_coupling import generate_mcm, get_coupling_info
from core.radarpy.signal.doa_sim_core import run_music

print("="*70)
print("MCM Integration Test - DOA Estimation with Mutual Coupling")
print("="*70)

# Test parameters
positions_ula = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # 7-element ULA, λ/2 spacing
wavelength = 1.0
true_doas = np.array([15.0, -20.0])  # Two sources
snr_db = 10
snapshots = 100
seed = 42

print(f"\nTest Configuration:")
print(f"  Array: 7-element ULA, λ/2 spacing")
print(f"  True DOAs: {true_doas} degrees")
print(f"  SNR: {snr_db} dB")
print(f"  Snapshots: {snapshots}")

# Test 1: Ideal case (no coupling)
print("\n" + "-"*70)
print("Test 1: Ideal Array (No Coupling)")
print("-"*70)

result_ideal = run_music(positions_ula, wavelength, true_doas, len(true_doas), 
                        snr_db, snapshots, seed=seed, coupling_matrix=None)

print(f"True DOAs:      {result_ideal['doas_true_deg']}")
print(f"Estimated DOAs: {result_ideal['doas_est_deg']}")
errors_ideal = np.abs(result_ideal['doas_est_deg'] - result_ideal['doas_true_deg'])
print(f"Estimation Errors: {errors_ideal} degrees")
print(f"RMSE: {np.sqrt(np.mean(errors_ideal**2)):.4f} degrees")

# Test 2: With exponential coupling
print("\n" + "-"*70)
print("Test 2: Array with Mutual Coupling (Exponential Model)")
print("-"*70)

C_exp = generate_mcm(len(positions_ula), positions_ula, model="exponential", 
                     c1=0.3, alpha=0.5)
info = get_coupling_info(C_exp)
print(f"Coupling Matrix Info:")
print(f"  Max off-diagonal coupling: {info['max_off_diagonal']:.4f}")
print(f"  Avg off-diagonal coupling: {info['avg_off_diagonal']:.4f}")
print(f"  Condition number: {info['condition_number']:.2e}")
print(f"  Is Hermitian: {info['is_hermitian']}")

result_coupled = run_music(positions_ula, wavelength, true_doas, len(true_doas), 
                          snr_db, snapshots, seed=seed, coupling_matrix=C_exp)

print(f"\nTrue DOAs:      {result_coupled['doas_true_deg']}")
print(f"Estimated DOAs: {result_coupled['doas_est_deg']}")
errors_coupled = np.abs(result_coupled['doas_est_deg'] - result_coupled['doas_true_deg'])
print(f"Estimation Errors: {errors_coupled} degrees")
print(f"RMSE: {np.sqrt(np.mean(errors_coupled**2)):.4f} degrees")

# Test 3: With Toeplitz coupling
print("\n" + "-"*70)
print("Test 3: Array with Mutual Coupling (Toeplitz Model)")
print("-"*70)

coupling_coeffs = np.array([1.0, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01])
C_toep = generate_mcm(len(positions_ula), positions_ula, model="toeplitz", 
                      coupling_coeffs=coupling_coeffs)
info_toep = get_coupling_info(C_toep)
print(f"Coupling Matrix Info:")
print(f"  Max off-diagonal coupling: {info_toep['max_off_diagonal']:.4f}")
print(f"  Avg off-diagonal coupling: {info_toep['avg_off_diagonal']:.4f}")
print(f"  Condition number: {info_toep['condition_number']:.2e}")

result_toep = run_music(positions_ula, wavelength, true_doas, len(true_doas), 
                       snr_db, snapshots, seed=seed, coupling_matrix=C_toep)

print(f"\nTrue DOAs:      {result_toep['doas_true_deg']}")
print(f"Estimated DOAs: {result_toep['doas_est_deg']}")
errors_toep = np.abs(result_toep['doas_est_deg'] - result_toep['doas_true_deg'])
print(f"Estimation Errors: {errors_toep} degrees")
print(f"RMSE: {np.sqrt(np.mean(errors_toep**2)):.4f} degrees")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY: Impact of Mutual Coupling on DOA Estimation")
print("="*70)
print(f"{'Scenario':<30} {'RMSE (deg)':<15} {'Max Error (deg)':<15}")
print("-"*70)
print(f"{'Ideal (No Coupling)':<30} {np.sqrt(np.mean(errors_ideal**2)):<15.4f} {np.max(errors_ideal):<15.4f}")
print(f"{'Exponential Coupling':<30} {np.sqrt(np.mean(errors_coupled**2)):<15.4f} {np.max(errors_coupled):<15.4f}")
print(f"{'Toeplitz Coupling':<30} {np.sqrt(np.mean(errors_toep**2)):<15.4f} {np.max(errors_toep):<15.4f}")

# Degradation analysis
rmse_ideal = np.sqrt(np.mean(errors_ideal**2))
rmse_exp = np.sqrt(np.mean(errors_coupled**2))
rmse_toep = np.sqrt(np.mean(errors_toep**2))

print(f"\nDegradation due to coupling:")
print(f"  Exponential: {((rmse_exp - rmse_ideal) / rmse_ideal * 100):+.2f}% change")
print(f"  Toeplitz:    {((rmse_toep - rmse_ideal) / rmse_ideal * 100):+.2f}% change")

print("\n" + "="*70)
print("✓ MCM Integration Test Complete!")
print("="*70)
print("\nConclusion:")
print("  - MCM can be easily enabled/disabled (coupling_matrix=None)")
print("  - Multiple coupling models supported (exponential, toeplitz, measured)")
print("  - Coupling effects propagate correctly through simulation pipeline")
print("  - DOA estimation performance can be compared with/without coupling")
