"""
Test Script for ALSS-II on Z5 Array.

Reproduces the Z5 experiment from the paper with:
1. Baseline (No ALSS)
2. ALSS (Original, Zero-Mode)
3. ALSS-II (New, Data-Driven Prior)

Conditions:
- Array: Z5 (N=7)
- SNR: 10 dB
- Snapshots: 200
- Trials: 100 (for speed, paper uses 1000)
- Coupling: c1=0.3, alpha=0.5
"""
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry_processors.z5_processor import Z5ArrayProcessor
from core.radarpy.signal.doa_sim_core import simulate_snapshots
from core.radarpy.signal.mutual_coupling import generate_mcm
from core.radarpy.algorithms.coarray_music import estimate_doa_coarray_music

def rmse(est, true):
    """Compute RMSE between estimated and true angles."""
    # Simple matching for K sources
    est = np.sort(est)
    true = np.sort(true)
    return np.sqrt(np.mean((est - true)**2))

def run_experiment():
    # Setup
    print("--- Setting up Z5 Experiment ---")
    processor = Z5ArrayProcessor(N=7, d=0.5)
    spec = processor.run_full_analysis(verbose=False)
    positions = np.array(spec.sensors_positions)
    print(f"Positions: {positions}")
    
    # Parameters
    K = 3
    true_angles = [-30, 0, 30]
    M = 200
    snr_db = 10
    trials = 100
    wavelength = 1.0
    d_phys = 0.5
    
    # Coupling
    c1 = 0.3
    alpha = 0.5
    # Hypothesis: alpha is defined per integer lag in the paper
    C = generate_mcm(len(positions), positions, model="exponential", c1=c1, alpha=alpha)
    print(f"Coupling Matrix Generated (c1={c1}, alpha={alpha})")
    
    # Storage
    results = {
        "Baseline": [],
        "ALSS_Zero": [],
        "ALSS_II": []
    }
    
    print(f"\n--- Running {trials} Trials ---")
    for i in tqdm(range(trials)):
        # Generate Data using PHYSICAL positions for Phase
        # But use C generated from integer positions
        pos_phys = positions * d_phys
        X, _, _ = simulate_snapshots(
            pos_phys, wavelength, true_angles, snr_db, M,
            coupling_matrix=C
        )
        
        # 1. Baseline (No ALSS)
        est_base, _ = estimate_doa_coarray_music(
            X, positions, d_phys, wavelength, K,
            alss_enabled=False
        )
        results["Baseline"].append(rmse(est_base, true_angles))
        
        # 2. ALSS (Original Zero-Mode)
        est_alss, _ = estimate_doa_coarray_music(
            X, positions, d_phys, wavelength, K,
            alss_enabled=True, alss_mode="zero", alss_tau=1.0
        )
        results["ALSS_Zero"].append(rmse(est_alss, true_angles))
        
        # 3. ALSS-II (New Data-Driven)
        est_alss_ii, _ = estimate_doa_coarray_music(
            X, positions, d_phys, wavelength, K,
            alss_enabled=True, alss_mode="alss_ii", alss_tau=1.0 # tau passed as beta
        )
        results["ALSS_II"].append(rmse(est_alss_ii, true_angles))
        
    # Analysis
    print("\n--- Results Summary ---")
    means = {k: np.mean(v) for k, v in results.items()}
    stds = {k: np.std(v) for k, v in results.items()}
    
    df = pd.DataFrame({
        "RMSE (deg)": means,
        "Std Dev": stds
    })
    print(df)
    
    # Gap Reduction Calculation
    # Gap = Baseline - Ideal (Assuming Ideal ~ 0 for simplicity or relative to ALSS)
    # Actually, Gap Reduction is relative to Baseline improvement
    
    base_rmse = means["Baseline"]
    alss_rmse = means["ALSS_Zero"]
    alss_ii_rmse = means["ALSS_II"]
    
    imp_alss = (base_rmse - alss_rmse) / base_rmse * 100
    imp_alss_ii = (base_rmse - alss_ii_rmse) / base_rmse * 100
    
    print(f"\nImprovement over Baseline:")
    print(f"ALSS (Zero): {imp_alss:.2f}%")
    print(f"ALSS-II:     {imp_alss_ii:.2f}%")
    
    if imp_alss_ii > imp_alss:
        print("\nSUCCESS: ALSS-II outperformed original ALSS!")
    else:
        print("\nNote: ALSS-II performance is similar or lower. Check parameters.")

if __name__ == "__main__":
    run_experiment()
