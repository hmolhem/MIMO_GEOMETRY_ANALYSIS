#!/usr/bin/env python3
"""
ALSS Paper Experiments - Minimal Metrics Set
============================================

Streamlined implementation for IEEE RadarCon 2025 paper submission.
Focus: Essential metrics only for core claims validation.

Author: MIMO Geometry Analysis Team
Date: November 6, 2025

MINIMAL METRICS SET:
-------------------
Must-Have (5 metrics):
    1. RMSE (degrees) - Primary performance metric
    2. Improvement_% - (Baseline - ALSS)/Baseline × 100
    3. 95% CI - Statistical significance
    4. Resolution_Rate_% - Practical usability
    5. Runtime_ms - Computational feasibility

Should-Have (3 metrics):
    6. RMSE/CRB_Ratio - Theoretical efficiency
    7. Parameter_Sensitivity - Robustness evidence
    8. Cross_Array_Consistency - Generality proof

EXPERIMENT MATRIX:
-----------------
    SCENARIO 1: Baseline Characterization
        - Arrays: ULA, Nested, Z1, Z4, Z5, Z6
        - Coupling: 0.0 (ideal)
        - ALSS: False
        - Sweeps: SNR [-5, 0, 5, 10, 15], Snapshots [32, 64, 128]
        - Metrics: RMSE, CRB_Ratio, Resolution_Rate
    
    SCENARIO 3: ALSS Effectiveness
        - Arrays: Z5 (focus array)
        - Coupling: [0.0, 0.3]
        - ALSS: [True, False]
        - Sweeps: SNR [-5, 0, 5, 10, 15], Snapshots [32, 64, 128]
        - Metrics: Improvement_%, Statistical_Significance, Harmlessness
    
    SCENARIO 4: Cross-Array Validation
        - Arrays: ULA, Nested, Z1, Z4, Z5, Z6
        - Coupling: 0.3
        - ALSS: [True, False]
        - Fixed: SNR=5dB, Snapshots=64
        - Metrics: Relative_Improvement_%, Ranking_Consistency

Usage:
------
    # Full experiment suite (all scenarios)
    python run_paper_experiments.py --scenario all --trials 500
    
    # Individual scenarios
    python run_paper_experiments.py --scenario 1 --trials 500
    python run_paper_experiments.py --scenario 3 --trials 500 --arrays Z5
    python run_paper_experiments.py --scenario 4 --trials 500
    
    # Quick test
    python run_paper_experiments.py --scenario all --trials 50 --test
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as t_dist
import warnings

# Import core functionality
from core.radarpy.signal.doa_sim_core import simulate_snapshots
from core.radarpy.algorithms.coarray_music import estimate_doa_coarray_music
from core.radarpy.algorithms.crb import crb_pair_worst_deg
from core.radarpy.signal.mutual_coupling import generate_mcm


# ============================================================================
# ARRAY DEFINITIONS
# ============================================================================

def get_array_positions(array_type: str, N: int = 7) -> np.ndarray:
    """
    Get sensor positions for various array types.
    
    NOTE: Returns INTEGER GRID INDICES (not physical positions in meters).
    These are used by estimate_doa_coarray_music with d_phys parameter.
    
    Args:
        array_type: 'ULA', 'Nested', 'Z1', 'Z4', 'Z5', 'Z6'
        N: Number of sensors (default: 7)
    
    Returns:
        positions: Sensor positions as integer grid indices (numpy array)
    """
    array_type = array_type.upper()
    
    if array_type == 'ULA':
        # Uniform Linear Array: [0, 1, 2, ..., N-1]
        return np.arange(N, dtype=float)
    
    elif array_type == 'NESTED':
        # Nested array: N1 = ceil(N/2), N2 = floor(N/2)
        N1 = (N + 1) // 2
        N2 = N // 2
        inner = np.arange(1, N1 + 1, dtype=float)
        outer = (N1 + 1) * np.arange(1, N2 + 1, dtype=float)
        return np.sort(np.concatenate([inner, outer]))
    
    elif array_type == 'Z1':
        # Z1 array for N=10: [0, 1, 4, 9, 11, 13, 17, 25, 27, 34]
        if N == 10:
            return np.array([0, 1, 4, 9, 11, 13, 17, 25, 27, 34], dtype=float)
        elif N == 3:
            return np.array([0, 1, 4], dtype=float)
        else:
            # Fallback to basic pattern
            return np.array([0, 1, 4] + list(range(9, 9 + N - 3)), dtype=float)
    
    elif array_type == 'Z4':
        # Z4 array for N=7: [0, 5, 8, 11, 14, 17, 21]
        if N == 7:
            return np.array([0, 5, 8, 11, 14, 17, 21], dtype=float)
        else:
            # Generic Z4 pattern
            return np.array([0] + list(range(5, 5 + 3 * (N - 1), 3)), dtype=float)
    
    elif array_type == 'Z5':
        # Z5 array for N=7: [0, 5, 8, 11, 14, 17, 21] (from run_scenario1_baseline.py)
        if N == 7:
            return np.array([0, 5, 8, 11, 14, 17, 21], dtype=float)
        else:
            # Generic Z5 pattern
            positions = [0, 5, 8]
            for i in range(3, N):
                positions.append(positions[-1] + 3)
            return np.array(positions, dtype=float)
    
    elif array_type == 'Z6':
        # Z6 array for N=7: [0, 1, 4, 8, 13, 17, 22] (from run_scenario1_baseline.py)
        if N == 7:
            return np.array([0, 1, 4, 8, 13, 17, 22], dtype=float)
        else:
            # Generic Z6 pattern
            positions = [0, 1, 4]
            for i in range(3, N):
                positions.append(positions[-1] + 4 + (i - 3))
            return np.array(positions, dtype=float)
    
    else:
        raise ValueError(f"Unknown array type: {array_type}")


# ============================================================================
# CORE METRICS COMPUTATION
# ============================================================================

def compute_rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """Compute RMSE with Hungarian matching."""
    if len(estimated) != len(true):
        return np.inf
    
    cost_matrix = np.abs(true[:, np.newaxis] - estimated[np.newaxis, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    errors = true[row_ind] - estimated[col_ind]
    return np.sqrt(np.mean(errors ** 2))


def compute_resolution_rate(trials_results: List[Tuple[np.ndarray, float]], 
                            true_doas: np.ndarray,
                            threshold: float = 3.0) -> float:
    """
    Compute resolution rate (% of trials with successful detection).
    
    Args:
        trials_results: List of (estimated_doas, runtime) tuples
        true_doas: True DOA angles
        threshold: RMSE threshold for success (default: 3.0°)
    
    Returns:
        resolution_rate: Percentage of successful trials
    """
    K = len(true_doas)
    successes = 0
    total = len(trials_results)
    
    for est_doas, _ in trials_results:
        if len(est_doas) == K:
            rmse = compute_rmse(est_doas, true_doas)
            if rmse < threshold:
                successes += 1
    
    return 100.0 * successes / total if total > 0 else 0.0


def compute_crb_ratio(rmse: float, positions: np.ndarray, wavelength: float,
                     true_doas: np.ndarray, snr_db: float, snapshots: int) -> float:
    """
    Compute RMSE/CRB ratio (efficiency metric).
    
    Args:
        rmse: Estimated RMSE in degrees
        positions: Sensor positions
        wavelength: Signal wavelength
        true_doas: True DOA angles
        snr_db: SNR in dB
        snapshots: Number of snapshots
    
    Returns:
        crb_ratio: RMSE/CRB (1.0 = CRB-optimal, >1.0 = suboptimal)
    """
    try:
        # Compute CRB for worst-case source pair
        crb_deg = crb_pair_worst_deg(
            positions=positions,
            wavelength=wavelength,
            doas_deg=true_doas,
            snr_db=snr_db,
            M=snapshots,
            coupling_matrix=None
        )
        
        if crb_deg > 0:
            return rmse / crb_deg
        else:
            return np.inf
    except Exception:
        return np.inf


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute 95% confidence interval using t-distribution.
    
    Args:
        values: Array of measurements
        confidence: Confidence level (default: 0.95)
    
    Returns:
        mean, ci_low, ci_high
    """
    n = len(values)
    if n < 2:
        return np.mean(values), np.nan, np.nan
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    sem = std / np.sqrt(n)
    
    # t-distribution critical value
    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem
    
    return mean, ci_low, ci_high


# ============================================================================
# DOA ESTIMATION WITH TIMING
# ============================================================================

def run_doa_trial(positions: np.ndarray, 
                  true_doas: np.ndarray,
                  wavelength: float,
                  snr_db: float,
                  snapshots: int,
                  coupling_matrix: Optional[np.ndarray] = None,
                  alss_enabled: bool = False,
                  seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Run single DOA estimation trial with timing.
    
    Args:
        positions: Sensor positions
        true_doas: True DOA angles
        wavelength: Signal wavelength
        snr_db: SNR in dB
        snapshots: Number of snapshots
        coupling_matrix: Optional coupling matrix
        alss_enabled: Enable ALSS regularization
        seed: Random seed for reproducibility
    
    Returns:
        estimated_doas: Estimated DOA angles
        runtime_ms: Execution time in milliseconds
    """
    # Generate snapshots
    # Note: positions are integer grid indices, d_phys converts to meters
    d_phys = 0.5  # Half-wavelength spacing
    positions_meters = positions * d_phys
    
    X, _, _ = simulate_snapshots(
        sensor_positions=positions_meters,
        wavelength=wavelength,
        doas_deg=true_doas,
        snr_db=snr_db,
        snapshots=snapshots,
        seed=seed,
        coupling_matrix=coupling_matrix
    )
    
    # Run DOA estimation with timing
    K = len(true_doas)
    d_phys = 0.5  # Half-wavelength spacing
    
    start = time.time()
    doas_est, _ = estimate_doa_coarray_music(
        X=X,
        positions=positions,  # These are integer grid indices (not scaled)
        d_phys=d_phys,
        wavelength=wavelength,
        K=K,
        scan_deg=(-60, 60, 0.1),
        alss_enabled=alss_enabled,
        alss_tau=1.0 if alss_enabled else 0.0,
        alss_mode='zero'
    )
    runtime_ms = (time.time() - start) * 1000
    
    return doas_est, runtime_ms


# ============================================================================
# SCENARIO 1: BASELINE CHARACTERIZATION
# ============================================================================

def run_scenario1_baseline(arrays: List[str],
                           snr_sweep: List[float],
                           snapshot_sweep: List[int],
                           trials: int,
                           output_dir: str) -> pd.DataFrame:
    """
    SCENARIO 1: Baseline performance characterization.
    
    Metrics:
        - RMSE (degrees)
        - RMSE/CRB Ratio
        - Resolution Rate (%)
        - Runtime (ms)
        - 95% CI
    
    Args:
        arrays: List of array types ['ULA', 'Z5', ...]
        snr_sweep: List of SNR values in dB
        snapshot_sweep: List of snapshot counts
        trials: Number of Monte Carlo trials
        output_dir: Output directory for results
    
    Returns:
        results_df: DataFrame with all results
    """
    print("\n" + "="*70)
    print("SCENARIO 1: BASELINE CHARACTERIZATION")
    print("="*70)
    print(f"Arrays: {', '.join(arrays)}")
    print(f"SNR sweep: {snr_sweep} dB")
    print(f"Snapshot sweep: {snapshot_sweep}")
    print(f"Trials per configuration: {trials}")
    print()
    
    # Fixed parameters
    wavelength = 1.0
    true_doas = np.array([15.0, -20.0])
    N = 7
    d = 1.0
    
    results = []
    
    # Iterate over arrays
    for array_type in arrays:
        print(f"\nArray: {array_type}")
        positions = get_array_positions(array_type, N)  # Returns integer grid indices
        
        # SNR sweep (fixed snapshots = 64)
        print("  SNR Sweep:")
        for snr in snr_sweep:
            rmse_list = []
            resolution_list = []
            runtime_list = []
            trials_data = []
            
            for trial in range(trials):
                seed = 42 + trial
                est_doas, runtime = run_doa_trial(
                    positions, true_doas, wavelength, snr, 64, 
                    coupling_matrix=None, alss_enabled=False, seed=seed
                )
                
                rmse = compute_rmse(est_doas, true_doas)
                rmse_list.append(rmse)
                runtime_list.append(runtime)
                trials_data.append((est_doas, runtime))
            
            # Compute metrics
            rmse_mean, rmse_ci_low, rmse_ci_high = compute_confidence_interval(np.array(rmse_list))
            crb_ratio = compute_crb_ratio(rmse_mean, positions, wavelength, true_doas, snr, 64)
            resolution_rate = compute_resolution_rate(trials_data, true_doas)
            runtime_mean = np.mean(runtime_list)
            
            results.append({
                'Scenario': 'Baseline',
                'Array': array_type,
                'SNR_dB': snr,
                'Snapshots': 64,
                'RMSE_deg': rmse_mean,
                'RMSE_CI_Low': rmse_ci_low,
                'RMSE_CI_High': rmse_ci_high,
                'RMSE_CRB_Ratio': crb_ratio,
                'Resolution_Rate_%': resolution_rate,
                'Runtime_ms': runtime_mean,
                'Trials': trials
            })
            
            print(f"    SNR={snr:+3d}dB: RMSE={rmse_mean:.3f}° (CI: [{rmse_ci_low:.3f}, {rmse_ci_high:.3f}]), "
                  f"CRB={crb_ratio:.2f}×, Res={resolution_rate:.1f}%, Time={runtime_mean:.1f}ms")
        
        # Snapshot sweep (fixed SNR = 5 dB)
        print("  Snapshot Sweep:")
        for M in snapshot_sweep:
            rmse_list = []
            resolution_list = []
            runtime_list = []
            trials_data = []
            
            for trial in range(trials):
                seed = 42 + trial
                est_doas, runtime = run_doa_trial(
                    positions, true_doas, wavelength, 5.0, M,
                    coupling_matrix=None, alss_enabled=False, seed=seed
                )
                
                rmse = compute_rmse(est_doas, true_doas)
                rmse_list.append(rmse)
                runtime_list.append(runtime)
                trials_data.append((est_doas, runtime))
            
            # Compute metrics
            rmse_mean, rmse_ci_low, rmse_ci_high = compute_confidence_interval(np.array(rmse_list))
            crb_ratio = compute_crb_ratio(rmse_mean, positions, wavelength, true_doas, 5.0, M)
            resolution_rate = compute_resolution_rate(trials_data, true_doas)
            runtime_mean = np.mean(runtime_list)
            
            results.append({
                'Scenario': 'Baseline',
                'Array': array_type,
                'SNR_dB': 5.0,
                'Snapshots': M,
                'RMSE_deg': rmse_mean,
                'RMSE_CI_Low': rmse_ci_low,
                'RMSE_CI_High': rmse_ci_high,
                'RMSE_CRB_Ratio': crb_ratio,
                'Resolution_Rate_%': resolution_rate,
                'Runtime_ms': runtime_mean,
                'Trials': trials
            })
            
            print(f"    M={M:3d}: RMSE={rmse_mean:.3f}° (CI: [{rmse_ci_low:.3f}, {rmse_ci_high:.3f}]), "
                  f"CRB={crb_ratio:.2f}×, Res={resolution_rate:.1f}%, Time={runtime_mean:.1f}ms")
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'scenario1_baseline.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Results saved: {csv_path}")
    
    return results_df


# ============================================================================
# SCENARIO 3: ALSS EFFECTIVENESS
# ============================================================================

def run_scenario3_alss_effectiveness(arrays: List[str],
                                    coupling_levels: List[float],
                                    snr_sweep: List[float],
                                    snapshot_sweep: List[int],
                                    trials: int,
                                    output_dir: str) -> pd.DataFrame:
    """
    SCENARIO 3: ALSS regularization effectiveness.
    
    Metrics:
        - Improvement_% = (Baseline - ALSS)/Baseline × 100
        - Statistical_Significance (p-value from paired t-test)
        - Harmlessness_Index (penalty when ALSS worse)
        - 95% CI
    
    Args:
        arrays: List of array types (typically ['Z5'])
        coupling_levels: List of coupling strengths [0.0, 0.3]
        snr_sweep: List of SNR values
        snapshot_sweep: List of snapshot counts
        trials: Number of trials
        output_dir: Output directory
    
    Returns:
        results_df: DataFrame with results
    """
    print("\n" + "="*70)
    print("SCENARIO 3: ALSS EFFECTIVENESS")
    print("="*70)
    print(f"Arrays: {', '.join(arrays)}")
    print(f"Coupling levels: {coupling_levels}")
    print(f"SNR sweep: {snr_sweep} dB")
    print(f"Snapshot sweep: {snapshot_sweep}")
    print(f"Trials per configuration: {trials}")
    print()
    
    # Fixed parameters
    wavelength = 1.0
    true_doas = np.array([15.0, -20.0])
    N = 7
    d = 1.0
    
    results = []
    
    # Iterate over arrays and coupling levels
    for array_type in arrays:
        print(f"\nArray: {array_type}")
        positions = get_array_positions(array_type, N)  # Returns integer grid indices
        
        for c1 in coupling_levels:
            print(f"  Coupling c1={c1:.1f}:")
            
            # Generate coupling matrix
            if c1 > 0:
                C = generate_mcm(len(positions), positions, model='exponential', c1=c1)
            else:
                C = None
            
            # SNR sweep
            print("    SNR Sweep:")
            for snr in snr_sweep:
                # Baseline (no ALSS)
                rmse_baseline = []
                for trial in range(trials):
                    seed = 42 + trial
                    est_doas, _ = run_doa_trial(
                        positions, true_doas, wavelength, snr, 64,
                        coupling_matrix=C, alss_enabled=False, seed=seed
                    )
                    rmse_baseline.append(compute_rmse(est_doas, true_doas))
                
                # ALSS enabled
                rmse_alss = []
                for trial in range(trials):
                    seed = 42 + trial
                    est_doas, _ = run_doa_trial(
                        positions, true_doas, wavelength, snr, 64,
                        coupling_matrix=C, alss_enabled=True, seed=seed
                    )
                    rmse_alss.append(compute_rmse(est_doas, true_doas))
                
                # Compute metrics
                rmse_base_mean = np.mean(rmse_baseline)
                rmse_alss_mean = np.mean(rmse_alss)
                improvement = 100.0 * (rmse_base_mean - rmse_alss_mean) / rmse_base_mean if rmse_base_mean > 0 else 0.0
                
                # Paired t-test for significance
                from scipy.stats import ttest_rel
                if len(rmse_baseline) >= 2:
                    _, p_value = ttest_rel(rmse_baseline, rmse_alss)
                else:
                    p_value = 1.0
                
                # Harmlessness index (penalty when ALSS worse)
                harmless = np.sum(np.array(rmse_alss) <= np.array(rmse_baseline)) / trials * 100
                
                # Confidence intervals
                _, base_ci_low, base_ci_high = compute_confidence_interval(np.array(rmse_baseline))
                _, alss_ci_low, alss_ci_high = compute_confidence_interval(np.array(rmse_alss))
                
                results.append({
                    'Scenario': 'ALSS_Effectiveness',
                    'Array': array_type,
                    'Coupling_c1': c1,
                    'SNR_dB': snr,
                    'Snapshots': 64,
                    'RMSE_Baseline': rmse_base_mean,
                    'RMSE_ALSS': rmse_alss_mean,
                    'Improvement_%': improvement,
                    'P_Value': p_value,
                    'Harmlessness_%': harmless,
                    'Baseline_CI_Low': base_ci_low,
                    'Baseline_CI_High': base_ci_high,
                    'ALSS_CI_Low': alss_ci_low,
                    'ALSS_CI_High': alss_ci_high,
                    'Trials': trials
                })
                
                sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                print(f"      SNR={snr:+3d}dB: Base={rmse_base_mean:.3f}°, ALSS={rmse_alss_mean:.3f}°, "
                      f"Δ={improvement:+.1f}% {sig_marker}, p={p_value:.4f}, Harmless={harmless:.1f}%")
            
            # Snapshot sweep
            print("    Snapshot Sweep:")
            for M in snapshot_sweep:
                # Baseline
                rmse_baseline = []
                for trial in range(trials):
                    seed = 42 + trial
                    est_doas, _ = run_doa_trial(
                        positions, true_doas, wavelength, 5.0, M,
                        coupling_matrix=C, alss_enabled=False, seed=seed
                    )
                    rmse_baseline.append(compute_rmse(est_doas, true_doas))
                
                # ALSS
                rmse_alss = []
                for trial in range(trials):
                    seed = 42 + trial
                    est_doas, _ = run_doa_trial(
                        positions, true_doas, wavelength, 5.0, M,
                        coupling_matrix=C, alss_enabled=True, seed=seed
                    )
                    rmse_alss.append(compute_rmse(est_doas, true_doas))
                
                # Metrics
                rmse_base_mean = np.mean(rmse_baseline)
                rmse_alss_mean = np.mean(rmse_alss)
                improvement = 100.0 * (rmse_base_mean - rmse_alss_mean) / rmse_base_mean if rmse_base_mean > 0 else 0.0
                
                from scipy.stats import ttest_rel
                if len(rmse_baseline) >= 2:
                    _, p_value = ttest_rel(rmse_baseline, rmse_alss)
                else:
                    p_value = 1.0
                
                harmless = np.sum(np.array(rmse_alss) <= np.array(rmse_baseline)) / trials * 100
                
                _, base_ci_low, base_ci_high = compute_confidence_interval(np.array(rmse_baseline))
                _, alss_ci_low, alss_ci_high = compute_confidence_interval(np.array(rmse_alss))
                
                results.append({
                    'Scenario': 'ALSS_Effectiveness',
                    'Array': array_type,
                    'Coupling_c1': c1,
                    'SNR_dB': 5.0,
                    'Snapshots': M,
                    'RMSE_Baseline': rmse_base_mean,
                    'RMSE_ALSS': rmse_alss_mean,
                    'Improvement_%': improvement,
                    'P_Value': p_value,
                    'Harmlessness_%': harmless,
                    'Baseline_CI_Low': base_ci_low,
                    'Baseline_CI_High': base_ci_high,
                    'ALSS_CI_Low': alss_ci_low,
                    'ALSS_CI_High': alss_ci_high,
                    'Trials': trials
                })
                
                sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                print(f"      M={M:3d}: Base={rmse_base_mean:.3f}°, ALSS={rmse_alss_mean:.3f}°, "
                      f"Δ={improvement:+.1f}% {sig_marker}, p={p_value:.4f}, Harmless={harmless:.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'scenario3_alss_effectiveness.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Results saved: {csv_path}")
    
    return results_df


# ============================================================================
# SCENARIO 4: CROSS-ARRAY VALIDATION
# ============================================================================

def run_scenario4_cross_array(arrays: List[str],
                              coupling_strength: float,
                              trials: int,
                              output_dir: str) -> pd.DataFrame:
    """
    SCENARIO 4: Cross-array consistency validation.
    
    Metrics:
        - Relative_Improvement_% (vs ULA)
        - Ranking_Consistency (Kendall's tau)
        - RMSE (degrees)
        - Resolution_Rate_%
    
    Args:
        arrays: List of array types
        coupling_strength: Coupling strength (c1)
        trials: Number of trials
        output_dir: Output directory
    
    Returns:
        results_df: DataFrame with results
    """
    print("\n" + "="*70)
    print("SCENARIO 4: CROSS-ARRAY VALIDATION")
    print("="*70)
    print(f"Arrays: {', '.join(arrays)}")
    print(f"Coupling strength: c1={coupling_strength:.1f}")
    print(f"Fixed: SNR=5dB, Snapshots=64")
    print(f"Trials: {trials}")
    print()
    
    # Fixed parameters
    wavelength = 1.0
    true_doas = np.array([15.0, -20.0])
    N = 7
    d = 1.0
    snr = 5.0
    M = 64
    
    results = []
    
    # Test baseline (no coupling) and coupled conditions
    for condition in ['Baseline', 'Coupled']:
        print(f"\n{condition} Condition:")
        
        c1 = 0.0 if condition == 'Baseline' else coupling_strength
        
        for array_type in arrays:
            print(f"  {array_type}:")
            positions = get_array_positions(array_type, N)  # Returns integer grid indices
            
            # Generate coupling matrix
            if c1 > 0:
                C = generate_mcm(len(positions), positions, model='exponential', c1=c1)
            else:
                C = None
            
            # Baseline (no ALSS)
            rmse_baseline = []
            trials_baseline = []
            for trial in range(trials):
                seed = 42 + trial
                est_doas, runtime = run_doa_trial(
                    positions, true_doas, wavelength, snr, M,
                    coupling_matrix=C, alss_enabled=False, seed=seed
                )
                rmse = compute_rmse(est_doas, true_doas)
                rmse_baseline.append(rmse)
                trials_baseline.append((est_doas, runtime))
            
            # ALSS enabled
            rmse_alss = []
            trials_alss = []
            for trial in range(trials):
                seed = 42 + trial
                est_doas, runtime = run_doa_trial(
                    positions, true_doas, wavelength, snr, M,
                    coupling_matrix=C, alss_enabled=True, seed=seed
                )
                rmse = compute_rmse(est_doas, true_doas)
                rmse_alss.append(rmse)
                trials_alss.append((est_doas, runtime))
            
            # Compute metrics
            rmse_base_mean = np.mean(rmse_baseline)
            rmse_alss_mean = np.mean(rmse_alss)
            improvement = 100.0 * (rmse_base_mean - rmse_alss_mean) / rmse_base_mean if rmse_base_mean > 0 else 0.0
            
            res_base = compute_resolution_rate(trials_baseline, true_doas)
            res_alss = compute_resolution_rate(trials_alss, true_doas)
            
            results.append({
                'Scenario': 'Cross_Array',
                'Condition': condition,
                'Array': array_type,
                'Coupling_c1': c1,
                'RMSE_Baseline': rmse_base_mean,
                'RMSE_ALSS': rmse_alss_mean,
                'Improvement_%': improvement,
                'Resolution_Baseline_%': res_base,
                'Resolution_ALSS_%': res_alss,
                'Trials': trials
            })
            
            print(f"    Base: RMSE={rmse_base_mean:.3f}°, Res={res_base:.1f}%")
            print(f"    ALSS: RMSE={rmse_alss_mean:.3f}°, Res={res_alss:.1f}%, Δ={improvement:+.1f}%")
    
    # Compute relative improvement vs ULA
    results_df = pd.DataFrame(results)
    
    # Ranking consistency (Kendall's tau)
    print("\n  Computing ranking consistency...")
    baseline_rmse = results_df[results_df['Condition'] == 'Baseline'].sort_values('Array')['RMSE_Baseline'].values
    coupled_rmse = results_df[results_df['Condition'] == 'Coupled'].sort_values('Array')['RMSE_Baseline'].values
    
    from scipy.stats import kendalltau
    if len(baseline_rmse) == len(coupled_rmse) and len(baseline_rmse) > 1:
        tau, p_value = kendalltau(baseline_rmse, coupled_rmse)
        print(f"    Kendall's tau = {tau:.3f}, p = {p_value:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'scenario4_cross_array.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Results saved: {csv_path}")
    
    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ALSS Paper Experiments - Minimal Metrics Set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full suite (all scenarios)
  python run_paper_experiments.py --scenario all --trials 500
  
  # Individual scenarios
  python run_paper_experiments.py --scenario 1 --trials 500
  python run_paper_experiments.py --scenario 3 --trials 500 --arrays Z5
  python run_paper_experiments.py --scenario 4 --trials 500 --arrays ULA Nested Z5 Z6
  
  # Quick test
  python run_paper_experiments.py --scenario all --trials 50 --test
        """
    )
    
    parser.add_argument('--scenario', type=str, choices=['1', '3', '4', 'all'], required=True,
                       help='Scenario to run: 1 (baseline), 3 (ALSS), 4 (cross-array), or all')
    parser.add_argument('--trials', type=int, default=500,
                       help='Number of Monte Carlo trials (default: 500)')
    parser.add_argument('--arrays', nargs='+', default=['ULA', 'Nested', 'Z1', 'Z4', 'Z5', 'Z6'],
                       help='Array types to test (default: all)')
    parser.add_argument('--output-dir', type=str, default='results/paper_experiments',
                       help='Output directory (default: results/paper_experiments)')
    parser.add_argument('--test', action='store_true',
                       help='Quick test mode (trials=50, reduced sweeps)')
    
    args = parser.parse_args()
    
    # Test mode adjustments
    if args.test:
        args.trials = 50
        snr_sweep = [0, 5, 10]
        snapshot_sweep = [32, 64]
        print("\n*** TEST MODE: trials=50, reduced sweeps ***\n")
    else:
        snr_sweep = [-5, 0, 5, 10, 15]
        snapshot_sweep = [32, 64, 128]
    
    # Run scenarios
    start_time = time.time()
    
    if args.scenario == '1' or args.scenario == 'all':
        run_scenario1_baseline(
            arrays=args.arrays,
            snr_sweep=snr_sweep,
            snapshot_sweep=snapshot_sweep,
            trials=args.trials,
            output_dir=args.output_dir
        )
    
    if args.scenario == '3' or args.scenario == 'all':
        # Focus on Z5 for ALSS effectiveness
        alss_arrays = ['Z5'] if 'Z5' in args.arrays else [args.arrays[0]]
        run_scenario3_alss_effectiveness(
            arrays=alss_arrays,
            coupling_levels=[0.0, 0.3],
            snr_sweep=snr_sweep,
            snapshot_sweep=snapshot_sweep,
            trials=args.trials,
            output_dir=args.output_dir
        )
    
    if args.scenario == '4' or args.scenario == 'all':
        run_scenario4_cross_array(
            arrays=args.arrays,
            coupling_strength=0.3,
            trials=args.trials,
            output_dir=args.output_dir
        )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("PAPER EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print()
    
    # Print summary
    print("NEXT STEPS:")
    print("  1. Review CSV files in output directory")
    print("  2. Generate plots for paper figures")
    print("  3. Copy figures to papers/radarcon2025_alss/figures/")
    print("  4. Integrate results into paper draft")
    print()


if __name__ == '__main__':
    main()
