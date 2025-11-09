"""
SCENARIO 3: ALSS Regularization Effectiveness Study

Purpose: Validate ALSS (Adaptive Lag-Selective Shrinkage) benefits and identify 
         optimal operating regions through comprehensive statistical testing and 
         parameter sensitivity analysis.

ALSS Innovation: Original contribution - lag-selective shrinkage for coarray MUSIC
                 to reduce estimation variance at low SNR and low snapshot counts.

Key Questions:
1. How much does ALSS improve over conventional coarray MUSIC? (Improvement heatmaps)
2. Is the improvement statistically significant? (p-values, confidence intervals)
3. When does ALSS NOT help? (Failure mode analysis, harmlessness index)
4. How sensitive is performance to ALSS parameters? (τ, L₀ sensitivity)
5. Does ALSS reduce performance variance? (Robustness gain metric)

Related Work:
- Array Design: Kulkarni & Vaidyanathan (2024) "Weight-Constrained Sparse Arrays 
                for Direction of Arrival Estimation," IEEE TSP, Vol. 72
                (Z5 array with w(1)=w(2)=0 for mutual coupling mitigation)
- ALSS Method: Original innovation for coarray lag estimation regularization

Author: [Your Name]
Date: 2025-01-06
Target: RadarCon 2025 / IEEE TSP
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import argparse
import time

from core.radarpy.signal.doa_sim_core import simulate_snapshots, music_spectrum, find_k_peaks
from core.radarpy.signal.metrics import compute_scenario_metrics
from core.radarpy.algorithms.coarray_music import estimate_doa_coarray_music
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor
from geometry_processors.z6_processor import Z6ArrayProcessor


def get_array_positions(array_type: str, N: int = 7, d: float = 0.5) -> tuple:
    """
    Get sensor positions for specified array type.
    
    Returns:
        tuple: (positions_meters, positions_indices) where:
            - positions_meters: Physical positions in meters for simulation
            - positions_indices: Integer grid indices for coarray MUSIC
    """
    if array_type == 'ULA':
        proc = ULArrayProcessor(N=N, d=d)
        positions_indices = np.arange(N)
        positions_meters = positions_indices * d
        return positions_meters, positions_indices
    elif array_type == 'Z5':
        # Z5 canonical positions for N=7
        if N == 7:
            positions_indices = np.array([0, 5, 8, 11, 14, 17, 21])
        else:
            proc = Z5ArrayProcessor(N=N, d=d)
            positions_indices = np.array(proc.data.sensors_positions)
        positions_meters = positions_indices * d
        return positions_meters, positions_indices
    elif array_type == 'Z6':
        # Z6 canonical positions for N=7
        if N == 7:
            positions_indices = np.array([0, 1, 4, 8, 13, 17, 22])
        else:
            proc = Z6ArrayProcessor(N=N, d=d)
            positions_indices = np.array(proc.data.sensors_positions)
        positions_meters = positions_indices * d
        return positions_meters, positions_indices
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def compute_alss_regularization_metrics(
    baseline_trials: List[np.ndarray],
    alss_trials: List[np.ndarray],
    true_doas: np.ndarray,
    wavelength: float,
    snr_db: float,
    M: int,
    K: int,
    baseline_runtimes: List[float],
    alss_runtimes: List[float]
) -> Dict:
    """
    Compute ALSS regularization effectiveness metrics.
    
    CORRECTED: Compares SAME array geometry with/without ALSS
    
    New Metrics:
    1. ALSS_Improvement_%: (RMSE_baseline - RMSE_ALSS) / RMSE_baseline * 100
    2. Statistical_Significance: p-value from paired t-test
    3. Harmlessness_Index: max(0, RMSE_ALSS - RMSE_baseline) in easy regimes
    4. Parameter_Sensitivity: (not computed here, needs sweep)
    5. Robustness_Gain: (Var_baseline - Var_ALSS) / Var_baseline * 100
    """
    from scipy.optimize import linear_sum_assignment
    
    num_trials = len(baseline_trials)
    K = len(true_doas)
    
    # Compute per-trial RMSE for both conditions
    baseline_rmse_trials = []
    alss_rmse_trials = []
    
    for trial_idx in range(num_trials):
        # Baseline RMSE (Z5 without ALSS)
        if baseline_trials[trial_idx] is not None and len(baseline_trials[trial_idx]) == K:
            cost_matrix = np.abs(baseline_trials[trial_idx][:, None] - true_doas[None, :])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            errors = baseline_trials[trial_idx][row_ind] - true_doas[col_ind]
            baseline_rmse_trials.append(np.sqrt(np.mean(errors**2)))
        else:
            baseline_rmse_trials.append(np.nan)
        
        # ALSS RMSE (Z5 with ALSS)
        if alss_trials[trial_idx] is not None and len(alss_trials[trial_idx]) == K:
            cost_matrix = np.abs(alss_trials[trial_idx][:, None] - true_doas[None, :])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            errors = alss_trials[trial_idx][row_ind] - true_doas[col_ind]
            alss_rmse_trials.append(np.sqrt(np.mean(errors**2)))
        else:
            alss_rmse_trials.append(np.nan)
    
    baseline_rmse_trials = np.array(baseline_rmse_trials)
    alss_rmse_trials = np.array(alss_rmse_trials)
    
    # Remove NaN trials (failed estimates)
    valid_mask = ~(np.isnan(baseline_rmse_trials) | np.isnan(alss_rmse_trials))
    baseline_rmse_valid = baseline_rmse_trials[valid_mask]
    alss_rmse_valid = alss_rmse_trials[valid_mask]
    
    if len(baseline_rmse_valid) == 0 or len(alss_rmse_valid) == 0:
        return {
            'ALSS_Improvement_%': np.nan,
            'Statistical_Significance_p': np.nan,
            'Confidence_Interval_95': (np.nan, np.nan),
            'Harmlessness_Index': np.nan,
            'Robustness_Gain_%': np.nan,
            'Baseline_RMSE_mean': np.nan,
            'ALSS_RMSE_mean': np.nan,
            'Baseline_RMSE_std': np.nan,
            'ALSS_RMSE_std': np.nan,
            'Valid_Trials': 0
        }
    
    # 1. ALSS Improvement %
    baseline_rmse_mean = np.mean(baseline_rmse_valid)
    alss_rmse_mean = np.mean(alss_rmse_valid)
    improvement_pct = (baseline_rmse_mean - alss_rmse_mean) / baseline_rmse_mean * 100
    
    # 2. Statistical Significance (paired t-test)
    # H0: ALSS and baseline have same mean RMSE
    # H1: ALSS has lower mean RMSE
    t_stat, p_value = stats.ttest_rel(baseline_rmse_valid, alss_rmse_valid, alternative='greater')
    
    # 95% confidence interval for improvement
    improvement_per_trial = baseline_rmse_valid - alss_rmse_valid
    ci_low, ci_high = stats.t.interval(
        0.95, 
        len(improvement_per_trial) - 1,
        loc=np.mean(improvement_per_trial),
        scale=stats.sem(improvement_per_trial)
    )
    
    # 3. Harmlessness Index (penalty when ALSS is worse)
    # In easy regimes (high SNR, many snapshots), ALSS should not degrade performance
    harmlessness = np.maximum(0, alss_rmse_mean - baseline_rmse_mean)
    
    # 4. Robustness Gain (variance reduction)
    baseline_variance = np.var(baseline_rmse_valid)
    alss_variance = np.var(alss_rmse_valid)
    robustness_gain_pct = (baseline_variance - alss_variance) / baseline_variance * 100 if baseline_variance > 0 else 0.0
    
    return {
        'ALSS_Improvement_%': improvement_pct,
        'Statistical_Significance_p': p_value,
        'Confidence_Interval_95_low': ci_low,
        'Confidence_Interval_95_high': ci_high,
        'Harmlessness_Index': harmlessness,
        'Robustness_Gain_%': robustness_gain_pct,
        'Baseline_RMSE_mean': baseline_rmse_mean,
        'ALSS_RMSE_mean': alss_rmse_mean,
        'Baseline_RMSE_std': np.std(baseline_rmse_valid),
        'ALSS_RMSE_std': np.std(alss_rmse_valid),
        'Baseline_Runtime_mean': np.mean(baseline_runtimes),
        'ALSS_Runtime_mean': np.mean(alss_runtimes),
        'Valid_Trials': len(baseline_rmse_valid),
        't_statistic': t_stat
    }


def run_improvement_heatmap(
    alss_array: str = 'Z5',
    snr_range: np.ndarray = None,
    snapshot_range: np.ndarray = None,
    true_doas: np.ndarray = None,
    num_trials: int = 500,
    wavelength: float = 1.0,
    K: int = 2,
    scan_grid: np.ndarray = None,
    resolution_threshold: float = 3.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Experiment 3A: Generate improvement heatmap (SNR × Snapshots).
    
    Compares ULA vs ALSS array performance across operating conditions.
    """
    if snr_range is None:
        snr_range = np.array([-5, 0, 5, 10, 15])
    if snapshot_range is None:
        snapshot_range = np.array([32, 64, 128, 256, 512])
    if true_doas is None:
        true_doas = np.array([15.0, -20.0])
    if scan_grid is None:
        scan_grid = np.linspace(-90, 90, 361)
    
    # Get array positions (same array, different ALSS settings)
    positions_meters, positions_indices = get_array_positions(alss_array, N=7, d=0.5)
    d_phys = 0.5  # Physical spacing in meters
    
    if verbose:
        print("\n" + "="*70)
        print("  SCENARIO 3A: ALSS Regularization Improvement Heatmap")
        print("="*70)
        print(f"  Array:            {alss_array} (7 sensors)")
        print(f"  Baseline:         {alss_array} WITHOUT ALSS (alss_enabled=False)")
        print(f"  Treatment:        {alss_array} WITH ALSS (alss_enabled=True)")
        print(f"  SNR Range:        {snr_range[0]} to {snr_range[-1]} dB ({len(snr_range)} points)")
        print(f"  Snapshots Range:  {snapshot_range[0]} to {snapshot_range[-1]} ({len(snapshot_range)} points)")
        print(f"  Trials per cell:  {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print(f"  Total conditions: {len(snr_range) * len(snapshot_range)}")
        print("="*70)
    
    results = []
    
    for snr_db in snr_range:
        for M_snapshots in snapshot_range:
            if verbose:
                print(f"\nRunning SNR={snr_db}dB, M={M_snapshots}...", end=' ', flush=True)
            
            start_time = time.time()
            
            # Run trials with SAME array, different ALSS settings
            baseline_estimates = []  # Z5 WITHOUT ALSS
            alss_estimates = []      # Z5 WITH ALSS
            baseline_runtimes = []
            alss_runtimes = []
            
            for trial in range(num_trials):
                seed = abs(trial + int(snr_db * 1000) + int(M_snapshots)) % (2**32 - 1)
                
                # Generate snapshots ONCE (same data for both conditions)
                X, A, snr_lin = simulate_snapshots(
                    sensor_positions=positions_meters,
                    wavelength=wavelength,
                    doas_deg=true_doas,
                    snr_db=snr_db,
                    snapshots=M_snapshots,
                    seed=seed
                )
                
                # Baseline trial: Z5 WITHOUT ALSS
                trial_start = time.perf_counter()
                est_doas_baseline, _ = estimate_doa_coarray_music(
                    X=X,
                    positions=positions_indices,
                    d_phys=d_phys,
                    wavelength=wavelength,
                    K=K,
                    scan_deg=(scan_grid[0], scan_grid[-1], scan_grid[1]-scan_grid[0]),
                    alss_enabled=False,  # NO ALSS
                    return_debug=False
                )
                trial_time = (time.perf_counter() - trial_start) * 1000
                baseline_runtimes.append(trial_time)
                baseline_estimates.append(est_doas_baseline)
                
                # ALSS trial: Z5 WITH ALSS
                trial_start = time.perf_counter()
                est_doas_alss, _ = estimate_doa_coarray_music(
                    X=X,
                    positions=positions_indices,
                    d_phys=d_phys,
                    wavelength=wavelength,
                    K=K,
                    scan_deg=(scan_grid[0], scan_grid[-1], scan_grid[1]-scan_grid[0]),
                    alss_enabled=True,   # WITH ALSS
                    alss_mode="zero",
                    alss_tau=1.0,
                    alss_coreL=3,
                    return_debug=False
                )
                trial_time = (time.perf_counter() - trial_start) * 1000
                alss_runtimes.append(trial_time)
                alss_estimates.append(est_doas_alss)
            
            # Compute ALSS regularization metrics
            reg_metrics = compute_alss_regularization_metrics(
                baseline_trials=baseline_estimates,
                alss_trials=alss_estimates,
                true_doas=true_doas,
                wavelength=wavelength,
                snr_db=snr_db,
                M=M_snapshots,
                K=K,
                baseline_runtimes=baseline_runtimes,
                alss_runtimes=alss_runtimes
            )
            
            elapsed = time.time() - start_time
            
            if verbose:
                improvement = reg_metrics['ALSS_Improvement_%']
                p_val = reg_metrics['Statistical_Significance_p']
                sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"Done! ({elapsed:.1f}s) Improvement={improvement:+.1f}% (p={p_val:.4f}{sig_marker})")
            
            # Store results
            result_row = {
                'SNR_dB': snr_db,
                'Snapshots': M_snapshots,
                'Array': alss_array,
                'Baseline_Condition': f'{alss_array}_noALSS',
                'ALSS_Condition': f'{alss_array}_withALSS',
                **reg_metrics
            }
            results.append(result_row)
    
    df = pd.DataFrame(results)
    return df


def run_parameter_sensitivity(
    alss_array: str = 'Z5',
    snr_db: float = 10.0,
    snapshots: int = 256,
    true_doas: np.ndarray = None,
    num_trials: int = 500,
    wavelength: float = 1.0,
    K: int = 2,
    scan_grid: np.ndarray = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Experiment 3B: Parameter sensitivity analysis.
    
    Tests performance across different array configurations:
    - Different sensor counts (N=5, 7, 9, 11)
    - Different spacings (d=0.3, 0.4, 0.5, 0.6)
    
    Note: This tests ALSS structural parameters, not algorithm parameters.
    """
    if true_doas is None:
        true_doas = np.array([15.0, -20.0])
    if scan_grid is None:
        scan_grid = np.linspace(-90, 90, 361)
    
    # Test different configurations
    N_values = [5, 7, 9, 11]
    d_values = [0.3, 0.4, 0.5, 0.6]
    
    if verbose:
        print("\n" + "="*70)
        print("  SCENARIO 3B: ALSS Parameter Sensitivity")
        print("="*70)
        print(f"  Array Type:       {alss_array}")
        print(f"  Sensor Counts:    {N_values}")
        print(f"  Spacings (λ):     {d_values}")
        print(f"  SNR:              {snr_db} dB")
        print(f"  Snapshots:        {snapshots}")
        print(f"  Trials per config: {num_trials}")
        print(f"  Total configs:    {len(N_values) * len(d_values)}")
        print("="*70)
    
    results = []
    
    for N in N_values:
        for d in d_values:
            if verbose:
                print(f"\nRunning N={N}, d={d:.1f}λ...", end=' ', flush=True)
            
            start_time = time.time()
            
            try:
                # Get ALSS positions with current parameters
                alss_positions = get_array_positions(alss_array, N=N, d=d)
                
                # Get ULA baseline with same parameters
                ula_positions = get_array_positions('ULA', N=N, d=d)
                
                # Run trials
                ula_estimates = []
                alss_estimates = []
                ula_runtimes = []
                alss_runtimes = []
                
                for trial in range(num_trials):
                    seed = abs(trial + N * 1000 + int(d * 10000)) % (2**32 - 1)
                    
                    # ULA trial
                    trial_start = time.perf_counter()
                    X_ula, A_ula, snr_lin = simulate_snapshots(
                        sensor_positions=ula_positions,
                        wavelength=wavelength,
                        doas_deg=true_doas,
                        snr_db=snr_db,
                        snapshots=snapshots,
                        seed=seed
                    )
                    Rxx_ula = (X_ula @ X_ula.conj().T) / snapshots
                    P_ula, _ = music_spectrum(
                        Rxx=Rxx_ula,
                        sensor_positions=ula_positions,
                        wavelength=wavelength,
                        scan_deg=scan_grid,
                        k_sources=K
                    )
                    est_doas_ula = find_k_peaks(P_ula, scan_grid, K)
                    trial_time = (time.perf_counter() - trial_start) * 1000
                    ula_runtimes.append(trial_time)
                    ula_estimates.append(est_doas_ula)
                    
                    # ALSS trial
                    trial_start = time.perf_counter()
                    X_alss, A_alss, snr_lin = simulate_snapshots(
                        sensor_positions=alss_positions,
                        wavelength=wavelength,
                        doas_deg=true_doas,
                        snr_db=snr_db,
                        snapshots=snapshots,
                        seed=seed
                    )
                    Rxx_alss = (X_alss @ X_alss.conj().T) / snapshots
                    P_alss, _ = music_spectrum(
                        Rxx=Rxx_alss,
                        sensor_positions=alss_positions,
                        wavelength=wavelength,
                        scan_deg=scan_grid,
                        k_sources=K
                    )
                    est_doas_alss = find_k_peaks(P_alss, scan_grid, K)
                    trial_time = (time.perf_counter() - trial_start) * 1000
                    alss_runtimes.append(trial_time)
                    alss_estimates.append(est_doas_alss)
                
                # Compute metrics
                reg_metrics = compute_alss_regularization_metrics(
                    ula_trials=ula_estimates,
                    alss_trials=alss_estimates,
                    true_doas=true_doas,
                    ula_positions=ula_positions,
                    alss_positions=alss_positions,
                    wavelength=wavelength,
                    snr_db=snr_db,
                    M=snapshots,
                    K=K,
                    ula_runtimes=ula_runtimes,
                    alss_runtimes=alss_runtimes
                )
                
                elapsed = time.time() - start_time
                
                if verbose:
                    improvement = reg_metrics['ALSS_Improvement_%']
                    p_val = reg_metrics['Statistical_Significance_p']
                    sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    print(f"Done! ({elapsed:.1f}s) Improvement={improvement:+.1f}% (p={p_val:.4f}{sig_marker})")
                
                # Store results
                result_row = {
                    'N_Sensors': N,
                    'Spacing_lambda': d,
                    'Baseline_Array': 'ULA',
                    'ALSS_Array': alss_array,
                    'SNR_dB': snr_db,
                    'Snapshots': snapshots,
                    **reg_metrics
                }
                results.append(result_row)
                
            except Exception as e:
                if verbose:
                    print(f"FAILED! ({e})")
                # Store failure
                result_row = {
                    'N_Sensors': N,
                    'Spacing_lambda': d,
                    'Baseline_Array': 'ULA',
                    'ALSS_Array': alss_array,
                    'SNR_dB': snr_db,
                    'Snapshots': snapshots,
                    'ALSS_Improvement_%': np.nan,
                    'Statistical_Significance_p': np.nan,
                    'Error': str(e)
                }
                results.append(result_row)
    
    df = pd.DataFrame(results)
    return df


def plot_improvement_heatmap(df: pd.DataFrame, output_dir: Path, alss_array: str):
    """Generate 2×3 panel heatmap visualization."""
    # Create pivot tables for each metric
    snr_vals = sorted(df['SNR_dB'].unique())
    snap_vals = sorted(df['Snapshots'].unique())
    
    pivot_improvement = df.pivot(index='SNR_dB', columns='Snapshots', values='ALSS_Improvement_%')
    pivot_pvalue = df.pivot(index='SNR_dB', columns='Snapshots', values='Statistical_Significance_p')
    pivot_harmless = df.pivot(index='SNR_dB', columns='Snapshots', values='Harmlessness_Index')
    pivot_robustness = df.pivot(index='SNR_dB', columns='Snapshots', values='Robustness_Gain_%')
    pivot_baseline_rmse = df.pivot(index='SNR_dB', columns='Snapshots', values='Baseline_RMSE_mean')
    pivot_alss_rmse = df.pivot(index='SNR_dB', columns='Snapshots', values='ALSS_RMSE_mean')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'ALSS Regularization Effectiveness: {alss_array} (with vs without ALSS)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel 1: Improvement %
    ax = axes[0, 0]
    im1 = ax.imshow(pivot_improvement, cmap='RdYlGn', aspect='auto', 
                    vmin=-20, vmax=100, origin='lower')
    ax.set_xticks(range(len(snap_vals)))
    ax.set_xticklabels(snap_vals)
    ax.set_yticks(range(len(snr_vals)))
    ax.set_yticklabels(snr_vals)
    ax.set_xlabel('Snapshots', fontsize=11, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title('(a) ALSS Improvement (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax, label='Improvement (%)')
    
    # Annotate with significance markers
    for i, snr in enumerate(snr_vals):
        for j, snap in enumerate(snap_vals):
            p_val = pivot_pvalue.loc[snr, snap]
            text = ''
            if p_val < 0.001:
                text = '***'
            elif p_val < 0.01:
                text = '**'
            elif p_val < 0.05:
                text = '*'
            if text:
                ax.text(j, i, text, ha='center', va='center', 
                       color='black', fontsize=14, fontweight='bold')
    
    # Panel 2: p-value (log scale)
    ax = axes[0, 1]
    pivot_pvalue_log = np.log10(pivot_pvalue + 1e-10)  # Avoid log(0)
    im2 = ax.imshow(pivot_pvalue_log, cmap='RdYlGn_r', aspect='auto',
                    vmin=-5, vmax=0, origin='lower')
    ax.set_xticks(range(len(snap_vals)))
    ax.set_xticklabels(snap_vals)
    ax.set_yticks(range(len(snr_vals)))
    ax.set_yticklabels(snr_vals)
    ax.set_xlabel('Snapshots', fontsize=11, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title('(b) Statistical Significance (log₁₀ p)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im2, ax=ax, label='log₁₀(p-value)')
    cbar.ax.axhline(-3, color='red', linestyle='--', linewidth=2)  # p=0.001 threshold
    
    # Panel 3: Harmlessness Index
    ax = axes[0, 2]
    im3 = ax.imshow(pivot_harmless * 1000, cmap='RdYlGn_r', aspect='auto',
                    vmin=0, vmax=10, origin='lower')  # Convert to millidegrees
    ax.set_xticks(range(len(snap_vals)))
    ax.set_xticklabels(snap_vals)
    ax.set_yticks(range(len(snr_vals)))
    ax.set_yticklabels(snr_vals)
    ax.set_xlabel('Snapshots', fontsize=11, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Harmlessness Index (mdeg)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax, label='Penalty (mdeg)')
    
    # Panel 4: Robustness Gain
    ax = axes[1, 0]
    im4 = ax.imshow(pivot_robustness, cmap='RdYlGn', aspect='auto',
                    vmin=-20, vmax=100, origin='lower')
    ax.set_xticks(range(len(snap_vals)))
    ax.set_xticklabels(snap_vals)
    ax.set_yticks(range(len(snr_vals)))
    ax.set_yticklabels(snr_vals)
    ax.set_xlabel('Snapshots', fontsize=11, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title('(d) Robustness Gain (% Var Reduction)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax, label='Variance Reduction (%)')
    
    # Panel 5: Baseline RMSE (Z5 without ALSS)
    ax = axes[1, 1]
    im5 = ax.imshow(pivot_baseline_rmse, cmap='YlOrRd', aspect='auto',
                    vmin=0, vmax=0.5, origin='lower')
    ax.set_xticks(range(len(snap_vals)))
    ax.set_xticklabels(snap_vals)
    ax.set_yticks(range(len(snr_vals)))
    ax.set_yticklabels(snr_vals)
    ax.set_xlabel('Snapshots', fontsize=11, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title(f'(e) {alss_array} Baseline RMSE (no ALSS) (°)', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax, label='RMSE (deg)')
    
    # Panel 6: ALSS RMSE (Z5 with ALSS)
    ax = axes[1, 2]
    im6 = ax.imshow(pivot_alss_rmse, cmap='YlOrRd', aspect='auto',
                    vmin=0, vmax=0.5, origin='lower')
    ax.set_xticks(range(len(snap_vals)))
    ax.set_xticklabels(snap_vals)
    ax.set_yticks(range(len(snr_vals)))
    ax.set_yticklabels(snr_vals)
    ax.set_xlabel('Snapshots', fontsize=11, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_title(f'(f) {alss_array} with ALSS RMSE (°)', fontsize=12, fontweight='bold')
    plt.colorbar(im6, ax=ax, label='RMSE (deg)')
    
    plt.tight_layout()
    output_path = output_dir / f'scenario3a_improvement_heatmap_{alss_array}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap: {output_path}")


def plot_parameter_sensitivity(df: pd.DataFrame, output_dir: Path, alss_array: str):
    """Generate 2×2 panel parameter sensitivity visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{alss_array} Parameter Sensitivity Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel 1: Improvement vs N (different d values)
    ax = axes[0, 0]
    d_values = sorted(df['Spacing_lambda'].unique())
    for d in d_values:
        subset = df[df['Spacing_lambda'] == d]
        ax.plot(subset['N_Sensors'], subset['ALSS_Improvement_%'], 
               'o-', linewidth=2, markersize=8, label=f'd={d:.1f}λ')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Sensors (N)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ALSS Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('(a) Improvement vs Sensor Count', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Improvement vs d (different N values)
    ax = axes[0, 1]
    N_values = sorted(df['N_Sensors'].unique())
    for N in N_values:
        subset = df[df['N_Sensors'] == N]
        ax.plot(subset['Spacing_lambda'], subset['ALSS_Improvement_%'], 
               's-', linewidth=2, markersize=8, label=f'N={N}')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spacing (λ)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ALSS Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('(b) Improvement vs Spacing', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Robustness Gain vs Configuration
    ax = axes[1, 0]
    for d in d_values:
        subset = df[df['Spacing_lambda'] == d]
        ax.plot(subset['N_Sensors'], subset['Robustness_Gain_%'], 
               'o-', linewidth=2, markersize=8, label=f'd={d:.1f}λ')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Sensors (N)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Robustness Gain (%)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Variance Reduction vs Sensor Count', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Statistical Significance (bar chart)
    ax = axes[1, 1]
    # Count significant improvements per configuration
    df_sig = df.copy()
    df_sig['Config'] = df_sig['N_Sensors'].astype(str) + 's, ' + df_sig['Spacing_lambda'].astype(str) + 'λ'
    df_sig['Significant'] = df_sig['Statistical_Significance_p'] < 0.05
    
    configs = df_sig['Config'].values
    p_values = df_sig['Statistical_Significance_p'].values
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    
    x_pos = np.arange(len(configs))
    ax.bar(x_pos, -np.log10(p_values + 1e-10), color=colors, alpha=0.7)
    ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=2, label='p=0.05')
    ax.axhline(-np.log10(0.001), color='red', linestyle='--', linewidth=2, label='p=0.001')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Configuration (N, d)', fontsize=11, fontweight='bold')
    ax.set_ylabel('-log₁₀(p-value)', fontsize=11, fontweight='bold')
    ax.set_title('(d) Statistical Significance by Configuration', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f'scenario3b_parameter_sensitivity_{alss_array}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved parameter sensitivity plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SCENARIO 3: ALSS Regularization Effectiveness',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        default=['heatmap'],
        choices=['heatmap', 'sensitivity', 'all'],
        help="Which experiments to run"
    )
    
    parser.add_argument('--alss-array', type=str, default='Z5', choices=['Z5', 'Z6'],
                       help="ALSS array type to test")
    
    # Heatmap parameters
    parser.add_argument('--snr-min', type=float, default=-5.0, help="Minimum SNR (dB)")
    parser.add_argument('--snr-max', type=float, default=15.0, help="Maximum SNR (dB)")
    parser.add_argument('--snr-points', type=int, default=5, help="Number of SNR points")
    
    parser.add_argument('--snap-min', type=int, default=32, help="Minimum snapshots")
    parser.add_argument('--snap-max', type=int, default=512, help="Maximum snapshots")
    parser.add_argument('--snap-points', type=int, default=5, help="Number of snapshot points")
    
    # Sensitivity parameters
    parser.add_argument('--sensitivity-snr', type=float, default=10.0, 
                       help="SNR for sensitivity analysis (dB)")
    parser.add_argument('--sensitivity-snapshots', type=int, default=256,
                       help="Snapshots for sensitivity analysis")
    
    # General parameters
    parser.add_argument('--trials', type=int, default=500, help="Monte Carlo trials")
    parser.add_argument('--doas', type=float, nargs='+', default=[15.0, -20.0],
                       help="True DOAs (degrees)")
    parser.add_argument('--resolution-threshold', type=float, default=3.0,
                       help="Resolution threshold (degrees)")
    
    parser.add_argument('--output-dir', type=str, default='results/scenario3_regularization',
                       help="Output directory")
    parser.add_argument('--no-plots', action='store_true', help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    true_doas = np.array(args.doas)
    experiments = args.experiments
    if 'all' in experiments:
        experiments = ['heatmap', 'sensitivity']
    
    print("\n" + "="*70)
    print("  SCENARIO 3: ALSS REGULARIZATION EFFECTIVENESS")
    print("="*70)
    print(f"  Experiments:      {', '.join(experiments)}")
    print(f"  ALSS Array:       {args.alss_array}")
    print(f"  Baseline:         ULA")
    print(f"  Trials:           {args.trials}")
    print(f"  True DOAs:        {true_doas}°")
    print(f"  Output:           {output_path}")
    print("="*70)
    
    # Experiment 3A: Improvement Heatmap
    if 'heatmap' in experiments:
        snr_range = np.linspace(args.snr_min, args.snr_max, args.snr_points)
        # Use logarithmic spacing for snapshots
        snap_range = np.unique(np.logspace(
            np.log10(args.snap_min), 
            np.log10(args.snap_max), 
            args.snap_points
        ).astype(int))
        
        results_heatmap = run_improvement_heatmap(
            alss_array=args.alss_array,
            snr_range=snr_range,
            snapshot_range=snap_range,
            true_doas=true_doas,
            num_trials=args.trials,
            verbose=True
        )
        
        # Save results
        csv_path = output_path / f'scenario3a_improvement_heatmap_{args.alss_array}.csv'
        results_heatmap.to_csv(csv_path, index=False)
        print(f"\n✓ Saved heatmap results: {csv_path}")
        
        # Print summary statistics
        print(f"\n📊 Heatmap Summary Statistics:")
        print(f"  Mean improvement:      {results_heatmap['ALSS_Improvement_%'].mean():.1f}%")
        print(f"  Max improvement:       {results_heatmap['ALSS_Improvement_%'].max():.1f}%")
        print(f"  Min improvement:       {results_heatmap['ALSS_Improvement_%'].min():.1f}%")
        sig_count = (results_heatmap['Statistical_Significance_p'] < 0.05).sum()
        total_count = len(results_heatmap)
        print(f"  Significant cells:     {sig_count}/{total_count} ({sig_count/total_count*100:.1f}%)")
        print(f"  Mean harmlessness:     {results_heatmap['Harmlessness_Index'].mean()*1000:.2f} mdeg")
        print(f"  Mean robustness gain:  {results_heatmap['Robustness_Gain_%'].mean():.1f}%")
        
        # Generate plots
        if not args.no_plots:
            plot_improvement_heatmap(results_heatmap, output_path, args.alss_array)
    
    # Experiment 3B: Parameter Sensitivity
    if 'sensitivity' in experiments:
        results_sensitivity = run_parameter_sensitivity(
            alss_array=args.alss_array,
            snr_db=args.sensitivity_snr,
            snapshots=args.sensitivity_snapshots,
            true_doas=true_doas,
            num_trials=args.trials,
            verbose=True
        )
        
        # Save results
        csv_path = output_path / f'scenario3b_parameter_sensitivity_{args.alss_array}.csv'
        results_sensitivity.to_csv(csv_path, index=False)
        print(f"\n✓ Saved sensitivity results: {csv_path}")
        
        # Print summary
        print(f"\n📊 Sensitivity Summary:")
        print(f"  Configurations tested: {len(results_sensitivity)}")
        valid_results = results_sensitivity.dropna(subset=['ALSS_Improvement_%'])
        print(f"  Successful configs:    {len(valid_results)}")
        if len(valid_results) > 0:
            print(f"  Mean improvement:      {valid_results['ALSS_Improvement_%'].mean():.1f}%")
            print(f"  Std improvement:       {valid_results['ALSS_Improvement_%'].std():.1f}%")
            best_config = valid_results.loc[valid_results['ALSS_Improvement_%'].idxmax()]
            print(f"  Best config:           N={int(best_config['N_Sensors'])}, d={best_config['Spacing_lambda']:.1f}λ")
            print(f"  Best improvement:      {best_config['ALSS_Improvement_%']:.1f}%")
        
        # Generate plots
        if not args.no_plots and len(valid_results) > 0:
            plot_parameter_sensitivity(results_sensitivity, output_path, args.alss_array)
    
    print("\n" + "="*70)
    print("  SCENARIO 3 COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
