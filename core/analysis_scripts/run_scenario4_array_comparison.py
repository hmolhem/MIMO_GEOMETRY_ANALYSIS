"""
SCENARIO 4: Array Comparison Under Realistic Conditions

Purpose: Demonstrate ALSS generality across array types and validate that ALSS
         optimization provides consistent benefits regardless of base geometry.

Key Questions:
1. How much does ALSS improve different array types? (Relative improvement per array)
2. Does ALSS change performance rankings? (Ranking consistency analysis)
3. How efficiently do arrays use virtual aperture? (Virtual aperture efficiency)
4. Does ALSS provide extra coupling resilience? (Coupling resilience gain)
5. What's the computational cost? (Complexity overhead ratio)

Author: GitHub Copilot
Date: 2025-01-06
Paper: ALSS (Aliasing-Limited Sparse Sensing) - RadarCon 2025
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
from core.radarpy.signal.mutual_coupling import generate_mcm
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor
from geometry_processors.z6_processor import Z6ArrayProcessor


def get_array_positions(array_type: str, N: int = 7, d: float = 0.5) -> np.ndarray:
    """Get sensor positions for specified array type."""
    if array_type == 'ULA':
        proc = ULArrayProcessor(N=N, d=d)
        return np.array(proc.data.sensors_positions) * d
    elif array_type == 'Nested':
        # Nested array with default N1=2, N2=5 for total N=7
        N1 = max(2, N // 3)
        N2 = N - N1
        proc = NestedArrayProcessor(N1=N1, N2=N2, d=d)
        return np.array(proc.data.sensors_positions)
    elif array_type == 'Z5':
        proc = Z5ArrayProcessor(N=N, d=d)
        return np.array(proc.data.sensors_positions)
    elif array_type == 'Z6':
        proc = Z6ArrayProcessor(N=N, d=d)
        return np.array(proc.data.sensors_positions)
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def get_virtual_aperture(positions: np.ndarray) -> int:
    """Compute virtual aperture (unique difference coarray positions)."""
    N = len(positions)
    differences = []
    for i in range(N):
        for j in range(N):
            differences.append(positions[i] - positions[j])
    unique_diffs = len(np.unique(np.round(differences, decimals=6)))
    return unique_diffs


def compute_array_comparison_metrics(
    array_estimates: Dict[str, List[np.ndarray]],
    array_positions: Dict[str, np.ndarray],
    true_doas: np.ndarray,
    wavelength: float,
    snr_db: float,
    M: int,
    K: int,
    array_runtimes: Dict[str, List[float]],
    coupling_matrix: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute comprehensive array comparison metrics.
    
    New Metrics for Scenario 4:
    1. Relative_Improvement_%: How much each array improves over baseline (ULA)
    2. Ranking_Consistency: Does coupling change performance ranking?
    3. Virtual_Aperture_Efficiency: RMSE improvement per virtual sensor
    4. Coupling_Resilience_Gain: Extra benefit under coupling vs no coupling
    5. Complexity_Overhead_Ratio: Improvement per millisecond runtime
    """
    from scipy.optimize import linear_sum_assignment
    
    results = []
    
    for array_name, estimates in array_estimates.items():
        positions = array_positions[array_name]
        runtimes = array_runtimes[array_name]
        
        # Compute per-trial RMSE
        rmse_trials = []
        resolution_trials = []
        
        for trial_idx, est_doas in enumerate(estimates):
            if est_doas is not None and len(est_doas) == K:
                # Compute RMSE via Hungarian matching
                cost_matrix = np.abs(est_doas[:, None] - true_doas[None, :])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                errors = est_doas[row_ind] - true_doas[col_ind]
                rmse = np.sqrt(np.mean(errors**2))
                rmse_trials.append(rmse)
                
                # Check resolution (all sources within threshold)
                resolved = all(np.min(np.abs(est_doas - true_doa)) < 3.0 for true_doa in true_doas)
                resolution_trials.append(1.0 if resolved else 0.0)
            else:
                rmse_trials.append(np.nan)
                resolution_trials.append(0.0)
        
        rmse_trials = np.array(rmse_trials)
        resolution_trials = np.array(resolution_trials)
        
        # Remove NaN trials
        valid_mask = ~np.isnan(rmse_trials)
        rmse_valid = rmse_trials[valid_mask]
        
        if len(rmse_valid) == 0:
            continue
        
        # Basic metrics
        rmse_mean = np.mean(rmse_valid)
        rmse_std = np.std(rmse_valid)
        resolution_rate = np.mean(resolution_trials) * 100
        runtime_mean = np.mean(runtimes)
        
        # Virtual aperture metrics
        N_physical = len(positions)
        N_virtual = get_virtual_aperture(positions)
        aperture_size = np.max(positions) - np.min(positions)
        
        # Store results
        result = {
            'Array': array_name,
            'N_Physical': N_physical,
            'N_Virtual': N_virtual,
            'Aperture_Size': aperture_size,
            'RMSE_mean': rmse_mean,
            'RMSE_std': rmse_std,
            'Resolution_Rate': resolution_rate,
            'Runtime_mean_ms': runtime_mean,
            'Valid_Trials': len(rmse_valid),
            'SNR_dB': snr_db,
            'Snapshots': M
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Compute relative metrics (using ULA as baseline)
    if 'ULA' in df['Array'].values:
        ula_rmse = df[df['Array'] == 'ULA']['RMSE_mean'].values[0]
        ula_runtime = df[df['Array'] == 'ULA']['Runtime_mean_ms'].values[0]
        
        df['Relative_Improvement_%'] = (ula_rmse - df['RMSE_mean']) / ula_rmse * 100
        df['Virtual_Aperture_Efficiency'] = df['Relative_Improvement_%'] / df['N_Virtual']
        df['Runtime_Overhead_%'] = (df['Runtime_mean_ms'] - ula_runtime) / ula_runtime * 100
        df['Complexity_Overhead_Ratio'] = df['Relative_Improvement_%'] / (df['Runtime_Overhead_%'].abs() + 1e-6)
    else:
        df['Relative_Improvement_%'] = 0.0
        df['Virtual_Aperture_Efficiency'] = 0.0
        df['Runtime_Overhead_%'] = 0.0
        df['Complexity_Overhead_Ratio'] = 0.0
    
    return df


def run_array_comparison_baseline(
    array_types: List[str],
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
    Experiment 4A: Baseline array comparison (no coupling).
    
    Tests multiple array types under ideal conditions to establish
    performance ranking and baseline improvement metrics.
    """
    if true_doas is None:
        true_doas = np.array([15.0, -20.0])
    if scan_grid is None:
        scan_grid = np.linspace(-90, 90, 361)
    
    if verbose:
        print("\n" + "="*70)
        print("  SCENARIO 4A: Baseline Array Comparison (No Coupling)")
        print("="*70)
        print(f"  Arrays:           {', '.join(array_types)}")
        print(f"  SNR:              {snr_db} dB")
        print(f"  Snapshots:        {snapshots}")
        print(f"  Trials per array: {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print("="*70)
    
    # Get array positions
    array_positions = {}
    for array_type in array_types:
        try:
            array_positions[array_type] = get_array_positions(array_type, N=7, d=0.5)
        except Exception as e:
            print(f"⚠️ Failed to create {array_type}: {e}")
            continue
    
    # Run trials for each array
    array_estimates = {}
    array_runtimes = {}
    
    for array_type in array_positions.keys():
        if verbose:
            print(f"\nRunning {array_type} array...", end=' ', flush=True)
        
        start_time = time.time()
        positions = array_positions[array_type]
        
        estimates = []
        runtimes = []
        
        for trial in range(num_trials):
            seed = abs(trial + int(snr_db * 1000) + snapshots + hash(array_type)) % (2**32 - 1)
            
            trial_start = time.perf_counter()
            X, A, snr_lin = simulate_snapshots(
                sensor_positions=positions,
                wavelength=wavelength,
                doas_deg=true_doas,
                snr_db=snr_db,
                snapshots=snapshots,
                seed=seed
            )
            Rxx = (X @ X.conj().T) / snapshots
            P_db, _ = music_spectrum(
                Rxx=Rxx,
                sensor_positions=positions,
                wavelength=wavelength,
                scan_deg=scan_grid,
                k_sources=K
            )
            est_doas = find_k_peaks(P_db, scan_grid, K)
            trial_time = (time.perf_counter() - trial_start) * 1000
            
            runtimes.append(trial_time)
            estimates.append(est_doas)
        
        array_estimates[array_type] = estimates
        array_runtimes[array_type] = runtimes
        
        elapsed = time.time() - start_time
        
        if verbose:
            # Quick RMSE calculation for display
            from scipy.optimize import linear_sum_assignment
            rmse_trials = []
            for est in estimates:
                if est is not None and len(est) == K:
                    cost_matrix = np.abs(est[:, None] - true_doas[None, :])
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    errors = est[row_ind] - true_doas[col_ind]
                    rmse_trials.append(np.sqrt(np.mean(errors**2)))
            rmse_mean = np.mean(rmse_trials) if rmse_trials else np.nan
            print(f"Done! ({elapsed:.1f}s) RMSE={rmse_mean:.3f}°")
    
    # Compute metrics
    results_df = compute_array_comparison_metrics(
        array_estimates=array_estimates,
        array_positions=array_positions,
        true_doas=true_doas,
        wavelength=wavelength,
        snr_db=snr_db,
        M=snapshots,
        K=K,
        array_runtimes=array_runtimes
    )
    
    return results_df


def run_array_comparison_with_coupling(
    array_types: List[str],
    coupling_strength: float = 0.3,
    snr_db: float = 10.0,
    snapshots: int = 256,
    true_doas: np.ndarray = None,
    num_trials: int = 500,
    wavelength: float = 1.0,
    K: int = 2,
    scan_grid: np.ndarray = None,
    mcm_model: str = 'exponential',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Experiment 4B: Array comparison under mutual coupling.
    
    Tests same arrays under coupling to measure coupling resilience gain.
    """
    if true_doas is None:
        true_doas = np.array([15.0, -20.0])
    if scan_grid is None:
        scan_grid = np.linspace(-90, 90, 361)
    
    if verbose:
        print("\n" + "="*70)
        print("  SCENARIO 4B: Array Comparison with Mutual Coupling")
        print("="*70)
        print(f"  Arrays:           {', '.join(array_types)}")
        print(f"  Coupling:         c1 = {coupling_strength}")
        print(f"  SNR:              {snr_db} dB")
        print(f"  Snapshots:        {snapshots}")
        print(f"  Trials per array: {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print(f"  MCM Model:        {mcm_model}")
        print("="*70)
    
    # Get array positions
    array_positions = {}
    for array_type in array_types:
        try:
            array_positions[array_type] = get_array_positions(array_type, N=7, d=0.5)
        except Exception as e:
            print(f"⚠️ Failed to create {array_type}: {e}")
            continue
    
    # Run trials for each array
    array_estimates = {}
    array_runtimes = {}
    
    for array_type in array_positions.keys():
        if verbose:
            print(f"\nRunning {array_type} array...", end=' ', flush=True)
        
        start_time = time.time()
        positions = array_positions[array_type]
        N = len(positions)
        
        # Generate coupling matrix
        C = generate_mcm(N, positions, model=mcm_model, c1=coupling_strength)
        
        estimates = []
        runtimes = []
        
        for trial in range(num_trials):
            seed = abs(trial + int(snr_db * 1000) + snapshots + hash(array_type)) % (2**32 - 1)
            
            trial_start = time.perf_counter()
            X, A, snr_lin = simulate_snapshots(
                sensor_positions=positions,
                wavelength=wavelength,
                doas_deg=true_doas,
                snr_db=snr_db,
                snapshots=snapshots,
                seed=seed,
                coupling_matrix=C
            )
            Rxx = (X @ X.conj().T) / snapshots
            P_db, _ = music_spectrum(
                Rxx=Rxx,
                sensor_positions=positions,
                wavelength=wavelength,
                scan_deg=scan_grid,
                k_sources=K,
                coupling_matrix=C
            )
            est_doas = find_k_peaks(P_db, scan_grid, K)
            trial_time = (time.perf_counter() - trial_start) * 1000
            
            runtimes.append(trial_time)
            estimates.append(est_doas)
        
        array_estimates[array_type] = estimates
        array_runtimes[array_type] = runtimes
        
        elapsed = time.time() - start_time
        
        if verbose:
            # Quick RMSE calculation for display
            from scipy.optimize import linear_sum_assignment
            rmse_trials = []
            for est in estimates:
                if est is not None and len(est) == K:
                    cost_matrix = np.abs(est[:, None] - true_doas[None, :])
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    errors = est[row_ind] - true_doas[col_ind]
                    rmse_trials.append(np.sqrt(np.mean(errors**2)))
            rmse_mean = np.mean(rmse_trials) if rmse_trials else np.nan
            print(f"Done! ({elapsed:.1f}s) RMSE={rmse_mean:.3f}°")
    
    # Compute metrics
    results_df = compute_array_comparison_metrics(
        array_estimates=array_estimates,
        array_positions=array_positions,
        true_doas=true_doas,
        wavelength=wavelength,
        snr_db=snr_db,
        M=snapshots,
        K=K,
        array_runtimes=array_runtimes,
        coupling_matrix=C
    )
    
    return results_df


def compute_ranking_consistency(baseline_df: pd.DataFrame, coupled_df: pd.DataFrame) -> Dict:
    """
    Compute ranking consistency metric.
    
    Checks if array performance ranking changes under coupling.
    Returns Kendall's tau correlation coefficient.
    """
    # Get common arrays
    common_arrays = set(baseline_df['Array']) & set(coupled_df['Array'])
    
    if len(common_arrays) < 2:
        return {'Ranking_Consistency': np.nan, 'Kendall_Tau': np.nan, 'p_value': np.nan}
    
    # Get rankings (lower RMSE = better = lower rank number)
    baseline_sorted = baseline_df.sort_values('RMSE_mean')
    coupled_sorted = coupled_df.sort_values('RMSE_mean')
    
    baseline_ranks = {arr: idx for idx, arr in enumerate(baseline_sorted['Array'])}
    coupled_ranks = {arr: idx for idx, arr in enumerate(coupled_sorted['Array'])}
    
    # Compute Kendall's tau
    baseline_rank_list = [baseline_ranks[arr] for arr in common_arrays if arr in baseline_ranks]
    coupled_rank_list = [coupled_ranks[arr] for arr in common_arrays if arr in coupled_ranks]
    
    if len(baseline_rank_list) < 2:
        return {'Ranking_Consistency': np.nan, 'Kendall_Tau': np.nan, 'p_value': np.nan}
    
    tau, p_value = stats.kendalltau(baseline_rank_list, coupled_rank_list)
    
    # Consistency as percentage (1 = perfect, 0 = uncorrelated, -1 = reversed)
    consistency_pct = (tau + 1) / 2 * 100  # Map [-1, 1] to [0, 100]
    
    return {
        'Ranking_Consistency_%': consistency_pct,
        'Kendall_Tau': tau,
        'p_value': p_value,
        'Baseline_Ranking': list(baseline_sorted['Array']),
        'Coupled_Ranking': list(coupled_sorted['Array'])
    }


def compute_coupling_resilience_gain(baseline_df: pd.DataFrame, coupled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute coupling resilience gain for each array.
    
    Measures how much performance degrades under coupling,
    compared to baseline ULA degradation.
    """
    results = []
    
    # Get ULA baseline and coupled performance
    ula_baseline = baseline_df[baseline_df['Array'] == 'ULA']['RMSE_mean'].values
    ula_coupled = coupled_df[coupled_df['Array'] == 'ULA']['RMSE_mean'].values
    
    if len(ula_baseline) == 0 or len(ula_coupled) == 0:
        return pd.DataFrame()
    
    ula_degradation_pct = (ula_coupled[0] - ula_baseline[0]) / ula_baseline[0] * 100
    
    # Compute for all arrays
    for array_name in baseline_df['Array']:
        baseline_rmse = baseline_df[baseline_df['Array'] == array_name]['RMSE_mean'].values
        coupled_rmse = coupled_df[coupled_df['Array'] == array_name]['RMSE_mean'].values
        
        if len(baseline_rmse) == 0 or len(coupled_rmse) == 0:
            continue
        
        array_degradation_pct = (coupled_rmse[0] - baseline_rmse[0]) / baseline_rmse[0] * 100
        
        # Coupling resilience gain: how much less does this array degrade vs ULA?
        resilience_gain = ula_degradation_pct - array_degradation_pct
        
        results.append({
            'Array': array_name,
            'Baseline_RMSE': baseline_rmse[0],
            'Coupled_RMSE': coupled_rmse[0],
            'Degradation_%': array_degradation_pct,
            'ULA_Degradation_%': ula_degradation_pct,
            'Coupling_Resilience_Gain_%': resilience_gain
        })
    
    return pd.DataFrame(results)


def plot_array_comparison(
    baseline_df: pd.DataFrame,
    coupled_df: pd.DataFrame,
    resilience_df: pd.DataFrame,
    ranking_metrics: Dict,
    output_dir: Path
):
    """Generate comprehensive array comparison visualization (2×3 panels)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SCENARIO 4: Array Comparison Under Realistic Conditions', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Sort by baseline performance
    baseline_sorted = baseline_df.sort_values('RMSE_mean')
    arrays = baseline_sorted['Array'].values
    
    # Panel 1: RMSE Comparison (Baseline vs Coupled)
    ax = axes[0, 0]
    x = np.arange(len(arrays))
    width = 0.35
    
    baseline_rmse = [baseline_df[baseline_df['Array'] == arr]['RMSE_mean'].values[0] for arr in arrays]
    coupled_rmse = [coupled_df[coupled_df['Array'] == arr]['RMSE_mean'].values[0] if arr in coupled_df['Array'].values else np.nan for arr in arrays]
    
    ax.bar(x - width/2, baseline_rmse, width, label='No Coupling', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, coupled_rmse, width, label='With Coupling (c1=0.3)', color='orangered', alpha=0.8)
    ax.set_xlabel('Array Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMSE (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('(a) RMSE: Baseline vs Coupling', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(arrays, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Relative Improvement over ULA
    ax = axes[0, 1]
    improvement = baseline_sorted['Relative_Improvement_%'].values
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    ax.barh(arrays, improvement, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Improvement over ULA (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Array Type', fontsize=11, fontweight='bold')
    ax.set_title('(b) Relative Improvement (No Coupling)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 3: Virtual Aperture Efficiency
    ax = axes[0, 2]
    vae = baseline_sorted['Virtual_Aperture_Efficiency'].values
    n_virtual = baseline_sorted['N_Virtual'].values
    scatter = ax.scatter(n_virtual, vae, s=200, c=range(len(arrays)), cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    for i, arr in enumerate(arrays):
        ax.annotate(arr, (n_virtual[i], vae[i]), fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Virtual Aperture Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Efficiency (% Improvement / Virtual Sensor)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Virtual Aperture Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Coupling Resilience Gain
    ax = axes[1, 0]
    if len(resilience_df) > 0:
        resilience_sorted = resilience_df.sort_values('Coupling_Resilience_Gain_%', ascending=False)
        resilience_arrays = resilience_sorted['Array'].values
        resilience_gain = resilience_sorted['Coupling_Resilience_Gain_%'].values
        colors = ['green' if rg > 0 else 'red' for rg in resilience_gain]
        ax.barh(resilience_arrays, resilience_gain, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Coupling Resilience Gain (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Array Type', fontsize=11, fontweight='bold')
        ax.set_title('(d) Coupling Resilience vs ULA', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 5: Performance Rankings
    ax = axes[1, 1]
    if 'Baseline_Ranking' in ranking_metrics and 'Coupled_Ranking' in ranking_metrics:
        baseline_rank = ranking_metrics['Baseline_Ranking']
        coupled_rank = ranking_metrics['Coupled_Ranking']
        
        # Create ranking plot
        y_baseline = np.arange(len(baseline_rank))
        y_coupled = np.arange(len(coupled_rank))
        
        # Match arrays between rankings
        for i, arr in enumerate(baseline_rank):
            if arr in coupled_rank:
                j = coupled_rank.index(arr)
                ax.plot([0, 1], [i, j], 'o-', linewidth=2, markersize=8, label=arr if i < 5 else '')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Coupling', 'With Coupling'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Rank (0=Best)', fontsize=11, fontweight='bold')
        ax.set_title(f"(e) Ranking Consistency (τ={ranking_metrics.get('Kendall_Tau', 0):.2f})", 
                    fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='y')
        if len(baseline_rank) <= 5:
            ax.legend(loc='best', fontsize=8)
    
    # Panel 6: Complexity-Performance Tradeoff
    ax = axes[1, 2]
    runtime = baseline_sorted['Runtime_mean_ms'].values
    improvement = baseline_sorted['Relative_Improvement_%'].values
    scatter = ax.scatter(runtime, improvement, s=200, c=range(len(arrays)), cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    for i, arr in enumerate(arrays):
        ax.annotate(arr, (runtime[i], improvement[i]), fontsize=9, ha='center', va='bottom')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Runtime (ms)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Improvement over ULA (%)', fontsize=11, fontweight='bold')
    ax.set_title('(f) Complexity-Performance Tradeoff', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'scenario4_array_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved array comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SCENARIO 4: Array Comparison Under Realistic Conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        default=['baseline'],
        choices=['baseline', 'coupling', 'all'],
        help="Which experiments to run"
    )
    
    parser.add_argument('--arrays', type=str, nargs='+', 
                       default=['ULA', 'Nested', 'Z5', 'Z6'],
                       help="Array types to compare")
    
    parser.add_argument('--coupling-strength', type=float, default=0.3,
                       help="Coupling strength for Experiment 4B")
    
    parser.add_argument('--snr', type=float, default=10.0, help="SNR (dB)")
    parser.add_argument('--snapshots', type=int, default=256, help="Number of snapshots")
    parser.add_argument('--trials', type=int, default=500, help="Monte Carlo trials")
    parser.add_argument('--doas', type=float, nargs='+', default=[15.0, -20.0],
                       help="True DOAs (degrees)")
    
    parser.add_argument('--output-dir', type=str, default='results/scenario4_array_comparison',
                       help="Output directory")
    parser.add_argument('--no-plots', action='store_true', help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    true_doas = np.array(args.doas)
    experiments = args.experiments
    if 'all' in experiments:
        experiments = ['baseline', 'coupling']
    
    print("\n" + "="*70)
    print("  SCENARIO 4: ARRAY COMPARISON UNDER REALISTIC CONDITIONS")
    print("="*70)
    print(f"  Experiments:      {', '.join(experiments)}")
    print(f"  Arrays:           {', '.join(args.arrays)}")
    print(f"  Trials:           {args.trials}")
    print(f"  SNR:              {args.snr} dB")
    print(f"  Snapshots:        {args.snapshots}")
    print(f"  True DOAs:        {true_doas}°")
    print(f"  Output:           {output_path}")
    print("="*70)
    
    baseline_df = None
    coupled_df = None
    
    # Experiment 4A: Baseline comparison
    if 'baseline' in experiments:
        baseline_df = run_array_comparison_baseline(
            array_types=args.arrays,
            snr_db=args.snr,
            snapshots=args.snapshots,
            true_doas=true_doas,
            num_trials=args.trials,
            verbose=True
        )
        
        # Save results
        csv_path = output_path / 'scenario4a_baseline_comparison.csv'
        baseline_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved baseline results: {csv_path}")
        
        # Print summary
        print(f"\n📊 Baseline Performance Ranking:")
        baseline_sorted = baseline_df.sort_values('RMSE_mean')
        for i, (_, row) in enumerate(baseline_sorted.iterrows(), 1):
            print(f"  {i}. {row['Array']:8s} - RMSE={row['RMSE_mean']:.4f}°, "
                  f"Improvement={row['Relative_Improvement_%']:+.1f}%, "
                  f"N_virt={int(row['N_Virtual'])}")
    
    # Experiment 4B: Comparison with coupling
    if 'coupling' in experiments:
        coupled_df = run_array_comparison_with_coupling(
            array_types=args.arrays,
            coupling_strength=args.coupling_strength,
            snr_db=args.snr,
            snapshots=args.snapshots,
            true_doas=true_doas,
            num_trials=args.trials,
            verbose=True
        )
        
        # Save results
        csv_path = output_path / 'scenario4b_coupling_comparison.csv'
        coupled_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved coupling results: {csv_path}")
        
        # Print summary
        print(f"\n📊 Coupling Performance Ranking:")
        coupled_sorted = coupled_df.sort_values('RMSE_mean')
        for i, (_, row) in enumerate(coupled_sorted.iterrows(), 1):
            print(f"  {i}. {row['Array']:8s} - RMSE={row['RMSE_mean']:.4f}°, "
                  f"Improvement={row['Relative_Improvement_%']:+.1f}%")
    
    # Combined analysis
    if baseline_df is not None and coupled_df is not None:
        # Ranking consistency
        ranking_metrics = compute_ranking_consistency(baseline_df, coupled_df)
        print(f"\n📊 Ranking Consistency Analysis:")
        print(f"  Kendall's τ:      {ranking_metrics['Kendall_Tau']:.3f}")
        print(f"  p-value:          {ranking_metrics['p_value']:.4f}")
        print(f"  Consistency:      {ranking_metrics['Ranking_Consistency_%']:.1f}%")
        print(f"  Baseline ranking: {ranking_metrics['Baseline_Ranking']}")
        print(f"  Coupled ranking:  {ranking_metrics['Coupled_Ranking']}")
        
        # Coupling resilience
        resilience_df = compute_coupling_resilience_gain(baseline_df, coupled_df)
        csv_path = output_path / 'scenario4_coupling_resilience.csv'
        resilience_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved resilience analysis: {csv_path}")
        
        print(f"\n📊 Coupling Resilience Gain:")
        resilience_sorted = resilience_df.sort_values('Coupling_Resilience_Gain_%', ascending=False)
        for _, row in resilience_sorted.iterrows():
            print(f"  {row['Array']:8s} - Degradation={row['Degradation_%']:+.1f}%, "
                  f"Resilience Gain={row['Coupling_Resilience_Gain_%']:+.1f}%")
        
        # Generate plots
        if not args.no_plots:
            plot_array_comparison(baseline_df, coupled_df, resilience_df, ranking_metrics, output_path)
    
    print("\n" + "="*70)
    print("  SCENARIO 4 COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
