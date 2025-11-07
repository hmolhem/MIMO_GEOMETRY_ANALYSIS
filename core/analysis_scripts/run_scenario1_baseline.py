"""
SCENARIO 1: Baseline Performance (No Coupling)

Purpose: Establish performance baseline and validate implementation

This script runs comprehensive baseline experiments to characterize DOA estimation
performance under ideal conditions (no mutual coupling) across varying:
- SNR levels (-5 to 15 dB)
- Snapshot counts (32 to 512)
- Array geometries (ULA, Z5, Z6)

Primary Metrics:
- RMSE_degrees: Core performance metric
- RMSE_CRB_ratio: Efficiency vs theoretical limit
- Resolution_Rate: % of correctly resolved sources
- Bias_degrees: Systematic error component
- Runtime_ms: Computational efficiency

Author: MIMO Geometry Analysis Team
Date: November 6, 2025
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.radarpy.signal.doa_sim_core import simulate_snapshots, music_spectrum, find_k_peaks
from core.radarpy.signal.metrics import compute_scenario_metrics, print_metrics_summary


def get_array_positions(array_type: str, d: float = 0.5) -> np.ndarray:
    """Get sensor positions for specified array type."""
    if array_type.upper() == 'ULA':
        return np.arange(7) * d
    elif array_type.upper() == 'Z5':
        return np.array([0, 5, 8, 11, 14, 17, 21]) * d
    elif array_type.upper() == 'Z6':
        return np.array([0, 1, 4, 8, 13, 17, 22]) * d
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def run_snr_sweep(array_type: str = 'Z5',
                  snr_range: np.ndarray = np.arange(-5, 16, 5),
                  snapshots: int = 256,
                  num_trials: int = 500,
                  true_doas: np.ndarray = np.array([15.0, -20.0]),
                  wavelength: float = 1.0,
                  resolution_threshold: float = 3.0,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Run RMSE vs SNR sweep analysis.
    
    Args:
        array_type: Array geometry type
        snr_range: Array of SNR values in dB
        snapshots: Number of snapshots (M)
        num_trials: Number of Monte Carlo trials per SNR
        true_doas: True DOA angles in degrees
        wavelength: Signal wavelength
        resolution_threshold: Threshold for resolution detection
        verbose: Whether to print progress
    
    Returns:
        results_df: DataFrame with metrics for each SNR value
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SCENARIO 1A: RMSE vs SNR Sweep")
        print(f"{'='*70}")
        print(f"  Array:            {array_type}")
        print(f"  SNR Range:        {snr_range[0]} to {snr_range[-1]} dB")
        print(f"  Snapshots:        {snapshots}")
        print(f"  Trials per SNR:   {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print(f"{'='*70}\n")
    
    positions = get_array_positions(array_type)
    N = len(positions)
    K = len(true_doas)
    scan_grid = np.linspace(-90, 90, 1801)
    
    results = []
    
    for snr_db in snr_range:
        if verbose:
            print(f"Running SNR = {snr_db} dB... ", end='', flush=True)
        
        start_time = time.time()
        estimated_trials = []
        runtimes = []
        
        for trial in range(num_trials):
            # Generate snapshots
            X, A, snr_lin = simulate_snapshots(
                sensor_positions=positions,
                wavelength=wavelength,
                doas_deg=true_doas,
                snr_db=snr_db,
                snapshots=snapshots,
                coupling_matrix=None  # No coupling for baseline
            )
            
            # Compute covariance
            Rxx = (X @ X.conj().T) / snapshots
            
            # Run MUSIC with timing
            trial_start = time.perf_counter()
            P_db, _ = music_spectrum(
                Rxx=Rxx,
                sensor_positions=positions,
                wavelength=wavelength,
                scan_deg=scan_grid,
                k_sources=K,
                coupling_matrix=None
            )
            est_doas = find_k_peaks(P_db, scan_grid, K)
            trial_runtime = (time.perf_counter() - trial_start) * 1000
            
            estimated_trials.append(est_doas)
            runtimes.append(trial_runtime)
        
        # Compute metrics
        metrics = compute_scenario_metrics(
            estimated_doas_trials=estimated_trials,
            true_doas=true_doas,
            sensor_positions=positions,
            wavelength=wavelength,
            snr_db=snr_db,
            snapshots=snapshots,
            coupling_matrix=None,
            resolution_threshold=resolution_threshold,
            runtimes_ms=runtimes
        )
        
        # Store results
        result = {
            'SNR_dB': snr_db,
            'Array': array_type,
            'Snapshots': snapshots,
            **metrics
        }
        results.append(result)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Done! ({elapsed:.1f}s) RMSE={metrics['RMSE_degrees']:.3f}°, Res={metrics['Resolution_Rate']:.1f}%")
    
    results_df = pd.DataFrame(results)
    return results_df


def run_snapshots_sweep(array_type: str = 'Z5',
                       snr_db: float = 10.0,
                       snapshot_range: np.ndarray = np.array([32, 64, 128, 256, 512]),
                       num_trials: int = 500,
                       true_doas: np.ndarray = np.array([15.0, -20.0]),
                       wavelength: float = 1.0,
                       resolution_threshold: float = 3.0,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Run RMSE vs Snapshots sweep analysis.
    
    Args:
        array_type: Array geometry type
        snr_db: SNR in dB (fixed)
        snapshot_range: Array of snapshot counts
        num_trials: Number of Monte Carlo trials per snapshot count
        true_doas: True DOA angles in degrees
        wavelength: Signal wavelength
        resolution_threshold: Threshold for resolution detection
        verbose: Whether to print progress
    
    Returns:
        results_df: DataFrame with metrics for each snapshot count
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SCENARIO 1B: RMSE vs Snapshots Sweep")
        print(f"{'='*70}")
        print(f"  Array:            {array_type}")
        print(f"  SNR:              {snr_db} dB")
        print(f"  Snapshot Range:   {snapshot_range[0]} to {snapshot_range[-1]}")
        print(f"  Trials per M:     {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print(f"{'='*70}\n")
    
    positions = get_array_positions(array_type)
    N = len(positions)
    K = len(true_doas)
    scan_grid = np.linspace(-90, 90, 1801)
    
    results = []
    
    for snapshots in snapshot_range:
        if verbose:
            print(f"Running M = {snapshots} snapshots... ", end='', flush=True)
        
        start_time = time.time()
        estimated_trials = []
        runtimes = []
        
        for trial in range(num_trials):
            # Generate snapshots
            X, A, snr_lin = simulate_snapshots(
                sensor_positions=positions,
                wavelength=wavelength,
                doas_deg=true_doas,
                snr_db=snr_db,
                snapshots=snapshots,
                coupling_matrix=None
            )
            
            # Compute covariance
            Rxx = (X @ X.conj().T) / snapshots
            
            # Run MUSIC with timing
            trial_start = time.perf_counter()
            P_db, _ = music_spectrum(
                Rxx=Rxx,
                sensor_positions=positions,
                wavelength=wavelength,
                scan_deg=scan_grid,
                k_sources=K,
                coupling_matrix=None
            )
            est_doas = find_k_peaks(P_db, scan_grid, K)
            trial_runtime = (time.perf_counter() - trial_start) * 1000
            
            estimated_trials.append(est_doas)
            runtimes.append(trial_runtime)
        
        # Compute metrics
        metrics = compute_scenario_metrics(
            estimated_doas_trials=estimated_trials,
            true_doas=true_doas,
            sensor_positions=positions,
            wavelength=wavelength,
            snr_db=snr_db,
            snapshots=snapshots,
            coupling_matrix=None,
            resolution_threshold=resolution_threshold,
            runtimes_ms=runtimes
        )
        
        # Store results
        result = {
            'Snapshots': snapshots,
            'SNR_dB': snr_db,
            'Array': array_type,
            **metrics
        }
        results.append(result)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Done! ({elapsed:.1f}s) RMSE={metrics['RMSE_degrees']:.3f}°, Res={metrics['Resolution_Rate']:.1f}%")
    
    results_df = pd.DataFrame(results)
    return results_df


def run_array_comparison(snr_db: float = 10.0,
                        snapshots: int = 256,
                        num_trials: int = 500,
                        true_doas: np.ndarray = np.array([15.0, -20.0]),
                        wavelength: float = 1.0,
                        resolution_threshold: float = 3.0,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Compare performance across different array geometries.
    
    Args:
        snr_db: SNR in dB
        snapshots: Number of snapshots
        num_trials: Number of Monte Carlo trials
        true_doas: True DOA angles in degrees
        wavelength: Signal wavelength
        resolution_threshold: Threshold for resolution detection
        verbose: Whether to print progress
    
    Returns:
        results_df: DataFrame with metrics for each array type
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SCENARIO 1C: Array Geometry Comparison")
        print(f"{'='*70}")
        print(f"  SNR:              {snr_db} dB")
        print(f"  Snapshots:        {snapshots}")
        print(f"  Trials per array: {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print(f"{'='*70}\n")
    
    array_types = ['ULA', 'Z5', 'Z6']
    scan_grid = np.linspace(-90, 90, 1801)
    K = len(true_doas)
    
    results = []
    
    for array_type in array_types:
        if verbose:
            print(f"Running {array_type} array... ", end='', flush=True)
        
        start_time = time.time()
        positions = get_array_positions(array_type)
        N = len(positions)
        
        estimated_trials = []
        runtimes = []
        
        for trial in range(num_trials):
            # Generate snapshots
            X, A, snr_lin = simulate_snapshots(
                sensor_positions=positions,
                wavelength=wavelength,
                doas_deg=true_doas,
                snr_db=snr_db,
                snapshots=snapshots,
                coupling_matrix=None
            )
            
            # Compute covariance
            Rxx = (X @ X.conj().T) / snapshots
            
            # Run MUSIC with timing
            trial_start = time.perf_counter()
            P_db, _ = music_spectrum(
                Rxx=Rxx,
                sensor_positions=positions,
                wavelength=wavelength,
                scan_deg=scan_grid,
                k_sources=K,
                coupling_matrix=None
            )
            est_doas = find_k_peaks(P_db, scan_grid, K)
            trial_runtime = (time.perf_counter() - trial_start) * 1000
            
            estimated_trials.append(est_doas)
            runtimes.append(trial_runtime)
        
        # Compute metrics
        metrics = compute_scenario_metrics(
            estimated_doas_trials=estimated_trials,
            true_doas=true_doas,
            sensor_positions=positions,
            wavelength=wavelength,
            snr_db=snr_db,
            snapshots=snapshots,
            coupling_matrix=None,
            resolution_threshold=resolution_threshold,
            runtimes_ms=runtimes
        )
        
        # Store results
        result = {
            'Array': array_type,
            'N_Sensors': N,
            'SNR_dB': snr_db,
            'Snapshots': snapshots,
            **metrics
        }
        results.append(result)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Done! ({elapsed:.1f}s) RMSE={metrics['RMSE_degrees']:.3f}°, CRB={metrics['RMSE_CRB_ratio']:.2f}x")
    
    results_df = pd.DataFrame(results)
    return results_df


def plot_snr_results(results_df: pd.DataFrame, output_path: Path):
    """Generate plots for SNR sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Scenario 1A: Baseline Performance vs SNR', fontsize=14, fontweight='bold')
    
    snr_values = results_df['SNR_dB'].values
    
    # Plot 1: RMSE vs SNR
    ax = axes[0, 0]
    ax.plot(snr_values, results_df['RMSE_degrees'], 'o-', linewidth=2, markersize=8, label='RMSE')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('RMSE (degrees)', fontsize=11)
    ax.set_title('RMSE vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: RMSE/CRB Ratio vs SNR
    ax = axes[0, 1]
    ax.plot(snr_values, results_df['RMSE_CRB_ratio'], 's-', linewidth=2, markersize=8, color='orange', label='RMSE/CRB')
    ax.axhline(y=1.0, color='r', linestyle='--', label='CRB (optimal)')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('RMSE/CRB Ratio', fontsize=11)
    ax.set_title('Efficiency vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Resolution Rate vs SNR
    ax = axes[1, 0]
    ax.plot(snr_values, results_df['Resolution_Rate'], '^-', linewidth=2, markersize=8, color='green', label='Resolution %')
    ax.axhline(y=95, color='r', linestyle='--', label='95% threshold')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Resolution Rate (%)', fontsize=11)
    ax.set_title('Resolution Rate vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])
    
    # Plot 4: Bias vs SNR
    ax = axes[1, 1]
    ax.plot(snr_values, results_df['Bias_degrees'], 'd-', linewidth=2, markersize=8, color='purple', label='Bias')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Bias (degrees)', fontsize=11)
    ax.set_title('Bias vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'scenario1a_snr_sweep.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'scenario1a_snr_sweep.png'}")
    plt.close()


def plot_snapshots_results(results_df: pd.DataFrame, output_path: Path):
    """Generate plots for snapshots sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Scenario 1B: Baseline Performance vs Snapshots', fontsize=14, fontweight='bold')
    
    snapshots = results_df['Snapshots'].values
    
    # Plot 1: RMSE vs Snapshots
    ax = axes[0, 0]
    ax.plot(snapshots, results_df['RMSE_degrees'], 'o-', linewidth=2, markersize=8, label='RMSE')
    ax.set_xlabel('Number of Snapshots (M)', fontsize=11)
    ax.set_ylabel('RMSE (degrees)', fontsize=11)
    ax.set_title('RMSE vs Snapshots', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Plot 2: RMSE/CRB Ratio vs Snapshots
    ax = axes[0, 1]
    ax.plot(snapshots, results_df['RMSE_CRB_ratio'], 's-', linewidth=2, markersize=8, color='orange', label='RMSE/CRB')
    ax.axhline(y=1.0, color='r', linestyle='--', label='CRB (optimal)')
    ax.set_xlabel('Number of Snapshots (M)', fontsize=11)
    ax.set_ylabel('RMSE/CRB Ratio', fontsize=11)
    ax.set_title('Efficiency vs Snapshots', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Plot 3: Resolution Rate vs Snapshots
    ax = axes[1, 0]
    ax.plot(snapshots, results_df['Resolution_Rate'], '^-', linewidth=2, markersize=8, color='green', label='Resolution %')
    ax.axhline(y=95, color='r', linestyle='--', label='95% threshold')
    ax.set_xlabel('Number of Snapshots (M)', fontsize=11)
    ax.set_ylabel('Resolution Rate (%)', fontsize=11)
    ax.set_title('Resolution Rate vs Snapshots', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_ylim([0, 105])
    
    # Plot 4: Runtime vs Snapshots
    ax = axes[1, 1]
    ax.plot(snapshots, results_df['Runtime_ms'], 'd-', linewidth=2, markersize=8, color='purple', label='Runtime')
    ax.set_xlabel('Number of Snapshots (M)', fontsize=11)
    ax.set_ylabel('Runtime (ms)', fontsize=11)
    ax.set_title('Runtime vs Snapshots', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'scenario1b_snapshots_sweep.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'scenario1b_snapshots_sweep.png'}")
    plt.close()


def plot_array_comparison(results_df: pd.DataFrame, output_path: Path):
    """Generate plots for array comparison results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Scenario 1C: Array Geometry Comparison', fontsize=14, fontweight='bold')
    
    arrays = results_df['Array'].values
    x_pos = np.arange(len(arrays))
    
    # Plot 1: RMSE Comparison
    ax = axes[0]
    bars = ax.bar(x_pos, results_df['RMSE_degrees'], color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Array Type', fontsize=11)
    ax.set_ylabel('RMSE (degrees)', fontsize=11)
    ax.set_title('RMSE Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arrays)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}°', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: RMSE/CRB Ratio Comparison
    ax = axes[1]
    bars = ax.bar(x_pos, results_df['RMSE_CRB_ratio'], color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='r', linestyle='--', label='CRB (optimal)', linewidth=2)
    ax.set_xlabel('Array Type', fontsize=11)
    ax.set_ylabel('RMSE/CRB Ratio', fontsize=11)
    ax.set_title('Efficiency Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arrays)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Resolution Rate Comparison
    ax = axes[2]
    bars = ax.bar(x_pos, results_df['Resolution_Rate'], color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
    ax.axhline(y=95, color='r', linestyle='--', label='95% threshold', linewidth=2)
    ax.set_xlabel('Array Type', fontsize=11)
    ax.set_ylabel('Resolution Rate (%)', fontsize=11)
    ax.set_title('Resolution Rate Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arrays)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    ax.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scenario1c_array_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'scenario1c_array_comparison.png'}")
    plt.close()


def main():
    """Main entry point for Scenario 1 baseline experiments."""
    parser = argparse.ArgumentParser(
        description="SCENARIO 1: Baseline Performance (No Coupling)"
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiments', type=str, nargs='+',
        choices=['snr', 'snapshots', 'arrays', 'all'],
        default=['all'],
        help="Which experiments to run"
    )
    
    # Array configuration
    parser.add_argument('--array', type=str, default='Z5', choices=['ULA', 'Z5', 'Z6'],
                       help="Array type (for SNR and snapshots sweeps)")
    
    # SNR sweep parameters
    parser.add_argument('--snr-min', type=float, default=-5.0, help="Minimum SNR (dB)")
    parser.add_argument('--snr-max', type=float, default=15.0, help="Maximum SNR (dB)")
    parser.add_argument('--snr-step', type=float, default=5.0, help="SNR step size (dB)")
    
    # Snapshots sweep parameters
    parser.add_argument('--snapshots-values', type=int, nargs='+',
                       default=[32, 64, 128, 256, 512],
                       help="Snapshot values to test")
    parser.add_argument('--fixed-snr', type=float, default=10.0,
                       help="Fixed SNR for snapshots sweep (dB)")
    
    # Common parameters
    parser.add_argument('--trials', type=int, default=500, help="Monte Carlo trials")
    parser.add_argument('--doas', type=float, nargs='+', default=[15.0, -20.0],
                       help="True DOA angles")
    parser.add_argument('--resolution-threshold', type=float, default=3.0,
                       help="Resolution threshold (degrees)")
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results/scenario1_baseline',
                       help="Output directory")
    parser.add_argument('--no-plots', action='store_true', help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    true_doas = np.array(args.doas)
    experiments = args.experiments
    if 'all' in experiments:
        experiments = ['snr', 'snapshots', 'arrays']
    
    print("\n" + "="*70)
    print("  SCENARIO 1: BASELINE PERFORMANCE (NO COUPLING)")
    print("="*70)
    print(f"  Experiments:      {', '.join(experiments)}")
    print(f"  Trials:           {args.trials}")
    print(f"  True DOAs:        {true_doas}°")
    print(f"  Output:           {output_path}")
    print("="*70)
    
    # Experiment 1A: SNR Sweep
    if 'snr' in experiments:
        snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)
        results_snr = run_snr_sweep(
            array_type=args.array,
            snr_range=snr_range,
            snapshots=256,
            num_trials=args.trials,
            true_doas=true_doas,
            resolution_threshold=args.resolution_threshold
        )
        
        # Save results
        csv_path = output_path / f'scenario1a_snr_sweep_{args.array}.csv'
        results_snr.to_csv(csv_path, index=False)
        print(f"\n✓ Saved SNR sweep results: {csv_path}")
        
        # Generate plots
        if not args.no_plots:
            plot_snr_results(results_snr, output_path)
    
    # Experiment 1B: Snapshots Sweep
    if 'snapshots' in experiments:
        snapshot_range = np.array(args.snapshots_values)
        results_snapshots = run_snapshots_sweep(
            array_type=args.array,
            snr_db=args.fixed_snr,
            snapshot_range=snapshot_range,
            num_trials=args.trials,
            true_doas=true_doas,
            resolution_threshold=args.resolution_threshold
        )
        
        # Save results
        csv_path = output_path / f'scenario1b_snapshots_sweep_{args.array}.csv'
        results_snapshots.to_csv(csv_path, index=False)
        print(f"\n✓ Saved snapshots sweep results: {csv_path}")
        
        # Generate plots
        if not args.no_plots:
            plot_snapshots_results(results_snapshots, output_path)
    
    # Experiment 1C: Array Comparison
    if 'arrays' in experiments:
        results_arrays = run_array_comparison(
            snr_db=args.fixed_snr,
            snapshots=256,
            num_trials=args.trials,
            true_doas=true_doas,
            resolution_threshold=args.resolution_threshold
        )
        
        # Save results
        csv_path = output_path / 'scenario1c_array_comparison.csv'
        results_arrays.to_csv(csv_path, index=False)
        print(f"\n✓ Saved array comparison results: {csv_path}")
        
        # Generate plots
        if not args.no_plots:
            plot_array_comparison(results_arrays, output_path)
    
    print("\n" + "="*70)
    print("  SCENARIO 1 COMPLETE!")
    print("="*70)
    print(f"  Results saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
