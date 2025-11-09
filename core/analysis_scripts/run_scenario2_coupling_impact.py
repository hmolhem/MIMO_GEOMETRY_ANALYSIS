"""
SCENARIO 2: Mutual Coupling Impact Study (ALSS Work)

Purpose: Quantify coupling degradation and identify sensitive regimes

This script analyzes DOA estimation performance under varying mutual coupling 
conditions to understand:
- Performance degradation patterns
- Critical coupling thresholds
- Array-specific sensitivities
- Failure modes and recovery

Primary Metrics:
- RMSE_Degradation_%: % increase from baseline
- Coupling_Sensitivity: d(RMSE)/d(coupling_strength)
- Failure_Threshold: Coupling level where performance collapses
- Resolution_Loss: How many sources become unresolvable
- CRB_Violation: Does coupling push you further from CRB?

Author: MIMO Geometry Analysis Team (ALSS Paper)
Date: November 6, 2025
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.radarpy.signal.doa_sim_core import simulate_snapshots, music_spectrum, find_k_peaks
from core.radarpy.signal.mutual_coupling import generate_mcm
from core.radarpy.signal.metrics import compute_scenario_metrics


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


def run_coupling_strength_sweep(
        array_type: str = 'Z5',
        coupling_range: np.ndarray = np.linspace(0.0, 0.5, 11),
        snr_db: float = 10.0,
        snapshots: int = 256,
        num_trials: int = 500,
        true_doas: np.ndarray = np.array([15.0, -20.0]),
        wavelength: float = 1.0,
        resolution_threshold: float = 3.0,
        coupling_model: str = "exponential",
        baseline_metrics: Optional[Dict] = None,
        verbose: bool = True) -> pd.DataFrame:
    """
    Run performance sweep across varying coupling strengths.
    
    Args:
        array_type: Array geometry type
        coupling_range: Array of coupling coefficients (c1 values)
        snr_db: SNR in dB (fixed)
        snapshots: Number of snapshots (M)
        num_trials: Number of Monte Carlo trials per coupling level
        true_doas: True DOA angles in degrees
        wavelength: Signal wavelength
        resolution_threshold: Threshold for resolution detection
        coupling_model: MCM model ("exponential", "polynomial", "uniform")
        baseline_metrics: Scenario 1 baseline metrics for comparison
        verbose: Whether to print progress
    
    Returns:
        results_df: DataFrame with metrics for each coupling strength
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SCENARIO 2A: Performance vs Coupling Strength")
        print(f"{'='*70}")
        print(f"  Array:            {array_type}")
        print(f"  Coupling Range:   {coupling_range[0]:.2f} to {coupling_range[-1]:.2f}")
        print(f"  SNR:              {snr_db} dB")
        print(f"  Snapshots:        {snapshots}")
        print(f"  Trials per c1:    {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print(f"  MCM Model:        {coupling_model}")
        print(f"{'='*70}\n")
    
    positions = get_array_positions(array_type)
    N = len(positions)
    K = len(true_doas)
    scan_grid = np.linspace(-90, 90, 1801)
    
    # Get baseline RMSE if provided
    baseline_rmse = baseline_metrics['RMSE_degrees'] if baseline_metrics else None
    baseline_crb_ratio = baseline_metrics['RMSE_CRB_ratio'] if baseline_metrics else None
    
    results = []
    
    for c1 in coupling_range:
        if verbose:
            print(f"Running c1 = {c1:.3f}... ", end='', flush=True)
        
        start_time = time.time()
        estimated_trials = []
        runtimes = []
        
        # Generate MCM once for this coupling level
        if c1 > 0:
            C = generate_mcm(N, positions, model=coupling_model, c1=c1)
        else:
            C = None  # No coupling (baseline)
        
        for trial in range(num_trials):
            # Generate snapshots with coupling
            X, A, snr_lin = simulate_snapshots(
                sensor_positions=positions,
                wavelength=wavelength,
                doas_deg=true_doas,
                snr_db=snr_db,
                snapshots=snapshots,
                coupling_matrix=C
            )
            
            # Compute covariance
            Rxx = (X @ X.conj().T) / snapshots
            
            # Run MUSIC with timing (with coupling correction)
            trial_start = time.perf_counter()
            P_db, _ = music_spectrum(
                Rxx=Rxx,
                sensor_positions=positions,
                wavelength=wavelength,
                scan_deg=scan_grid,
                k_sources=K,
                coupling_matrix=C  # Use coupling in MUSIC
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
            coupling_matrix=C,
            resolution_threshold=resolution_threshold,
            runtimes_ms=runtimes
        )
        
        # Compute degradation metrics
        if baseline_rmse is not None:
            rmse_degradation_pct = ((metrics['RMSE_degrees'] - baseline_rmse) / baseline_rmse) * 100
            metrics['RMSE_Degradation_%'] = rmse_degradation_pct
        else:
            metrics['RMSE_Degradation_%'] = 0.0
        
        if baseline_crb_ratio is not None:
            crb_violation = metrics['RMSE_CRB_ratio'] - baseline_crb_ratio
            metrics['CRB_Violation'] = crb_violation
        else:
            metrics['CRB_Violation'] = 0.0
        
        # Store results
        result = {
            'Coupling_c1': c1,
            'Array': array_type,
            'SNR_dB': snr_db,
            'Snapshots': snapshots,
            **metrics
        }
        results.append(result)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Done! ({elapsed:.1f}s) RMSE={metrics['RMSE_degrees']:.3f}°, "
                  f"Deg={metrics['RMSE_Degradation_%']:+.1f}%, Res={metrics['Resolution_Rate']:.1f}%")
    
    results_df = pd.DataFrame(results)
    
    # Compute coupling sensitivity (derivative)
    if len(results_df) > 1:
        results_df['Coupling_Sensitivity'] = np.gradient(
            results_df['RMSE_degrees'].values, 
            results_df['Coupling_c1'].values
        )
    else:
        results_df['Coupling_Sensitivity'] = 0.0
    
    return results_df


def identify_failure_threshold(results_df: pd.DataFrame, 
                               degradation_threshold: float = 100.0) -> Dict:
    """
    Identify the coupling strength where performance collapses.
    
    Args:
        results_df: Results from coupling sweep
        degradation_threshold: % RMSE degradation considered "failure"
    
    Returns:
        threshold_info: Dict with failure threshold and related metrics
    """
    # Find where degradation exceeds threshold
    failures = results_df[results_df['RMSE_Degradation_%'] >= degradation_threshold]
    
    if len(failures) > 0:
        failure_c1 = failures.iloc[0]['Coupling_c1']
        failure_rmse = failures.iloc[0]['RMSE_degrees']
        failure_resolution = failures.iloc[0]['Resolution_Rate']
    else:
        failure_c1 = None
        failure_rmse = None
        failure_resolution = None
    
    # Find maximum sensitivity point
    max_sens_idx = results_df['Coupling_Sensitivity'].abs().idxmax()
    max_sens_c1 = results_df.loc[max_sens_idx, 'Coupling_c1']
    max_sens_value = results_df.loc[max_sens_idx, 'Coupling_Sensitivity']
    
    return {
        'failure_threshold_c1': failure_c1,
        'failure_rmse': failure_rmse,
        'failure_resolution': failure_resolution,
        'max_sensitivity_c1': max_sens_c1,
        'max_sensitivity_value': max_sens_value
    }


def run_array_sensitivity_comparison(
        coupling_strength: float = 0.3,
        snr_db: float = 10.0,
        snapshots: int = 256,
        num_trials: int = 500,
        true_doas: np.ndarray = np.array([15.0, -20.0]),
        wavelength: float = 1.0,
        resolution_threshold: float = 3.0,
        baseline_metrics_dict: Optional[Dict] = None,
        verbose: bool = True) -> pd.DataFrame:
    """
    Compare coupling sensitivity across different array geometries.
    
    Args:
        coupling_strength: Fixed coupling coefficient
        snr_db: SNR in dB
        snapshots: Number of snapshots
        num_trials: Number of Monte Carlo trials
        true_doas: True DOA angles
        wavelength: Signal wavelength
        resolution_threshold: Resolution threshold
        baseline_metrics_dict: Dict mapping array types to baseline metrics
        verbose: Progress printing
    
    Returns:
        results_df: Comparison across arrays
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SCENARIO 2B: Array Sensitivity Comparison")
        print(f"{'='*70}")
        print(f"  Coupling:         c1 = {coupling_strength}")
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
        
        # Generate MCM
        C = generate_mcm(N, positions, model="exponential", c1=coupling_strength)
        
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
                coupling_matrix=C
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
                coupling_matrix=C
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
            coupling_matrix=C,
            resolution_threshold=resolution_threshold,
            runtimes_ms=runtimes
        )
        
        # Compute degradation from baseline
        if baseline_metrics_dict and array_type in baseline_metrics_dict:
            baseline = baseline_metrics_dict[array_type]
            metrics['RMSE_Degradation_%'] = (
                (metrics['RMSE_degrees'] - baseline['RMSE_degrees']) / 
                baseline['RMSE_degrees'] * 100
            )
            metrics['Resolution_Loss'] = baseline['Resolution_Rate'] - metrics['Resolution_Rate']
            metrics['CRB_Violation'] = metrics['RMSE_CRB_ratio'] - baseline['RMSE_CRB_ratio']
        else:
            metrics['RMSE_Degradation_%'] = 0.0
            metrics['Resolution_Loss'] = 0.0
            metrics['CRB_Violation'] = 0.0
        
        # Store results
        result = {
            'Array': array_type,
            'N_Sensors': N,
            'Coupling_c1': coupling_strength,
            'SNR_dB': snr_db,
            'Snapshots': snapshots,
            **metrics
        }
        results.append(result)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Done! ({elapsed:.1f}s) RMSE={metrics['RMSE_degrees']:.3f}°, "
                  f"Deg={metrics['RMSE_Degradation_%']:+.1f}%")
    
    results_df = pd.DataFrame(results)
    return results_df


def plot_coupling_sweep_results(results_df: pd.DataFrame, 
                                output_path: Path,
                                array_type: str):
    """Generate plots for coupling strength sweep."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Scenario 2A: Coupling Impact Analysis ({array_type} Array)', 
                 fontsize=14, fontweight='bold')
    
    c1_values = results_df['Coupling_c1'].values
    
    # Plot 1: RMSE vs Coupling
    ax = axes[0, 0]
    ax.plot(c1_values, results_df['RMSE_degrees'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Coefficient (c₁)', fontsize=11)
    ax.set_ylabel('RMSE (degrees)', fontsize=11)
    ax.set_title('RMSE vs Coupling Strength', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RMSE Degradation %
    ax = axes[0, 1]
    ax.plot(c1_values, results_df['RMSE_Degradation_%'], 's-', 
            linewidth=2, markersize=8, color='orange')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=100, color='r', linestyle='--', label='100% degradation', linewidth=2)
    ax.set_xlabel('Coupling Coefficient (c₁)', fontsize=11)
    ax.set_ylabel('RMSE Degradation (%)', fontsize=11)
    ax.set_title('Performance Degradation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Resolution Rate vs Coupling
    ax = axes[0, 2]
    ax.plot(c1_values, results_df['Resolution_Rate'], '^-', 
            linewidth=2, markersize=8, color='green')
    ax.axhline(y=95, color='r', linestyle='--', label='95% threshold', linewidth=2)
    ax.set_xlabel('Coupling Coefficient (c₁)', fontsize=11)
    ax.set_ylabel('Resolution Rate (%)', fontsize=11)
    ax.set_title('Resolution vs Coupling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])
    
    # Plot 4: CRB Ratio vs Coupling
    ax = axes[1, 0]
    ax.plot(c1_values, results_df['RMSE_CRB_ratio'], 'd-', 
            linewidth=2, markersize=8, color='purple')
    ax.axhline(y=1.0, color='r', linestyle='--', label='CRB (optimal)', linewidth=2)
    ax.set_xlabel('Coupling Coefficient (c₁)', fontsize=11)
    ax.set_ylabel('RMSE/CRB Ratio', fontsize=11)
    ax.set_title('Efficiency vs Coupling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Coupling Sensitivity
    ax = axes[1, 1]
    ax.plot(c1_values, results_df['Coupling_Sensitivity'], 'v-', 
            linewidth=2, markersize=8, color='red')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Coupling Coefficient (c₁)', fontsize=11)
    ax.set_ylabel('d(RMSE)/d(c₁)', fontsize=11)
    ax.set_title('Coupling Sensitivity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Runtime vs Coupling
    ax = axes[1, 2]
    ax.plot(c1_values, results_df['Runtime_ms'], 'h-', 
            linewidth=2, markersize=8, color='brown')
    ax.set_xlabel('Coupling Coefficient (c₁)', fontsize=11)
    ax.set_ylabel('Runtime (ms)', fontsize=11)
    ax.set_title('Computational Cost', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'scenario2a_coupling_sweep_{array_type}.png', 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / f'scenario2a_coupling_sweep_{array_type}.png'}")
    plt.close()


def plot_array_sensitivity_comparison(results_df: pd.DataFrame, output_path: Path):
    """Generate plots for array sensitivity comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Scenario 2B: Array Sensitivity to Coupling', 
                 fontsize=14, fontweight='bold')
    
    arrays = results_df['Array'].values
    x_pos = np.arange(len(arrays))
    
    # Plot 1: RMSE Degradation Comparison
    ax = axes[0]
    bars = ax.bar(x_pos, results_df['RMSE_Degradation_%'], 
                  color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Array Type', fontsize=11)
    ax.set_ylabel('RMSE Degradation (%)', fontsize=11)
    ax.set_title('Degradation Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arrays)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Resolution Loss Comparison
    ax = axes[1]
    bars = ax.bar(x_pos, results_df['Resolution_Loss'], 
                  color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Array Type', fontsize=11)
    ax.set_ylabel('Resolution Loss (%)', fontsize=11)
    ax.set_title('Resolution Degradation', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arrays)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: CRB Violation Comparison
    ax = axes[2]
    bars = ax.bar(x_pos, results_df['CRB_Violation'], 
                  color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Array Type', fontsize=11)
    ax.set_ylabel('CRB Ratio Increase', fontsize=11)
    ax.set_title('Efficiency Loss', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arrays)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}x', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scenario2b_array_sensitivity.png', 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'scenario2b_array_sensitivity.png'}")
    plt.close()


def main():
    """Main entry point for Scenario 2 coupling impact experiments."""
    parser = argparse.ArgumentParser(
        description="SCENARIO 2: Mutual Coupling Impact Study (ALSS Work)"
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiments', type=str, nargs='+',
        choices=['coupling-sweep', 'array-comparison', 'all'],
        default=['all'],
        help="Which experiments to run"
    )
    
    # Array configuration
    parser.add_argument('--array', type=str, default='Z5', choices=['ULA', 'Z5', 'Z6'],
                       help="Array type (for coupling sweep)")
    
    # Coupling sweep parameters
    parser.add_argument('--coupling-min', type=float, default=0.0, 
                       help="Minimum coupling coefficient")
    parser.add_argument('--coupling-max', type=float, default=0.5, 
                       help="Maximum coupling coefficient")
    parser.add_argument('--coupling-points', type=int, default=11, 
                       help="Number of coupling levels to test")
    parser.add_argument('--fixed-coupling', type=float, default=0.3,
                       help="Fixed coupling for array comparison")
    
    # Common parameters
    parser.add_argument('--snr', type=float, default=10.0, help="SNR (dB)")
    parser.add_argument('--snapshots', type=int, default=256, help="Number of snapshots")
    parser.add_argument('--trials', type=int, default=500, help="Monte Carlo trials")
    parser.add_argument('--doas', type=float, nargs='+', default=[15.0, -20.0],
                       help="True DOA angles")
    parser.add_argument('--resolution-threshold', type=float, default=3.0,
                       help="Resolution threshold (degrees)")
    parser.add_argument('--coupling-model', type=str, default='exponential',
                       choices=['exponential', 'polynomial', 'uniform'],
                       help="MCM model")
    
    # Baseline comparison
    parser.add_argument('--baseline-csv', type=str, 
                       default='results/scenario1_baseline/scenario1a_snr_sweep_Z5.csv',
                       help="Path to Scenario 1 baseline CSV")
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results/scenario2_coupling',
                       help="Output directory")
    parser.add_argument('--no-plots', action='store_true', help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    true_doas = np.array(args.doas)
    experiments = args.experiments
    if 'all' in experiments:
        experiments = ['coupling-sweep', 'array-comparison']
    
    # Load baseline metrics if available
    baseline_metrics = None
    baseline_metrics_dict = {}
    if os.path.exists(args.baseline_csv):
        baseline_df = pd.read_csv(args.baseline_csv)
        # Get metrics at SNR = args.snr
        baseline_row = baseline_df[baseline_df['SNR_dB'] == args.snr]
        if len(baseline_row) > 0:
            baseline_metrics = baseline_row.iloc[0].to_dict()
            print(f"\n✓ Loaded baseline metrics from Scenario 1 (SNR={args.snr}dB)")
            print(f"  Baseline RMSE: {baseline_metrics['RMSE_degrees']:.4f}°")
    
    print("\n" + "="*70)
    print("  SCENARIO 2: MUTUAL COUPLING IMPACT STUDY (ALSS)")
    print("="*70)
    print(f"  Experiments:      {', '.join(experiments)}")
    print(f"  Trials:           {args.trials}")
    print(f"  True DOAs:        {true_doas}°")
    print(f"  MCM Model:        {args.coupling_model}")
    print(f"  Output:           {output_path}")
    print("="*70)
    
    # Experiment 2A: Coupling Strength Sweep
    if 'coupling-sweep' in experiments:
        coupling_range = np.linspace(args.coupling_min, args.coupling_max, args.coupling_points)
        results_coupling = run_coupling_strength_sweep(
            array_type=args.array,
            coupling_range=coupling_range,
            snr_db=args.snr,
            snapshots=args.snapshots,
            num_trials=args.trials,
            true_doas=true_doas,
            resolution_threshold=args.resolution_threshold,
            coupling_model=args.coupling_model,
            baseline_metrics=baseline_metrics
        )
        
        # Save results
        csv_path = output_path / f'scenario2a_coupling_sweep_{args.array}.csv'
        results_coupling.to_csv(csv_path, index=False)
        print(f"\n✓ Saved coupling sweep results: {csv_path}")
        
        # Identify failure threshold
        threshold_info = identify_failure_threshold(results_coupling)
        print(f"\n📊 Failure Threshold Analysis:")
        if threshold_info['failure_threshold_c1'] is not None:
            print(f"  Failure at c1 = {threshold_info['failure_threshold_c1']:.3f}")
            print(f"  RMSE at failure: {threshold_info['failure_rmse']:.3f}°")
            print(f"  Resolution at failure: {threshold_info['failure_resolution']:.1f}%")
        else:
            print(f"  No failure detected (degradation < 100% for all coupling levels)")
        print(f"  Max sensitivity at c1 = {threshold_info['max_sensitivity_c1']:.3f}")
        print(f"  Max sensitivity value: {threshold_info['max_sensitivity_value']:.3f}")
        
        # Generate plots
        if not args.no_plots:
            plot_coupling_sweep_results(results_coupling, output_path, args.array)
    
    # Experiment 2B: Array Sensitivity Comparison
    if 'array-comparison' in experiments:
        # Load baselines for all arrays if available
        for arr in ['ULA', 'Z5', 'Z6']:
            baseline_file = f'results/scenario1_baseline/scenario1c_array_comparison.csv'
            if os.path.exists(baseline_file):
                df = pd.read_csv(baseline_file)
                arr_row = df[df['Array'] == arr]
                if len(arr_row) > 0:
                    baseline_metrics_dict[arr] = arr_row.iloc[0].to_dict()
        
        results_arrays = run_array_sensitivity_comparison(
            coupling_strength=args.fixed_coupling,
            snr_db=args.snr,
            snapshots=args.snapshots,
            num_trials=args.trials,
            true_doas=true_doas,
            resolution_threshold=args.resolution_threshold,
            baseline_metrics_dict=baseline_metrics_dict if baseline_metrics_dict else None
        )
        
        # Save results
        csv_path = output_path / 'scenario2b_array_sensitivity.csv'
        results_arrays.to_csv(csv_path, index=False)
        print(f"\n✓ Saved array sensitivity results: {csv_path}")
        
        # Generate plots
        if not args.no_plots:
            plot_array_sensitivity_comparison(results_arrays, output_path)
    
    print("\n" + "="*70)
    print("  SCENARIO 2 COMPLETE!")
    print("="*70)
    print(f"  Results saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
