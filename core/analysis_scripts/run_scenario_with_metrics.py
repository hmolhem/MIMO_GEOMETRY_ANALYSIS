"""
Example Scenario Runner with Comprehensive Metrics

Demonstrates how to run experiments and compute all 5 requested metrics:
- RMSE_degrees
- RMSE_CRB_ratio
- Resolution_Rate
- Bias_degrees
- Runtime_ms

Usage:
    python run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100
    python run_scenario_with_metrics.py --array Z5 --snr 10 --with-coupling --coupling-strength 0.3

Author: MIMO Geometry Analysis Team
Date: November 6, 2025
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.radarpy.signal.doa_sim_core import simulate_snapshots, run_music
from core.radarpy.signal.mutual_coupling import generate_mcm
from core.radarpy.signal.metrics import (
    compute_scenario_metrics, 
    print_metrics_summary,
    run_trial_with_timing
)


def get_array_positions(array_type: str, num_sensors: int = 7, d: float = 0.5) -> np.ndarray:
    """
    Get sensor positions for specified array type.
    
    Args:
        array_type: Array type ('ULA', 'Z5', 'Z4', etc.)
        num_sensors: Number of sensors (for ULA)
        d: Element spacing in wavelengths
    
    Returns:
        positions: Sensor positions array
    """
    if array_type.upper() == 'ULA':
        return np.arange(num_sensors) * d
    elif array_type.upper() == 'Z5':
        return np.array([0, 5, 8, 11, 14, 17, 21]) * d
    elif array_type.upper() == 'Z4':
        return np.array([0, 5, 8, 11, 14, 17, 21]) * d  # Same layout for N=7
    elif array_type.upper() == 'NESTED':
        # Simple nested: N1=2, N2=5
        n1_positions = np.arange(1, 3) * d  # [1, 2]
        n2_positions = np.arange(1, 6) * 2 * d  # [2, 4, 6, 8, 10]
        return np.sort(np.concatenate([n1_positions, n2_positions]))
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def run_scenario(array_type: str = 'Z5',
                snr_db: float = 10.0,
                snapshots: int = 256,
                num_trials: int = 100,
                true_doas: np.ndarray = np.array([15.0, -20.0]),
                wavelength: float = 1.0,
                with_coupling: bool = False,
                coupling_strength: float = 0.3,
                resolution_threshold: float = 3.0,
                verbose: bool = True) -> Dict[str, float]:
    """
    Run a complete scenario with metrics computation.
    
    Args:
        array_type: Array geometry type
        snr_db: SNR in dB
        snapshots: Number of snapshots (M)
        num_trials: Number of Monte Carlo trials
        true_doas: True DOA angles in degrees
        wavelength: Signal wavelength
        with_coupling: Whether to apply mutual coupling
        coupling_strength: Coupling coefficient magnitude
        resolution_threshold: Threshold for resolution detection (degrees)
        verbose: Whether to print progress
    
    Returns:
        metrics: Dictionary with all 5 metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Running Scenario: {array_type} Array")
        print(f"{'='*70}")
        print(f"  SNR:              {snr_db} dB")
        print(f"  Snapshots:        {snapshots}")
        print(f"  Trials:           {num_trials}")
        print(f"  True DOAs:        {true_doas}°")
        print(f"  Coupling:         {'YES' if with_coupling else 'NO'}")
        if with_coupling:
            print(f"  Coupling Strength: {coupling_strength}")
        print(f"{'='*70}\n")
    
    # Get array configuration
    positions = get_array_positions(array_type)
    N = len(positions)
    K = len(true_doas)
    
    # Generate coupling matrix if needed
    if with_coupling:
        coupling_matrix = generate_mcm(N, coupling_strength, model='exponential')
    else:
        coupling_matrix = None
    
    # Scan grid for MUSIC
    scan_grid = np.linspace(-90, 90, 1801)
    
    # Run Monte Carlo trials
    estimated_trials = []
    runtimes = []
    
    if verbose:
        print(f"Running {num_trials} trials...")
    
    for trial in range(num_trials):
        # Progress indicator
        if verbose and (trial + 1) % 20 == 0:
            print(f"  Progress: {trial + 1}/{num_trials} trials completed")
        
        # Generate snapshots with noise
        snapshots_data = simulate_snapshots(
            sensor_positions=positions,
            wavelength=wavelength,
            doas_deg=true_doas,
            num_snapshots=snapshots,
            snr_db=snr_db,
            coupling_matrix=coupling_matrix
        )
        
        # Compute covariance matrix
        Rxx = (snapshots_data @ snapshots_data.conj().T) / snapshots
        
        # Run MUSIC with timing
        est_doas, runtime = run_trial_with_timing(
            run_music,
            Rxx=Rxx,
            sensor_positions=positions,
            wavelength=wavelength,
            scan_grid=scan_grid,
            K=K,
            coupling_matrix=coupling_matrix
        )
        
        estimated_trials.append(est_doas)
        runtimes.append(runtime)
    
    if verbose:
        print(f"  Completed all {num_trials} trials!\n")
    
    # Compute comprehensive metrics
    metrics = compute_scenario_metrics(
        estimated_doas_trials=estimated_trials,
        true_doas=true_doas,
        sensor_positions=positions,
        wavelength=wavelength,
        snr_db=snr_db,
        snapshots=snapshots,
        coupling_matrix=coupling_matrix,
        resolution_threshold=resolution_threshold,
        runtimes_ms=runtimes
    )
    
    return metrics


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run DOA scenario with comprehensive metrics computation"
    )
    
    # Array configuration
    parser.add_argument(
        '--array', type=str, default='Z5',
        choices=['ULA', 'Z5', 'Z4', 'NESTED'],
        help="Array geometry type"
    )
    
    # Signal parameters
    parser.add_argument(
        '--snr', type=float, default=10.0,
        help="SNR in dB"
    )
    parser.add_argument(
        '--snapshots', type=int, default=256,
        help="Number of snapshots (M)"
    )
    parser.add_argument(
        '--trials', type=int, default=100,
        help="Number of Monte Carlo trials"
    )
    
    # DOA configuration
    parser.add_argument(
        '--doas', type=float, nargs='+', default=[15.0, -20.0],
        help="True DOA angles in degrees (e.g., --doas 15 -20 30)"
    )
    
    # Coupling parameters
    parser.add_argument(
        '--with-coupling', action='store_true',
        help="Enable mutual coupling"
    )
    parser.add_argument(
        '--coupling-strength', type=float, default=0.3,
        help="Coupling coefficient magnitude"
    )
    
    # Analysis parameters
    parser.add_argument(
        '--resolution-threshold', type=float, default=3.0,
        help="Resolution threshold in degrees"
    )
    
    # Output options
    parser.add_argument(
        '--save-csv', action='store_true',
        help="Save metrics to CSV file"
    )
    parser.add_argument(
        '--save-json', action='store_true',
        help="Save metrics to JSON file"
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/summaries',
        help="Output directory for saved files"
    )
    
    args = parser.parse_args()
    
    # Convert DOAs to numpy array
    true_doas = np.array(args.doas)
    
    # Run scenario
    metrics = run_scenario(
        array_type=args.array,
        snr_db=args.snr,
        snapshots=args.snapshots,
        num_trials=args.trials,
        true_doas=true_doas,
        with_coupling=args.with_coupling,
        coupling_strength=args.coupling_strength,
        resolution_threshold=args.resolution_threshold,
        verbose=True
    )
    
    # Print results
    scenario_name = f"{args.array} Array"
    if args.with_coupling:
        scenario_name += f" (Coupling: {args.coupling_strength:.2f})"
    scenario_name += f" @ {args.snr}dB SNR"
    
    print_metrics_summary(metrics, scenario_name=scenario_name)
    
    # Display individual metrics
    print("Detailed Metrics:")
    print(f"  RMSE:               {metrics['RMSE_degrees']:.4f}°")
    print(f"  RMSE/CRB Ratio:     {metrics['RMSE_CRB_ratio']:.4f}x")
    print(f"  Resolution Rate:    {metrics['Resolution_Rate']:.2f}%")
    print(f"  Bias:               {metrics['Bias_degrees']:.4f}°")
    print(f"  Runtime:            {metrics['Runtime_ms']:.2f} ms")
    print()
    
    # Save results if requested
    if args.save_csv or args.save_json:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        coupling_tag = f"_coupling{args.coupling_strength:.2f}" if args.with_coupling else ""
        base_filename = f"{args.array}_SNR{args.snr}dB{coupling_tag}_trials{args.trials}"
        
        if args.save_csv:
            csv_path = output_dir / f"{base_filename}_metrics.csv"
            df = pd.DataFrame([metrics])
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved metrics to: {csv_path}")
        
        if args.save_json:
            import json
            json_path = output_dir / f"{base_filename}_metrics.json"
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Saved metrics to: {json_path}")
    
    print("\n" + "="*70)
    print("  SCENARIO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
