#!/usr/bin/env python3
"""
Generate comprehensive data for ALSS paper publication.
Runs additional experiments for SNR sweeps, array comparisons, and statistical analysis.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from geometry_processors.z1_processor import Z1ArrayProcessor
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor
from doa_estimation.music import MUSICEstimator
from doa_estimation.metrics import DOAMetrics

# Test configuration
TRUE_ANGLES = [-30, 0, 30]
WAVELENGTH = 1.0
NUM_TRIALS = 100  # More trials for better statistics

def run_snr_sweep():
    """Run SNR sweep for paper Figure."""
    print("="*80)
    print("SNR SWEEP ANALYSIS")
    print("="*80)
    
    snr_values = [0, 5, 10, 15, 20]
    snapshots = 128
    
    # Test with Z5 array
    processor = Z5ArrayProcessor(N=7, d=0.5)
    data = processor.run_full_analysis()
    positions = data.sensors_positions
    
    results = []
    
    for snr in snr_values:
        print(f"\nTesting SNR = {snr} dB...")
        
        estimator_no_mcm = MUSICEstimator(
            sensor_positions=positions,
            wavelength=WAVELENGTH,
            enable_mcm=False
        )
        
        estimator_with_mcm = MUSICEstimator(
            sensor_positions=positions,
            wavelength=WAVELENGTH,
            enable_mcm=True,
            mcm_model='exponential',
            mcm_params={'c1': 0.3, 'alpha': 0.5}
        )
        
        rmse_no_mcm = []
        rmse_with_mcm = []
        
        for trial in range(NUM_TRIALS):
            if (trial + 1) % 20 == 0:
                print(f"  Progress: {trial+1}/{NUM_TRIALS}")
            
            try:
                # No MCM
                X = estimator_no_mcm.simulate_signals(
                    true_angles=TRUE_ANGLES,
                    snapshots=snapshots,
                    SNR_dB=snr
                )
                theta_est = estimator_no_mcm.estimate(X, K_sources=len(TRUE_ANGLES))
                rmse_no_mcm.append(DOAMetrics.compute_rmse(TRUE_ANGLES, theta_est))
                
                # With MCM
                X = estimator_with_mcm.simulate_signals(
                    true_angles=TRUE_ANGLES,
                    snapshots=snapshots,
                    SNR_dB=snr
                )
                theta_est = estimator_with_mcm.estimate(X, K_sources=len(TRUE_ANGLES))
                rmse_with_mcm.append(DOAMetrics.compute_rmse(TRUE_ANGLES, theta_est))
                
            except Exception as e:
                print(f"  Trial {trial} failed: {e}")
                rmse_no_mcm.append(np.nan)
                rmse_with_mcm.append(np.nan)
        
        # Filter out failed trials
        rmse_no_mcm = [r for r in rmse_no_mcm if not np.isnan(r)]
        rmse_with_mcm = [r for r in rmse_with_mcm if not np.isnan(r)]
        
        if len(rmse_no_mcm) > 0 and len(rmse_with_mcm) > 0:
            results.append({
                'SNR_dB': snr,
                'RMSE_NoMCM_Mean': np.mean(rmse_no_mcm),
                'RMSE_NoMCM_Std': np.std(rmse_no_mcm),
                'RMSE_WithMCM_Mean': np.mean(rmse_with_mcm),
                'RMSE_WithMCM_Std': np.std(rmse_with_mcm),
                'MCM_Impact_Pct': (np.mean(rmse_with_mcm) - np.mean(rmse_no_mcm)) / np.mean(rmse_no_mcm) * 100,
                'Trials_Success': len(rmse_no_mcm)
            })
            
            print(f"  No MCM:   {np.mean(rmse_no_mcm):.3f}° ± {np.std(rmse_no_mcm):.3f}°")
            print(f"  With MCM: {np.mean(rmse_with_mcm):.3f}° ± {np.std(rmse_with_mcm):.3f}°")
            print(f"  MCM Impact: {results[-1]['MCM_Impact_Pct']:+.1f}%")
    
    df = pd.DataFrame(results)
    output_path = os.path.join('..', 'results', 'summaries', 'snr_sweep_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")
    
    return df


def run_snapshot_sweep():
    """Run snapshot count sweep."""
    print("\n" + "="*80)
    print("SNAPSHOT SWEEP ANALYSIS")
    print("="*80)
    
    snapshot_values = [32, 64, 128, 256, 512]
    snr = 10  # Fixed SNR
    
    # Test with Z5 array
    processor = Z5ArrayProcessor(N=7, d=0.5)
    data = processor.run_full_analysis()
    positions = data.sensors_positions
    
    results = []
    
    for snapshots in snapshot_values:
        print(f"\nTesting Snapshots = {snapshots}...")
        
        estimator_no_mcm = MUSICEstimator(
            sensor_positions=positions,
            wavelength=WAVELENGTH,
            enable_mcm=False
        )
        
        estimator_with_mcm = MUSICEstimator(
            sensor_positions=positions,
            wavelength=WAVELENGTH,
            enable_mcm=True,
            mcm_model='exponential',
            mcm_params={'c1': 0.3, 'alpha': 0.5}
        )
        
        rmse_no_mcm = []
        rmse_with_mcm = []
        
        for trial in range(NUM_TRIALS):
            if (trial + 1) % 20 == 0:
                print(f"  Progress: {trial+1}/{NUM_TRIALS}")
            
            try:
                # No MCM
                X = estimator_no_mcm.simulate_signals(
                    true_angles=TRUE_ANGLES,
                    snapshots=snapshots,
                    SNR_dB=snr
                )
                theta_est = estimator_no_mcm.estimate(X, K_sources=len(TRUE_ANGLES))
                rmse_no_mcm.append(DOAMetrics.compute_rmse(TRUE_ANGLES, theta_est))
                
                # With MCM
                X = estimator_with_mcm.simulate_signals(
                    true_angles=TRUE_ANGLES,
                    snapshots=snapshots,
                    SNR_dB=snr
                )
                theta_est = estimator_with_mcm.estimate(X, K_sources=len(TRUE_ANGLES))
                rmse_with_mcm.append(DOAMetrics.compute_rmse(TRUE_ANGLES, theta_est))
                
            except Exception as e:
                rmse_no_mcm.append(np.nan)
                rmse_with_mcm.append(np.nan)
        
        # Filter out failed trials
        rmse_no_mcm = [r for r in rmse_no_mcm if not np.isnan(r)]
        rmse_with_mcm = [r for r in rmse_with_mcm if not np.isnan(r)]
        
        if len(rmse_no_mcm) > 0 and len(rmse_with_mcm) > 0:
            results.append({
                'Snapshots': snapshots,
                'RMSE_NoMCM_Mean': np.mean(rmse_no_mcm),
                'RMSE_NoMCM_Std': np.std(rmse_no_mcm),
                'RMSE_WithMCM_Mean': np.mean(rmse_with_mcm),
                'RMSE_WithMCM_Std': np.std(rmse_with_mcm),
                'MCM_Impact_Pct': (np.mean(rmse_with_mcm) - np.mean(rmse_no_mcm)) / np.mean(rmse_no_mcm) * 100,
                'Trials_Success': len(rmse_no_mcm)
            })
            
            print(f"  No MCM:   {np.mean(rmse_no_mcm):.3f}° ± {np.std(rmse_no_mcm):.3f}°")
            print(f"  With MCM: {np.mean(rmse_with_mcm):.3f}° ± {np.std(rmse_with_mcm):.3f}°")
    
    df = pd.DataFrame(results)
    output_path = os.path.join('..', 'results', 'summaries', 'snapshot_sweep_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")
    
    return df


def run_array_comparison():
    """Compare multiple array types."""
    print("\n" + "="*80)
    print("ARRAY COMPARISON ANALYSIS")
    print("="*80)
    
    snr = 10
    snapshots = 200
    
    # Test arrays
    test_arrays = [
        ("ULA", ULArrayProcessor(N=7, d=0.5)),
        ("Nested", NestedArrayProcessor(N1=2, N2=4, d=0.5)),
        ("Z1", Z1ArrayProcessor(N=7, d=0.5)),
        ("Z3_2", Z3_2ArrayProcessor(N=6, d=0.5)),
        ("Z5", Z5ArrayProcessor(N=7, d=0.5))
    ]
    
    results = []
    
    for array_name, processor in test_arrays:
        print(f"\nTesting {array_name}...")
        
        data = processor.run_full_analysis()
        positions = data.sensors_positions
        
        estimator_no_mcm = MUSICEstimator(
            sensor_positions=positions,
            wavelength=WAVELENGTH,
            enable_mcm=False
        )
        
        estimator_with_mcm = MUSICEstimator(
            sensor_positions=positions,
            wavelength=WAVELENGTH,
            enable_mcm=True,
            mcm_model='exponential',
            mcm_params={'c1': 0.3, 'alpha': 0.5}
        )
        
        rmse_no_mcm = []
        rmse_with_mcm = []
        
        for trial in range(NUM_TRIALS):
            if (trial + 1) % 20 == 0:
                print(f"  Progress: {trial+1}/{NUM_TRIALS}")
            
            try:
                # No MCM
                X = estimator_no_mcm.simulate_signals(
                    true_angles=TRUE_ANGLES,
                    snapshots=snapshots,
                    SNR_dB=snr
                )
                theta_est = estimator_no_mcm.estimate(X, K_sources=len(TRUE_ANGLES))
                rmse_no_mcm.append(DOAMetrics.compute_rmse(TRUE_ANGLES, theta_est))
                
                # With MCM
                X = estimator_with_mcm.simulate_signals(
                    true_angles=TRUE_ANGLES,
                    snapshots=snapshots,
                    SNR_dB=snr
                )
                theta_est = estimator_with_mcm.estimate(X, K_sources=len(TRUE_ANGLES))
                rmse_with_mcm.append(DOAMetrics.compute_rmse(TRUE_ANGLES, theta_est))
                
            except Exception as e:
                rmse_no_mcm.append(np.nan)
                rmse_with_mcm.append(np.nan)
        
        # Filter out failed trials
        rmse_no_mcm = [r for r in rmse_no_mcm if not np.isnan(r)]
        rmse_with_mcm = [r for r in rmse_with_mcm if not np.isnan(r)]
        
        if len(rmse_no_mcm) > 0 and len(rmse_with_mcm) > 0:
            mcm_impact = np.mean(rmse_with_mcm) - np.mean(rmse_no_mcm)
            mcm_impact_pct = mcm_impact / np.mean(rmse_no_mcm) * 100
            
            results.append({
                'Array': array_name,
                'N_Sensors': data.num_sensors,
                'Aperture': data.array_aperture,
                'Virtual_Sensors': data.num_unique_differences,
                'RMSE_NoMCM_Mean': np.mean(rmse_no_mcm),
                'RMSE_NoMCM_Std': np.std(rmse_no_mcm),
                'RMSE_WithMCM_Mean': np.mean(rmse_with_mcm),
                'RMSE_WithMCM_Std': np.std(rmse_with_mcm),
                'MCM_Impact_Deg': mcm_impact,
                'MCM_Impact_Pct': mcm_impact_pct,
                'Trials_Success': len(rmse_no_mcm)
            })
            
            print(f"  No MCM:   {np.mean(rmse_no_mcm):.3f}° ± {np.std(rmse_no_mcm):.3f}°")
            print(f"  With MCM: {np.mean(rmse_with_mcm):.3f}° ± {np.std(rmse_with_mcm):.3f}°")
            print(f"  MCM Impact: {mcm_impact:+.3f}° ({mcm_impact_pct:+.1f}%)")
    
    df = pd.DataFrame(results)
    output_path = os.path.join('..', 'results', 'summaries', 'array_comparison_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")
    
    return df


def main():
    """Main execution."""
    print("="*80)
    print("PAPER DATA GENERATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  True angles: {TRUE_ANGLES}")
    print(f"  Wavelength: {WAVELENGTH}")
    print(f"  Trials per condition: {NUM_TRIALS}")
    print()
    
    # Run all analyses
    snr_results = run_snr_sweep()
    snapshot_results = run_snapshot_sweep()
    array_results = run_array_comparison()
    
    print("\n" + "="*80)
    print("PAPER DATA GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/summaries/snr_sweep_results.csv")
    print("  - results/summaries/snapshot_sweep_results.csv")
    print("  - results/summaries/array_comparison_results.csv")
    print("\nReady for paper integration!")


if __name__ == "__main__":
    main()
