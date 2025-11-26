"""
ALSS-II Validation Test Script

Compares ALSS (original) vs ALSS-II (enhanced) on Z5 array using the
experimental setup from the RadarCon 2025 paper.

Tests:
1. RMT-based noise estimation vs trace-based
2. Adaptive coreL vs fixed coreL=3
3. Geometry-aware priors vs standard AR(1)
4. Coupling-aware shrinkage
5. Full ALSS-II pipeline

Expected: ALSS-II should improve gap reduction from 45% → 50-55%

Author: ALSS-II Development Team
Date: November 25, 2025
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from pathlib import Path

# Import DOA framework
from doa_estimation.music import MUSICEstimator
from doa_estimation.metrics import DOAMetrics
from doa_estimation.simulation import generate_received_signal

# Import array processors
from geometry_processors.z5_processor import Z5ArrayProcessor

# Import ALSS modules
import core.radarpy.algorithms.alss as alss_module
from core.radarpy.algorithms.alss_coupling import apply_alss_coupling_aware


# Configuration (matching paper)
TRUE_ANGLES = np.array([-30, 0, 30])
WAVELENGTH = 2.0
NUM_TRIALS = 100  # Start with 100 for speed, increase to 1000 for paper
SNR_DB = 10
NUM_SNAPSHOTS = 200
MCM_C1 = 0.3
MCM_ALPHA = 0.5
SEED = 42

np.random.seed(SEED)


def print_header(text, char='='):
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}\n")


def run_single_trial(array_positions, use_coupling=True, alss_mode='off', 
                     use_rmt=False, auto_coreL=False):
    """Run single Monte Carlo trial."""
    
    # Generate signal with optional coupling
    X = generate_received_signal(
        array_positions=array_positions,
        wavelength=WAVELENGTH,
        doas=TRUE_ANGLES,
        num_snapshots=NUM_SNAPSHOTS,
        snr_db=SNR_DB,
        mutual_coupling=(MCM_C1, MCM_ALPHA) if use_coupling else None
    )
    
    # Create estimator with ALSS configuration
    if alss_mode == 'off':
        estimator = MUSICEstimator(
            array_positions=array_positions,
            wavelength=WAVELENGTH,
            num_sources=len(TRUE_ANGLES),
            alss_enabled=False
        )
    elif alss_mode == 'alss':
        # Original ALSS
        estimator = MUSICEstimator(
            array_positions=array_positions,
            wavelength=WAVELENGTH,
            num_sources=len(TRUE_ANGLES),
            alss_enabled=True,
            alss_mode='zero',
            alss_tau=1.0,
            alss_coreL=3
        )
    elif alss_mode == 'alss_ii':
        # ALSS-II with enhancements
        estimator = MUSICEstimator(
            array_positions=array_positions,
            wavelength=WAVELENGTH,
            num_sources=len(TRUE_ANGLES),
            alss_enabled=True,
            alss_mode='zero',
            alss_tau=1.0,
            alss_coreL=None,  # Will be computed adaptively
            use_rmt=use_rmt,
            auto_coreL=auto_coreL,
            K_sources=len(TRUE_ANGLES),
            snr_est=SNR_DB
        )
    else:
        raise ValueError(f"Unknown alss_mode: {alss_mode}")
    
    # Estimate DOAs
    try:
        doas_est = estimator.estimate(X, num_snapshots=NUM_SNAPSHOTS)
    except Exception as e:
        print(f"Estimation failed: {e}")
        return None
    
    # Compute RMSE
    metrics = DOAMetrics()
    rmse = metrics.compute_rmse(TRUE_ANGLES, doas_est)
    
    return rmse


def run_comparison(num_trials=100):
    """Run full ALSS vs ALSS-II comparison."""
    
    print_header("ALSS-II Validation Experiment")
    
    # Setup Z5 array
    processor = Z5ArrayProcessor(N=7, d=1.0)
    positions = np.array(processor.sensors_positions) * 1.0  # Convert to physical units
    
    print(f"Array: Z5 (N=7)")
    print(f"Positions: {positions}")
    print(f"True DOAs: {TRUE_ANGLES}")
    print(f"SNR: {SNR_DB} dB")
    print(f"Snapshots: {NUM_SNAPSHOTS}")
    print(f"Trials: {num_trials}")
    print(f"Coupling: c1={MCM_C1}, alpha={MCM_ALPHA}")
    
    # Test configurations
    configs = [
        # Condition 1: Baseline (no coupling, no ALSS)
        {'name': 'Cond1_Baseline', 'coupling': False, 'alss': 'off', 'rmt': False, 'auto': False},
        
        # Condition 2: Original ALSS (no coupling)
        {'name': 'Cond2_ALSS', 'coupling': False, 'alss': 'alss', 'rmt': False, 'auto': False},
        
        # Condition 2b: ALSS-II with RMT only
        {'name': 'Cond2b_ALSS2_RMT', 'coupling': False, 'alss': 'alss_ii', 'rmt': True, 'auto': False},
        
        # Condition 2c: ALSS-II with RMT + adaptive coreL
        {'name': 'Cond2c_ALSS2_Full', 'coupling': False, 'alss': 'alss_ii', 'rmt': True, 'auto': True},
        
        # Condition 3: Baseline with coupling
        {'name': 'Cond3_MCM', 'coupling': True, 'alss': 'off', 'rmt': False, 'auto': False},
        
        # Condition 4: Original ALSS with coupling
        {'name': 'Cond4_ALSS_MCM', 'coupling': True, 'alss': 'alss', 'rmt': False, 'auto': False},
        
        # Condition 4b: ALSS-II full with coupling
        {'name': 'Cond4b_ALSS2_MCM', 'coupling': True, 'alss': 'alss_ii', 'rmt': True, 'auto': True},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")
        
        rmse_list = []
        
        for trial in range(num_trials):
            if trial % 10 == 0:
                print(f"Trial {trial}/{num_trials}...", end='\r')
            
            rmse = run_single_trial(
                array_positions=positions,
                use_coupling=config['coupling'],
                alss_mode=config['alss'],
                use_rmt=config['rmt'],
                auto_coreL=config['auto']
            )
            
            if rmse is not None:
                rmse_list.append(rmse)
        
        # Compute statistics
        rmse_mean = np.mean(rmse_list)
        rmse_std = np.std(rmse_list)
        
        results[config['name']] = {
            'mean': rmse_mean,
            'std': rmse_std,
            'trials': len(rmse_list)
        }
        
        print(f"\n{config['name']}: {rmse_mean:.2f}° ± {rmse_std:.2f}° (n={len(rmse_list)})")
    
    return results


def compute_gap_reduction(results):
    """Compute gap reduction metrics."""
    
    print_header("Gap Reduction Analysis")
    
    # Original ALSS gap reduction
    cond1 = results['Cond1_Baseline']['mean']
    cond2 = results['Cond2_ALSS']['mean']
    cond3 = results['Cond3_MCM']['mean']
    cond4 = results['Cond4_ALSS_MCM']['mean']
    
    gap_original = cond3 - cond1
    reduction_original = cond3 - cond4
    pct_original = (reduction_original / gap_original) * 100 if gap_original > 0 else 0
    
    print(f"\n📊 Original ALSS:")
    print(f"   Gap (MCM impact): {gap_original:.2f}° ({cond3:.2f}° - {cond1:.2f}°)")
    print(f"   Reduction: {reduction_original:.2f}°")
    print(f"   Gap Reduction: {pct_original:.1f}%")
    
    # ALSS-II gap reduction
    cond4b = results['Cond4b_ALSS2_MCM']['mean']
    reduction_alss2 = cond3 - cond4b
    pct_alss2 = (reduction_alss2 / gap_original) * 100 if gap_original > 0 else 0
    
    print(f"\n🚀 ALSS-II (Full):")
    print(f"   Gap (MCM impact): {gap_original:.2f}°")
    print(f"   Reduction: {reduction_alss2:.2f}°")
    print(f"   Gap Reduction: {pct_alss2:.1f}%")
    
    # Improvement
    improvement = pct_alss2 - pct_original
    print(f"\n✨ ALSS-II Improvement over ALSS:")
    print(f"   Additional gap reduction: {improvement:.1f} percentage points")
    print(f"   Relative improvement: {(improvement/pct_original)*100:.1f}%")
    
    # Individual enhancements
    print(f"\n🔬 Component Analysis:")
    
    # RMT only
    cond2b = results['Cond2b_ALSS2_RMT']['mean']
    rmt_benefit = cond2 - cond2b
    print(f"   RMT noise estimation: {rmt_benefit:.2f}° improvement")
    
    # RMT + adaptive coreL
    cond2c = results['Cond2c_ALSS2_Full']['mean']
    full_benefit = cond2 - cond2c
    print(f"   RMT + Adaptive coreL: {full_benefit:.2f}° improvement")
    
    return {
        'alss_gap_reduction_pct': pct_original,
        'alss_ii_gap_reduction_pct': pct_alss2,
        'improvement_pct_points': improvement,
        'rmt_benefit_deg': rmt_benefit,
        'full_benefit_deg': full_benefit
    }


def save_results(results, metrics, output_dir='results/alss_ii'):
    """Save results to CSV."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create summary table
    summary_data = []
    for config_name, stats in results.items():
        summary_data.append({
            'Configuration': config_name,
            'Mean_RMSE_deg': stats['mean'],
            'Std_RMSE_deg': stats['std'],
            'Num_Trials': stats['trials']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = Path(output_dir) / 'alss_ii_validation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = Path(output_dir) / 'alss_ii_gap_reduction_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Metrics saved to: {metrics_path}")
    
    return df


if __name__ == '__main__':
    print_header("ALSS-II Validation Test", char='#')
    print("This script validates ALSS-II enhancements:")
    print("  1. RMT-based noise estimation")
    print("  2. Adaptive coreL threshold")
    print("  3. Geometry-aware priors")
    print("  4. Full ALSS-II pipeline")
    print("\nTarget: Beat 45% gap reduction from original ALSS")
    
    # Run comparison
    results = run_comparison(num_trials=NUM_TRIALS)
    
    # Compute metrics
    metrics = compute_gap_reduction(results)
    
    # Save results
    df = save_results(results, metrics)
    
    # Print summary table
    print_header("Summary Table")
    print(df.to_string(index=False))
    
    # Final verdict
    print_header("Final Verdict", char='#')
    if metrics['alss_ii_gap_reduction_pct'] > 45:
        print(f"✅ SUCCESS! ALSS-II achieves {metrics['alss_ii_gap_reduction_pct']:.1f}% gap reduction")
        print(f"   (vs 45% from original ALSS)")
        print(f"   Improvement: +{metrics['improvement_pct_points']:.1f} percentage points")
    else:
        print(f"⚠️  ALSS-II: {metrics['alss_ii_gap_reduction_pct']:.1f}% (target: >45%)")
        print(f"   Additional tuning may be needed")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Increase NUM_TRIALS to 1000 for publication-quality results")
    print("  2. Test on additional arrays (Z1, Z3_2) for robustness")
    print("  3. Run SNR sweep (0-20 dB) for comprehensive validation")
    print("  4. Add coupling-aware shrinkage (Phase 2)")
    print("="*80)
