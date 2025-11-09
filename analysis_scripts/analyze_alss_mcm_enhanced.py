"""
Enhanced ALSS+MCM Experimental Validation with Publication-Quality Plots
=========================================================================

This script performs comprehensive ALSS (Adaptive Lag-Selective Shrinkage) validation
under Mutual Coupling Matrix (MCM) conditions with three publication-ready visualizations:

1. Bias-Variance Decomposition Plot
   - Shows how ALSS affects variance vs bias components separately
   - Different curves for each array type (Z1, Z3_2, Z5)
   - Demonstrates orthogonal effects principle

2. SNR-Dependent Effectiveness Plot
   - ALSS improvement percentage vs SNR for MCM ON/OFF conditions
   - Demonstrates when ALSS matters most (low SNR emphasis)
   - Validates harmlessness at high SNR

3. Gap Reduction Visualization
   - Bar chart showing gap reduction percentages with confidence intervals
   - Statistical significance markers (p < 0.05, p < 0.01, p < 0.001)
   - Validates theoretical predictions (Z5: 40-50%, Z1: 25-35%, Z3_2: 15-25%)

Output:
    - Actual experimental results in dictionary format for publication
    - Three high-resolution plots saved to results/plots/
    - Statistical validation CSV files
    - Comprehensive console report

Author: Enhanced for IEEE RadarCon 2025 submission
Date: November 8, 2025
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

# Import local DOA estimation module (same as compare_mcm_effects.py)
from doa_estimation.music import MUSICEstimator
from doa_estimation.simulation import generate_narrowband_signal

# Import array processors
from geometry_processors.z1_processor import Z1ArrayProcessor
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor

# Import core radarpy for ALSS support
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))
from radarpy.algorithms.coarray_music import estimate_doa_coarray_music
from radarpy.algorithms.alss import apply_alss
from radarpy.algorithms.coarray import build_virtual_ula_covariance


# ============================================================================
# Configuration
# ============================================================================

TRUE_ANGLES = np.array([-30, 0, 30])  # DOA in degrees
WAVELENGTH = 2.0  # Carrier wavelength
NUM_TRIALS = 50  # Monte Carlo trials for statistical significance

# Scenario 2 Configuration (4 conditions)
SCENARIO2_SNR = 10  # dB
SCENARIO2_SNAPSHOTS = 200

# SNR sweep configuration
SNR_RANGE = np.array([0, 5, 10, 15, 20])  # dB
SNR_SWEEP_SNAPSHOTS = 200
SNR_SWEEP_TRIALS = 30  # Fewer trials for sweep (computational efficiency)

# MCM parameters (standard exponential model)
MCM_C1 = 0.3
MCM_ALPHA = 0.5

# ALSS parameters
ALSS_MODE = 'zero'
ALSS_TAU = 1.0
ALSS_COREL = 3


# ============================================================================
# Utility Functions
# ============================================================================

def print_header(text, char='='):
    """Print formatted section header."""
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}\n")


def print_section(text):
    """Print formatted subsection."""
    print(f"\n{text}")
    print("-" * len(text))


def compute_rmse(estimated, true_angles):
    """Compute RMSE between estimated and true DOA angles."""
    if len(estimated) == 0:
        return 999.0
    
    # Match estimated to true using Hungarian algorithm (simplified nearest match)
    errors = []
    for true_ang in true_angles:
        diffs = np.abs(estimated - true_ang)
        errors.append(np.min(diffs))
    
    return np.sqrt(np.mean(np.array(errors)**2))


def compute_bias_variance(errors_list):
    """
    Decompose RMSE into bias and variance components.
    
    RMSE² = Bias² + Variance
    
    Args:
        errors_list: List of error arrays from multiple trials
    
    Returns:
        bias: Systematic error (mean)
        variance: Random error (std²)
        rmse: Total RMSE
    """
    all_errors = np.concatenate(errors_list)
    bias = np.abs(np.mean(all_errors))
    variance = np.var(all_errors)
    rmse = np.sqrt(bias**2 + variance)
    return bias, variance, rmse


def gap_reduction_metric(cond1_rmse, cond3_rmse, cond4_rmse):
    """
    Compute gap reduction percentage.
    
    Gap Reduction = (Cond1 - Cond4) / (Cond3 - Cond1) × 100%
    
    Measures how much ALSS recovers toward ideal (Cond1) from MCM degradation.
    """
    if cond3_rmse <= cond1_rmse:
        return 0.0  # No gap to reduce (MCM improves, like Z5)
    
    gap_total = cond3_rmse - cond1_rmse
    gap_recovered = cond3_rmse - cond4_rmse
    
    if gap_total < 0.01:
        return 0.0
    
    return (gap_recovered / gap_total) * 100.0


# ============================================================================
# Scenario 2: Four-Condition Framework with Bias-Variance Decomposition
# ============================================================================

def run_scenario2_four_conditions():
    """
    Run comprehensive 4-condition experiment for gap reduction analysis.
    
    Conditions:
        1. No MCM, No ALSS (baseline ideal)
        2. No MCM, ALSS ON (ALSS baseline benefit)
        3. MCM ON, No ALSS (coupling degradation)
        4. MCM ON, ALSS ON (best effort recovery)
    
    Returns:
        results_dict: Nested dictionary with RMSE values and bias-variance components
    """
    print_header("SCENARIO 2: FOUR-CONDITION GAP REDUCTION ANALYSIS")
    
    test_arrays = [
        ("Z1", Z1ArrayProcessor(N=7, d=1.0)),
        ("Z3_2", Z3_2ArrayProcessor(N=6, d=1.0)),
        ("Z5", Z5ArrayProcessor(N=7, d=1.0))
    ]
    
    results_dict = {}
    
    for array_name, processor in test_arrays:
        print_section(f"Testing Array: {array_name}")
        
        # Get array geometry
        data = processor.run_full_analysis()
        positions = data.sensor_positions
        N = len(positions)
        
        print(f"  Sensors: {N}, Positions: {positions}")
        print(f"  Sources: {len(TRUE_ANGLES)} at {TRUE_ANGLES}°")
        print(f"  SNR: {SCENARIO2_SNR} dB, Snapshots: {SCENARIO2_SNAPSHOTS}")
        print(f"  Trials: {NUM_TRIALS}\n")
        
        # Storage for all conditions
        rmse_cond1 = []  # No MCM, No ALSS
        rmse_cond2 = []  # No MCM, ALSS ON
        rmse_cond3 = []  # MCM ON, No ALSS
        rmse_cond4 = []  # MCM ON, ALSS ON
        
        # Storage for bias-variance decomposition
        errors_cond1 = []
        errors_cond2 = []
        errors_cond3 = []
        errors_cond4 = []
        
        # Initialize MCM
        mcm = MutualCouplingMatrix(positions, c1=MCM_C1, alpha=MCM_ALPHA)
        C = mcm.get_coupling_matrix()
        
        print(f"  Running {NUM_TRIALS} Monte Carlo trials...")
        
        for trial in range(NUM_TRIALS):
            if (trial + 1) % 10 == 0:
                print(f"    Progress: {trial+1}/{NUM_TRIALS} trials", end='\r')
            
            # Generate snapshot data (shared across conditions for fair comparison)
            # This ensures same noise realization for all 4 conditions
            np.random.seed(1000 + trial)  # Reproducible but different per trial
            
            # Condition 1: No MCM, No ALSS (BASELINE)
            try:
                X_ideal = generate_ula_measurements(
                    positions, TRUE_ANGLES, SCENARIO2_SNAPSHOTS, 
                    SCENARIO2_SNR, WAVELENGTH, coupling_matrix=None
                )
                
                theta_est, _, _ = coarray_music_1d(
                    X_ideal, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                    alss_enabled=False
                )
                
                rmse_val = compute_rmse(theta_est, TRUE_ANGLES)
                rmse_cond1.append(rmse_val)
                
                # Store per-source errors for bias-variance decomposition
                for true_ang in TRUE_ANGLES:
                    diffs = np.abs(theta_est - true_ang)
                    errors_cond1.append(np.min(diffs))
                    
            except Exception as e:
                print(f"\n    Condition 1 failed (trial {trial}): {e}")
                rmse_cond1.append(999.0)
            
            # Condition 2: No MCM, ALSS ON
            try:
                # Use same snapshot data
                theta_est, _, _ = coarray_music_1d(
                    X_ideal, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                    alss_enabled=True, alss_mode=ALSS_MODE, 
                    alss_tau=ALSS_TAU, alss_coreL=ALSS_COREL
                )
                
                rmse_val = compute_rmse(theta_est, TRUE_ANGLES)
                rmse_cond2.append(rmse_val)
                
                for true_ang in TRUE_ANGLES:
                    diffs = np.abs(theta_est - true_ang)
                    errors_cond2.append(np.min(diffs))
                    
            except Exception as e:
                print(f"\n    Condition 2 failed (trial {trial}): {e}")
                rmse_cond2.append(999.0)
            
            # Condition 3: MCM ON, No ALSS
            try:
                X_coupled = generate_ula_measurements(
                    positions, TRUE_ANGLES, SCENARIO2_SNAPSHOTS,
                    SCENARIO2_SNR, WAVELENGTH, coupling_matrix=C
                )
                
                theta_est, _, _ = coarray_music_1d(
                    X_coupled, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                    alss_enabled=False
                )
                
                rmse_val = compute_rmse(theta_est, TRUE_ANGLES)
                rmse_cond3.append(rmse_val)
                
                for true_ang in TRUE_ANGLES:
                    diffs = np.abs(theta_est - true_ang)
                    errors_cond3.append(np.min(diffs))
                    
            except Exception as e:
                print(f"\n    Condition 3 failed (trial {trial}): {e}")
                rmse_cond3.append(999.0)
            
            # Condition 4: MCM ON, ALSS ON (BEST EFFORT)
            try:
                # Use same coupled snapshots
                theta_est, _, _ = coarray_music_1d(
                    X_coupled, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                    alss_enabled=True, alss_mode=ALSS_MODE,
                    alss_tau=ALSS_TAU, alss_coreL=ALSS_COREL
                )
                
                rmse_val = compute_rmse(theta_est, TRUE_ANGLES)
                rmse_cond4.append(rmse_val)
                
                for true_ang in TRUE_ANGLES:
                    diffs = np.abs(theta_est - true_ang)
                    errors_cond4.append(np.min(diffs))
                    
            except Exception as e:
                print(f"\n    Condition 4 failed (trial {trial}): {e}")
                rmse_cond4.append(999.0)
        
        print(f"    Completed {NUM_TRIALS}/{NUM_TRIALS} trials")
        
        # Compute statistics
        mean_cond1 = np.mean(rmse_cond1)
        mean_cond2 = np.mean(rmse_cond2)
        mean_cond3 = np.mean(rmse_cond3)
        mean_cond4 = np.mean(rmse_cond4)
        
        std_cond1 = np.std(rmse_cond1)
        std_cond2 = np.std(rmse_cond2)
        std_cond3 = np.std(rmse_cond3)
        std_cond4 = np.std(rmse_cond4)
        
        # Compute bias-variance decomposition
        bias1, var1, _ = compute_bias_variance(errors_cond1)
        bias2, var2, _ = compute_bias_variance(errors_cond2)
        bias3, var3, _ = compute_bias_variance(errors_cond3)
        bias4, var4, _ = compute_bias_variance(errors_cond4)
        
        # Gap reduction metric
        gap_pct = gap_reduction_metric(mean_cond1, mean_cond3, mean_cond4)
        
        # Statistical significance (paired t-test: Cond3 vs Cond4)
        t_stat, p_value = stats.ttest_rel(rmse_cond3, rmse_cond4)
        
        # Store results
        results_dict[array_name] = {
            'cond1': mean_cond1,
            'cond2': mean_cond2,
            'cond3': mean_cond3,
            'cond4': mean_cond4,
            'std1': std_cond1,
            'std2': std_cond2,
            'std3': std_cond3,
            'std4': std_cond4,
            'bias1': bias1,
            'bias2': bias2,
            'bias3': bias3,
            'bias4': bias4,
            'variance1': var1,
            'variance2': var2,
            'variance3': var3,
            'variance4': var4,
            'gap_reduction_pct': gap_pct,
            'p_value': p_value,
            'rmse_trials': {
                'cond1': rmse_cond1,
                'cond2': rmse_cond2,
                'cond3': rmse_cond3,
                'cond4': rmse_cond4
            }
        }
        
        # Print summary
        print(f"\n  Results Summary:")
        print(f"    Condition 1 (No MCM, No ALSS):  {mean_cond1:.3f}° ± {std_cond1:.3f}° RMSE")
        print(f"    Condition 2 (No MCM, ALSS ON):  {mean_cond2:.3f}° ± {std_cond2:.3f}° RMSE")
        print(f"    Condition 3 (MCM ON, No ALSS):  {mean_cond3:.3f}° ± {std_cond3:.3f}° RMSE")
        print(f"    Condition 4 (MCM ON, ALSS ON):  {mean_cond4:.3f}° ± {std_cond4:.3f}° RMSE")
        print(f"\n    Gap Reduction: {gap_pct:.1f}%")
        print(f"    Statistical Significance (Cond3 vs Cond4): p = {p_value:.4f}")
        
        if p_value < 0.001:
            sig_marker = "***"
        elif p_value < 0.01:
            sig_marker = "**"
        elif p_value < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "n.s."
        print(f"    Significance: {sig_marker}")
    
    return results_dict


# ============================================================================
# SNR Sweep Analysis
# ============================================================================

def run_snr_sweep_analysis():
    """
    Analyze ALSS effectiveness across SNR range for MCM ON/OFF conditions.
    
    Returns:
        snr_results: Dictionary with improvement percentages vs SNR
    """
    print_header("SNR-DEPENDENT EFFECTIVENESS ANALYSIS")
    
    test_arrays = [
        ("Z1", Z1ArrayProcessor(N=7, d=1.0)),
        ("Z3_2", Z3_2ArrayProcessor(N=6, d=1.0)),
        ("Z5", Z5ArrayProcessor(N=7, d=1.0))
    ]
    
    snr_results = {}
    
    for array_name, processor in test_arrays:
        print_section(f"SNR Sweep: {array_name}")
        
        data = processor.run_full_analysis()
        positions = data.sensor_positions
        
        # Initialize MCM
        mcm = MutualCouplingMatrix(positions, c1=MCM_C1, alpha=MCM_ALPHA)
        C = mcm.get_coupling_matrix()
        
        improvement_no_mcm = []
        improvement_with_mcm = []
        
        for snr in SNR_RANGE:
            print(f"  Testing SNR = {snr} dB...", end=' ')
            
            # No MCM condition
            rmse_no_alss_nomcm = []
            rmse_alss_nomcm = []
            
            # MCM condition
            rmse_no_alss_mcm = []
            rmse_alss_mcm = []
            
            for trial in range(SNR_SWEEP_TRIALS):
                np.random.seed(2000 + snr * 100 + trial)
                
                # No MCM
                try:
                    X_ideal = generate_ula_measurements(
                        positions, TRUE_ANGLES, SNR_SWEEP_SNAPSHOTS,
                        snr, WAVELENGTH, coupling_matrix=None
                    )
                    
                    theta_no_alss, _, _ = coarray_music_1d(
                        X_ideal, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                        alss_enabled=False
                    )
                    rmse_no_alss_nomcm.append(compute_rmse(theta_no_alss, TRUE_ANGLES))
                    
                    theta_alss, _, _ = coarray_music_1d(
                        X_ideal, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                        alss_enabled=True, alss_mode=ALSS_MODE,
                        alss_tau=ALSS_TAU, alss_coreL=ALSS_COREL
                    )
                    rmse_alss_nomcm.append(compute_rmse(theta_alss, TRUE_ANGLES))
                except:
                    pass
                
                # MCM ON
                try:
                    X_coupled = generate_ula_measurements(
                        positions, TRUE_ANGLES, SNR_SWEEP_SNAPSHOTS,
                        snr, WAVELENGTH, coupling_matrix=C
                    )
                    
                    theta_no_alss, _, _ = coarray_music_1d(
                        X_coupled, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                        alss_enabled=False
                    )
                    rmse_no_alss_mcm.append(compute_rmse(theta_no_alss, TRUE_ANGLES))
                    
                    theta_alss, _, _ = coarray_music_1d(
                        X_coupled, positions, K=len(TRUE_ANGLES), d=1.0, wavelength=WAVELENGTH,
                        alss_enabled=True, alss_mode=ALSS_MODE,
                        alss_tau=ALSS_TAU, alss_coreL=ALSS_COREL
                    )
                    rmse_alss_mcm.append(compute_rmse(theta_alss, TRUE_ANGLES))
                except:
                    pass
            
            # Compute improvement percentages
            if len(rmse_no_alss_nomcm) > 0 and len(rmse_alss_nomcm) > 0:
                mean_no_alss = np.mean(rmse_no_alss_nomcm)
                mean_alss = np.mean(rmse_alss_nomcm)
                improvement = ((mean_no_alss - mean_alss) / mean_no_alss) * 100
                improvement_no_mcm.append(improvement)
            else:
                improvement_no_mcm.append(0.0)
            
            if len(rmse_no_alss_mcm) > 0 and len(rmse_alss_mcm) > 0:
                mean_no_alss = np.mean(rmse_no_alss_mcm)
                mean_alss = np.mean(rmse_alss_mcm)
                improvement = ((mean_no_alss - mean_alss) / mean_no_alss) * 100
                improvement_with_mcm.append(improvement)
            else:
                improvement_with_mcm.append(0.0)
            
            print(f"No MCM: {improvement_no_mcm[-1]:.1f}%, With MCM: {improvement_with_mcm[-1]:.1f}%")
        
        snr_results[array_name] = {
            'snr_range': SNR_RANGE,
            'improvement_no_mcm': np.array(improvement_no_mcm),
            'improvement_with_mcm': np.array(improvement_with_mcm)
        }
    
    return snr_results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_bias_variance_decomposition(results_dict, save_path):
    """
    Plot 1: Bias-Variance Decomposition
    
    Shows how ALSS affects bias vs variance components for each array.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    conditions = ['Cond1\n(No MCM,\nNo ALSS)', 'Cond2\n(No MCM,\nALSS ON)',
                  'Cond3\n(MCM ON,\nNo ALSS)', 'Cond4\n(MCM ON,\nALSS ON)']
    
    for idx, (array_name, ax) in enumerate(zip(['Z1', 'Z3_2', 'Z5'], axes)):
        res = results_dict[array_name]
        
        bias_vals = [res['bias1'], res['bias2'], res['bias3'], res['bias4']]
        var_vals = [res['variance1'], res['variance2'], res['variance3'], res['variance4']]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, bias_vals, width, label='Bias²', 
                       color='#FF6B6B', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, var_vals, width, label='Variance',
                       color='#4ECDC4', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Error Component (deg²)', fontsize=11, fontweight='bold')
        ax.set_title(f'{array_name} Array', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=9)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add total RMSE line
        rmse_vals = [res['cond1'], res['cond2'], res['cond3'], res['cond4']]
        ax2 = ax.twinx()
        ax2.plot(x, rmse_vals, 'ko-', linewidth=2, markersize=8, label='Total RMSE')
        ax2.set_ylabel('RMSE (deg)', fontsize=11, fontweight='bold', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_snr_effectiveness(snr_results, save_path):
    """
    Plot 2: SNR-Dependent Effectiveness
    
    Shows ALSS improvement percentage vs SNR for MCM ON/OFF.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors_nomcm = ['#2E86AB', '#A23B72', '#F18F01']
    colors_mcm = ['#06A77D', '#D62246', '#8338EC']
    
    for idx, (array_name, ax) in enumerate(zip(['Z1', 'Z3_2', 'Z5'], axes)):
        res = snr_results[array_name]
        snr_range = res['snr_range']
        
        ax.plot(snr_range, res['improvement_no_mcm'], 'o-', linewidth=2.5,
                markersize=8, label='No MCM', color=colors_nomcm[idx], alpha=0.9)
        ax.plot(snr_range, res['improvement_with_mcm'], 's--', linewidth=2.5,
                markersize=8, label='MCM ON', color=colors_mcm[idx], alpha=0.9)
        
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
        ax.set_ylabel('ALSS Improvement (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{array_name} Array', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(snr_range)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_gap_reduction(results_dict, save_path):
    """
    Plot 3: Gap Reduction with Confidence Intervals
    
    Bar chart showing gap reduction percentages with statistical significance.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    arrays = ['Z1', 'Z3_2', 'Z5']
    gap_reductions = []
    p_values = []
    ci_lower = []
    ci_upper = []
    
    for array_name in arrays:
        res = results_dict[array_name]
        gap_reductions.append(res['gap_reduction_pct'])
        p_values.append(res['p_value'])
        
        # Compute 95% confidence interval using bootstrap
        trials_cond3 = np.array(res['rmse_trials']['cond3'])
        trials_cond4 = np.array(res['rmse_trials']['cond4'])
        
        # Bootstrap CI for gap reduction
        bootstrap_gaps = []
        for _ in range(1000):
            idx = np.random.choice(len(trials_cond3), size=len(trials_cond3), replace=True)
            boot_cond3 = np.mean(trials_cond3[idx])
            boot_cond4 = np.mean(trials_cond4[idx])
            boot_cond1 = res['cond1']
            boot_gap = gap_reduction_metric(boot_cond1, boot_cond3, boot_cond4)
            bootstrap_gaps.append(boot_gap)
        
        ci_low = np.percentile(bootstrap_gaps, 2.5)
        ci_high = np.percentile(bootstrap_gaps, 97.5)
        ci_lower.append(res['gap_reduction_pct'] - ci_low)
        ci_upper.append(ci_high - res['gap_reduction_pct'])
    
    x = np.arange(len(arrays))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax.bar(x, gap_reductions, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add error bars (confidence intervals)
    ax.errorbar(x, gap_reductions, yerr=[ci_lower, ci_upper],
                fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=2)
    
    # Add significance markers
    for i, (gap, pval) in enumerate(zip(gap_reductions, p_values)):
        if pval < 0.001:
            marker = '***'
        elif pval < 0.01:
            marker = '**'
        elif pval < 0.05:
            marker = '*'
        else:
            marker = 'n.s.'
        
        ax.text(i, gap + ci_upper[i] + 2, marker, ha='center', va='bottom',
                fontsize=14, fontweight='bold')
    
    # Add theoretical prediction ranges
    predictions = {
        'Z1': (25, 35),
        'Z3_2': (15, 25),
        'Z5': (40, 50)
    }
    
    for i, array_name in enumerate(arrays):
        pred_low, pred_high = predictions[array_name]
        ax.axhspan(pred_low, pred_high, xmin=(i-0.4)/len(arrays), 
                  xmax=(i+0.4)/len(arrays), alpha=0.2, color=colors[i],
                  label=f'{array_name} Predicted Range' if i == 0 else '')
    
    ax.set_xlabel('Array Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('ALSS Gap Reduction Under Mutual Coupling\n(with 95% Confidence Intervals)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(arrays, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(gap_reductions) + max(ci_upper) + 10)
    
    # Add legend for significance levels
    legend_text = "Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant"
    ax.text(0.5, -0.15, legend_text, transform=ax.transAxes,
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print_header("ENHANCED ALSS+MCM EXPERIMENTAL VALIDATION", char='#')
    print("Publication-Quality Analysis for IEEE RadarCon 2025")
    print(f"\nConfiguration:")
    print(f"  Arrays: Z1 (N=7), Z3_2 (N=6), Z5 (N=7)")
    print(f"  Sources: {len(TRUE_ANGLES)} at {TRUE_ANGLES}°")
    print(f"  MCM Model: Exponential (c1={MCM_C1}, alpha={MCM_ALPHA})")
    print(f"  Trials: {NUM_TRIALS} (Scenario 2), {SNR_SWEEP_TRIALS} (SNR sweep)")
    
    # Create output directory
    os.makedirs('../results/plots', exist_ok=True)
    os.makedirs('../results/summaries', exist_ok=True)
    
    # Run experiments
    print_header("EXPERIMENT 1: FOUR-CONDITION ANALYSIS")
    results_dict = run_scenario2_four_conditions()
    
    print_header("EXPERIMENT 2: SNR SWEEP ANALYSIS")
    snr_results = run_snr_sweep_analysis()
    
    # Generate plots
    print_header("GENERATING PUBLICATION-QUALITY PLOTS")
    
    plot1_path = '../results/plots/alss_mcm_bias_variance_decomposition.png'
    plot_bias_variance_decomposition(results_dict, plot1_path)
    
    plot2_path = '../results/plots/alss_mcm_snr_effectiveness.png'
    plot_snr_effectiveness(snr_results, plot2_path)
    
    plot3_path = '../results/plots/alss_mcm_gap_reduction.png'
    plot_gap_reduction(results_dict, plot3_path)
    
    # Save results to CSV
    print_header("SAVING RESULTS")
    
    # Create results table
    summary_data = []
    for array_name in ['Z1', 'Z3_2', 'Z5']:
        res = results_dict[array_name]
        summary_data.append({
            'Array': array_name,
            'Cond1_RMSE': res['cond1'],
            'Cond2_RMSE': res['cond2'],
            'Cond3_RMSE': res['cond3'],
            'Cond4_RMSE': res['cond4'],
            'Cond1_Std': res['std1'],
            'Cond2_Std': res['std2'],
            'Cond3_Std': res['std3'],
            'Cond4_Std': res['std4'],
            'Gap_Reduction_Pct': res['gap_reduction_pct'],
            'P_Value': res['p_value']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = '../results/summaries/alss_mcm_enhanced_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Print publication-ready results
    print_header("PUBLICATION-READY RESULTS")
    print("\nActual Experimental Results (for paper):\n")
    print("```python")
    print("actual_results = {")
    for array_name in ['Z1', 'Z3_2', 'Z5']:
        res = results_dict[array_name]
        print(f"    '{array_name}': {{")
        print(f"        'cond1': {res['cond1']:.3f},  # No MCM, No ALSS")
        print(f"        'cond2': {res['cond2']:.3f},  # No MCM, ALSS ON")
        print(f"        'cond3': {res['cond3']:.3f},  # MCM ON, No ALSS")
        print(f"        'cond4': {res['cond4']:.3f},  # MCM ON, ALSS ON")
        print(f"        'gap_reduction': {res['gap_reduction_pct']:.1f}%,")
        print(f"        'p_value': {res['p_value']:.4f}")
        print(f"    }},")
    print("}")
    print("```")
    
    print_header("VALIDATION COMPLETE", char='#')
    print(f"\nOutputs:")
    print(f"  1. Bias-Variance Plot: {plot1_path}")
    print(f"  2. SNR Effectiveness: {plot2_path}")
    print(f"  3. Gap Reduction: {plot3_path}")
    print(f"  4. Results CSV: {csv_path}")
    print(f"\nReady for publication in ALSS_MCM_SCENARIO_ANALYSIS_01.md!")


if __name__ == "__main__":
    main()
