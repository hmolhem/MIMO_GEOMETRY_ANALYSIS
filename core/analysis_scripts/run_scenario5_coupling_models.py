"""
SCENARIO 5: Coupling Model Comparison for ALSS Arrays
=======================================================

Purpose: Validate ALSS robustness across different mutual coupling models

This script implements comprehensive analysis to test whether ALSS benefits are
model-agnostic or specific to particular coupling formulations. Tests multiple
coupling models from simple to realistic, measuring performance consistency.

Primary Metrics:
- Model_Sensitivity_Index: Performance variation across coupling models
- Worst_Case_Improvement: Minimum improvement across all models  
- Model_Robustness_Ratio: std(improvement)/mean(improvement)
- Generalization_Gap: Performance difference between ideal vs realistic models

Author: MIMO Geometry Analysis Framework
Date: November 2025
Version: 1.0
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from time import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import geometry processors
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor
from geometry_processors.z6_processor import Z6ArrayProcessor

# Also need scipy for Hungarian matching
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


def get_array_positions(array_type: str, N: int, d: float) -> np.ndarray:
    """Get sensor positions for specified array type."""
    if array_type == 'ULA':
        proc = ULArrayProcessor(N=N, d=d)
        return np.array(proc.data.sensors_positions) * d
    elif array_type == 'Z5':
        proc = Z5ArrayProcessor(N=N, d=d)
        return np.array(proc.data.sensors_positions) * d
    elif array_type == 'Z6':
        proc = Z6ArrayProcessor(N=N, d=d)
        return np.array(proc.data.sensors_positions) * d
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def generate_coupling_matrix_simple(positions: np.ndarray, c1: float, model: str = 'exponential') -> np.ndarray:
    """
    Generate mutual coupling matrix using different models.
    
    Parameters:
    -----------
    positions : np.ndarray
        Sensor positions in wavelengths
    c1 : float
        Coupling strength parameter (0 to 1)
    model : str
        Coupling model type:
        - 'exponential': C_mn = c1 * exp(-|m-n|)
        - 'inverse': C_mn = c1 / (1 + |m-n|)
        - 'power': C_mn = c1 / (1 + |m-n|)^2
        - 'sinc': C_mn = c1 * sinc(π * d_mn)
        - 'gaussian': C_mn = c1 * exp(-(d_mn/λ)^2)
        - 'uniform': C_mn = c1 for adjacent, 0 otherwise
    
    Returns:
    --------
    C : np.ndarray
        Complex coupling matrix (N × N)
    """
    N = len(positions)
    C = np.eye(N, dtype=complex)
    
    for m in range(N):
        for n in range(N):
            if m != n:
                dist = np.abs(positions[m] - positions[n])
                index_diff = np.abs(m - n)
                
                if model == 'exponential':
                    # Traditional exponential decay
                    C[m, n] = c1 * np.exp(-index_diff)
                    
                elif model == 'inverse':
                    # Inverse distance coupling
                    C[m, n] = c1 / (1 + index_diff)
                    
                elif model == 'power':
                    # Power law decay
                    C[m, n] = c1 / ((1 + index_diff) ** 2)
                    
                elif model == 'sinc':
                    # Sinc function (electromagnetic wave coupling)
                    if dist > 0:
                        C[m, n] = c1 * np.sinc(np.pi * dist)
                    else:
                        C[m, n] = 0.0
                    
                elif model == 'gaussian':
                    # Gaussian decay (near-field coupling)
                    C[m, n] = c1 * np.exp(-(dist ** 2))
                    
                elif model == 'uniform':
                    # Uniform nearest-neighbor only
                    if index_diff == 1:
                        C[m, n] = c1
                    else:
                        C[m, n] = 0.0
                
                elif model == 'realistic':
                    # Realistic MCM with distance-dependent phase
                    # Combines amplitude decay with phase rotation
                    amplitude = c1 * np.exp(-0.5 * index_diff)
                    phase = 2 * np.pi * dist  # Phase depends on physical distance
                    C[m, n] = amplitude * np.exp(1j * phase)
                    
                elif model == 'none':
                    # No coupling (ideal case)
                    C[m, n] = 0.0
                    
                else:
                    raise ValueError(f"Unknown coupling model: {model}")
    
    return C


def music_doa_with_coupling(
    theta_true: np.ndarray,
    positions: np.ndarray,
    C: np.ndarray,
    M_snapshots: int = 256,
    snr_db: float = 10.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    MUSIC DOA estimation with mutual coupling.
    
    Returns:
    --------
    theta_est : np.ndarray
        Estimated DOAs (degrees)
    runtime : float
        Computation time (seconds)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_start = time()
    
    N = len(positions)
    K = len(theta_true)
    
    # Steering matrix (ideal)
    A = np.zeros((N, K), dtype=complex)
    for k in range(K):
        theta_rad = np.deg2rad(theta_true[k])
        A[:, k] = np.exp(-1j * 2 * np.pi * positions * np.sin(theta_rad))
    
    # Apply coupling to steering matrix
    A_coupled = C @ A
    
    # Generate snapshots
    s = np.random.randn(K, M_snapshots) + 1j * np.random.randn(K, M_snapshots)
    s = s / np.sqrt(2)
    
    # Noise
    sigma2 = 10 ** (-snr_db / 10)
    n = np.sqrt(sigma2 / 2) * (np.random.randn(N, M_snapshots) + 1j * np.random.randn(N, M_snapshots))
    
    # Received signal
    X = A_coupled @ s + n
    
    # Sample covariance
    R = (X @ X.conj().T) / M_snapshots
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvecs_sorted = eigvecs[:, idx_sorted]
    
    # Noise subspace
    Un = eigvecs_sorted[:, K:]
    
    # MUSIC spectrum (search grid with coupled steering)
    theta_grid = np.linspace(-90, 90, 1801)
    P_music = np.zeros(len(theta_grid))
    
    for i, theta in enumerate(theta_grid):
        theta_rad = np.deg2rad(theta)
        a = np.exp(-1j * 2 * np.pi * positions * np.sin(theta_rad))
        a_coupled = C @ a  # Apply coupling to steering vector
        
        denominator = np.abs(a_coupled.conj().T @ Un @ Un.conj().T @ a_coupled)
        P_music[i] = 1.0 / (denominator + 1e-10)
    
    # Peak detection
    peaks, _ = find_peaks(P_music, height=np.max(P_music) * 0.1)
    peak_powers = P_music[peaks]
    top_k_idx = np.argsort(peak_powers)[-K:]
    theta_est = theta_grid[peaks[top_k_idx]]
    theta_est = np.sort(theta_est)
    
    runtime = time() - t_start
    
    return theta_est, runtime


def compute_model_comparison_metrics(
    results_df: pd.DataFrame,
    baseline_array: str = 'ULA',
    alss_array: str = 'Z5'
) -> Dict:
    """
    Compute Scenario 5 metrics comparing performance across coupling models.
    
    Metrics:
    --------
    1. Model_Sensitivity_Index: std(RMSE) across models
    2. Worst_Case_Improvement: min(improvement) across all models
    3. Model_Robustness_Ratio: std(improvement)/mean(improvement)
    4. Generalization_Gap: Performance difference ideal vs realistic
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Columns: Model, Array, RMSE_mean, RMSE_std, ...
    baseline_array : str
        Baseline array name
    alss_array : str
        ALSS array name
        
    Returns:
    --------
    metrics : Dict
        Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Extract baseline and ALSS data
    baseline_df = results_df[results_df['Array'] == baseline_array].copy()
    alss_df = results_df[results_df['Array'] == alss_array].copy()
    
    # Ensure matching models
    common_models = set(baseline_df['Model']) & set(alss_df['Model'])
    baseline_df = baseline_df[baseline_df['Model'].isin(common_models)]
    alss_df = alss_df[alss_df['Model'].isin(common_models)]
    
    # Sort by model for alignment
    baseline_df = baseline_df.sort_values('Model').reset_index(drop=True)
    alss_df = alss_df.sort_values('Model').reset_index(drop=True)
    
    # 1. Model Sensitivity Index (lower is better)
    # Measures how much ALSS performance varies across models
    alss_rmse_values = alss_df['RMSE_mean'].values
    baseline_rmse_values = baseline_df['RMSE_mean'].values
    
    metrics['ALSS_Model_Sensitivity'] = np.std(alss_rmse_values)
    metrics['Baseline_Model_Sensitivity'] = np.std(baseline_rmse_values)
    metrics['Model_Sensitivity_Index'] = metrics['ALSS_Model_Sensitivity']
    
    # 2. Worst-Case Improvement (should be positive for ALSS benefit)
    improvements = []
    for model in common_models:
        baseline_rmse = baseline_df[baseline_df['Model'] == model]['RMSE_mean'].values[0]
        alss_rmse = alss_df[alss_df['Model'] == model]['RMSE_mean'].values[0]
        
        if baseline_rmse > 0:
            improvement_pct = (baseline_rmse - alss_rmse) / baseline_rmse * 100
        else:
            improvement_pct = 0.0
            
        improvements.append(improvement_pct)
    
    metrics['Worst_Case_Improvement_%'] = np.min(improvements)
    metrics['Best_Case_Improvement_%'] = np.max(improvements)
    metrics['Mean_Improvement_%'] = np.mean(improvements)
    metrics['Median_Improvement_%'] = np.median(improvements)
    
    # 3. Model Robustness Ratio (lower is better - more consistent)
    # Ratio of std to mean for improvements
    if metrics['Mean_Improvement_%'] != 0:
        metrics['Model_Robustness_Ratio'] = np.std(improvements) / np.abs(metrics['Mean_Improvement_%'])
    else:
        metrics['Model_Robustness_Ratio'] = np.inf
    
    metrics['Improvement_Consistency_%'] = (1 - np.std(improvements) / (np.abs(metrics['Mean_Improvement_%']) + 1e-10)) * 100
    metrics['Improvement_Consistency_%'] = np.clip(metrics['Improvement_Consistency_%'], 0, 100)
    
    # 4. Generalization Gap (ideal 'none' vs realistic models)
    try:
        # Ideal performance (no coupling)
        baseline_ideal = baseline_df[baseline_df['Model'] == 'none']['RMSE_mean'].values[0]
        alss_ideal = alss_df[alss_df['Model'] == 'none']['RMSE_mean'].values[0]
        
        # Realistic performance (average over realistic models)
        realistic_models = ['exponential', 'sinc', 'gaussian', 'realistic']
        baseline_realistic_values = []
        alss_realistic_values = []
        
        for model in realistic_models:
            if model in common_models:
                baseline_realistic_values.append(
                    baseline_df[baseline_df['Model'] == model]['RMSE_mean'].values[0]
                )
                alss_realistic_values.append(
                    alss_df[alss_df['Model'] == model]['RMSE_mean'].values[0]
                )
        
        baseline_realistic = np.mean(baseline_realistic_values)
        alss_realistic = np.mean(alss_realistic_values)
        
        # Generalization gap: degradation from ideal to realistic
        metrics['Baseline_Generalization_Gap_%'] = (baseline_realistic - baseline_ideal) / (baseline_ideal + 1e-10) * 100
        metrics['ALSS_Generalization_Gap_%'] = (alss_realistic - alss_ideal) / (alss_ideal + 1e-10) * 100
        metrics['Generalization_Gap'] = metrics['ALSS_Generalization_Gap_%']
        
        # Relative gap reduction (ALSS should have smaller gap)
        if metrics['Baseline_Generalization_Gap_%'] != 0:
            metrics['Gap_Reduction_%'] = (
                (metrics['Baseline_Generalization_Gap_%'] - metrics['ALSS_Generalization_Gap_%']) / 
                np.abs(metrics['Baseline_Generalization_Gap_%']) * 100
            )
        else:
            metrics['Gap_Reduction_%'] = 0.0
            
    except (IndexError, KeyError):
        # If 'none' model not available
        metrics['Baseline_Generalization_Gap_%'] = np.nan
        metrics['ALSS_Generalization_Gap_%'] = np.nan
        metrics['Generalization_Gap'] = np.nan
        metrics['Gap_Reduction_%'] = np.nan
    
    # 5. Additional metrics
    metrics['Num_Models_Tested'] = len(common_models)
    metrics['Models_With_Improvement'] = np.sum(np.array(improvements) > 0)
    metrics['Models_With_Degradation'] = np.sum(np.array(improvements) < 0)
    
    # Store per-model improvements for detailed analysis
    metrics['Per_Model_Improvements'] = {
        model: imp for model, imp in zip(common_models, improvements)
    }
    
    return metrics


def run_model_comparison(
    coupling_strength: float = 0.3,
    models: List[str] = None,
    arrays: List[str] = None,
    N_sensors: int = 7,
    d_spacing: float = 0.5,
    snr_db: float = 10.0,
    M_snapshots: int = 256,
    trials: int = 500,
    theta_true: np.ndarray = np.array([-20.0, 10.0]),
    output_dir: str = 'results/scenario5_coupling_models',
    seed_offset: int = 50000
) -> Tuple[pd.DataFrame, Dict]:
    """
    Experiment 5A: Compare ALSS performance across different coupling models.
    
    Parameters:
    -----------
    coupling_strength : float
        Fixed coupling strength for all models
    models : List[str]
        List of coupling models to test
    arrays : List[str]
        List of array types to test
    trials : int
        Number of Monte Carlo trials per configuration
        
    Returns:
    --------
    results_df : pd.DataFrame
        Results for each array × model combination
    metrics : Dict
        Scenario 5 metrics
    """
    if models is None:
        models = ['none', 'exponential', 'inverse', 'power', 'sinc', 'gaussian', 'uniform', 'realistic']
    
    if arrays is None:
        arrays = ['ULA', 'Z5']
    
    print("\n" + "="*80)
    print("SCENARIO 5A: Coupling Model Comparison")
    print("="*80)
    print(f"Arrays: {', '.join(arrays)}")
    print(f"Coupling models: {', '.join(models)}")
    print(f"Coupling strength: c1 = {coupling_strength}")
    print(f"SNR: {snr_db} dB")
    print(f"Snapshots: {M_snapshots}")
    print(f"Trials per configuration: {trials}")
    print(f"Total configurations: {len(arrays) * len(models)}")
    print("="*80 + "\n")
    
    # Storage for results
    results_list = []
    
    # Run experiments
    for array_type in arrays:
        print(f"\nArray: {array_type}")
        print("-" * 40)
        
        # Get sensor positions
        positions = get_array_positions(array_type, N_sensors, d_spacing)
        N_virt = len(np.unique([positions[i] - positions[j] for i in range(len(positions)) for j in range(len(positions))]))
        
        for model in models:
            print(f"  Model: {model:12s} ", end='', flush=True)
            t_start = time()
            
            # Generate coupling matrix for this model
            C = generate_coupling_matrix_simple(positions, coupling_strength, model=model)
            
            # Run trials
            rmse_list = []
            runtime_list = []
            resolution_list = []
            
            for trial in range(trials):
                seed = seed_offset + trial + len(array_type) * 1000 + len(model) * 100
                seed = abs(seed) % (2**32 - 1)
                
                theta_est, runtime = music_doa_with_coupling(
                    theta_true=theta_true,
                    positions=positions,
                    C=C,
                    M_snapshots=M_snapshots,
                    snr_db=snr_db,
                    seed=seed
                )
                
                # Hungarian matching for RMSE
                if len(theta_est) == len(theta_true):
                    cost_matrix = np.abs(theta_true[:, np.newaxis] - theta_est[np.newaxis, :])
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    rmse = np.sqrt(np.mean((theta_true[row_ind] - theta_est[col_ind]) ** 2))
                    resolution_success = 1
                else:
                    rmse = 90.0  # Failure
                    resolution_success = 0
                
                rmse_list.append(rmse)
                runtime_list.append(runtime)
                resolution_list.append(resolution_success)
            
            # Compute statistics
            rmse_mean = np.mean(rmse_list)
            rmse_std = np.std(rmse_list)
            resolution_rate = np.mean(resolution_list) * 100
            runtime_mean = np.mean(runtime_list)
            
            elapsed = time() - t_start
            print(f"Done! ({elapsed:.1f}s) RMSE={rmse_mean:.3f}° (res={resolution_rate:.0f}%)")
            
            # Store results
            results_list.append({
                'Array': array_type,
                'Model': model,
                'Coupling_Strength': coupling_strength,
                'RMSE_mean': rmse_mean,
                'RMSE_std': rmse_std,
                'Resolution_Rate_%': resolution_rate,
                'Runtime_mean': runtime_mean,
                'Runtime_std': np.std(runtime_list),
                'N_sensors': N_sensors,
                'N_virtual': N_virt,
                'Valid_Trials': trials
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Compute Scenario 5 metrics
    print("\n" + "="*80)
    print("Computing Scenario 5 Metrics...")
    print("="*80)
    
    metrics = compute_model_comparison_metrics(
        results_df,
        baseline_array='ULA',
        alss_array='Z5' if 'Z5' in arrays else arrays[-1]
    )
    
    # Print metrics summary
    print(f"\nModel Sensitivity Index: {metrics['Model_Sensitivity_Index']:.4f} mdeg")
    print(f"Worst-Case Improvement: {metrics['Worst_Case_Improvement_%']:.1f}%")
    print(f"Mean Improvement: {metrics['Mean_Improvement_%']:.1f}%")
    print(f"Model Robustness Ratio: {metrics['Model_Robustness_Ratio']:.3f}")
    print(f"Improvement Consistency: {metrics['Improvement_Consistency_%']:.1f}%")
    print(f"Generalization Gap (ALSS): {metrics['ALSS_Generalization_Gap_%']:.1f}%")
    print(f"Gap Reduction: {metrics['Gap_Reduction_%']:.1f}%")
    print(f"\nModels with improvement: {metrics['Models_With_Improvement']}/{metrics['Num_Models_Tested']}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'scenario5a_model_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'scenario5a_metrics.csv')
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    return results_df, metrics


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: Dict,
    output_dir: str = 'results/scenario5_coupling_models'
) -> None:
    """
    Generate comprehensive visualization for Scenario 5.
    
    Creates 2×3 panel figure:
    (a) RMSE by model (grouped bar chart)
    (b) Improvement by model (horizontal bars)
    (c) Model sensitivity comparison (box plot)
    (d) Generalization gap (ideal vs realistic)
    (e) Robustness ratio (scatter)
    (f) Worst-case analysis (waterfall)
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Extract data
    models = results_df['Model'].unique()
    arrays = results_df['Array'].unique()
    
    # Panel (a): RMSE by model
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(models))
    width = 0.35
    
    for i, array in enumerate(arrays):
        array_data = results_df[results_df['Array'] == array].sort_values('Model')
        rmse_values = array_data['RMSE_mean'].values
        ax1.bar(x + i * width, rmse_values, width, label=array, alpha=0.8)
    
    ax1.set_xlabel('Coupling Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('RMSE (degrees)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Performance Across Coupling Models', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel (b): Improvement by model
    ax2 = plt.subplot(2, 3, 2)
    improvements = []
    model_labels = []
    
    baseline_array = 'ULA'
    alss_array = 'Z5' if 'Z5' in arrays else arrays[-1]
    
    for model in models:
        baseline_rmse = results_df[(results_df['Array'] == baseline_array) & 
                                   (results_df['Model'] == model)]['RMSE_mean'].values[0]
        alss_rmse = results_df[(results_df['Array'] == alss_array) & 
                               (results_df['Model'] == model)]['RMSE_mean'].values[0]
        
        if baseline_rmse > 0:
            improvement = (baseline_rmse - alss_rmse) / baseline_rmse * 100
        else:
            improvement = 0.0
        
        improvements.append(improvement)
        model_labels.append(model)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.barh(model_labels, improvements, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('ALSS Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'(b) {alss_array} vs {baseline_array} Improvement', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add worst-case line
    worst_case = metrics['Worst_Case_Improvement_%']
    ax2.axvline(worst_case, color='darkred', linestyle='--', linewidth=2, 
                label=f'Worst-case: {worst_case:.1f}%')
    ax2.legend(fontsize=9)
    
    # Panel (c): Model sensitivity (box plot simulation)
    ax3 = plt.subplot(2, 3, 3)
    sensitivity_data = []
    labels = []
    
    for array in arrays:
        array_data = results_df[results_df['Array'] == array]['RMSE_mean'].values
        sensitivity_data.append(array_data)
        labels.append(array)
    
    bp = ax3.boxplot(sensitivity_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('RMSE (degrees)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Model Sensitivity Distribution', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add sensitivity index annotation
    alss_sensitivity = metrics['Model_Sensitivity_Index']
    ax3.text(0.98, 0.98, f'ALSS Sensitivity:\n{alss_sensitivity:.4f}°', 
             transform=ax3.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel (d): Generalization gap
    ax4 = plt.subplot(2, 3, 4)
    
    gap_data = {
        'Baseline\n(ULA)': [
            metrics['Baseline_Generalization_Gap_%']
        ],
        'ALSS\n(Z5)': [
            metrics['ALSS_Generalization_Gap_%']
        ]
    }
    
    x_pos = np.arange(len(gap_data))
    gaps = [gap_data[k][0] for k in gap_data.keys()]
    colors_gap = ['indianred', 'lightgreen']
    
    bars = ax4.bar(x_pos, gaps, color=colors_gap, alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(gap_data.keys(), fontsize=10)
    ax4.set_ylabel('Generalization Gap (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Ideal vs Realistic Performance Gap', fontsize=12, fontweight='bold')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.1f}%', ha='center', va='bottom' if gap > 0 else 'top', fontsize=10)
    
    # Add gap reduction annotation
    gap_reduction = metrics['Gap_Reduction_%']
    ax4.text(0.5, 0.02, f'Gap Reduction: {gap_reduction:.1f}%', 
             transform=ax4.transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=11, fontweight='bold')
    
    # Panel (e): Robustness ratio scatter
    ax5 = plt.subplot(2, 3, 5)
    
    for array in arrays:
        array_data = results_df[results_df['Array'] == array]
        rmse_means = array_data['RMSE_mean'].values
        rmse_stds = array_data['RMSE_std'].values
        
        ax5.scatter(rmse_means, rmse_stds, s=100, alpha=0.6, label=array)
    
    ax5.set_xlabel('Mean RMSE (degrees)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Std Dev RMSE (degrees)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) Robustness: Mean vs Variability', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    
    # Add robustness ratio annotation
    robustness_ratio = metrics['Model_Robustness_Ratio']
    consistency = metrics['Improvement_Consistency_%']
    ax5.text(0.98, 0.02, 
             f'Robustness Ratio: {robustness_ratio:.3f}\nConsistency: {consistency:.1f}%', 
             transform=ax5.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=10)
    
    # Panel (f): Worst-case waterfall
    ax6 = plt.subplot(2, 3, 6)
    
    # Sort improvements
    sorted_improvements = sorted(zip(model_labels, improvements), key=lambda x: x[1])
    sorted_models = [x[0] for x in sorted_improvements]
    sorted_imps = [x[1] for x in sorted_improvements]
    
    colors_waterfall = ['darkgreen' if imp > 0 else 'darkred' for imp in sorted_imps]
    bars = ax6.barh(range(len(sorted_models)), sorted_imps, color=colors_waterfall, alpha=0.7)
    
    ax6.set_yticks(range(len(sorted_models)))
    ax6.set_yticklabels(sorted_models, fontsize=9)
    ax6.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax6.set_title('(f) Worst-to-Best Case Analysis', fontsize=12, fontweight='bold')
    ax6.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(axis='x', alpha=0.3)
    
    # Highlight worst-case
    worst_idx = sorted_imps.index(min(sorted_imps))
    bars[worst_idx].set_edgecolor('red')
    bars[worst_idx].set_linewidth(3)
    
    # Add worst-case annotation
    ax6.text(0.02, 0.98, 
             f'Worst-case: {min(sorted_imps):.1f}%\n' +
             f'Best-case: {max(sorted_imps):.1f}%\n' +
             f'Range: {max(sorted_imps) - min(sorted_imps):.1f}%',
             transform=ax6.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'scenario5a_model_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {fig_path}")


def main():
    """Main execution function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='SCENARIO 5: Coupling Model Comparison for ALSS Arrays',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (50 trials, 4 models)
  python run_scenario5_coupling_models.py --trials 50 --models none exponential sinc realistic
  
  # Full production run (500 trials, all 8 models)
  python run_scenario5_coupling_models.py --trials 500 --arrays ULA Z5 Z6
  
  # Custom coupling strength
  python run_scenario5_coupling_models.py --trials 500 --coupling-strength 0.4
        """
    )
    
    parser.add_argument('--trials', type=int, default=500,
                        help='Number of Monte Carlo trials per configuration (default: 500)')
    parser.add_argument('--coupling-strength', type=float, default=0.3,
                        help='Coupling strength c1 (default: 0.3)')
    parser.add_argument('--models', nargs='+', 
                        default=['none', 'exponential', 'inverse', 'power', 'sinc', 'gaussian', 'uniform', 'realistic'],
                        help='Coupling models to test (default: all 8)')
    parser.add_argument('--arrays', nargs='+', default=['ULA', 'Z5'],
                        help='Array types to compare (default: ULA Z5)')
    parser.add_argument('--snr', type=float, default=10.0,
                        help='SNR in dB (default: 10.0)')
    parser.add_argument('--snapshots', type=int, default=256,
                        help='Number of snapshots (default: 256)')
    parser.add_argument('--output-dir', type=str, default='results/scenario5_coupling_models',
                        help='Output directory (default: results/scenario5_coupling_models)')
    
    args = parser.parse_args()
    
    # Run experiment
    results_df, metrics = run_model_comparison(
        coupling_strength=args.coupling_strength,
        models=args.models,
        arrays=args.arrays,
        snr_db=args.snr,
        M_snapshots=args.snapshots,
        trials=args.trials,
        output_dir=args.output_dir
    )
    
    # Generate visualization
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80)
    plot_model_comparison(results_df, metrics, output_dir=args.output_dir)
    
    print("\n" + "="*80)
    print("SCENARIO 5 COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}/")
    print("  - scenario5a_model_comparison.csv (per-model results)")
    print("  - scenario5a_metrics.csv (aggregated metrics)")
    print("  - scenario5a_model_comparison.png (6-panel visualization)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
