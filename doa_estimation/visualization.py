"""
DOA Estimation Visualization
==============================

Plotting utilities for DOA estimation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple, Union


def plot_doa_spectrum(
    angles: np.ndarray,
    spectrum: np.ndarray,
    true_angles: Optional[List[float]] = None,
    estimated_angles: Optional[List[float]] = None,
    title: str = "MUSIC Spectrum",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot MUSIC pseudo-spectrum with true and estimated DOAs.
    
    Parameters
    ----------
    angles : np.ndarray
        Angle grid in degrees.
        
    spectrum : np.ndarray
        MUSIC spectrum values (dB).
        
    true_angles : list of float, optional
        True DOA angles in degrees.
        
    estimated_angles : list of float, optional
        Estimated DOA angles in degrees.
        
    title : str
        Plot title.
        
    figsize : tuple
        Figure size (width, height).
        
    save_path : str, optional
        Path to save figure.
        
    Examples
    --------
    >>> from doa_estimation import MUSICEstimator
    >>> from doa_estimation.visualization import plot_doa_spectrum
    >>> 
    >>> estimator = MUSICEstimator(positions, wavelength=1.0)
    >>> X = estimator.simulate_signals([-30, 10, 45], SNR_dB=10, snapshots=200)
    >>> angles_est, spectrum = estimator.estimate(X, K_sources=3, return_spectrum=True)
    >>> 
    >>> plot_doa_spectrum(
    >>>     estimator.angles, spectrum,
    >>>     true_angles=[-30, 10, 45],
    >>>     estimated_angles=angles_est,
    >>>     title="Z5 Array DOA Estimation"
    >>> )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectrum
    ax.plot(angles, spectrum, 'b-', linewidth=2, label='MUSIC Spectrum')
    
    # Mark true angles
    if true_angles is not None:
        y_max = np.max(spectrum)
        for angle in true_angles:
            ax.axvline(angle, color='g', linestyle='--', linewidth=2,
                      alpha=0.7, label='True DOA' if angle == true_angles[0] else '')
    
    # Mark estimated angles
    if estimated_angles is not None:
        for angle in estimated_angles:
            # Find spectrum value at this angle
            idx = np.argmin(np.abs(angles - angle))
            ax.plot(angle, spectrum[idx], 'ro', markersize=10, 
                   label='Estimated DOA' if angle == estimated_angles[0] else '')
    
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('MUSIC Spectrum (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectrum plot to {save_path}")
    
    plt.show()


def plot_array_geometry(
    sensor_positions: Union[List[float], np.ndarray],
    array_name: str = "Array Geometry",
    wavelength: float = 1.0,
    figsize: Tuple[int, int] = (10, 3),
    save_path: Optional[str] = None
):
    """
    Plot array geometry (sensor positions).
    
    Parameters
    ----------
    sensor_positions : array-like
        Sensor positions in wavelengths or meters.
        
    array_name : str
        Array name for title.
        
    wavelength : float
        Signal wavelength for display.
        
    figsize : tuple
        Figure size.
        
    save_path : str, optional
        Path to save figure.
    """
    positions = np.array(sensor_positions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot sensors
    ax.plot(positions, np.zeros_like(positions), 'bo', markersize=12, 
            label='Physical Sensors')
    
    # Add sensor numbers
    for i, pos in enumerate(positions):
        ax.text(pos, 0.1, f'{i+1}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel(f'Position (λ = {wavelength})', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'{array_name} - {len(positions)} Sensors', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.5, 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved geometry plot to {save_path}")
    
    plt.show()


def plot_estimation_results(
    true_angles: List[float],
    estimated_angles: List[float],
    rmse: float,
    title: str = "DOA Estimation Results",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot true vs estimated angles with error bars.
    
    Parameters
    ----------
    true_angles : list of float
        True DOA angles in degrees.
        
    estimated_angles : list of float
        Estimated DOA angles in degrees.
        
    rmse : float
        Root Mean Squared Error in degrees.
        
    title : str
        Plot title.
        
    figsize : tuple
        Figure size.
        
    save_path : str, optional
        Path to save figure.
    """
    true = np.array(true_angles)
    est = np.array(estimated_angles)
    
    # Match angles
    errors = []
    used = set()
    matched_true = []
    matched_est = []
    
    for t in true:
        candidates = [(abs(t - e), i, e) 
                     for i, e in enumerate(est) if i not in used]
        if candidates:
            _, idx, e = min(candidates)
            matched_true.append(t)
            matched_est.append(e)
            errors.append(t - e)
            used.add(idx)
    
    matched_true = np.array(matched_true)
    matched_est = np.array(matched_est)
    errors = np.array(errors)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: True vs Estimated
    ax1.plot([-90, 90], [-90, 90], 'k--', alpha=0.3, label='Perfect Estimation')
    ax1.plot(matched_true, matched_est, 'ro', markersize=10, label='Estimates')
    
    for t, e in zip(matched_true, matched_est):
        ax1.plot([t, t], [t, e], 'b-', alpha=0.5)
    
    ax1.set_xlabel('True Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Estimated Angle (degrees)', fontsize=12)
    ax1.set_title('True vs Estimated DOA', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.axis('equal')
    
    # Right plot: Error distribution
    ax2.bar(range(len(errors)), errors, color='skyblue', edgecolor='navy')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(rmse, color='r', linestyle='--', linewidth=2, 
               label=f'RMSE = {rmse:.2f}°')
    ax2.axhline(-rmse, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Source Index', fontsize=12)
    ax2.set_ylabel('Error (degrees)', fontsize=12)
    ax2.set_title('Estimation Errors', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved results plot to {save_path}")
    
    plt.show()


def plot_music_comparison(
    results_dict: Dict[str, Dict],
    metric: str = 'rmse',
    xlabel: str = 'SNR (dB)',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Compare MUSIC performance across different arrays or conditions.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with format:
        {
            'array_name': {
                'x_values': [0, 5, 10, 15, 20],  # e.g., SNR values
                'metric': [5.2, 2.1, 0.8, 0.4, 0.2]  # e.g., RMSE values
            }
        }
        
    metric : str
        Metric name for y-axis label (e.g., 'rmse', 'mae', 'bias').
        
    xlabel : str
        X-axis label.
        
    figsize : tuple
        Figure size.
        
    save_path : str, optional
        Path to save figure.
        
    Examples
    --------
    >>> results = {
    >>>     'ULA (N=6)': {
    >>>         'x_values': [0, 5, 10, 15, 20],
    >>>         'rmse': [8.5, 3.2, 1.1, 0.5, 0.3]
    >>>     },
    >>>     'Nested (N1=2, N2=3)': {
    >>>         'x_values': [0, 5, 10, 15, 20],
    >>>         'rmse': [7.8, 2.8, 0.9, 0.4, 0.2]
    >>>     }
    >>> }
    >>> plot_music_comparison(results, metric='rmse', xlabel='SNR (dB)')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for i, (name, data) in enumerate(results_dict.items()):
        x = data['x_values']
        y = data[metric]
        marker = markers[i % len(markers)]
        
        ax.plot(x, y, marker=marker, markersize=8, linewidth=2,
               label=name, alpha=0.8)
    
    metric_labels = {
        'rmse': 'RMSE (degrees)',
        'mae': 'MAE (degrees)',
        'bias': 'Bias (degrees)',
        'max_error': 'Max Error (degrees)',
        'success_rate': 'Success Rate'
    }
    
    ylabel = metric_labels.get(metric, metric.upper())
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('MUSIC Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Log scale for RMSE/MAE if values span orders of magnitude
    if metric in ['rmse', 'mae', 'max_error']:
        all_values = [v for data in results_dict.values() for v in data[metric]]
        if max(all_values) / min(all_values) > 10:
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()
