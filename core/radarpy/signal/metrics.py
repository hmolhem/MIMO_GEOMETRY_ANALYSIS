"""
Performance Metrics Computation Module

Comprehensive metrics for DOA estimation evaluation across experimental scenarios.
Includes RMSE, Bias, CRB comparison, Resolution Rate, and Runtime analysis.

Author: MIMO Geometry Analysis Team
Date: November 6, 2025
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment


# ============================================================================
# EXISTING FUNCTIONS (Preserved for backward compatibility)
# ============================================================================

def angle_rmse_deg(y_true, y_pred):
    """
    Matches predictions to truths via Hungarian assignment on absolute angle error.
    Returns RMSE in degrees and per-pair errors (deg).
    """
    t = np.sort(np.asarray(y_true, dtype=float))
    p = np.sort(np.asarray(y_pred, dtype=float))
    if t.size == 0 or p.size == 0:
        return np.nan, np.array([])

    # Cost matrix (abs error)
    T, P = t.size, p.size
    C = np.zeros((T, P), dtype=float)
    for i in range(T):
        for j in range(P):
            C[i, j] = np.abs(t[i] - p[j])

    r, c = linear_sum_assignment(C)
    errs = C[r, c]
    rmse = np.sqrt(np.mean(errs**2))
    return rmse, errs

def resolved_indicator(y_true, y_pred, threshold_deg=1.0):
    """
    Returns 1 if:
      - number of predicted peaks >= number of true sources, and
      - all matched absolute errors <= threshold_deg
    else 0.
    """
    t = np.sort(np.asarray(y_true, dtype=float))
    p = np.sort(np.asarray(y_pred, dtype=float))
    if p.size < t.size or t.size == 0:
        return 0
    # Match
    rmse, errs = angle_rmse_deg(t, p[:t.size])
    return int(np.all(errs <= threshold_deg))


# ============================================================================
# COMPREHENSIVE METRICS FOR SCENARIO EVALUATION
# ============================================================================

def compute_rmse(estimated_doas: np.ndarray, true_doas: np.ndarray) -> float:
    """
    Compute Root Mean Square Error between estimated and true DOAs.
    
    Args:
        estimated_doas: Estimated DOA angles (K,) in degrees
        true_doas: True DOA angles (K,) in degrees
    
    Returns:
        rmse: RMSE value in degrees
    
    Usage:
        >>> est = np.array([14.8, -20.2])
        >>> true = np.array([15.0, -20.0])
        >>> rmse = compute_rmse(est, true)
        >>> print(f"RMSE: {rmse:.4f}°")
        RMSE: 0.2236°
    """
    # Reuse existing function for consistency
    rmse, _ = angle_rmse_deg(true_doas, estimated_doas)
    return rmse


def compute_bias(estimated_doas: np.ndarray, true_doas: np.ndarray) -> float:
    """
    Compute systematic bias (mean signed error) in DOA estimation.
    
    Args:
        estimated_doas: Estimated DOA angles (K,) in degrees
        true_doas: True DOA angles (K,) in degrees
    
    Returns:
        bias: Mean bias in degrees (signed)
    
    Usage:
        >>> est = np.array([15.2, -19.8])
        >>> true = np.array([15.0, -20.0])
        >>> bias = compute_bias(est, true)
        >>> print(f"Bias: {bias:.4f}°")
        Bias: 0.2000°
    """
    matched_est = match_doas(estimated_doas, true_doas)
    bias = np.mean(matched_est - true_doas)
    return bias


def compute_crb(sensor_positions: np.ndarray, wavelength: float, 
                doas_deg: np.ndarray, snr_lin: float, snapshots: int,
                coupling_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Cramér-Rao Bound for DOA estimation.
    
    Computes theoretical lower bound on estimation variance.
    
    Args:
        sensor_positions: Sensor positions (N,) in same units as wavelength
        wavelength: Signal wavelength
        doas_deg: True DOA angles (K,) in degrees
        snr_lin: Linear SNR (not dB)
        snapshots: Number of snapshots (M)
        coupling_matrix: Optional (N × N) mutual coupling matrix
    
    Returns:
        crb_std: CRB standard deviation for each DOA (K,) in degrees
    
    Note:
        Implements formula from Stoica & Nehorai (1989)
        CRB with coupling follows van Trees (2002)
    """
    N = len(sensor_positions)
    K = len(doas_deg)
    doas_rad = np.deg2rad(doas_deg)
    
    # Steering matrix
    A = np.zeros((N, K), dtype=complex)
    for k in range(K):
        # a(θ) = exp(-j * 2π * d * sin(θ) / λ)
        phase = -2 * np.pi * sensor_positions * np.sin(doas_rad[k]) / wavelength
        A[:, k] = np.exp(1j * phase)
    
    # Apply coupling if present
    if coupling_matrix is not None:
        A = coupling_matrix @ A
    
    # Derivative w.r.t. angle: dA/dθ (numerical)
    delta = 0.01  # Small angle perturbation in degrees
    dA_dtheta = np.zeros((N, K), dtype=complex)
    for k in range(K):
        theta_plus = doas_deg.copy()
        theta_plus[k] += delta
        theta_plus_rad = np.deg2rad(theta_plus)
        
        A_plus = np.zeros((N, K), dtype=complex)
        for kk in range(K):
            phase = -2 * np.pi * sensor_positions * np.sin(theta_plus_rad[kk]) / wavelength
            A_plus[:, kk] = np.exp(1j * phase)
        
        if coupling_matrix is not None:
            A_plus = coupling_matrix @ A_plus
        
        dA_dtheta[:, k] = (A_plus[:, k] - A[:, k]) / delta
    
    # Fisher Information Matrix (FIM)
    # FIM = 2 * M * SNR * Re{D^H * P_A_perp * D}
    # where D = dA/dθ, P_A_perp = I - A(A^H A)^{-1}A^H
    
    # Projection onto noise subspace
    AHA = A.conj().T @ A  # (K, K)
    try:
        AHA_inv = np.linalg.inv(AHA)
    except np.linalg.LinAlgError:
        warnings.warn("Singular steering matrix in CRB computation")
        return np.full(K, np.inf)
    
    P_A_perp = np.eye(N) - A @ AHA_inv @ A.conj().T  # (N, N)
    
    # FIM computation
    FIM = 2 * snapshots * snr_lin * np.real(dA_dtheta.conj().T @ P_A_perp @ dA_dtheta)  # (K, K)
    
    # CRB = inverse of FIM
    try:
        CRB = np.linalg.inv(FIM)  # (K, K)
        crb_var = np.diag(CRB)  # Extract variances
        crb_std = np.sqrt(np.abs(crb_var))  # Standard deviations in radians
        crb_std_deg = np.rad2deg(crb_std)  # Convert to degrees
    except np.linalg.LinAlgError:
        warnings.warn("Singular FIM in CRB computation")
        crb_std_deg = np.full(K, np.inf)
    
    return crb_std_deg


def compute_resolution_rate(estimated_doas_trials: List[np.ndarray], 
                            true_doas: np.ndarray,
                            threshold_deg: float = 3.0) -> float:
    """
    Compute resolution rate (percentage of correctly resolved sources).
    
    A trial is considered "resolved" if all estimated DOAs are within
    threshold distance from their true values.
    
    Args:
        estimated_doas_trials: List of estimated DOA arrays from multiple trials
        true_doas: True DOA angles (K,) in degrees
        threshold_deg: Maximum error threshold in degrees (default: 3°)
    
    Returns:
        resolution_rate: Percentage of resolved trials (0-100)
    
    Usage:
        >>> trials = [np.array([14.9, -20.1]), np.array([15.2, -19.8]), 
        ...           np.array([10.0, -25.0])]  # Last trial failed
        >>> true = np.array([15.0, -20.0])
        >>> rate = compute_resolution_rate(trials, true, threshold_deg=1.0)
        >>> print(f"Resolution rate: {rate:.1f}%")
        Resolution rate: 66.7%
    """
    num_trials = len(estimated_doas_trials)
    num_resolved = 0
    
    for est_doas in estimated_doas_trials:
        # Use existing resolved_indicator function
        if resolved_indicator(true_doas, est_doas, threshold_deg) == 1:
            num_resolved += 1
    
    resolution_rate = (num_resolved / num_trials) * 100.0
    return resolution_rate


def match_doas(estimated: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Match estimated DOAs to true DOAs using Hungarian algorithm.
    
    Handles permutation ambiguity in DOA estimation.
    
    Args:
        estimated: Estimated DOA angles (K,)
        true: True DOA angles (K,)
    
    Returns:
        matched_estimated: Permuted estimated DOAs to match true order (K,)
    """
    K = len(true)
    
    if len(estimated) != K:
        # Number mismatch - return as is (will result in large error)
        return estimated
    
    # Cost matrix: |est_i - true_j|
    cost_matrix = np.abs(estimated[:, np.newaxis] - true[np.newaxis, :])
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_estimated = estimated[row_ind]
    
    return matched_estimated


def compute_scenario_metrics(estimated_doas_trials: List[np.ndarray],
                             true_doas: np.ndarray,
                             sensor_positions: np.ndarray,
                             wavelength: float,
                             snr_db: float,
                             snapshots: int,
                             coupling_matrix: Optional[np.ndarray] = None,
                             resolution_threshold: float = 3.0,
                             runtimes_ms: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a single scenario.
    
    This is the main function for computing all metrics requested:
    - RMSE (degrees)
    - RMSE/CRB ratio (efficiency)
    - Resolution Rate (%)
    - Bias (degrees)
    - Runtime (ms)
    
    Args:
        estimated_doas_trials: List of estimated DOA arrays from multiple trials
        true_doas: True DOA angles (K,) in degrees
        sensor_positions: Sensor positions (N,)
        wavelength: Signal wavelength
        snr_db: SNR in dB
        snapshots: Number of snapshots (M)
        coupling_matrix: Optional (N × N) mutual coupling matrix
        resolution_threshold: Threshold for resolution detection (degrees)
        runtimes_ms: Optional list of runtimes for each trial (milliseconds)
    
    Returns:
        metrics: Dictionary containing all computed metrics
            - 'RMSE_degrees': RMSE in degrees
            - 'RMSE_CRB_ratio': RMSE / CRB (efficiency metric)
            - 'Resolution_Rate': Percentage of resolved trials
            - 'Bias_degrees': Mean bias in degrees
            - 'Runtime_ms': Mean runtime in milliseconds
    
    Usage:
        >>> # Run 100 trials
        >>> trials = []
        >>> runtimes = []
        >>> for _ in range(100):
        ...     start = time.time()
        ...     est = run_music(...)  # Your DOA estimation
        ...     runtime = (time.time() - start) * 1000  # Convert to ms
        ...     trials.append(est)
        ...     runtimes.append(runtime)
        >>> 
        >>> metrics = compute_scenario_metrics(
        ...     trials, true_doas, positions, wavelength, 
        ...     snr_db, snapshots, coupling_matrix, runtimes_ms=runtimes
        ... )
        >>> print(f"RMSE: {metrics['RMSE_degrees']:.4f}°")
        >>> print(f"RMSE/CRB: {metrics['RMSE_CRB_ratio']:.2f}")
        >>> print(f"Resolution: {metrics['Resolution_Rate']:.1f}%")
        >>> print(f"Bias: {metrics['Bias_degrees']:.4f}°")
        >>> print(f"Runtime: {metrics['Runtime_ms']:.2f} ms")
    """
    num_trials = len(estimated_doas_trials)
    snr_lin = 10 ** (snr_db / 10.0)
    
    # 1. Compute RMSE across all trials
    rmse_values = []
    bias_values = []
    
    for est_doas in estimated_doas_trials:
        if len(est_doas) == len(true_doas):
            rmse = compute_rmse(est_doas, true_doas)
            bias = compute_bias(est_doas, true_doas)
            if not np.isnan(rmse):
                rmse_values.append(rmse)
                bias_values.append(bias)
    
    rmse_mean = np.mean(rmse_values) if rmse_values else np.inf
    bias_mean = np.mean(bias_values) if bias_values else 0.0
    
    # 2. Compute CRB
    crb_std = compute_crb(sensor_positions, wavelength, true_doas, 
                         snr_lin, snapshots, coupling_matrix)
    crb_rmse = np.sqrt(np.mean(crb_std ** 2))  # Overall CRB RMSE
    
    # 3. Compute RMSE/CRB ratio
    if crb_rmse > 0 and np.isfinite(crb_rmse):
        rmse_crb_ratio = rmse_mean / crb_rmse
    else:
        rmse_crb_ratio = np.inf
    
    # 4. Compute Resolution Rate
    resolution_rate = compute_resolution_rate(estimated_doas_trials, true_doas, 
                                             resolution_threshold)
    
    # 5. Compute mean runtime
    if runtimes_ms is not None:
        runtime_mean = np.mean(runtimes_ms)
    else:
        runtime_mean = np.nan
    
    # Package results
    metrics = {
        'RMSE_degrees': rmse_mean,
        'RMSE_CRB_ratio': rmse_crb_ratio,
        'Resolution_Rate': resolution_rate,
        'Bias_degrees': bias_mean,
        'Runtime_ms': runtime_mean
    }
    
    return metrics


def run_trial_with_timing(run_function, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """
    Wrapper to run DOA estimation trial with timing measurement.
    
    Args:
        run_function: DOA estimation function to call
        *args: Positional arguments for run_function
        **kwargs: Keyword arguments for run_function
    
    Returns:
        estimated_doas: Estimated DOA angles (K,)
        runtime_ms: Execution time in milliseconds
    
    Usage:
        >>> from core.radarpy.signal.doa_sim_core import run_music
        >>> 
        >>> # Run single trial with timing
        >>> est_doas, runtime = run_trial_with_timing(
        ...     run_music, Rxx, positions, wavelength, scan_grid, K_sources,
        ...     coupling_matrix=C
        ... )
        >>> print(f"Estimated: {est_doas}, Runtime: {runtime:.2f} ms")
    """
    start_time = time.perf_counter()
    estimated_doas = run_function(*args, **kwargs)
    end_time = time.perf_counter()
    
    runtime_ms = (end_time - start_time) * 1000.0  # Convert to milliseconds
    
    return estimated_doas, runtime_ms


def print_metrics_summary(metrics: Dict[str, float], 
                         scenario_name: str = "Scenario",
                         show_crb_comparison: bool = True):
    """
    Pretty print metrics summary.
    
    Args:
        metrics: Dictionary from compute_scenario_metrics()
        scenario_name: Name of the scenario for display
        show_crb_comparison: Whether to show CRB comparison
    """
    print(f"\n{'='*60}")
    print(f"  {scenario_name}")
    print(f"{'='*60}")
    print(f"  RMSE:               {metrics['RMSE_degrees']:>8.4f}°")
    
    if show_crb_comparison and np.isfinite(metrics['RMSE_CRB_ratio']):
        print(f"  RMSE/CRB Ratio:     {metrics['RMSE_CRB_ratio']:>8.2f}x")
        efficiency = (1.0 / metrics['RMSE_CRB_ratio']) * 100
        print(f"  Efficiency:         {efficiency:>8.1f}%")
    
    print(f"  Resolution Rate:    {metrics['Resolution_Rate']:>8.1f}%")
    print(f"  Bias:               {metrics['Bias_degrees']:>8.4f}°")
    
    if not np.isnan(metrics['Runtime_ms']):
        print(f"  Runtime:            {metrics['Runtime_ms']:>8.2f} ms")
    
    print(f"{'='*60}\n")


# Demo and testing
if __name__ == "__main__":
    print("="*70)
    print("  METRICS MODULE DEMO")
    print("="*70)
    print()
    
    # Example: Simulate scenario metrics
    np.random.seed(42)
    
    # True DOAs
    true_doas = np.array([15.0, -20.0])
    
    # Simulate 100 trials with small random errors
    num_trials = 100
    estimated_trials = []
    runtimes = []
    
    for i in range(num_trials):
        # Add Gaussian noise to true DOAs
        noise = np.random.randn(2) * 0.3  # RMSE ≈ 0.3°
        est = true_doas + noise
        estimated_trials.append(est)
        
        # Simulate runtime (5-15 ms)
        runtime = np.random.uniform(5, 15)
        runtimes.append(runtime)
    
    # Configuration
    positions = np.array([0, 5, 8, 11, 14, 17, 21]) * 0.5  # Z5 array
    wavelength = 1.0
    snr_db = 10.0
    snapshots = 256
    
    # Compute metrics
    metrics = compute_scenario_metrics(
        estimated_trials, true_doas, positions, wavelength,
        snr_db, snapshots, coupling_matrix=None,
        runtimes_ms=runtimes
    )
    
    # Display results
    print_metrics_summary(metrics, "Demo Scenario: Ideal Array")
    
    # Show individual components
    print("Individual Metric Components:")
    print(f"  ├─ RMSE:            {metrics['RMSE_degrees']:.4f}°")
    print(f"  ├─ RMSE/CRB Ratio:  {metrics['RMSE_CRB_ratio']:.2f}x")
    print(f"  ├─ Resolution Rate: {metrics['Resolution_Rate']:.1f}%")
    print(f"  ├─ Bias:            {metrics['Bias_degrees']:.4f}°")
    print(f"  └─ Runtime:         {metrics['Runtime_ms']:.2f} ms")
    print()
    
    print("✅ Metrics module ready for experimental scenarios!")
    print()
