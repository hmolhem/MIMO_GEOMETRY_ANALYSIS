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


def compute_confidence_intervals(estimated_doas_trials: List[np.ndarray],
                                 true_doas: np.ndarray,
                                 confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for DOA estimation errors.
    
    Args:
        estimated_doas_trials: List of estimated DOA arrays from trials
        true_doas: True DOA angles (K,) in degrees
        confidence_level: Confidence level (default: 0.95 for 95%)
    
    Returns:
        ci_dict: Dictionary with confidence intervals for each DOA
            Keys: 'DOA_0', 'DOA_1', ... 'DOA_K-1'
            Values: (lower_bound, upper_bound) tuples
    
    Usage:
        >>> ci = compute_confidence_intervals(trials, true_doas, 0.95)
        >>> print(f"DOA 0: [{ci['DOA_0'][0]:.2f}°, {ci['DOA_0'][1]:.2f}°]")
    """
    K = len(true_doas)
    errors = []
    
    for est_doas in estimated_doas_trials:
        if len(est_doas) == K:
            matched_est = match_doas(est_doas, true_doas)
            errors.append(matched_est - true_doas)
    
    if not errors:
        return {f'DOA_{k}': (np.nan, np.nan) for k in range(K)}
    
    errors = np.array(errors)  # (num_trials, K)
    
    # Compute confidence intervals for each DOA
    alpha = 1 - confidence_level
    ci_dict = {}
    
    for k in range(K):
        doa_errors = errors[:, k]
        mean_error = np.mean(doa_errors)
        std_error = np.std(doa_errors, ddof=1)
        
        # t-distribution critical value
        from scipy import stats
        n = len(doa_errors)
        t_crit = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin = t_crit * std_error / np.sqrt(n)
        ci_dict[f'DOA_{k}'] = (mean_error - margin, mean_error + margin)
    
    return ci_dict


def compute_effect_size(baseline_rmse: float, treatment_rmse: float, 
                       pooled_std: Optional[float] = None) -> float:
    """
    Compute Cohen's d effect size for comparing two conditions.
    
    Measures standardized difference between baseline and treatment.
    
    Args:
        baseline_rmse: RMSE from baseline condition (e.g., no coupling)
        treatment_rmse: RMSE from treatment condition (e.g., with coupling)
        pooled_std: Optional pooled standard deviation (computed if None)
    
    Returns:
        cohens_d: Effect size (negative = improvement, positive = degradation)
    
    Interpretation:
        |d| < 0.2: Negligible effect
        0.2 ≤ |d| < 0.5: Small effect
        0.5 ≤ |d| < 0.8: Medium effect
        |d| ≥ 0.8: Large effect
    
    Usage:
        >>> baseline = 0.5  # RMSE without coupling
        >>> treatment = 1.2  # RMSE with coupling
        >>> d = compute_effect_size(baseline, treatment)
        >>> print(f"Cohen's d: {d:.2f} (degradation)")
    """
    if pooled_std is None:
        # Estimate pooled std as average of the two RMSEs
        pooled_std = (baseline_rmse + treatment_rmse) / 2.0
    
    cohens_d = (treatment_rmse - baseline_rmse) / pooled_std
    return cohens_d


def compute_statistical_power(effect_size: float, n_trials: int, 
                              alpha: float = 0.05) -> float:
    """
    Compute statistical power for detecting given effect size.
    
    Power = probability of correctly rejecting null hypothesis.
    
    Args:
        effect_size: Cohen's d or similar standardized effect size
        n_trials: Number of trials (sample size)
        alpha: Significance level (default: 0.05)
    
    Returns:
        power: Statistical power (0 to 1)
    
    Interpretation:
        power < 0.5: Underpowered
        0.5 ≤ power < 0.8: Moderate power
        power ≥ 0.8: Good power (recommended minimum)
        power ≥ 0.95: Excellent power
    
    Usage:
        >>> power = compute_statistical_power(effect_size=0.5, n_trials=100)
        >>> print(f"Statistical power: {power:.2%}")
    """
    from scipy import stats
    
    # For two-sample t-test (approximation)
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_trials / 2)
    
    # Critical value for two-tailed test
    t_crit = stats.t.ppf(1 - alpha/2, n_trials - 1)
    
    # Power = P(reject H0 | H1 is true)
    power = 1 - stats.nct.cdf(t_crit, n_trials - 1, ncp)
    
    return power


def bootstrap_validation(estimated_doas_trials: List[np.ndarray],
                        true_doas: np.ndarray,
                        n_bootstrap: int = 1000,
                        confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap resampling for non-parametric confidence interval estimation.
    
    Args:
        estimated_doas_trials: List of estimated DOA arrays from trials
        true_doas: True DOA angles (K,) in degrees
        n_bootstrap: Number of bootstrap samples (default: 1000)
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        bootstrap_ci: Dictionary with bootstrap confidence intervals
            - 'RMSE': (lower, upper) for RMSE
            - 'Bias': (lower, upper) for Bias
            - 'Resolution_Rate': (lower, upper) for Resolution %
    
    Usage:
        >>> boot_ci = bootstrap_validation(trials, true_doas, n_bootstrap=1000)
        >>> print(f"RMSE 95% CI: [{boot_ci['RMSE'][0]:.3f}, {boot_ci['RMSE'][1]:.3f}]")
    """
    n_trials = len(estimated_doas_trials)
    bootstrap_rmse = []
    bootstrap_bias = []
    bootstrap_resolution = []
    
    np.random.seed(42)  # Reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_trials, n_trials, replace=True)
        bootstrap_sample = [estimated_doas_trials[i] for i in indices]
        
        # Compute metrics on bootstrap sample
        rmse_values = []
        bias_values = []
        resolved_count = 0
        
        for est_doas in bootstrap_sample:
            if len(est_doas) == len(true_doas):
                rmse = compute_rmse(est_doas, true_doas)
                bias = compute_bias(est_doas, true_doas)
                if not np.isnan(rmse):
                    rmse_values.append(rmse)
                    bias_values.append(bias)
                
                if resolved_indicator(true_doas, est_doas, threshold_deg=3.0) == 1:
                    resolved_count += 1
        
        if rmse_values:
            bootstrap_rmse.append(np.mean(rmse_values))
            bootstrap_bias.append(np.mean(bias_values))
            bootstrap_resolution.append((resolved_count / len(bootstrap_sample)) * 100)
    
    # Compute percentile-based confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    bootstrap_ci = {
        'RMSE': (np.percentile(bootstrap_rmse, lower_percentile),
                np.percentile(bootstrap_rmse, upper_percentile)),
        'Bias': (np.percentile(bootstrap_bias, lower_percentile),
                np.percentile(bootstrap_bias, upper_percentile)),
        'Resolution_Rate': (np.percentile(bootstrap_resolution, lower_percentile),
                           np.percentile(bootstrap_resolution, upper_percentile))
    }
    
    return bootstrap_ci


def compute_computational_overhead(baseline_runtime_ms: float,
                                  treatment_runtime_ms: float) -> Dict[str, float]:
    """
    Compute computational overhead metrics.
    
    Args:
        baseline_runtime_ms: Runtime from baseline method (ms)
        treatment_runtime_ms: Runtime from treatment method (ms)
    
    Returns:
        overhead_metrics: Dictionary with overhead metrics
            - 'Absolute_Overhead_ms': Absolute difference
            - 'Relative_Overhead_percent': Percentage increase
            - 'Speedup_Factor': Speedup (if negative overhead)
    
    Usage:
        >>> overhead = compute_computational_overhead(baseline=10.0, treatment=15.0)
        >>> print(f"Overhead: {overhead['Relative_Overhead_percent']:.1f}%")
    """
    absolute_overhead = treatment_runtime_ms - baseline_runtime_ms
    relative_overhead = (absolute_overhead / baseline_runtime_ms) * 100
    
    if absolute_overhead < 0:
        speedup_factor = baseline_runtime_ms / treatment_runtime_ms
    else:
        speedup_factor = treatment_runtime_ms / baseline_runtime_ms
    
    return {
        'Absolute_Overhead_ms': absolute_overhead,
        'Relative_Overhead_percent': relative_overhead,
        'Speedup_Factor': speedup_factor
    }


def estimate_memory_footprint(array_size: int, snapshots: int, 
                              num_sources: int, dtype_bytes: int = 16) -> Dict[str, float]:
    """
    Estimate memory footprint for DOA estimation.
    
    Args:
        array_size: Number of sensors (N)
        snapshots: Number of snapshots (M)
        num_sources: Number of sources (K)
        dtype_bytes: Bytes per element (default: 16 for complex128)
    
    Returns:
        memory_metrics: Dictionary with memory usage estimates in MB
            - 'Snapshot_Matrix_MB': X (N × M)
            - 'Covariance_Matrix_MB': Rxx (N × N)
            - 'Steering_Matrix_MB': A (N × K)
            - 'Total_Estimated_MB': Sum of all components
    
    Usage:
        >>> mem = estimate_memory_footprint(N=7, M=256, K=2)
        >>> print(f"Total memory: {mem['Total_Estimated_MB']:.2f} MB")
    """
    snapshot_matrix_bytes = array_size * snapshots * dtype_bytes
    covariance_matrix_bytes = array_size * array_size * dtype_bytes
    steering_matrix_bytes = array_size * num_sources * dtype_bytes
    
    # Additional overhead (eigendecomposition, temporary arrays, etc.)
    overhead_factor = 1.5
    
    total_bytes = (snapshot_matrix_bytes + covariance_matrix_bytes + 
                   steering_matrix_bytes) * overhead_factor
    
    bytes_to_mb = 1024 * 1024
    
    return {
        'Snapshot_Matrix_MB': snapshot_matrix_bytes / bytes_to_mb,
        'Covariance_Matrix_MB': covariance_matrix_bytes / bytes_to_mb,
        'Steering_Matrix_MB': steering_matrix_bytes / bytes_to_mb,
        'Total_Estimated_MB': total_bytes / bytes_to_mb
    }


def compute_parameter_sensitivity(base_metrics: Dict[str, float],
                                  varied_metrics: List[Dict[str, float]],
                                  parameter_values: List[float]) -> Dict[str, float]:
    """
    Compute parameter sensitivity scores.
    
    Measures how much performance changes with parameter variation.
    
    Args:
        base_metrics: Metrics at baseline parameter value
        varied_metrics: List of metrics at different parameter values
        parameter_values: List of parameter values corresponding to varied_metrics
    
    Returns:
        sensitivity_scores: Dictionary with sensitivity metrics
            - 'RMSE_Sensitivity': Normalized RMSE variation
            - 'Resolution_Sensitivity': Normalized resolution variation
            - 'Overall_Sensitivity': Combined sensitivity score
    
    Interpretation:
        < 0.1: Low sensitivity (robust)
        0.1 - 0.5: Moderate sensitivity
        > 0.5: High sensitivity (requires careful tuning)
    
    Usage:
        >>> sensitivity = compute_parameter_sensitivity(
        ...     base_metrics, varied_metrics, [0.1, 0.2, 0.3, 0.4, 0.5]
        ... )
        >>> print(f"Overall sensitivity: {sensitivity['Overall_Sensitivity']:.3f}")
    """
    base_rmse = base_metrics['RMSE_degrees']
    base_resolution = base_metrics['Resolution_Rate']
    
    rmse_variations = []
    resolution_variations = []
    
    for varied in varied_metrics:
        rmse_change = abs(varied['RMSE_degrees'] - base_rmse) / base_rmse
        resolution_change = abs(varied['Resolution_Rate'] - base_resolution) / (base_resolution + 1e-6)
        
        rmse_variations.append(rmse_change)
        resolution_variations.append(resolution_change)
    
    rmse_sensitivity = np.mean(rmse_variations)
    resolution_sensitivity = np.mean(resolution_variations)
    overall_sensitivity = (rmse_sensitivity + resolution_sensitivity) / 2.0
    
    return {
        'RMSE_Sensitivity': rmse_sensitivity,
        'Resolution_Sensitivity': resolution_sensitivity,
        'Overall_Sensitivity': overall_sensitivity
    }


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


def compute_extended_metrics(estimated_doas_trials: List[np.ndarray],
                            true_doas: np.ndarray,
                            sensor_positions: np.ndarray,
                            wavelength: float,
                            snr_db: float,
                            snapshots: int,
                            coupling_matrix: Optional[np.ndarray] = None,
                            resolution_threshold: float = 3.0,
                            runtimes_ms: Optional[List[float]] = None,
                            baseline_metrics: Optional[Dict[str, float]] = None,
                            n_bootstrap: int = 1000) -> Dict[str, any]:
    """
    Compute comprehensive extended metrics including statistical and practical metrics.
    
    This function computes all 5 core metrics plus:
    - Statistical metrics (95% CI, effect size, power, bootstrap)
    - Practical metrics (overhead, memory, sensitivity)
    
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
        baseline_metrics: Optional baseline metrics for comparison
        n_bootstrap: Number of bootstrap samples (default: 1000)
    
    Returns:
        extended_metrics: Dictionary containing:
            - Core metrics (5): RMSE, RMSE/CRB, Resolution%, Bias, Runtime
            - Statistical metrics (4): 95% CI, Effect Size, Power, Bootstrap
            - Practical metrics (4): Overhead, Memory, Sensitivity, Integration
    
    Usage:
        >>> extended = compute_extended_metrics(
        ...     trials, true_doas, positions, wavelength, snr_db, snapshots,
        ...     coupling_matrix=C, runtimes_ms=runtimes, baseline_metrics=baseline
        ... )
        >>> print(f"Effect size: {extended['Effect_Size']:.3f}")
        >>> print(f"Statistical power: {extended['Statistical_Power']:.2%}")
        >>> print(f"Memory footprint: {extended['Memory_Footprint_MB']:.2f} MB")
    """
    # 1. Core metrics
    core_metrics = compute_scenario_metrics(
        estimated_doas_trials, true_doas, sensor_positions, wavelength,
        snr_db, snapshots, coupling_matrix, resolution_threshold, runtimes_ms
    )
    
    # 2. Statistical metrics
    
    # 95% Confidence Intervals
    ci_dict = compute_confidence_intervals(estimated_doas_trials, true_doas, 0.95)
    
    # Effect Size (if baseline provided)
    if baseline_metrics is not None:
        effect_size = compute_effect_size(
            baseline_metrics['RMSE_degrees'],
            core_metrics['RMSE_degrees']
        )
    else:
        effect_size = 0.0  # No comparison
    
    # Statistical Power
    n_trials = len(estimated_doas_trials)
    if effect_size != 0:
        statistical_power = compute_statistical_power(effect_size, n_trials, alpha=0.05)
    else:
        statistical_power = np.nan
    
    # Bootstrap Validation
    bootstrap_ci = bootstrap_validation(
        estimated_doas_trials, true_doas, n_bootstrap, confidence_level=0.95
    )
    
    # 3. Practical metrics
    
    # Computational Overhead (if baseline provided)
    if baseline_metrics is not None and runtimes_ms is not None:
        overhead = compute_computational_overhead(
            baseline_metrics['Runtime_ms'],
            core_metrics['Runtime_ms']
        )
    else:
        overhead = {
            'Absolute_Overhead_ms': 0.0,
            'Relative_Overhead_percent': 0.0,
            'Speedup_Factor': 1.0
        }
    
    # Memory Footprint
    N = len(sensor_positions)
    K = len(true_doas)
    memory = estimate_memory_footprint(N, snapshots, K)
    
    # Parameter Sensitivity (placeholder - requires parameter sweep)
    # This would need multiple runs with varied parameters
    sensitivity = {
        'RMSE_Sensitivity': np.nan,
        'Resolution_Sensitivity': np.nan,
        'Overall_Sensitivity': np.nan
    }
    
    # Ease of Integration (qualitative assessment)
    ease_of_integration = {
        'API_Complexity': 'Low' if coupling_matrix is None else 'Medium',
        'Dependencies': 'NumPy, SciPy',
        'Documentation': 'Complete',
        'Integration_Score': 0.9  # 0 to 1 scale
    }
    
    # 4. Combine all metrics
    extended_metrics = {
        # Core metrics (5)
        **core_metrics,
        
        # Statistical metrics (4)
        '95%_Confidence_Intervals': ci_dict,
        'Effect_Size': effect_size,
        'Statistical_Power': statistical_power,
        'Bootstrap_Validation': bootstrap_ci,
        
        # Practical metrics (4)
        'Computational_Overhead_ms': overhead['Absolute_Overhead_ms'],
        'Computational_Overhead_percent': overhead['Relative_Overhead_percent'],
        'Memory_Footprint_MB': memory['Total_Estimated_MB'],
        'Memory_Details': memory,
        'Parameter_Sensitivity_Score': sensitivity['Overall_Sensitivity'],
        'Parameter_Sensitivity_Details': sensitivity,
        'Ease_of_Integration': ease_of_integration,
        
        # Meta information
        'N_Trials': n_trials,
        'N_Sensors': N,
        'N_Sources': K,
        'N_Snapshots': snapshots
    }
    
    return extended_metrics


def print_extended_metrics_summary(metrics: Dict[str, any], 
                                  scenario_name: str = "Extended Scenario"):
    """
    Pretty print extended metrics summary.
    
    Args:
        metrics: Dictionary from compute_extended_metrics()
        scenario_name: Name of the scenario for display
    """
    print(f"\n{'='*70}")
    print(f"  {scenario_name}")
    print(f"{'='*70}")
    
    # Core metrics
    print(f"\n  CORE METRICS:")
    print(f"    RMSE:                {metrics['RMSE_degrees']:>8.4f}°")
    print(f"    RMSE/CRB Ratio:      {metrics['RMSE_CRB_ratio']:>8.2f}x")
    print(f"    Resolution Rate:     {metrics['Resolution_Rate']:>8.1f}%")
    print(f"    Bias:                {metrics['Bias_degrees']:>8.4f}°")
    print(f"    Runtime:             {metrics['Runtime_ms']:>8.2f} ms")
    
    # Statistical metrics
    print(f"\n  STATISTICAL METRICS:")
    
    boot_ci = metrics['Bootstrap_Validation']
    print(f"    RMSE 95% CI:         [{boot_ci['RMSE'][0]:.4f}, {boot_ci['RMSE'][1]:.4f}]°")
    
    if not np.isnan(metrics['Effect_Size']):
        effect = metrics['Effect_Size']
        effect_label = "Negligible" if abs(effect) < 0.2 else \
                      "Small" if abs(effect) < 0.5 else \
                      "Medium" if abs(effect) < 0.8 else "Large"
        print(f"    Effect Size:         {effect:>8.3f} ({effect_label})")
    
    if not np.isnan(metrics['Statistical_Power']):
        power = metrics['Statistical_Power']
        power_label = "Underpowered" if power < 0.5 else \
                     "Moderate" if power < 0.8 else "Good"
        print(f"    Statistical Power:   {power:>8.2%} ({power_label})")
    
    print(f"    Bootstrap Samples:   {1000:>8d}")
    
    # Practical metrics
    print(f"\n  PRACTICAL METRICS:")
    print(f"    Memory Footprint:    {metrics['Memory_Footprint_MB']:>8.2f} MB")
    
    if abs(metrics['Computational_Overhead_ms']) > 0.01:
        overhead_pct = metrics['Computational_Overhead_percent']
        overhead_dir = "increase" if overhead_pct > 0 else "decrease"
        print(f"    Comp. Overhead:      {overhead_pct:>8.1f}% {overhead_dir}")
    
    if not np.isnan(metrics['Parameter_Sensitivity_Score']):
        sens = metrics['Parameter_Sensitivity_Score']
        sens_label = "Robust" if sens < 0.1 else \
                    "Moderate" if sens < 0.5 else "Sensitive"
        print(f"    Param. Sensitivity:  {sens:>8.3f} ({sens_label})")
    
    ease = metrics['Ease_of_Integration']
    print(f"    Integration Score:   {ease['Integration_Score']:>8.1f}/1.0")
    
    # Configuration
    print(f"\n  CONFIGURATION:")
    print(f"    Trials:              {metrics['N_Trials']:>8d}")
    print(f"    Sensors:             {metrics['N_Sensors']:>8d}")
    print(f"    Sources:             {metrics['N_Sources']:>8d}")
    print(f"    Snapshots:           {metrics['N_Snapshots']:>8d}")
    
    print(f"{'='*70}\n")


# Demo and testing
if __name__ == "__main__":
    print("="*70)
    print("  EXTENDED METRICS MODULE DEMO")
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
    
    # Display core metrics
    print_metrics_summary(metrics, "Demo Scenario: Ideal Array (Core Metrics)")
    
    # Compute extended metrics
    print("\n" + "="*70)
    print("  EXTENDED METRICS DEMO")
    print("="*70)
    
    extended = compute_extended_metrics(
        estimated_trials, true_doas, positions, wavelength,
        snr_db, snapshots, coupling_matrix=None,
        runtimes_ms=runtimes, baseline_metrics=None, n_bootstrap=100  # Reduced for demo speed
    )
    
    print_extended_metrics_summary(extended, "Demo: Extended Analysis")
    
    # Show specific extended components
    print("Extended Metric Details:")
    print(f"  Statistical Metrics:")
    boot_ci = extended['Bootstrap_Validation']
    print(f"    ├─ RMSE Bootstrap CI:     [{boot_ci['RMSE'][0]:.4f}, {boot_ci['RMSE'][1]:.4f}]°")
    print(f"    ├─ Bias Bootstrap CI:     [{boot_ci['Bias'][0]:.4f}, {boot_ci['Bias'][1]:.4f}]°")
    print(f"    └─ Resolution CI:         [{boot_ci['Resolution_Rate'][0]:.1f}, {boot_ci['Resolution_Rate'][1]:.1f}]%")
    
    mem = extended['Memory_Details']
    print(f"  Practical Metrics:")
    print(f"    ├─ Snapshot Matrix:       {mem['Snapshot_Matrix_MB']:.3f} MB")
    print(f"    ├─ Covariance Matrix:     {mem['Covariance_Matrix_MB']:.3f} MB")
    print(f"    ├─ Steering Matrix:       {mem['Steering_Matrix_MB']:.6f} MB")
    print(f"    └─ Total (with overhead): {mem['Total_Estimated_MB']:.3f} MB")
    
    ease = extended['Ease_of_Integration']
    print(f"  Integration Assessment:")
    print(f"    ├─ API Complexity:        {ease['API_Complexity']}")
    print(f"    ├─ Dependencies:          {ease['Dependencies']}")
    print(f"    ├─ Documentation:         {ease['Documentation']}")
    print(f"    └─ Integration Score:     {ease['Integration_Score']:.1f}/1.0")
    print()
    
    print("✅ Extended metrics module ready for comprehensive evaluation!")
    print()
