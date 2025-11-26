"""
Coupling-Aware ALSS Extensions.

This module provides mutual coupling-aware shrinkage for ALSS-II.
Exploits the synergistic coupling behavior observed in Z5 arrays.

Author: ALSS-II Development Team
Date: November 2025
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np


def estimate_coupling_parameters(
    r_lag: Dict[int, complex],
    w_lag: Dict[int, int],
    max_lag: int = 5
) -> tuple[float, float]:
    """
    Estimate mutual coupling parameters (c1, alpha) from lag structure.
    
    Assumes MCM model: C[i,j] = c1 * alpha^|i-j| for |i-j| <= max_lag
    
    Parameters
    ----------
    r_lag : Dict[int, complex]
        Observed lag correlations
    w_lag : Dict[int, int]
        Lag weights
    max_lag : int, optional
        Maximum lag for coupling effect (default 5)
    
    Returns
    -------
    tuple[float, float]
        Estimated (c1, alpha) parameters
    
    Notes
    -----
    - Uses exponential fit to low-lag decay pattern
    - Returns (0.3, 0.5) as default if estimation fails
    """
    # Extract low-lag magnitudes
    lags = []
    mags = []
    
    for l in range(1, min(max_lag + 1, len(r_lag))):
        if l in r_lag and w_lag.get(l, 0) > 0:
            lags.append(l)
            mags.append(abs(r_lag[l]))
    
    if len(lags) < 2:
        # Insufficient data, use defaults
        return 0.3, 0.5
    
    # Fit exponential decay: mag[l] ~ c1 * alpha^l
    # Take log: log(mag[l]) ~ log(c1) + l * log(alpha)
    lags_arr = np.array(lags)
    log_mags = np.log(np.array(mags) + 1e-12)
    
    # Linear regression
    A = np.vstack([lags_arr, np.ones(len(lags_arr))]).T
    coeffs = np.linalg.lstsq(A, log_mags, rcond=None)[0]
    
    log_alpha = coeffs[0]
    log_c1 = coeffs[1]
    
    alpha_est = np.exp(log_alpha)
    c1_est = np.exp(log_c1)
    
    # Clamp to reasonable ranges
    alpha_est = np.clip(alpha_est, 0.1, 0.9)
    c1_est = np.clip(c1_est, 0.05, 0.5)
    
    return float(c1_est), float(alpha_est)


def compute_coupling_aware_tau(
    base_tau: float,
    c1: float,
    alpha: float,
    lag: int
) -> float:
    """
    Adjust shrinkage strength based on coupling decay.
    
    Key insight: Low lags are more affected by coupling (synergistic for Z5).
    Reduce shrinkage where coupling is strong.
    
    Parameters
    ----------
    base_tau : float
        Base shrinkage strength
    c1 : float
        Coupling strength parameter
    alpha : float
        Coupling decay parameter
    lag : int
        Current lag index
    
    Returns
    -------
    float
        Adjusted tau for this lag
    
    Notes
    -----
    - For Z5: coupling helps at low lags → reduce shrinkage
    - For arrays where coupling hurts → increase shrinkage
    - Linear interpolation based on coupling decay
    """
    # Coupling effect at this lag
    coupling_decay = c1 * (alpha ** abs(lag))
    
    # For synergistic coupling (Z5), reduce tau at low lags
    # For detrimental coupling, increase tau
    # Simple heuristic: tau_adjusted = tau * (1 - coupling_decay)
    
    # Synergy detection: if low-lag variance is unusually low, coupling helps
    # For now, use simple decay model
    tau_adjusted = base_tau * (1.0 - 0.5 * coupling_decay)
    
    return max(0.1 * base_tau, tau_adjusted)  # Ensure minimum shrinkage


def apply_alss_coupling_aware(
    r_lag: Dict[int, complex],
    w_lag: Dict[int, int],
    base_tau: float,
    c1: Optional[float] = None,
    alpha: Optional[float] = None,
    auto_estimate: bool = True
) -> Dict[int, float]:
    """
    Compute coupling-aware per-lag tau values.
    
    Parameters
    ----------
    r_lag : Dict[int, complex]
        Lag correlations
    w_lag : Dict[int, int]
        Lag weights
    base_tau : float
        Base shrinkage strength
    c1 : float, optional
        Coupling strength (if known)
    alpha : float, optional
        Coupling decay (if known)
    auto_estimate : bool, optional
        Automatically estimate c1, alpha if not provided (default True)
    
    Returns
    -------
    Dict[int, float]
        Per-lag tau values adjusted for coupling
    """
    # Estimate coupling if needed
    if (c1 is None or alpha is None) and auto_estimate:
        c1_est, alpha_est = estimate_coupling_parameters(r_lag, w_lag)
    else:
        c1_est = c1 if c1 is not None else 0.3
        alpha_est = alpha if alpha is not None else 0.5
    
    # Compute per-lag tau
    tau_dict = {}
    for lag in r_lag.keys():
        tau_dict[lag] = compute_coupling_aware_tau(base_tau, c1_est, alpha_est, lag)
    
    return tau_dict
