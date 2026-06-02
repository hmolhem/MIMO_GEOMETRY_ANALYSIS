"""
Adaptive Lag-Selective Shrinkage (ALSS) for Coarray MUSIC.

This module implements lag-selective shrinkage for coarray lag estimates. The method is applied
after coarray lag averaging and before virtual Toeplitz covariance reconstruction.

Mathematical principle
----------------------
ALSS uses a per-lag variance proxy of the form

    Var[rhat(ell)] ≈ sigma^2 / (M * w[ell])

where M is the number of snapshots and w[ell] is the coarray weight, i.e., the
number of physical sensor pairs contributing to lag ell. Low-weight lags have
higher finite-snapshot variance and may benefit from shrinkage toward a prior.

Implemented features
--------------------
- Lag-specific shrinkage rather than uniform covariance shrinkage.
- Weight-aware variance scheduling using coarray weights from the array geometry.
- Core-lag protection for lags 0..coreL.
- Two shrinkage modes:
  - "zero": shrink low-confidence lags toward zero.
  - "ar1": shrink low-confidence lags toward an AR(1)-style lag prior.
- Hermitian symmetry enforcement after shrinkage.

Paper-facing validation
-----------------------
The current paper-facing validation in this repository focuses on the canonical Z5 sparse
array under Scenario 3 using 1000 Monte Carlo trials and the fixed configuration
mode="ar1", tau=0.25, coreL=3.

The archived Scenario 3 Z5 trial-1000 result reports:
- mean improvement over reported rows: approximately 9.85%;
- mean improvement over unique conditions: approximately 9.76%;
- mean improvement under c1=0.3 mutual coupling: approximately 13.92%.

Caution
-------
ALSS is not claimed to be universally optimal or harmless for every array, every
parameter setting, or every trial. The paper presents a conservative Z5-focused
conference result, not a full multi-geometry validation.

Author: Hossein Molhem
Date: May 2026
"""
from __future__ import annotations
from typing import Dict
import numpy as np


def _enforce_hermitian_symmetry(r_lag: Dict[int, complex]) -> Dict[int, complex]:
    """Ensure r[-ℓ] = conj(r[+ℓ]) for all available lags."""
    out = dict(r_lag)
    pos = {ell for ell in r_lag.keys() if ell >= 0}
    for ell in pos:
        r_p = out.get(ell, None)
        if r_p is None:
            continue
        if ell == 0:
            # force real( r[0] )
            out[0] = complex(np.real(r_p), 0.0)
        else:
            out[-ell] = np.conj(r_p)
    return out


def _fit_ar1_prior(r_lag: Dict[int, complex], coreL: int, eps: float = 1e-12):
    """Rough AR(1) fit from core low lags: r[ℓ] ~ r[0] * ρ^{|ℓ|}."""
    r0 = r_lag.get(0, 0.0)
    if abs(r0) < eps:
        return 0.0, (lambda ell: 0.0)
    num, den = 0.0, 0.0
    for ell in range(1, coreL + 1):
        if ell in r_lag:
            # use real ratio as a simple, stable estimator
            num += (np.real(r_lag[ell]) / (np.real(r0) + eps))
            den += 1.0
    rho = (num / den) if den > 0 else 0.0
    
    def r_prior(ell: int) -> complex:
        return (rho ** abs(ell)) * r0
    
    return rho, r_prior


def apply_alss(
    r_lag: Dict[int, complex],
    w_lag: Dict[int, int],
    R_x: np.ndarray,
    M: int,
    mode: str = "zero",
    tau: float = 1.0,
    coreL: int = 3,
    eps: float = 1e-12
) -> Dict[int, complex]:
    """
    Adaptive Lag-Selective Shrinkage (ALSS).
    
    Shrink noisy lags using a per-lag variance proxy: Var[r̂(ℓ)] ~ σ² / (M * w[ℓ]).
    
    Parameters
    ----------
    r_lag : Dict[int, complex]
        Lag autocorrelation estimates (unbiased or biased)
    w_lag : Dict[int, int]
        Lag weight counts (number of sensor pairs contributing to each lag)
    R_x : np.ndarray
        Physical array covariance matrix (for noise power estimation)
    M : int
        Number of snapshots
    mode : str, optional
        Shrinkage target:
        - 'zero': shrink toward 0 (default)
        - 'ar1': shrink toward AR(1) prior fitted from low lags 0..coreL
    tau : float, optional
        Strength parameter (larger = more shrinkage), default 1.0
    coreL : int, optional
        Protect low lags 0..coreL from shrinkage (set αℓ=0 there), default 3
    eps : float, optional
        Numerical stability constant, default 1e-12
    
    Returns
    -------
    Dict[int, complex]
        Shrunk lag estimates with Hermitian symmetry enforced
    
    Notes
    -----
    - Applied after unbiased lag averaging and before Toeplitz mapping
    - Particularly effective at low SNR or low snapshot counts
    - Core lags (0..coreL) are protected from shrinkage
    - Hermitian symmetry r[-ℓ] = conj(r[+ℓ]) is enforced
    """
    N = R_x.shape[0]
    # crude noise power proxy; robust enough for shrink scheduling
    sigma2 = float(np.trace(R_x).real) / float(N)

    # optional prior
    if mode == "ar1":
        _, r_prior = _fit_ar1_prior(r_lag, coreL=coreL, eps=eps)
    else:
        r_prior = (lambda ell: 0.0)

    r_out: Dict[int, complex] = {}
    # work on nonnegative lags and reflect to negative to preserve Hermitian structure
    # but accept any input dict (negative keys OK)
    all_lags = sorted(set(r_lag.keys()))

    for ell in all_lags:
        r = r_lag[ell]
        # protect core nonnegative lags
        if 0 <= ell <= coreL:
            r_out[ell] = r if ell != 0 else complex(np.real(r), 0.0)
            continue

        # variance proxy per lag; guard small counts
        w = max(int(w_lag.get(ell, w_lag.get(abs(ell), 1))), 1)
        var_hat = sigma2 / max(M * w, 1)

        if mode == "zero":
            denom = var_hat + tau * (abs(r) ** 2) + eps
            alpha = var_hat / denom
            r_shrunk = (1.0 - alpha) * r
        elif mode == "ar1":
            rp = r_prior(ell)
            denom = var_hat + tau * (abs(r - rp) ** 2) + eps
            alpha = var_hat / denom
            r_shrunk = (1.0 - alpha) * r + alpha * rp
        else:
            r_shrunk = r

        # don't amplify; mild cap (optional safety)
        if abs(r_shrunk) > abs(r) * 1.25:
            r_shrunk = r * 1.25

        r_out[ell] = r_shrunk

    # enforce r[-ℓ] = conj(r[+ℓ]) and real r[0]
    r_out = _enforce_hermitian_symmetry(r_out)
    return r_out
