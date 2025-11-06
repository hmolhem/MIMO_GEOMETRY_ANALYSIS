# algorithms/coarray_music.py
import numpy as np
from numpy.linalg import eigh
from .coarray import build_virtual_ula_covariance

def steering_ula(theta_deg, m_idx, d, wavelength):
    """
    ULA steering for indices m_idx = [0..M-1] (virtual array), spacing d (meters).
    """
    k = 2.0 * np.pi / wavelength
    theta = np.deg2rad(theta_deg)
    return np.exp(1j * k * d * np.outer(m_idx, np.sin(theta)))  # (M x G)

def music_spectrum(R, M_sources, d, wavelength, scan_deg=(-60, 60, 0.1), lags=None):
    """
    Standard MUSIC pseudospectrum for ULA covariance R (size MxM).
    
    Parameters:
    -----------
    R : ndarray (M, M)
        Covariance matrix
    M_sources : int
        Number of sources
    d : float
        Sensor spacing (meters)
    wavelength : float
        Signal wavelength (meters)
    scan_deg : tuple
        (theta_min, theta_max, theta_step)
    lags : ndarray, optional
        Actual lag indices for virtual array (if not starting at 0)
        If None, assumes consecutive indices [0, 1, ..., M-1]
    
    Returns: angles (deg), P(angles).
    """
    # EVD (ascending); noise subspace = smallest M-M_sources eigenvectors
    vals, vecs = eigh(R)
    idx = np.argsort(vals.real)
    vals = vals[idx]; vecs = vecs[:, idx]
    M = R.shape[0]
    d_noise = M - M_sources
    if d_noise <= 0:
        d_noise = 1  # guard
    En = vecs[:, :d_noise]  # noise subspace
    # Scan
    thetas = np.arange(*scan_deg)
    # Use actual lag indices if provided (for virtual arrays that don't start at 0)
    if lags is None:
        m_idx = np.arange(M)
    else:
        m_idx = np.asarray(lags)
    A = steering_ula(thetas, m_idx, d, wavelength)  # (M x G)
    denom = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
    P = 1.0 / np.maximum(denom.real, 1e-12)
    return thetas, P

def pick_peaks_safeguarded(thetas, P, K, guard_bins=1):
    """
    Simple peak picker with edge/adjacent guards. Returns K sorted DOA estimates.
    """
    P = P.copy()
    # edge guard
    P[0:guard_bins] = 0.0
    P[-guard_bins:] = 0.0
    # find K peaks
    idxs = []
    for _ in range(K):
        i = int(np.argmax(P))
        idxs.append(i)
        # zero out a 1-bin neighborhood to avoid duplicates
        lo = max(0, i - 1); hi = min(len(P), i + 2)
        P[lo:hi] = 0.0
    est = sorted([thetas[i] for i in idxs])
    return np.array(est, dtype=float)

def root_music_from_Rv(Rv, K, k, d):
    """
    Root-MUSIC for a ULA with inter-element spacing d.
    Rv must correspond to virtual indices m=0..Lv-1.
    
    Parameters:
    -----------
    Rv : ndarray (Lv, Lv)
        Virtual covariance (FBA+unbiased+loading already applied)
    K : int
        Number of sources
    k : float
        Wavenumber = 2π/λ
    d : float
        Virtual ULA spacing (meters)
    
    Returns:
    --------
    thetas : ndarray
        Sorted DOA estimates (degrees)
    dbg : dict
        Debug info (roots, inside, chosen, poly_len)
    """
    Lv = Rv.shape[0]
    
    # 1) Eigendecompose, build noise subspace projector
    evals, evecs = np.linalg.eigh(Rv)
    idx = np.argsort(evals)  # ascending
    En = evecs[:, idx[:-K]]  # (Lv x (Lv-K))
    Pn = En @ En.conj().T    # noise projector
    
    # 2) Form polynomial coefficients for q(z) = a(z)^H Pn a(z)
    # a(z) = [1, z, z^2, ..., z^{Lv-1}]^T with z = e^{-j k d sin(theta)}
    # => q(z) = sum_{i,j} Pn[i,j] z^{j-i}  (Hermitian Toeplitz in exponent)
    # Map exponents e = j-i in [-(Lv-1) ... +(Lv-1)] to polynomial
    coeff = np.zeros(2*Lv-1, dtype=complex)   # exponents from -(Lv-1)..+(Lv-1)
    for i in range(Lv):
        for j in range(Lv):
            e = j - i
            coeff[e + (Lv - 1)] += Pn[i, j]
    
    # Convert from z^{-1}-poly to z-poly by reversing
    p = coeff[::-1]
    
    # 3) Roots & selection
    roots = np.roots(p)
    # Project near-unit-circle (robust against small numeric drift)
    roots_uc = roots / np.maximum(1e-12, np.abs(roots))
    # Keep roots strictly inside unit circle (reciprocal choice)
    inside = roots[np.abs(roots) < 1.0]
    # Fallback if numerical issues: use projected and unique by angle
    if inside.size < K:
        inside = roots_uc
    
    # Pick K closest to unit circle by radius
    order = np.argsort(np.abs(np.abs(inside) - 1))
    chosen = []
    for r in inside[order]:
        # Avoid conjugate duplicates: keep those with imag >= 0
        if np.imag(r) >= -1e-12:
            chosen.append(r)
        if len(chosen) == K:
            break
    chosen = np.array(chosen, dtype=complex)
    
    # 4) Map roots to DOA
    # Try positive convention: z = e^{+j k d sin(theta)}
    # => angle(z) = +k d sin(theta)  ->  sin(theta) = +angle(z)/(k d)
    ang = np.angle(chosen)
    s = ang / (k * d)  # Try positive sign
    s = np.clip(s, -1.0, 1.0)
    thetas = np.rad2deg(np.arcsin(s))
    thetas = np.sort(thetas)
    
    dbg = {"roots": roots, "inside": inside, "chosen": chosen, "poly_len": len(p)}
    return thetas, dbg

def estimate_doa_coarray_music(X, positions, d_phys, wavelength, K,
                               scan_deg=(-60, 60, 0.1), return_debug=False, use_root=False,
                               alss_enabled=False, alss_mode="zero", alss_tau=1.0, alss_coreL=3):
    """
    Coarray-MUSIC on the largest one-sided contiguous segment of the difference coarray.
    X: snapshots matrix (N x M), positions: integer-grid sensor indices * d_phys,
    d_phys: physical grid spacing (meters), wavelength (meters), K: source count.
    
    ALSS Parameters (optional lag-selective shrinkage):
    - alss_enabled: Enable adaptive lag-selective shrinkage (default False)
    - alss_mode: 'zero' (shrink toward 0) or 'ar1' (AR(1) prior) (default 'zero')
    - alss_tau: Shrinkage strength parameter (default 1.0)
    - alss_coreL: Protect low lags 0..coreL from shrinkage (default 3)
    """
    # Sample covariance
    M = X.shape[1]  # Number of snapshots
    Rxx = X @ X.conj().T / max(1, M)
    # Build virtual ULA covariance from lag-averaged stats
    Rv, dvirt, (L1, L2), one_side, rmap, coarray_debug = build_virtual_ula_covariance(
        Rxx, positions, d_phys,
        alss_enabled=alss_enabled, alss_mode=alss_mode,
        alss_tau=alss_tau, alss_coreL=alss_coreL, M=M
    )
    
    # Forward-Backward Averaging (FBA) - halves variance, decorrelates symmetric components
    Lv = Rv.shape[0]
    J = np.fliplr(np.eye(Lv))
    Rv_fba = 0.5 * (Rv + J @ Rv.conj() @ J)
    Rv = Rv_fba
    
    # Diagonal loading - stabilize eigen-decomp at finite M
    eps = 1e-3 * np.trace(Rv).real / Lv
    Rv = Rv + eps * np.eye(Lv)
    
    # SVD analysis (Hermitian PSD ⇒ singular values == eigenvalues)
    Uv, Sv, Vhv = np.linalg.svd(Rv, full_matrices=False)
    
    # MUSIC on virtual ULA
    # Key: Rv is Toeplitz with Lv x Lv, representing a contiguous virtual ULA at indices 0..Lv-1
    if use_root:
        # Root-MUSIC: polynomial-based DOA estimation (experimental - see paper limitations)
        print("[WARN] Coarray Root-MUSIC is experimental; grid search is recommended for stable results (see paper).")
        k = 2.0 * np.pi / wavelength
        doas_est, root_dbg = root_music_from_Rv(Rv, K, k, dvirt)
        P, thetas = None, None  # no spectrum needed
        dbg_extra = {"root_music": root_dbg}
    else:
        # Standard grid search
        thetas, P = music_spectrum(Rv, K, dvirt, wavelength, scan_deg=scan_deg, lags=None)
        doas_est = pick_peaks_safeguarded(thetas, P, K)
        dbg_extra = {}
    # Build SVD info dict
    svd_info = {
        "Mv": int(coarray_debug["Mv"]),
        "Rv_singular": Sv.tolist(),
        "Rv_cond": float(Sv[0] / max(Sv[-1], 1e-12)),
    }
    
    if return_debug:
        # Merge coarray debug info with algorithm-specific debug
        dbg = {
            "Rv_shape": Rv.shape,
            "one_sided_segment": (L1, L2),
            "Lv": coarray_debug["Lv"],
            "Mv": coarray_debug["Mv"],
            "lags_used": coarray_debug["lags_used"],
            "L1": coarray_debug["L1"],
            "L2": coarray_debug["L2"],
        }
        dbg.update(dbg_extra)
        dbg.update(svd_info)  # Add SVD diagnostics
        return doas_est, P, thetas, dbg
    
    # Return DOAs and SVD info when return_debug=False
    return doas_est, svd_info
