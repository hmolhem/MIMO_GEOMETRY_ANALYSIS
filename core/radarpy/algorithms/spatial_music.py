# algorithms/spatial_music.py
"""
Spatial MUSIC Algorithm Implementation.

This module implements the standard Spatial MUSIC (Multiple Signal Classification)
algorithm for Direction-of-Arrival (DOA) estimation on physical sensor arrays.

Key Features:
    - Grid-based DOA estimation via pseudospectrum peaks
    - Eigen-decomposition with signal/noise subspace separation
    - SVD analysis for condition number tracking
    - Arbitrary sensor position support (not limited to ULA)

Mathematical Background:
    Spatial MUSIC operates directly on the physical array covariance R_xx.
    It exploits orthogonality between signal steering vectors and noise subspace
    to identify DOA angles where P(θ) = 1 / ||E_n^H a(θ)||² has peaks.

Comparison with Coarray MUSIC:
    - Spatial MUSIC: Uses N physical sensors → K_max ≈ N-1 sources
    - Coarray MUSIC: Uses N² virtual sensors → K_max ≈ N²/2 sources
    - Spatial MUSIC has better conditioning but lower DOF

Author: [Your Name]
Date: November 6, 2025
Version: 1.0.0
"""
import numpy as np
from numpy.linalg import eigh

def steering_vector_spatial(theta_deg, positions, wavelength):
    """
    Compute steering vector for arbitrary sensor array geometry.
    
    Generates the array response vector a(θ) for given DOA angle and
    sensor positions. Supports arbitrary (non-uniform) geometries.
    
    Args:
        theta_deg (float): DOA angle in degrees
        positions (np.ndarray): Sensor positions in meters (length N)
        wavelength (float): Carrier wavelength in meters
    
    Returns:
        np.ndarray: Complex steering vector of shape (N,)
            a(θ) = [e^(j k r₀ sin θ), e^(j k r₁ sin θ), ..., e^(j k r_{N-1} sin θ)]^T
    
    Mathematical Form:
        a_n(θ) = exp(j * 2π/λ * r_n * sin(θ))
        where r_n is the position of n-th sensor
    
    Usage:
        >>> positions = np.array([0, 0.5, 1.0, 1.5])  # 4 sensors
        >>> a = steering_vector_spatial(theta_deg=30, positions=positions, wavelength=1.0)
        >>> a.shape
        (4,)
    
    Note:
        Assumes far-field narrowband signal model with plane wave propagation.
    """
    k = 2.0 * np.pi / wavelength
    theta_rad = np.deg2rad(theta_deg)
    return np.exp(1j * k * positions * np.sin(theta_rad))

def music_spectrum_spatial(Rxx, positions, wavelength, K, scan_deg=(-60, 60, 0.1)):
    """
    Compute Spatial MUSIC pseudospectrum for physical array.
    
    Performs eigen-decomposition of physical array covariance Rxx and computes
    MUSIC pseudospectrum by scanning over angle grid.
    
    Args:
        Rxx (np.ndarray): Physical array covariance matrix of shape (N, N)
        positions (np.ndarray): Physical sensor positions in meters (length N)
        wavelength (float): Carrier wavelength in meters
        K (int): Number of sources
        scan_deg (tuple): Scan range (theta_min, theta_max, step) in degrees
    
    Returns:
        tuple: (angles, P) where:
            - angles (np.ndarray): Scanned angles in degrees
            - P (np.ndarray): MUSIC pseudospectrum values (higher = more likely DOA)
    
    Algorithm:
        1. Eigen-decompose Rxx: Rxx = V Λ V^H
        2. Identify noise subspace: E_n = V[:, :N-K] (smallest eigenvectors)
        3. For each angle θ in scan range:
            a. Compute steering vector a(θ)
            b. P(θ) = 1 / |a(θ)^H E_n E_n^H a(θ)|²
        4. Return angles and spectrum P
    
    Usage:
        >>> Rxx = (X @ X.conj().T) / M  # Sample covariance
        >>> angles, P = music_spectrum_spatial(
        ...     Rxx, positions, wavelength=1.0, K=2, scan_deg=(-60, 60, 0.1)
        ... )
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(angles, P); plt.show()
    
    Note:
        Uses standard MUSIC on physical sensors. For enhanced DOF, use
        estimate_doa_coarray_music() which operates on virtual coarray.
    """
    # EVD (ascending order)
    vals, vecs = eigh(Rxx)
    idx = np.argsort(vals.real)
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    N = Rxx.shape[0]
    d_noise = N - K
    if d_noise <= 0:
        d_noise = 1  # guard
    En = vecs[:, :d_noise]  # noise subspace
    
    # Scan
    theta_min, theta_max, theta_step = scan_deg
    thetas = np.arange(theta_min, theta_max + theta_step, theta_step)
    P = np.zeros_like(thetas)
    
    for i, theta in enumerate(thetas):
        a = steering_vector_spatial(theta, positions, wavelength)
        denom = np.abs(a.conj().T @ En @ En.conj().T @ a)
        P[i] = 1.0 / np.maximum(denom.real, 1e-12)
    
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
        lo = max(0, i - 1)
        hi = min(len(P), i + 2)
        P[lo:hi] = 0.0
    est = sorted([thetas[i] for i in idxs])
    return np.array(est, dtype=float)

def estimate_doa_spatial_music(X, positions, d_phys, wavelength, K, scan_deg=(-60, 60, 0.1)):
    """
    Spatial MUSIC on physical array.
    X: snapshots matrix (N x M)
    positions: integer-grid sensor indices
    d_phys: physical grid spacing (meters)
    wavelength: signal wavelength (meters)
    K: number of sources
    
    Returns:
        doas_est: estimated DOAs (degrees)
        info: dict with SVD diagnostics (Rx_singular, Rx_cond)
    """
    # Convert integer grid to physical positions
    pos_m = np.asarray(positions, dtype=float) * d_phys
    
    # Sample covariance
    Rxx = X @ X.conj().T / max(1, X.shape[1])
    
    # SVD analysis (Hermitian PSD ⇒ singular values == eigenvalues)
    Ux, Sx, Vhx = np.linalg.svd(Rxx, full_matrices=False)
    
    # MUSIC spectrum
    thetas, P = music_spectrum_spatial(Rxx, pos_m, wavelength, K, scan_deg)
    
    # Peak picking
    doas_est = pick_peaks_safeguarded(thetas, P, K)
    
    # Build info dict with SVD diagnostics
    info = {
        "Rx_singular": Sx.tolist(),
        "Rx_cond": float(Sx[0] / max(Sx[-1], 1e-12)),
    }
    
    return doas_est, info
