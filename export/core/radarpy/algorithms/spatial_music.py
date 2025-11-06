# algorithms/spatial_music.py
import numpy as np
from numpy.linalg import eigh

def steering_vector_spatial(theta_deg, positions, wavelength):
    """
    Steering vector for arbitrary sensor positions (in meters).
    positions: array of N sensor positions in meters
    theta_deg: scalar angle in degrees
    """
    k = 2.0 * np.pi / wavelength
    theta_rad = np.deg2rad(theta_deg)
    return np.exp(1j * k * positions * np.sin(theta_rad))

def music_spectrum_spatial(Rxx, positions, wavelength, K, scan_deg=(-60, 60, 0.1)):
    """
    Standard spatial MUSIC on physical array covariance Rxx.
    Returns: angles (deg), P(angles).
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
