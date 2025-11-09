# sim/doa_sim_core.py
"""
DOA simulation core with mutual coupling support.

Simulates received signals for Direction-of-Arrival (DOA) estimation scenarios
with optional electromagnetic mutual coupling between array elements.

Updated: November 6, 2025 - Added mutual coupling matrix (MCM) support
"""
import time
import numpy as np
from numpy.linalg import eigh
from .array_manifold import steering_vector
from typing import Optional

def simulate_snapshots(sensor_positions, wavelength, doas_deg, snr_db, snapshots, 
                      seed=None, coupling_matrix=None):
    """
    Simulate narrowband received signal snapshots with optional mutual coupling.
    
    Generates uncorrelated sources with equal power and white Gaussian noise.
    Optionally applies mutual coupling matrix to model electromagnetic interactions
    between array elements.
    
    Args:
        sensor_positions: Sensor positions (N,) in meters or normalized units
        wavelength: Signal wavelength in same units as positions
        doas_deg: True DOA angles (K,) in degrees (broadside = 0°)
        snr_db: Signal-to-noise ratio in dB (per sensor)
        snapshots: Number of time samples (M)
        seed: Optional random seed for reproducibility
        coupling_matrix: Optional (N × N) mutual coupling matrix
                        If None, ideal array with no coupling
                        If provided, applies coupling to received signal
    
    Returns:
        X: Received signal matrix (N × M) including coupling effects and noise
        A: True steering matrix (N × K) with coupling applied
        snr_lin: Linear SNR value
    
    Usage:
        >>> # Ideal case (no coupling)
        >>> X, A, snr = simulate_snapshots(positions, 1.0, [10, -15], 10, 100)
        
        >>> # With mutual coupling
        >>> from core.radarpy.signal.mutual_coupling import generate_mcm
        >>> C = generate_mcm(7, positions, model="exponential", c1=0.3)
        >>> X, A, snr = simulate_snapshots(positions, 1.0, [10, -15], 10, 100, 
        ...                                coupling_matrix=C)
    
    Mathematical Model:
        Ideal: X = A @ S + W
        With coupling: X = C @ (A @ S) + C @ W = C @ A @ S + C @ W
        where C is the mutual coupling matrix
    
    Note:
        When coupling is present, both signal and noise are affected by the MCM.
        This is the physical model for electromagnetic coupling.
    """
    if seed is not None:
        np.random.seed(int(seed))

    pos = np.asarray(sensor_positions, dtype=float)
    N = pos.size
    doas = np.asarray(doas_deg, dtype=float)
    K = doas.size
    M = int(snapshots)

    # Generate steering matrix with coupling applied
    A = steering_vector(pos, wavelength, doas, coupling_matrix=coupling_matrix)  # (N x K)

    # Unit signal power per source -> total signal covariance S = I_K
    S = np.eye(K, dtype=complex)
    Xs = A @ ((np.random.randn(K, M) + 1j*np.random.randn(K, M))/np.sqrt(2.0))  # signals

    snr_lin = 10.0**(snr_db/10.0)
    # Scale noise to achieve per-sensor SNR ~ snr_lin (assuming unit signal variance)
    # Here, average signal power at sensors is ~ trace(A A^H)/N = K (steering magnitude is 1)
    sig_power = K
    noise_var = sig_power / snr_lin
    W = np.sqrt(noise_var/2.0) * (np.random.randn(N, M) + 1j*np.random.randn(N, M))
    
    # Apply coupling to noise if MCM is provided
    # Physical model: both signal and noise pass through coupled array
    if coupling_matrix is not None:
        W = coupling_matrix @ W

    X = Xs + W
    return X, A, snr_lin

def sample_covariance(X):
    """ Rxx = (1/M) X X^H """
    M = X.shape[1]
    return (X @ X.conj().T) / float(M)

def music_spectrum(Rxx, sensor_positions, wavelength, scan_deg, k_sources, coupling_matrix=None):
    """
    Classic narrowband MUSIC spectrum with optional mutual coupling.
    
    Computes the MUSIC pseudospectrum P(θ) using eigendecomposition of the
    spatial covariance matrix. Optionally accounts for mutual coupling by
    applying the coupling matrix to the steering vectors.
    
    Args:
        Rxx: Spatial covariance matrix (N × N)
        sensor_positions: Sensor positions (N,)
        wavelength: Signal wavelength
        scan_deg: Scan angles in degrees (array-like)
        k_sources: Number of sources (for subspace partitioning)
        coupling_matrix: Optional (N × N) mutual coupling matrix
                        Should match the MCM used in signal generation
    
    Returns:
        P_db: MUSIC pseudospectrum in dB (normalized)
        eig_data: Tuple of (eigenvalues, eigenvectors)
    
    Usage:
        >>> # Ideal case
        >>> P_db, (evals, evecs) = music_spectrum(Rxx, positions, 1.0, scan_grid, 2)
        
        >>> # With coupling (must use same MCM as in simulation)
        >>> P_db, _ = music_spectrum(Rxx, positions, 1.0, scan_grid, 2, 
        ...                          coupling_matrix=C)
    
    Note:
        When coupling is present in the data (Rxx), you should use the same
        coupling matrix here for consistent steering vector computation.
    """
    # Eigendecomposition
    evals, evecs = eigh(Rxx)  # ascending
    idx = np.argsort(evals)
    evals, evecs = evals[idx], evecs[:, idx]
    N = Rxx.shape[0]
    K = int(k_sources)
    Un = evecs[:, :N-K]  # noise subspace

    thetas = np.asarray(scan_deg, dtype=float)
    P = np.empty_like(thetas, dtype=float)

    pos = np.asarray(sensor_positions, dtype=float)
    for i, th in enumerate(thetas):
        a = steering_vector(pos, wavelength, th, coupling_matrix=coupling_matrix)[:, 0:1]  # (N,1)
        denom = np.linalg.norm(Un.conj().T @ a)**2
        P[i] = 1.0 / max(denom, 1e-12)

    # Normalize and convert to dB
    P = P / (np.max(P) + 1e-18)
    P_db = 10*np.log10(P + 1e-18)
    return P_db, (evals, evecs)

def find_k_peaks(values, xs, k):
    """
    Simple peak picker: local maxima by comparison with neighbors, then top-k by height.
    xs: x-axis values (same size as 'values')
    Returns k peak locations (xs) sorted ascending.
    """
    v = np.asarray(values)
    x = np.asarray(xs)
    # Local maxima mask
    locs = []
    for i in range(1, len(v)-1):
        if v[i] > v[i-1] and v[i] > v[i+1]:
            locs.append(i)
    if len(locs) == 0:
        # fallback: just take global maxima indices
        locs = [int(np.argmax(v))]
    # rank by value
    locs = sorted(locs, key=lambda i: v[i], reverse=True)
    locs = locs[:k]
    est = np.sort(x[locs])
    return est

def run_music(sensor_positions, wavelength, doas_true_deg, k_sources, snr_db, snapshots, 
              scan_grid=None, seed=None, coupling_matrix=None):
    """
    Convenience wrapper: simulate snapshots, compute Rxx, run MUSIC, pick top-k peaks.
    
    Complete DOA estimation pipeline with optional mutual coupling support.
    
    Args:
        sensor_positions: Sensor positions (N,)
        wavelength: Signal wavelength
        doas_true_deg: True DOA angles in degrees
        k_sources: Number of sources
        snr_db: Signal-to-noise ratio in dB
        snapshots: Number of time samples
        scan_grid: Optional scan angles (default: -60° to +60°, 0.1° resolution)
        seed: Optional random seed
        coupling_matrix: Optional (N × N) mutual coupling matrix
    
    Returns:
        dict: Results containing:
            - doas_true_deg: True DOA angles (sorted)
            - doas_est_deg: Estimated DOA angles
            - P_db: MUSIC pseudospectrum
            - scan_grid: Scan angles used
            - Rxx: Sample covariance matrix
            - runtime_ms: Processing time in milliseconds
    
    Usage:
        >>> # Ideal case
        >>> result = run_music(positions, 1.0, [10, -15], 2, 10, 100)
        
        >>> # With mutual coupling
        >>> from core.radarpy.signal.mutual_coupling import generate_mcm
        >>> C = generate_mcm(7, positions, model="exponential", c1=0.3)
        >>> result = run_music(positions, 1.0, [10, -15], 2, 10, 100, 
        ...                    coupling_matrix=C)
    """
    if scan_grid is None:
        scan_grid = np.linspace(-60.0, 60.0, 1201)  # 0.1° resolution

    t0 = time.perf_counter()
    X, A, snr_lin = simulate_snapshots(sensor_positions, wavelength, doas_true_deg, snr_db, 
                                       snapshots, seed=seed, coupling_matrix=coupling_matrix)
    Rxx = sample_covariance(X)
    P_db, _ = music_spectrum(Rxx, sensor_positions, wavelength, scan_grid, k_sources, 
                            coupling_matrix=coupling_matrix)
    doas_est = find_k_peaks(P_db, scan_grid, int(k_sources))
    runtime_ms = 1000.0 * (time.perf_counter() - t0)
    return {
        "doas_true_deg": np.sort(np.asarray(doas_true_deg, dtype=float)),
        "doas_est_deg": doas_est,
        "P_db": P_db,
        "scan_grid": scan_grid,
        "Rxx": Rxx,
        "runtime_ms": runtime_ms,
    }
