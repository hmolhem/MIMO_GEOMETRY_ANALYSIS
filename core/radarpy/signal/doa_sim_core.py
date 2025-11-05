# sim/doa_sim_core.py
import time
import numpy as np
from numpy.linalg import eigh
from .array_manifold import steering_vector

def simulate_snapshots(sensor_positions, wavelength, doas_deg, snr_db, snapshots, seed=None):
    """
    Uncorrelated narrowband sources; equal power. White noise.
    Returns X (N x M), true steering A, and SNR linear.
    """
    if seed is not None:
        np.random.seed(int(seed))

    pos = np.asarray(sensor_positions, dtype=float)
    N = pos.size
    doas = np.asarray(doas_deg, dtype=float)
    K = doas.size
    M = int(snapshots)

    A = steering_vector(pos, wavelength, doas)  # (N x K)

    # Unit signal power per source -> total signal covariance S = I_K
    S = np.eye(K, dtype=complex)
    Xs = A @ ((np.random.randn(K, M) + 1j*np.random.randn(K, M))/np.sqrt(2.0))  # signals

    snr_lin = 10.0**(snr_db/10.0)
    # Scale noise to achieve per-sensor SNR ~ snr_lin (assuming unit signal variance)
    # Here, average signal power at sensors is ~ trace(A A^H)/N = K (steering magnitude is 1)
    sig_power = K
    noise_var = sig_power / snr_lin
    W = np.sqrt(noise_var/2.0) * (np.random.randn(N, M) + 1j*np.random.randn(N, M))

    X = Xs + W
    return X, A, snr_lin

def sample_covariance(X):
    """ Rxx = (1/M) X X^H """
    M = X.shape[1]
    return (X @ X.conj().T) / float(M)

def music_spectrum(Rxx, sensor_positions, wavelength, scan_deg, k_sources):
    """
    Classic narrowband MUSIC.
    Returns pseudospectrum P(θ) (in dB, normalized) and eigen-decomposition artifacts.
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
        a = steering_vector(pos, wavelength, th)[:, 0:1]  # (N,1)
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

def run_music(sensor_positions, wavelength, doas_true_deg, k_sources, snr_db, snapshots, scan_grid=None, seed=None):
    """
    Convenience wrapper: simulate snapshots, compute Rxx, run MUSIC, pick top-k peaks.
    """
    if scan_grid is None:
        scan_grid = np.linspace(-60.0, 60.0, 1201)  # 0.1° resolution

    t0 = time.perf_counter()
    X, A, snr_lin = simulate_snapshots(sensor_positions, wavelength, doas_true_deg, snr_db, snapshots, seed=seed)
    Rxx = sample_covariance(X)
    P_db, _ = music_spectrum(Rxx, sensor_positions, wavelength, scan_grid, k_sources)
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
