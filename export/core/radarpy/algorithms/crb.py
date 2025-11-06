# utils/crb.py
import numpy as np

def crb_single_source_ula_deg(Mv, d, wavelength, snr_linear, theta_deg, snapshots):
    """
    Approximate deterministic CRB for a single DOA on a ULA of Mv sensors with spacing d.
    Returns CRB in degrees^2, then you can sqrt for std (deg).
    Formula (common approximation):
      var(theta) >= 1 / [ 2 * snapshots * SNR * (2π d / λ)^2 * Σ_{m=0}^{Mv-1} (m - (Mv-1)/2)^2 * cos^2(theta) ]
    """
    if Mv <= 1 or snr_linear <= 0 or snapshots <= 0:
        return np.inf
    k = 2.0 * np.pi / wavelength
    m = np.arange(Mv)
    m0 = (Mv - 1) / 2.0
    beta = np.sum((m - m0) ** 2)
    theta = np.deg2rad(theta_deg)
    denom = 2.0 * snapshots * snr_linear * (k * d) ** 2 * beta * (np.cos(theta) ** 2 + 1e-12)
    var_rad2 = 1.0 / denom
    var_deg2 = (180.0 / np.pi) ** 2 * var_rad2
    return var_deg2

def crb_pair_worst_deg(Mv, d, wavelength, snr_linear, theta_pair_deg, snapshots):
    """Return the max of the two single-source CRBs (deg^2) as a simple overlay line."""
    th1, th2 = theta_pair_deg
    v1 = crb_single_source_ula_deg(Mv, d, wavelength, snr_linear, th1, snapshots)
    v2 = crb_single_source_ula_deg(Mv, d, wavelength, snr_linear, th2, snapshots)
    return max(v1, v2)
