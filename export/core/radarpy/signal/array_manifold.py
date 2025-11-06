# sim/array_manifold.py
import numpy as np

def deg2rad(x): return np.deg2rad(x)

def steering_vector(sensor_positions, wavelength, doa_deg):
    """
    sensor_positions : array-like, meters (or normalized units)
    wavelength       : float, meters (or same units as positions)
    doa_deg          : scalar or array of degrees (broadside=0°, positive to left)
    Returns: A (N x L) complex manifold, N sensors, L DOAs.
    """
    pos = np.asarray(sensor_positions, dtype=float).reshape(-1, 1)  # (N,1)
    doas = np.atleast_1d(doa_deg).astype(float).reshape(1, -1)      # (1,L)
    k = 2.0 * np.pi / float(wavelength)
    # Assume linear array on x-axis; phase = k * x * sin(theta)
    phase = k * pos @ np.sin(deg2rad(doas))
    return np.exp(1j * phase)
