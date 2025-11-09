# sim/array_manifold.py
"""
Array manifold computation with optional mutual coupling support.

This module computes steering vectors for antenna arrays, with optional
modeling of electromagnetic mutual coupling between array elements.

Updated: November 6, 2025 - Added mutual coupling matrix (MCM) support
"""
import numpy as np
from typing import Optional

def deg2rad(x): return np.deg2rad(x)

def steering_vector(sensor_positions, wavelength, doa_deg, coupling_matrix=None):
    """
    Compute array steering vector(s) with optional mutual coupling.
    
    The steering vector represents the array's response to a plane wave from
    direction doa_deg. With mutual coupling, the received signal is modified
    by electromagnetic interactions between array elements.
    
    Args:
        sensor_positions: Array-like, sensor positions in meters (or normalized units)
                         Shape: (N,) for N sensors
        wavelength: Float, signal wavelength in meters (or same units as positions)
        doa_deg: Scalar or array of DOA angles in degrees
                Broadside = 0°, positive angles to left (standard DOA convention)
                Shape: scalar or (L,) for L directions
        coupling_matrix: Optional (N × N) complex matrix modeling mutual coupling
                        If None, ideal array with no coupling (default behavior)
                        If provided, applies: A_coupled = C @ A_ideal
    
    Returns:
        A: Complex array manifold matrix
           Shape: (N, L) where N = num sensors, L = num DOAs
           Each column is the steering vector for one DOA angle
    
    Usage:
        >>> # Ideal array (no coupling)
        >>> positions = np.array([0, 0.5, 1.0, 1.5])  # λ/2 spacing
        >>> A = steering_vector(positions, wavelength=1.0, doa_deg=[0, 30, -20])
        >>> print(A.shape)  # (4, 3)
        
        >>> # With mutual coupling
        >>> from core.radarpy.signal.mutual_coupling import generate_mcm
        >>> C = generate_mcm(4, positions, model="exponential", c1=0.3, alpha=0.5)
        >>> A_coupled = steering_vector(positions, 1.0, [0, 30], coupling_matrix=C)
    
    Mathematical Model:
        Ideal steering vector: a(θ) = exp(j * k * x * sin(θ))
        where k = 2π/λ, x = sensor positions
        
        With coupling: a_coupled(θ) = C @ a(θ)
        where C is the mutual coupling matrix (MCM)
    
    References:
        [1] Van Trees, "Optimum Array Processing," Wiley, 2002
        [2] Friedlander & Weiss, "Direction finding in the presence of mutual
            coupling," IEEE TAP, 1991
    """
    pos = np.asarray(sensor_positions, dtype=float).reshape(-1, 1)  # (N,1)
    doas = np.atleast_1d(doa_deg).astype(float).reshape(1, -1)      # (1,L)
    k = 2.0 * np.pi / float(wavelength)
    
    # Assume linear array on x-axis; phase = k * x * sin(theta)
    phase = k * pos @ np.sin(deg2rad(doas))
    A_ideal = np.exp(1j * phase)  # (N, L) ideal manifold
    
    # Apply mutual coupling if provided
    if coupling_matrix is not None:
        # C @ A for each DOA column
        A_coupled = coupling_matrix @ A_ideal  # (N,N) @ (N,L) = (N,L)
        return A_coupled
    
    return A_ideal
