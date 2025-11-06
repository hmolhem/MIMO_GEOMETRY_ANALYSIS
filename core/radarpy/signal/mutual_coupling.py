"""
Mutual Coupling Matrix (MCM) modeling for antenna arrays.

This module provides tools to model electromagnetic mutual coupling between
array elements, which affects the received signal in real-world antenna arrays.

Functions:
    generate_mcm: Create mutual coupling matrix using various models
    apply_coupling: Apply MCM to steering vectors or received signals
    compensate_coupling: Remove coupling effects (if MCM is known)

Models Supported:
    - exponential: Exponential decay with distance
    - toeplitz: Symmetric Toeplitz structure
    - custom: User-provided matrix from measurements

Author: RadarPy Development Team
Date: November 6, 2025
"""

import numpy as np
from typing import Optional, Union, Tuple


def generate_mcm_exponential(
    num_sensors: int,
    positions: np.ndarray,
    c0: float = 1.0,
    c1: float = 0.3,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Generate mutual coupling matrix using exponential decay model.
    
    The coupling between sensor i and j follows:
        C[i,j] = c0 (if i==j) or c1 * exp(-alpha * |pos[i] - pos[j]|)
    
    This model assumes coupling strength decreases exponentially with distance.
    
    Args:
        num_sensors: Number of array elements (N)
        positions: Sensor positions in units of wavelength (shape: (N,))
        c0: Self-coupling coefficient (diagonal elements), typically 1.0
        c1: Mutual coupling strength coefficient (0 to 1)
        alpha: Decay rate (larger = faster decay with distance)
    
    Returns:
        coupling_matrix: Complex coupling matrix (N × N)
    
    Usage:
        >>> positions = np.array([0, 1, 2, 3])  # ULA with unit spacing
        >>> C = generate_mcm_exponential(4, positions, c1=0.3, alpha=0.5)
        >>> print(C.shape)
        (4, 4)
    
    References:
        [1] Friedlander & Weiss, "Direction finding in the presence of mutual
            coupling," IEEE TAP, 1991.
    """
    C = np.zeros((num_sensors, num_sensors), dtype=complex)
    
    for i in range(num_sensors):
        for j in range(num_sensors):
            if i == j:
                C[i, j] = c0  # Self-coupling (diagonal)
            else:
                distance = np.abs(positions[i] - positions[j])
                C[i, j] = c1 * np.exp(-alpha * distance)
    
    return C


def generate_mcm_toeplitz(
    num_sensors: int,
    coupling_coeffs: np.ndarray
) -> np.ndarray:
    """
    Generate symmetric Toeplitz mutual coupling matrix.
    
    For uniform linear arrays, coupling often exhibits Toeplitz structure
    where C[i,j] depends only on |i-j|.
    
    Args:
        num_sensors: Number of array elements (N)
        coupling_coeffs: Coupling coefficients for lags 0, 1, 2, ...
                        Example: [1.0, 0.3, 0.1, 0.05] means:
                        - Self coupling = 1.0
                        - Adjacent elements = 0.3
                        - 2-element spacing = 0.1, etc.
    
    Returns:
        coupling_matrix: Complex coupling matrix (N × N)
    
    Usage:
        >>> C = generate_mcm_toeplitz(5, np.array([1.0, 0.3, 0.1, 0.05, 0.02]))
        >>> print(C[0, :])  # First row shows coupling pattern
        [1.0+0.j 0.3+0.j 0.1+0.j 0.05+0.j 0.02+0.j]
    
    References:
        [1] Svantesson, "Modeling and estimation of mutual coupling in a
            uniform linear array of dipoles," IEEE ICASSP, 1999.
    """
    from scipy.linalg import toeplitz
    
    # Ensure we have enough coefficients
    if len(coupling_coeffs) < num_sensors:
        # Pad with zeros if not enough coefficients provided
        padding = np.zeros(num_sensors - len(coupling_coeffs))
        coupling_coeffs = np.concatenate([coupling_coeffs, padding])
    
    # Create symmetric Toeplitz matrix
    C = toeplitz(coupling_coeffs[:num_sensors])
    
    return C.astype(complex)


def generate_mcm_measured(
    matrix_file: str
) -> np.ndarray:
    """
    Load mutual coupling matrix from measurement file.
    
    Supports loading MCM from:
    - CSV files (real or complex values)
    - NPY files (numpy binary format)
    
    Args:
        matrix_file: Path to file containing MCM
                    CSV format: N×N matrix, can include complex numbers as "real+imag*j"
                    NPY format: numpy array saved with np.save()
    
    Returns:
        coupling_matrix: Complex coupling matrix (N × N)
    
    Usage:
        >>> C = generate_mcm_measured("mcm_z5_measured.csv")
        >>> print(f"Loaded {C.shape[0]}-element coupling matrix")
    
    Raises:
        FileNotFoundError: If matrix_file doesn't exist
        ValueError: If matrix is not square
    """
    import os
    
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Coupling matrix file not found: {matrix_file}")
    
    # Load based on file extension
    if matrix_file.endswith('.npy'):
        C = np.load(matrix_file)
    elif matrix_file.endswith('.csv'):
        # Try loading as complex (if saved with complex notation)
        try:
            C = np.loadtxt(matrix_file, dtype=complex, delimiter=',')
        except:
            # If that fails, try loading as real and convert to complex
            C = np.loadtxt(matrix_file, delimiter=',').astype(complex)
    else:
        raise ValueError(f"Unsupported file format: {matrix_file}. Use .csv or .npy")
    
    # Validate square matrix
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Coupling matrix must be square, got shape {C.shape}")
    
    return C


def generate_mcm(
    num_sensors: int,
    positions: np.ndarray,
    model: str = "exponential",
    **kwargs
) -> Optional[np.ndarray]:
    """
    Generate mutual coupling matrix using specified model.
    
    This is the main entry point for MCM generation. It dispatches to
    specific model implementations based on the 'model' parameter.
    
    Args:
        num_sensors: Number of array elements (N)
        positions: Sensor positions in wavelength units (shape: (N,))
        model: Coupling model type:
               - "exponential": Exponential decay (requires c0, c1, alpha)
               - "toeplitz": Symmetric Toeplitz (requires coupling_coeffs)
               - "measured": Load from file (requires matrix_file)
               - "none" or None: Return None (no coupling)
        **kwargs: Model-specific parameters (see individual functions)
    
    Returns:
        coupling_matrix: Complex coupling matrix (N × N), or None if model="none"
    
    Usage:
        >>> # Exponential model
        >>> C1 = generate_mcm(7, positions, model="exponential", c1=0.3, alpha=0.5)
        
        >>> # Toeplitz model
        >>> C2 = generate_mcm(7, positions, model="toeplitz", 
        ...                   coupling_coeffs=[1.0, 0.3, 0.1, 0.05])
        
        >>> # Measured data
        >>> C3 = generate_mcm(7, positions, model="measured", 
        ...                   matrix_file="mcm_data.csv")
        
        >>> # No coupling
        >>> C4 = generate_mcm(7, positions, model="none")
        >>> assert C4 is None
    
    Raises:
        ValueError: If model type is not recognized
    """
    if model is None or model.lower() == "none":
        return None
    
    elif model.lower() == "exponential":
        c0 = kwargs.get('c0', 1.0)
        c1 = kwargs.get('c1', 0.3)
        alpha = kwargs.get('alpha', 0.5)
        return generate_mcm_exponential(num_sensors, positions, c0, c1, alpha)
    
    elif model.lower() == "toeplitz":
        coupling_coeffs = kwargs.get('coupling_coeffs', None)
        if coupling_coeffs is None:
            raise ValueError("toeplitz model requires 'coupling_coeffs' parameter")
        return generate_mcm_toeplitz(num_sensors, coupling_coeffs)
    
    elif model.lower() == "measured":
        matrix_file = kwargs.get('matrix_file', None)
        if matrix_file is None:
            raise ValueError("measured model requires 'matrix_file' parameter")
        return generate_mcm_measured(matrix_file)
    
    else:
        raise ValueError(f"Unknown coupling model: {model}. "
                        f"Use 'exponential', 'toeplitz', 'measured', or 'none'")


def apply_coupling(
    signal_or_steering: np.ndarray,
    coupling_matrix: Optional[np.ndarray]
) -> np.ndarray:
    """
    Apply mutual coupling to steering vector or received signal.
    
    Transforms ideal signal model to coupled model:
        y_coupled = C @ y_ideal
    
    where C is the coupling matrix.
    
    Args:
        signal_or_steering: Either:
                           - Steering vector: shape (N,) or (N, 1)
                           - Received signal: shape (N, M) for M snapshots
        coupling_matrix: Coupling matrix (N × N), or None to skip
    
    Returns:
        coupled_signal: Signal with coupling applied, same shape as input
    
    Usage:
        >>> # Apply to steering vector
        >>> a = np.exp(1j * np.pi * np.sin(theta) * positions)
        >>> a_coupled = apply_coupling(a, C)
        
        >>> # Apply to received signal (N sensors, M snapshots)
        >>> y = np.random.randn(7, 100) + 1j * np.random.randn(7, 100)
        >>> y_coupled = apply_coupling(y, C)
    
    Note:
        If coupling_matrix is None, returns input unchanged (ideal case).
    """
    if coupling_matrix is None:
        return signal_or_steering  # No coupling, return as-is
    
    # Apply coupling: C @ signal
    return coupling_matrix @ signal_or_steering


def compensate_coupling(
    coupled_signal: np.ndarray,
    coupling_matrix: np.ndarray
) -> np.ndarray:
    """
    Compensate for mutual coupling (if MCM is known).
    
    Given coupled signal y_coupled = C @ y_ideal, recovers ideal signal:
        y_ideal = C^(-1) @ y_coupled
    
    Args:
        coupled_signal: Signal with coupling effects (N,) or (N, M)
        coupling_matrix: Known coupling matrix (N × N)
    
    Returns:
        decoupled_signal: Signal with coupling removed
    
    Usage:
        >>> # Assume we measured y_coupled and know C
        >>> y_ideal_estimate = compensate_coupling(y_coupled, C)
    
    Note:
        Requires accurate knowledge of coupling matrix. In practice, C
        is often estimated through calibration procedures.
    
    Raises:
        np.linalg.LinAlgError: If coupling matrix is singular
    """
    # Invert coupling matrix
    C_inv = np.linalg.inv(coupling_matrix)
    
    # Apply inverse coupling
    return C_inv @ coupled_signal


def get_coupling_info(coupling_matrix: np.ndarray) -> dict:
    """
    Analyze properties of coupling matrix.
    
    Args:
        coupling_matrix: Coupling matrix (N × N)
    
    Returns:
        info: Dictionary containing:
              - 'num_sensors': Number of array elements
              - 'is_symmetric': Whether C is symmetric
              - 'is_hermitian': Whether C is Hermitian
              - 'condition_number': Matrix condition number (stability indicator)
              - 'max_off_diagonal': Maximum coupling between different elements
              - 'avg_off_diagonal': Average coupling between different elements
    
    Usage:
        >>> info = get_coupling_info(C)
        >>> print(f"Max coupling: {info['max_off_diagonal']:.3f}")
        >>> print(f"Condition number: {info['condition_number']:.2e}")
    """
    N = coupling_matrix.shape[0]
    
    # Extract off-diagonal elements
    mask = ~np.eye(N, dtype=bool)
    off_diag = np.abs(coupling_matrix[mask])
    
    info = {
        'num_sensors': N,
        'is_symmetric': np.allclose(coupling_matrix, coupling_matrix.T),
        'is_hermitian': np.allclose(coupling_matrix, coupling_matrix.conj().T),
        'condition_number': np.linalg.cond(coupling_matrix),
        'max_off_diagonal': np.max(off_diag),
        'avg_off_diagonal': np.mean(off_diag),
    }
    
    return info


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Mutual Coupling Matrix (MCM) Module - Demo")
    print("="*70)
    
    # Example 1: Exponential model for Z5 array
    print("\n[Example 1] Exponential MCM for Z5 array (N=7)")
    positions_z5 = np.array([0, 1, 3, 5, 7, 9, 12])  # Example Z5 positions
    C_exp = generate_mcm(7, positions_z5, model="exponential", c1=0.3, alpha=0.5)
    print(f"Matrix shape: {C_exp.shape}")
    print(f"Diagonal (self-coupling): {np.diag(C_exp)[:3]} ...")
    print(f"C[0,1] (adjacent): {C_exp[0, 1]:.4f}")
    print(f"C[0,6] (far apart): {C_exp[0, 6]:.4f}")
    
    info = get_coupling_info(C_exp)
    print(f"Max off-diagonal coupling: {info['max_off_diagonal']:.4f}")
    print(f"Condition number: {info['condition_number']:.2e}")
    
    # Example 2: Toeplitz model for ULA
    print("\n[Example 2] Toeplitz MCM for ULA (N=5)")
    C_toep = generate_mcm(5, np.arange(5), model="toeplitz",
                         coupling_coeffs=np.array([1.0, 0.3, 0.1, 0.05, 0.02]))
    print("Toeplitz matrix:")
    print(C_toep.real)
    
    # Example 3: Apply coupling to steering vector
    print("\n[Example 3] Apply coupling to steering vector")
    theta = np.deg2rad(30)  # 30 degrees DOA
    positions_ula = np.arange(5)
    a_ideal = np.exp(1j * np.pi * np.sin(theta) * positions_ula)
    a_coupled = apply_coupling(a_ideal, C_toep)
    print(f"Ideal steering vector magnitude: {np.abs(a_ideal)}")
    print(f"Coupled steering vector magnitude: {np.abs(a_coupled)}")
    print(f"Phase distortion: {np.angle(a_coupled) - np.angle(a_ideal)} rad")
    
    # Example 4: No coupling (ideal case)
    print("\n[Example 4] No coupling (ideal case)")
    C_none = generate_mcm(5, np.arange(5), model="none")
    print(f"Coupling matrix: {C_none}")
    a_no_coupling = apply_coupling(a_ideal, C_none)
    print(f"Signal unchanged: {np.allclose(a_ideal, a_no_coupling)}")
    
    print("\n" + "="*70)
    print("Demo complete! See function docstrings for more usage examples.")
    print("="*70)
