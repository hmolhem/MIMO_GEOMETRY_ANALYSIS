"""
MUSIC (Multiple Signal Classification) Algorithm for DOA Estimation
====================================================================

Implements the MUSIC algorithm for direction-of-arrival estimation with
sparse MIMO arrays. Works with any array geometry from the framework.

References
----------
1. Schmidt, R. (1986). "Multiple emitter location and signal parameter estimation"
   IEEE Transactions on Antennas and Propagation, 34(3), 276-280.
2. Pal & Vaidyanathan (2010). "Nested Arrays: A Novel Approach to Array Processing"
   IEEE Transactions on Signal Processing.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings
import sys
import os

# Add path to access core radarpy modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.radarpy.signal.mutual_coupling import generate_mcm_exponential, generate_mcm_toeplitz
    MCM_AVAILABLE = True
except ImportError:
    MCM_AVAILABLE = False
    warnings.warn("Mutual coupling module not available. MCM features disabled.")


class MUSICEstimator:
    """
    MUSIC algorithm for DOA estimation with sparse arrays.
    
    The MUSIC (Multiple Signal Classification) algorithm exploits the orthogonality
    between signal and noise subspaces to estimate DOA angles with high resolution.
    
    Parameters
    ----------
    sensor_positions : array-like
        Physical sensor positions in wavelengths or meters.
        Can be obtained from any array processor's `.data.sensors_positions`.
        
    wavelength : float, optional
        Signal wavelength (default: 1.0).
        - If sensor_positions in meters: use actual wavelength in meters
        - If sensor_positions normalized: use 1.0
        
    angle_range : tuple, optional
        Search range for DOA angles in degrees (default: (-90, 90)).
        
    angle_resolution : float, optional
        Angular resolution in degrees (default: 0.5).
        Smaller values give finer resolution but slower computation.
    
    enable_mcm : bool, optional
        Enable mutual coupling matrix modeling (default: False).
        
    mcm_model : str, optional
        MCM model type: 'exponential' or 'toeplitz' (default: 'exponential').
        
    mcm_params : dict, optional
        Parameters for MCM model:
        - For 'exponential': {'c1': 0.3, 'alpha': 0.5}
        - For 'toeplitz': {'coupling_values': [...]}
        
    Attributes
    ----------
    sensor_positions : np.ndarray
        Array sensor positions.
        
    N : int
        Number of physical sensors.
        
    angle_grid : np.ndarray
        Grid of angles for spectrum computation.
    
    coupling_matrix : np.ndarray or None
        Mutual coupling matrix (N × N) if MCM enabled, else None.
        
    Examples
    --------
    **Example 1: Basic usage with Z5 array**
    
    >>> from doa_estimation import MUSICEstimator
    >>> from geometry_processors.z5_processor import Z5ArrayProcessor
    >>> 
    >>> # Create Z5 array
    >>> z5 = Z5ArrayProcessor(N=7, d=0.5)
    >>> z5.run_full_analysis()
    >>> 
    >>> # Create MUSIC estimator
    >>> estimator = MUSICEstimator(
    >>>     sensor_positions=z5.data.sensors_positions,
    >>>     wavelength=1.0
    >>> )
    >>> 
    >>> # Simulate signals from 3 sources
    >>> true_angles = [-30, 10, 45]
    >>> X = estimator.simulate_signals(true_angles, SNR_dB=10, snapshots=200)
    >>> 
    >>> # Estimate DOAs
    >>> estimated_angles = estimator.estimate(X, K_sources=3)
    >>> print(f"True: {true_angles}")
    >>> print(f"Estimated: {estimated_angles}")
    
    **Example 2: With custom angle range**
    
    >>> estimator = MUSICEstimator(
    >>>     sensor_positions=z5.data.sensors_positions,
    >>>     wavelength=1.0,
    >>>     angle_range=(-60, 60),  # Limit search range
    >>>     angle_resolution=0.25   # Higher resolution
    >>> )
    
    **Example 3: With Mutual Coupling Matrix (MCM)**
    
    >>> # Enable MCM with exponential decay model
    >>> estimator = MUSICEstimator(
    >>>     sensor_positions=z5.data.sensors_positions,
    >>>     wavelength=1.0,
    >>>     enable_mcm=True,
    >>>     mcm_model='exponential',
    >>>     mcm_params={'c1': 0.3, 'alpha': 0.5}
    >>> )
    >>> # MCM is automatically applied to steering vectors
    
    **Example 4: MCM with Toeplitz model**
    
    >>> estimator = MUSICEstimator(
    >>>     sensor_positions=z5.data.sensors_positions,
    >>>     wavelength=1.0,
    >>>     enable_mcm=True,
    >>>     mcm_model='toeplitz',
    >>>     mcm_params={'coupling_values': [1.0, 0.3, 0.15, 0.08]}
    >>> )
    
    **Example 5: Full spectrum analysis**
    
    >>> estimated_angles, spectrum = estimator.estimate(
    >>>     X, K_sources=3, return_spectrum=True
    >>> )
    >>> # Plot spectrum
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(estimator.angle_grid, 10*np.log10(spectrum))
    >>> plt.xlabel('Angle (degrees)')
    >>> plt.ylabel('MUSIC Spectrum (dB)')
    
    Notes
    -----
    - Requires K_sources < N (number of sources < number of sensors)
    - Performance depends on SNR, snapshots, and source separation
    - For sparse arrays, can detect up to K_max = L/2 sources (L from coarray)
    - White Gaussian noise assumed
    
    See Also
    --------
    SignalSimulator : Simulate received signals for testing
    DOAMetrics : Evaluate estimation performance
    """
    
    def __init__(
        self,
        sensor_positions: Union[List[float], np.ndarray],
        wavelength: float = 1.0,
        angle_range: Tuple[float, float] = (-90, 90),
        angle_resolution: float = 0.5,
        enable_mcm: bool = False,
        mcm_model: str = 'exponential',
        mcm_params: Optional[dict] = None
    ):
        """Initialize MUSIC estimator with array geometry and optional MCM."""
        self.sensor_positions = np.array(sensor_positions, dtype=float)
        self.N = len(self.sensor_positions)
        self.wavelength = wavelength
        self.angle_range = angle_range
        self.angle_resolution = angle_resolution
        
        # Mutual Coupling Matrix configuration
        self.enable_mcm = enable_mcm
        self.mcm_model = mcm_model
        self.mcm_params = mcm_params or {}
        self.coupling_matrix = None
        
        # Generate MCM if enabled
        if self.enable_mcm:
            self._generate_coupling_matrix()
        
        # Create angle grid for spectrum computation
        self.angle_grid = np.arange(
            angle_range[0], 
            angle_range[1] + angle_resolution, 
            angle_resolution
        )
        
        # Pre-compute steering vectors for efficiency
        self._steering_matrix = self._compute_steering_matrix()
    
    def _generate_coupling_matrix(self):
        """Generate mutual coupling matrix based on configuration."""
        if not MCM_AVAILABLE:
            warnings.warn(
                "MCM requested but mutual_coupling module not available. "
                "Operating without coupling effects."
            )
            self.enable_mcm = False
            return
        
        if self.mcm_model == 'exponential':
            c1 = self.mcm_params.get('c1', 0.3)
            alpha = self.mcm_params.get('alpha', 0.5)
            self.coupling_matrix = generate_mcm_exponential(
                self.N, self.sensor_positions, c1=c1, alpha=alpha
            )
        elif self.mcm_model == 'toeplitz':
            coupling_values = self.mcm_params.get('coupling_values', None)
            if coupling_values is None:
                # Default Toeplitz values
                coupling_values = [1.0] + [0.3 * 0.5**i for i in range(self.N-1)]
            self.coupling_matrix = generate_mcm_toeplitz(self.N, coupling_values)
        else:
            raise ValueError(f"Unknown MCM model: {self.mcm_model}")
        
        print(f"[OK] MCM enabled: {self.mcm_model} model (c1={self.mcm_params.get('c1', 0.3):.2f})")
    
    def _compute_steering_matrix(self) -> np.ndarray:
        """
        Pre-compute steering vectors for all angles in grid.
        
        Returns
        -------
        np.ndarray
            Steering matrix (N_sensors × N_angles) with complex exponentials.
            If MCM enabled, coupling effects are applied.
            
        Notes
        -----
        Steering vector for angle θ:
            a(θ) = exp(j × 2π × sensor_positions / λ × sin(θ))
        With MCM:
            a_coupled(θ) = C @ a(θ)
        """
        N_angles = len(self.angle_grid)
        A = np.zeros((self.N, N_angles), dtype=complex)
        
        angles_rad = np.deg2rad(self.angle_grid)
        
        for i, theta in enumerate(angles_rad):
            # Phase shift: 2π × (position / wavelength) × sin(θ)
            phase = 2 * np.pi * self.sensor_positions / self.wavelength * np.sin(theta)
            A[:, i] = np.exp(1j * phase)
        
        # Apply mutual coupling if enabled
        if self.enable_mcm and self.coupling_matrix is not None:
            A = self.coupling_matrix @ A  # (N,N) @ (N,M) = (N,M)
        
        return A
    
    def steering_vector(self, angle_deg: float) -> np.ndarray:
        """
        Compute steering vector for a specific angle.
        
        Parameters
        ----------
        angle_deg : float
            Angle in degrees.
            
        Returns
        -------
        np.ndarray
            Steering vector (N_sensors × 1) complex array.
            If MCM enabled, coupling effects are applied.
        """
        angle_rad = np.deg2rad(angle_deg)
        phase = 2 * np.pi * self.sensor_positions / self.wavelength * np.sin(angle_rad)
        a = np.exp(1j * phase)
        
        # Apply mutual coupling if enabled
        if self.enable_mcm and self.coupling_matrix is not None:
            a = self.coupling_matrix @ a
        
        return a
    
    def estimate(
        self,
        X: np.ndarray,
        K_sources: int,
        return_spectrum: bool = False,
        method: str = 'peaks'
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate DOA angles using MUSIC algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Received signal matrix (N_sensors × N_snapshots).
            Complex-valued array of received signals.
            
        K_sources : int
            Number of signal sources to detect.
            Must satisfy: 1 ≤ K_sources < N_sensors.
            
        return_spectrum : bool, optional
            If True, return both angles and full spectrum (default: False).
            
        method : str, optional
            Peak detection method (default: 'peaks'):
            - 'peaks': Find K highest peaks in spectrum
            - 'threshold': Use threshold-based detection
            
        Returns
        -------
        angles : np.ndarray
            Estimated DOA angles in degrees, shape (K_sources,).
            
        spectrum : np.ndarray, optional
            MUSIC pseudo-spectrum over angle_grid (if return_spectrum=True).
            
        Raises
        ------
        ValueError
            If K_sources ≥ N_sensors or K_sources < 1.
            If X shape doesn't match (N_sensors × snapshots).
            
        Examples
        --------
        >>> # Estimate 3 sources
        >>> angles = estimator.estimate(X, K_sources=3)
        >>> 
        >>> # Get spectrum for plotting
        >>> angles, spectrum = estimator.estimate(X, K_sources=3, return_spectrum=True)
        
        Notes
        -----
        Algorithm steps:
        1. Compute spatial covariance matrix: R = X @ X^H / snapshots
        2. Eigendecomposition: R = U @ Λ @ U^H
        3. Noise subspace: U_n = eigenvectors for K smallest eigenvalues
        4. MUSIC spectrum: P(θ) = 1 / (a(θ)^H @ U_n @ U_n^H @ a(θ))
        5. Find K largest peaks in spectrum
        """
        # Validate inputs
        if K_sources >= self.N:
            raise ValueError(f"K_sources ({K_sources}) must be < N_sensors ({self.N})")
        if K_sources < 1:
            raise ValueError(f"K_sources must be ≥ 1, got {K_sources}")
        if X.shape[0] != self.N:
            raise ValueError(f"X rows ({X.shape[0]}) must equal N_sensors ({self.N})")
        
        # Warn if K_sources is at the theoretical limit
        if K_sources >= self.N - 1:
            import warnings
            warnings.warn(
                f"K_sources={K_sources} is at/near the limit for N={self.N} sensors. "
                f"Standard MUSIC can reliably detect up to K_max ≈ N/2 sources. "
                f"Results may be unreliable. Consider using more sensors or leveraging "
                f"the difference coarray (K_max = L/2 where L is contiguous coarray length).",
                UserWarning
            )
        
        # Step 1: Spatial covariance matrix
        N_snapshots = X.shape[1]
        R = (X @ X.conj().T) / N_snapshots
        
        # Step 2: Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Step 3: Noise subspace (last N-K eigenvectors)
        U_noise = eigvecs[:, K_sources:]
        
        # Step 4: Compute MUSIC spectrum
        spectrum = self._compute_music_spectrum(U_noise)
        
        # Step 5: Find peaks
        if method == 'peaks':
            estimated_angles = self._find_peaks(spectrum, K_sources)
        elif method == 'threshold':
            estimated_angles = self._find_peaks_threshold(spectrum, K_sources)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if return_spectrum:
            return estimated_angles, spectrum
        else:
            return estimated_angles
    
    def _compute_music_spectrum(self, U_noise: np.ndarray) -> np.ndarray:
        """
        Compute MUSIC pseudo-spectrum.
        
        Parameters
        ----------
        U_noise : np.ndarray
            Noise subspace eigenvectors (N_sensors × (N-K)).
            
        Returns
        -------
        np.ndarray
            MUSIC spectrum values at each angle in angle_grid.
        """
        N_angles = len(self.angle_grid)
        spectrum = np.zeros(N_angles)
        
        # Projection matrix: P_n = U_noise @ U_noise^H
        P_noise = U_noise @ U_noise.conj().T
        
        for i in range(N_angles):
            a = self._steering_matrix[:, i].reshape(-1, 1)
            # MUSIC spectrum: 1 / (a^H @ P_n @ a)
            denominator = a.conj().T @ P_noise @ a
            spectrum[i] = 1.0 / np.abs(denominator[0, 0])
        
        return spectrum
    
    def _find_peaks(self, spectrum: np.ndarray, K: int) -> np.ndarray:
        """
        Find K highest peaks in spectrum.
        
        Parameters
        ----------
        spectrum : np.ndarray
            MUSIC spectrum values.
        K : int
            Number of peaks to find.
            
        Returns
        -------
        np.ndarray
            Angles corresponding to K highest peaks.
        """
        # Normalize spectrum
        spectrum_norm = spectrum / spectrum.max()
        
        # Find all local maxima
        peaks_idx = []
        for i in range(1, len(spectrum) - 1):
            if (spectrum_norm[i] > spectrum_norm[i-1] and 
                spectrum_norm[i] > spectrum_norm[i+1]):
                peaks_idx.append(i)
        
        if len(peaks_idx) < K:
            warnings.warn(f"Found only {len(peaks_idx)} peaks, expected {K}")
            # Pad with additional highest values
            remaining = K - len(peaks_idx)
            all_idx = np.argsort(spectrum_norm)[::-1]
            for idx in all_idx:
                if idx not in peaks_idx:
                    peaks_idx.append(idx)
                    remaining -= 1
                    if remaining == 0:
                        break
        
        # Sort peaks by height and take top K
        peaks_idx = sorted(peaks_idx, key=lambda i: spectrum_norm[i], reverse=True)
        peaks_idx = peaks_idx[:K]
        
        # Sort by angle for output
        peaks_idx = sorted(peaks_idx)
        
        return self.angle_grid[peaks_idx]
    
    def _find_peaks_threshold(self, spectrum: np.ndarray, K: int, 
                              threshold_ratio: float = 0.3) -> np.ndarray:
        """
        Find peaks above threshold.
        
        Parameters
        ----------
        spectrum : np.ndarray
            MUSIC spectrum values.
        K : int
            Expected number of peaks.
        threshold_ratio : float
            Threshold as ratio of max spectrum value.
            
        Returns
        -------
        np.ndarray
            Angles corresponding to peaks above threshold.
        """
        spectrum_norm = spectrum / spectrum.max()
        threshold = threshold_ratio
        
        peaks_idx = []
        for i in range(1, len(spectrum) - 1):
            if (spectrum_norm[i] > threshold and
                spectrum_norm[i] > spectrum_norm[i-1] and
                spectrum_norm[i] > spectrum_norm[i+1]):
                peaks_idx.append(i)
        
        # If found more than K, keep K highest
        if len(peaks_idx) > K:
            peaks_idx = sorted(peaks_idx, key=lambda i: spectrum_norm[i], reverse=True)
            peaks_idx = peaks_idx[:K]
        
        peaks_idx = sorted(peaks_idx)
        return self.angle_grid[peaks_idx]
    
    def simulate_signals(
        self,
        true_angles: List[float],
        SNR_dB: float = 10,
        snapshots: int = 200,
        signal_type: str = 'random'
    ) -> np.ndarray:
        """
        Simulate received signals for DOA algorithm testing.
        
        Parameters
        ----------
        true_angles : list of float
            True DOA angles in degrees.
            
        SNR_dB : float, optional
            Signal-to-noise ratio in dB (default: 10).
            
        snapshots : int, optional
            Number of time snapshots (default: 200).
            
        signal_type : str, optional
            Type of source signals (default: 'random'):
            - 'random': Random complex Gaussian
            - 'narrowband': Narrowband sinusoids
            
        Returns
        -------
        X : np.ndarray
            Simulated received signals (N_sensors × snapshots).
            
        Examples
        --------
        >>> # Simulate 3 sources at -30°, 10°, 45° with SNR=10dB
        >>> X = estimator.simulate_signals([-30, 10, 45], SNR_dB=10, snapshots=200)
        >>> 
        >>> # Test estimation
        >>> angles_est = estimator.estimate(X, K_sources=3)
        
        Notes
        -----
        Signal model: X = A @ S + N
        - A: Steering matrix (N × K)
        - S: Source signals (K × snapshots)
        - N: Noise (N × snapshots)
        """
        K = len(true_angles)
        
        # Build steering matrix for true angles
        A = np.zeros((self.N, K), dtype=complex)
        for k, angle in enumerate(true_angles):
            A[:, k] = self.steering_vector(angle)
        
        # Generate source signals
        if signal_type == 'random':
            S = (np.random.randn(K, snapshots) + 
                 1j * np.random.randn(K, snapshots)) / np.sqrt(2)
        elif signal_type == 'narrowband':
            t = np.arange(snapshots)
            S = np.zeros((K, snapshots), dtype=complex)
            for k in range(K):
                freq = 0.1 + k * 0.05  # Different frequencies
                S[k, :] = np.exp(1j * 2 * np.pi * freq * t)
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")
        
        # Compute noise power from SNR
        signal_power = np.mean(np.abs(S)**2)
        noise_power = signal_power / (10**(SNR_dB / 10))
        
        # Generate noise
        N = np.sqrt(noise_power / 2) * (np.random.randn(self.N, snapshots) + 
                                         1j * np.random.randn(self.N, snapshots))
        
        # Received signal
        X = A @ S + N
        
        return X
    
    def estimate_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate spatial covariance matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Received signal matrix (N_sensors × snapshots).
            
        Returns
        -------
        R : np.ndarray
            Spatial covariance matrix (N × N).
        """
        return (X @ X.conj().T) / X.shape[1]
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MUSICEstimator(N_sensors={self.N}, wavelength={self.wavelength}, "
                f"angle_range={self.angle_range}, resolution={self.angle_resolution}°)")
