"""
Signal Simulation for DOA Algorithm Testing
============================================

Provides utilities to simulate received signals for testing DOA estimation
algorithms with various array geometries.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


class SignalSimulator:
    """
    Simulate received signals for DOA estimation testing.
    
    Parameters
    ----------
    sensor_positions : array-like
        Physical sensor positions in wavelengths or meters.
        
    wavelength : float, optional
        Signal wavelength (default: 1.0).
        
    Examples
    --------
    >>> from doa_estimation import SignalSimulator
    >>> 
    >>> simulator = SignalSimulator(
    >>>     sensor_positions=[0, 1, 3, 6, 10],
    >>>     wavelength=1.0
    >>> )
    >>> 
    >>> X = simulator.generate_signals(
    >>>     angles=[-30, 10, 45],
    >>>     SNR_dB=10,
    >>>     snapshots=200
    >>> )
    """
    
    def __init__(
        self,
        sensor_positions: Union[List[float], np.ndarray],
        wavelength: float = 1.0
    ):
        """Initialize signal simulator."""
        self.sensor_positions = np.array(sensor_positions, dtype=float)
        self.N = len(self.sensor_positions)
        self.wavelength = wavelength
    
    def steering_vector(self, angle_deg: float) -> np.ndarray:
        """
        Compute steering vector for angle.
        
        Parameters
        ----------
        angle_deg : float
            Angle in degrees.
            
        Returns
        -------
        np.ndarray
            Complex steering vector (N,).
        """
        angle_rad = np.deg2rad(angle_deg)
        phase = 2 * np.pi * self.sensor_positions / self.wavelength * np.sin(angle_rad)
        return np.exp(1j * phase)
    
    def generate_signals(
        self,
        angles: List[float],
        SNR_dB: float = 10,
        snapshots: int = 200,
        signal_type: str = 'random',
        correlation: float = 0.0
    ) -> np.ndarray:
        """
        Generate simulated received signals.
        
        Parameters
        ----------
        angles : list of float
            True DOA angles in degrees.
            
        SNR_dB : float
            Signal-to-noise ratio in dB.
            
        snapshots : int
            Number of time snapshots.
            
        signal_type : str
            'random', 'narrowband', or 'wideband'.
            
        correlation : float
            Source correlation coefficient (0=uncorrelated, 1=fully correlated).
            
        Returns
        -------
        X : np.ndarray
            Received signals (N_sensors × snapshots).
        """
        K = len(angles)
        
        # Steering matrix
        A = np.array([self.steering_vector(angle) for angle in angles]).T
        
        # Generate source signals
        if signal_type == 'random':
            S = self._random_sources(K, snapshots, correlation)
        elif signal_type == 'narrowband':
            S = self._narrowband_sources(K, snapshots)
        elif signal_type == 'wideband':
            S = self._wideband_sources(K, snapshots)
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")
        
        # Compute noise
        signal_power = np.mean(np.abs(S)**2)
        noise_power = signal_power / (10**(SNR_dB / 10))
        N_noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.N, snapshots) + 
            1j * np.random.randn(self.N, snapshots)
        )
        
        # Received signal
        X = A @ S + N_noise
        return X
    
    def _random_sources(self, K: int, snapshots: int, 
                       correlation: float) -> np.ndarray:
        """Generate random complex Gaussian sources."""
        if correlation == 0:
            # Uncorrelated sources
            S = (np.random.randn(K, snapshots) + 
                 1j * np.random.randn(K, snapshots)) / np.sqrt(2)
        else:
            # Correlated sources
            # Generate base signal
            base = (np.random.randn(1, snapshots) + 
                   1j * np.random.randn(1, snapshots)) / np.sqrt(2)
            # Add independent components
            indep = (np.random.randn(K, snapshots) + 
                    1j * np.random.randn(K, snapshots)) / np.sqrt(2)
            S = correlation * base + np.sqrt(1 - correlation**2) * indep
        
        return S
    
    def _narrowband_sources(self, K: int, snapshots: int) -> np.ndarray:
        """Generate narrowband sinusoidal sources."""
        t = np.arange(snapshots)
        S = np.zeros((K, snapshots), dtype=complex)
        for k in range(K):
            freq = 0.1 + k * 0.05
            phase = np.random.rand() * 2 * np.pi
            S[k, :] = np.exp(1j * (2 * np.pi * freq * t + phase))
        return S
    
    def _wideband_sources(self, K: int, snapshots: int) -> np.ndarray:
        """Generate wideband sources."""
        # Random phase modulation
        S = np.zeros((K, snapshots), dtype=complex)
        for k in range(K):
            phase = np.cumsum(np.random.randn(snapshots)) * 0.1
            amplitude = np.random.randn(snapshots) * 0.3 + 1.0
            S[k, :] = amplitude * np.exp(1j * phase)
        return S
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SignalSimulator(N_sensors={self.N}, wavelength={self.wavelength})"
