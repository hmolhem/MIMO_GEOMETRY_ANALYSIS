"""
Geometry Analyzer Module

Provides analysis tools for MIMO antenna array geometries.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .antenna_array import AntennaArray


class GeometryAnalyzer:
    """
    Analyzes geometric properties of MIMO antenna arrays.
    """
    
    def __init__(self, array: AntennaArray):
        """
        Initialize the geometry analyzer.
        
        Args:
            array: AntennaArray to analyze
        """
        self.array = array
    
    def compute_array_aperture(self) -> float:
        """
        Compute the effective aperture of the array.
        
        Returns:
            Maximum extent of the array
        """
        positions = self.array.positions
        ranges = np.ptp(positions, axis=0)  # peak-to-peak (max - min)
        return np.linalg.norm(ranges)
    
    def compute_element_spacing_stats(self) -> Dict[str, float]:
        """
        Compute statistics on inter-element spacing.
        
        Returns:
            Dictionary with min, max, mean, and std of spacings
        """
        distances = self.array.get_distances()
        
        # Get upper triangle (excluding diagonal) to avoid duplicates
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        
        return {
            'min_spacing': np.min(upper_triangle),
            'max_spacing': np.max(upper_triangle),
            'mean_spacing': np.mean(upper_triangle),
            'std_spacing': np.std(upper_triangle)
        }
    
    def compute_spatial_correlation(self, 
                                   angle_spread: float = 10.0,
                                   wavelength: float = 1.0) -> np.ndarray:
        """
        Compute spatial correlation matrix for the array.
        
        Uses a simplified model based on angular spread.
        
        Args:
            angle_spread: Angular spread in degrees
            wavelength: Signal wavelength (normalized to 1.0)
            
        Returns:
            Spatial correlation matrix (N x N)
        """
        N = self.array.num_elements
        correlation = np.zeros((N, N), dtype=complex)
        
        distances = self.array.get_distances()
        angle_spread_rad = np.deg2rad(angle_spread)
        
        for i in range(N):
            for j in range(N):
                # Simplified correlation model
                d = distances[i, j]
                if d == 0:
                    correlation[i, j] = 1.0
                else:
                    # Using a sinc function model
                    arg = 2 * np.pi * d * np.sin(angle_spread_rad) / wavelength
                    correlation[i, j] = np.sinc(arg / np.pi)
        
        return correlation
    
    def estimate_channel_capacity(self,
                                  snr_db: float = 10.0,
                                  angle_spread: float = 10.0) -> float:
        """
        Estimate MIMO channel capacity.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            angle_spread: Angular spread in degrees
            
        Returns:
            Channel capacity in bits/s/Hz
        """
        N = self.array.num_elements
        snr_linear = 10 ** (snr_db / 10)
        
        # Get spatial correlation
        R = self.compute_spatial_correlation(angle_spread)
        
        # Simplified capacity calculation
        # C = log2(det(I + (SNR/N) * H * H'))
        # Assuming H is influenced by spatial correlation
        eigenvalues = np.linalg.eigvalsh(R.real)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        
        capacity = 0
        for eigval in eigenvalues:
            capacity += np.log2(1 + (snr_linear / N) * eigval)
        
        return capacity
    
    def compute_array_factor(self,
                            azimuth_angles: np.ndarray,
                            elevation: float = 0.0,
                            wavelength: float = 1.0) -> np.ndarray:
        """
        Compute the array factor pattern.
        
        Args:
            azimuth_angles: Array of azimuth angles in degrees
            elevation: Elevation angle in degrees
            wavelength: Signal wavelength (normalized to 1.0)
            
        Returns:
            Array factor values (normalized)
        """
        azimuth_rad = np.deg2rad(azimuth_angles)
        elevation_rad = np.deg2rad(elevation)
        
        k = 2 * np.pi / wavelength  # Wavenumber
        
        # Ensure azimuth_rad is an array
        azimuth_rad = np.atleast_1d(azimuth_rad)
        
        # Direction vectors
        kx = k * np.cos(elevation_rad) * np.cos(azimuth_rad)
        ky = k * np.cos(elevation_rad) * np.sin(azimuth_rad)
        kz = k * np.sin(elevation_rad) * np.ones_like(azimuth_rad)
        
        array_factor = np.zeros(len(azimuth_rad), dtype=complex)
        
        for i in range(len(azimuth_rad)):
            af = 0
            for pos in self.array.positions:
                phase = kx[i] * pos[0] + ky[i] * pos[1] + kz[i] * pos[2]
                af += np.exp(1j * phase)
            array_factor[i] = af
        
        # Normalize
        array_factor = np.abs(array_factor)
        if np.max(array_factor) > 0:
            array_factor /= np.max(array_factor)
        
        return array_factor
    
    def analyze_array(self) -> Dict:
        """
        Perform comprehensive array analysis.
        
        Returns:
            Dictionary with various analysis metrics
        """
        spacing_stats = self.compute_element_spacing_stats()
        aperture = self.compute_array_aperture()
        
        # Estimate capacity at different SNRs
        capacities = {}
        for snr in [0, 10, 20, 30]:
            capacities[f'capacity_snr_{snr}dB'] = self.estimate_channel_capacity(snr)
        
        analysis = {
            'array_type': self.array.array_type,
            'num_elements': self.array.num_elements,
            'array_center': self.array.center.tolist(),
            'aperture': aperture,
            **spacing_stats,
            **capacities
        }
        
        return analysis
