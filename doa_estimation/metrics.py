"""
DOA Estimation Performance Metrics
====================================

Provides metrics for evaluating DOA estimation algorithm performance.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


class DOAMetrics:
    """
    Performance metrics for DOA estimation.
    
    Examples
    --------
    >>> from doa_estimation import DOAMetrics
    >>> 
    >>> true_angles = [-30, 10, 45]
    >>> estimated = [-29.5, 10.2, 44.8]
    >>> 
    >>> metrics = DOAMetrics()
    >>> rmse = metrics.compute_rmse(true_angles, estimated)
    >>> print(f"RMSE: {rmse:.2f}°")
    """
    
    @staticmethod
    def compute_rmse(
        true_angles: Union[List[float], np.ndarray],
        estimated_angles: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute Root Mean Squared Error (RMSE).
        
        Parameters
        ----------
        true_angles : array-like
            True DOA angles in degrees.
            
        estimated_angles : array-like
            Estimated DOA angles in degrees.
            
        Returns
        -------
        float
            RMSE in degrees.
            
        Notes
        -----
        Automatically matches closest estimated angle to each true angle.
        
        Examples
        --------
        >>> true = [-30, 10, 45]
        >>> est = [-29.5, 10.2, 44.8]
        >>> rmse = DOAMetrics.compute_rmse(true, est)
        >>> print(f"{rmse:.2f}°")
        0.36°
        """
        true = np.array(true_angles)
        est = np.array(estimated_angles)
        
        if len(true) != len(est):
            raise ValueError(
                f"Length mismatch: {len(true)} true vs {len(est)} estimated"
            )
        
        # Match closest pairs
        errors = []
        used = set()
        for t in true:
            # Find closest unused estimate
            candidates = [(abs(t - e), i, e) 
                         for i, e in enumerate(est) if i not in used]
            if not candidates:
                raise ValueError("Not enough estimates")
            _, idx, e = min(candidates)
            errors.append(t - e)
            used.add(idx)
        
        return np.sqrt(np.mean(np.array(errors)**2))
    
    @staticmethod
    def compute_bias(
        true_angles: Union[List[float], np.ndarray],
        estimated_angles: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute estimation bias.
        
        Parameters
        ----------
        true_angles : array-like
            True DOA angles in degrees.
            
        estimated_angles : array-like
            Estimated DOA angles in degrees.
            
        Returns
        -------
        float
            Mean bias in degrees.
        """
        true = np.array(true_angles)
        est = np.array(estimated_angles)
        
        errors = []
        used = set()
        for t in true:
            candidates = [(abs(t - e), i, e) 
                         for i, e in enumerate(est) if i not in used]
            _, idx, e = min(candidates)
            errors.append(t - e)
            used.add(idx)
        
        return np.mean(errors)
    
    @staticmethod
    def compute_mae(
        true_angles: Union[List[float], np.ndarray],
        estimated_angles: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute Mean Absolute Error (MAE).
        
        Parameters
        ----------
        true_angles : array-like
            True DOA angles in degrees.
            
        estimated_angles : array-like
            Estimated DOA angles in degrees.
            
        Returns
        -------
        float
            MAE in degrees.
        """
        true = np.array(true_angles)
        est = np.array(estimated_angles)
        
        errors = []
        used = set()
        for t in true:
            candidates = [(abs(t - e), i, e) 
                         for i, e in enumerate(est) if i not in used]
            _, idx, e = min(candidates)
            errors.append(abs(t - e))
            used.add(idx)
        
        return np.mean(errors)
    
    @staticmethod
    def compute_max_error(
        true_angles: Union[List[float], np.ndarray],
        estimated_angles: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute maximum absolute error.
        
        Parameters
        ----------
        true_angles : array-like
            True DOA angles in degrees.
            
        estimated_angles : array-like
            Estimated DOA angles in degrees.
            
        Returns
        -------
        float
            Maximum error in degrees.
        """
        true = np.array(true_angles)
        est = np.array(estimated_angles)
        
        errors = []
        used = set()
        for t in true:
            candidates = [(abs(t - e), i, e) 
                         for i, e in enumerate(est) if i not in used]
            _, idx, e = min(candidates)
            errors.append(abs(t - e))
            used.add(idx)
        
        return np.max(errors)
    
    @staticmethod
    def cramer_rao_bound(
        sensor_positions: Union[List[float], np.ndarray],
        angles: Union[List[float], np.ndarray],
        SNR_dB: float,
        snapshots: int,
        wavelength: float = 1.0
    ) -> np.ndarray:
        """
        Compute Cramér-Rao Lower Bound (CRLB) for DOA estimation.
        
        Parameters
        ----------
        sensor_positions : array-like
            Sensor positions in wavelengths.
            
        angles : array-like
            True DOA angles in degrees.
            
        SNR_dB : float
            Signal-to-noise ratio in dB.
            
        snapshots : int
            Number of snapshots.
            
        wavelength : float
            Signal wavelength.
            
        Returns
        -------
        np.ndarray
            CRLB standard deviation for each angle (degrees).
            
        Notes
        -----
        Assumes uncorrelated sources and white Gaussian noise.
        
        References
        ----------
        P. Stoica and A. Nehorai, "MUSIC, Maximum Likelihood, and 
        Cramer-Rao Bound," IEEE Trans. ASSP, 1989.
        """
        positions = np.array(sensor_positions)
        angles_rad = np.deg2rad(angles)
        N = len(positions)
        K = len(angles)
        
        # SNR linear scale
        SNR = 10**(SNR_dB / 10)
        
        # Steering matrix and derivatives
        A = np.zeros((N, K), dtype=complex)
        D = np.zeros((N, K), dtype=complex)
        
        for k, angle in enumerate(angles_rad):
            phase = 2 * np.pi * positions / wavelength * np.sin(angle)
            A[:, k] = np.exp(1j * phase)
            D[:, k] = 1j * (2 * np.pi * positions / wavelength * 
                            np.cos(angle)) * A[:, k]
        
        # Fisher Information Matrix (simplified for uncorrelated sources)
        P = np.eye(N) - A @ np.linalg.pinv(A @ A.conj().T) @ A.conj().T
        
        FIM = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                FIM[i, j] = 2 * SNR * snapshots * np.real(
                    D[:, i].conj().T @ P @ D[:, j]
                )
        
        # CRLB is inverse of FIM
        try:
            CRLB = np.linalg.inv(FIM)
            std_rad = np.sqrt(np.diag(CRLB))
            std_deg = np.rad2deg(std_rad)
            return std_deg
        except np.linalg.LinAlgError:
            # Singular FIM - return inf
            return np.full(K, np.inf)
    
    @staticmethod
    def success_rate(
        true_angles: Union[List[float], np.ndarray],
        estimated_angles: Union[List[float], np.ndarray],
        threshold_deg: float = 2.0
    ) -> float:
        """
        Compute success rate (percentage with error < threshold).
        
        Parameters
        ----------
        true_angles : array-like
            True DOA angles in degrees.
            
        estimated_angles : array-like
            Estimated DOA angles in degrees.
            
        threshold_deg : float
            Error threshold in degrees (default: 2.0).
            
        Returns
        -------
        float
            Success rate (0.0 to 1.0).
        """
        true = np.array(true_angles)
        est = np.array(estimated_angles)
        
        if len(est) == 0:
            return 0.0
        
        errors = []
        used = set()
        for t in true:
            candidates = [(abs(t - e), i, e) 
                         for i, e in enumerate(est) if i not in used]
            if not candidates:
                errors.append(np.inf)
                continue
            _, idx, e = min(candidates)
            errors.append(abs(t - e))
            used.add(idx)
        
        successes = sum(1 for err in errors if err < threshold_deg)
        return successes / len(true)
