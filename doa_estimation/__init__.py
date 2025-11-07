"""
DOA (Direction of Arrival) Estimation Module
=============================================

Provides DOA estimation algorithms for sparse MIMO arrays.

Modules
-------
- music: MUSIC (Multiple Signal Classification) algorithm
- esprit: ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
- simulation: Signal simulation for testing DOA algorithms
- metrics: Performance evaluation metrics
- visualization: Plotting DOA results

Quick Start
-----------
>>> from doa_estimation import MUSICEstimator
>>> from geometry_processors.z5_processor import Z5ArrayProcessor
>>> 
>>> # Create array
>>> z5 = Z5ArrayProcessor(N=7, d=0.5)
>>> results = z5.run_full_analysis()
>>> 
>>> # Setup DOA estimator
>>> estimator = MUSICEstimator(
>>>     sensor_positions=z5.data.sensors_positions,
>>>     wavelength=1.0
>>> )
>>> 
>>> # Simulate signals
>>> true_angles = [-30, 10, 45]
>>> X = estimator.simulate_signals(true_angles, SNR_dB=10, snapshots=200)
>>> 
>>> # Estimate DOAs
>>> estimated_angles = estimator.estimate(X, K_sources=3)
>>> print(f"True: {true_angles}")
>>> print(f"Estimated: {estimated_angles}")
"""

from .music import MUSICEstimator
from .simulation import SignalSimulator
from .metrics import DOAMetrics
from .visualization import plot_doa_spectrum, plot_array_geometry, plot_estimation_results

__all__ = [
    'MUSICEstimator',
    'SignalSimulator', 
    'DOAMetrics',
    'plot_doa_spectrum',
    'plot_array_geometry',
    'plot_estimation_results'
]

__version__ = '1.0.0'
