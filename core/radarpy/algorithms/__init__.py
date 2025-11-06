"""
Algorithms module: DOA estimation algorithms and utilities

This module contains implementations of MUSIC-based DOA estimation algorithms
including Spatial MUSIC, Coarray MUSIC, and related utilities.
"""

from .spatial_music import estimate_doa_spatial_music
from .coarray_music import estimate_doa_coarray_music
from .crb import crb_pair_worst_deg
from .coarray import build_virtual_ula_covariance
from .alss import apply_alss

__all__ = [
    'estimate_doa_spatial_music',
    'estimate_doa_coarray_music',
    'crb_pair_worst_deg',
    'build_virtual_ula_covariance',
    'apply_alss',
]
