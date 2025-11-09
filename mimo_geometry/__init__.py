"""
MIMO Geometry Analysis Package

This package provides tools for analyzing and visualizing MIMO antenna array geometries.
"""

__version__ = "0.1.0"

from .antenna_array import AntennaArray
from .geometry_analyzer import GeometryAnalyzer
from .visualizer import ArrayVisualizer

__all__ = ['AntennaArray', 'GeometryAnalyzer', 'ArrayVisualizer']
