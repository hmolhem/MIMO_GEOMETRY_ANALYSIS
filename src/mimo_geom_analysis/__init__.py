"""
Lightweight package wrapper for MIMO Geometry Analysis.
This module exposes a minimal public API and imports core modules lazily.
"""

# Expose processors package namespace for backward compatibility
from importlib import import_module

__all__ = [
    "geometry_processors",
]

# Provide a helper to access the geometry_processors package
def geometry_processors():
    return import_module('geometry_processors')
