"""Re-export canonical geometry processors from geometry_processors package.

This module provides backward compatibility for code that imports from
core.radarpy.geometry.*. All implementations are located in the canonical
geometry_processors/ package at the root level.

Deprecated import paths:
    from core.radarpy.geometry.ula_processors import ULArrayProcessor
    from core.radarpy.geometry.z4_processor import Z4ArrayProcessor
    
Recommended import paths:
    from geometry_processors.ula_processors import ULArrayProcessor
    from geometry_processors.z4_processor import Z4ArrayProcessor
"""

import warnings

# Re-export ArraySpec and BaseArrayProcessor from canonical location
from geometry_processors.bases_classes import ArraySpec, BaseArrayProcessor

# Re-export all canonical processor classes
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor
from geometry_processors.z1_processor import Z1ArrayProcessor
from geometry_processors.z3_1_processor import Z3_1ArrayProcessor
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor
from geometry_processors.z4_processor import Z4ArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor
from geometry_processors.z6_processor import Z6ArrayProcessor

__all__ = [
    'ArraySpec',
    'BaseArrayProcessor',
    'ULArrayProcessor',
    'NestedArrayProcessor',
    'Z1ArrayProcessor',
    'Z3_1ArrayProcessor',
    'Z3_2ArrayProcessor',
    'Z4ArrayProcessor',
    'Z5ArrayProcessor',
    'Z6ArrayProcessor',
]

# Emit deprecation warning on module import for explicit awareness
warnings.warn(
    "Importing from core.radarpy.geometry is deprecated. "
    "Please import directly from geometry_processors instead: "
    "from geometry_processors.{module_name} import {ClassName}",
    DeprecationWarning,
    stacklevel=2,
)
