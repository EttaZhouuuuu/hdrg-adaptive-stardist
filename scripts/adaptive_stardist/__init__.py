"""
Multi-Scale Adaptive StarDist Framework
This package implements an adaptive version of StarDist that automatically adjusts
its parameters based on local image characteristics for better handling of
variable-sized neuronal structures.
"""

from .scale_detector import ScaleDetector
from .adaptive_model import AdaptiveStarDist2D
from .utils import compute_local_statistics

__all__ = ['ScaleDetector', 'AdaptiveStarDist2D', 'compute_local_statistics']
