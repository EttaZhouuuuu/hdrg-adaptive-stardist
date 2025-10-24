"""
Adaptive Shape StarDist
========================

An innovative cell segmentation framework that replaces StarDist's fixed ray representation
with an adaptive shape encoder using deformable convolutions and learned sampling points.

Key Components:
--------------
- Deformable Convolution: Learns adaptive sampling locations
- Adaptive Shape Encoder: Dynamically adjusts to irregular cell shapes
- Shape Prior Encoder: Incorporates domain-specific shape knowledge
- Flexible Sampling: Variable number of boundary points per cell

Authors: Zhou Yitong
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Zhou Yitong"

from .models.adaptive_shape_model import AdaptiveShapeStarDist
from .core.deformable_conv import DeformableConv2D
from .core.shape_encoder import AdaptiveShapeEncoder
from .core.shape_prior import ShapePriorEncoder

__all__ = [
    'AdaptiveShapeStarDist',
    'DeformableConv2D',
    'AdaptiveShapeEncoder',
    'ShapePriorEncoder',
]

