"""
Adaptive StarDist Model
Extends the base StarDist2D model with adaptive parameter selection based on local image characteristics.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from stardist.models import StarDist2D, Config2D
from skimage import feature, morphology
from .scale_detector import ScaleDetector
from .param_optimizer import ParamOptimizer

class AdaptiveConfig2D(Config2D):
    """Extended configuration class for adaptive StarDist."""
    def __init__(self,
                 min_n_rays: int = 32,
                 max_n_rays: int = 128,
                 min_grid: int = 1,
                 max_grid: int = 4,
                 boundary_weight: float = 0.7,
                 size_weight: float = 0.3,
                 scale_detector_params: Optional[Dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.min_n_rays = min_n_rays
        self.max_n_rays = max_n_rays
        self.min_grid = min_grid
        self.max_grid = max_grid
        self.boundary_weight = boundary_weight
        self.size_weight = size_weight
        self.scale_detector_params = scale_detector_params or {}

class AdaptiveStarDist2D(StarDist2D):
    """
    Adaptive StarDist model that automatically adjusts its parameters based on local image characteristics.
    """
    
    def __init__(self, config, name=None, basedir=None):
        """
        Initialize the adaptive model.
        
        Args:
            config: Configuration object (AdaptiveConfig2D)
            name: Model name
            basedir: Base directory for model files
        """
        super().__init__(config, name=name, basedir=basedir)
        self.scale_detector = ScaleDetector(**config.scale_detector_params)
        self.param_optimizer = ParamOptimizer(
            min_n_rays=config.min_n_rays,
            max_n_rays=config.max_n_rays,
            min_grid=config.min_grid,
            max_grid=config.max_grid,
            boundary_weight=config.boundary_weight,
            size_weight=config.size_weight
        )
        
    def _predict_instances(self, img: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Predict instances with adaptive parameters.
        
        Args:
            img: Input image
            
        Returns:
            labels: Instance segmentation labels
            details: Dictionary with additional information
        """
        # Detect local scales
        scale_map, confidence_map = self.scale_detector.detect_local_scales(img)
        
        # Get initial rough segmentation for parameter optimization
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(img)
        rough_mask = img > threshold
        rough_mask = morphology.remove_small_objects(rough_mask)
        
        # Optimize parameters based on image characteristics
        params = self.param_optimizer.optimize_parameters(
            image=img,
            mask=rough_mask,
            scale_map=scale_map
        )
        
        # Update model parameters
        self.config.n_rays = params['n_rays']
        self.config.grid = params['grid']
        
        # Perform prediction with adapted parameters
        labels, details = StarDist2D.predict_instances(self, img)
        
        # Add scale information to details
        details.update({
            'scale_map': scale_map,
            'confidence_map': confidence_map,
            'adapted_params': params
        })
        
        return labels, details
    
    def predict_instances(self, img: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Public method for instance prediction.
        
        Args:
            img: Input image
            
        Returns:
            labels: Instance segmentation labels
            details: Dictionary with additional information
        """
        return self._predict_instances(img)
