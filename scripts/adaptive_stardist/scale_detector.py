"""
Scale Detection Module
Implements methods for detecting local scale variations in histological images.
"""

import numpy as np
from scipy import ndimage
from skimage.feature import blob_dog
from typing import Tuple, Dict

class ScaleDetector:
    def __init__(self, 
                 min_sigma: float = 1.0,
                 max_sigma: float = 30.0,
                 sigma_ratio: float = 1.6,
                 threshold: float = 0.1):
        """
        Initialize the scale detector.
        
        Args:
            min_sigma: Minimum standard deviation for Gaussian kernel
            max_sigma: Maximum standard deviation for Gaussian kernel
            sigma_ratio: Ratio between consecutive standard deviations
            threshold: Threshold for blob detection
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_ratio = sigma_ratio
        self.threshold = threshold
        
    def detect_local_scales(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect local scales in the image using Difference of Gaussians.
        
        Args:
            image: Input image (2D numpy array)
            
        Returns:
            scale_map: 2D array of local scale estimates
            confidence_map: 2D array of detection confidences
        """
        # Normalize image to [0,1] range
        img_norm = (image - image.min()) / (image.max() - image.min())
        
        # Detect blobs using Difference of Gaussians
        blobs = blob_dog(img_norm, 
                        min_sigma=self.min_sigma,
                        max_sigma=self.max_sigma,
                        sigma_ratio=self.sigma_ratio,
                        threshold=self.threshold)
        
        # Create scale and confidence maps
        scale_map = np.zeros_like(image, dtype=float)
        confidence_map = np.zeros_like(image, dtype=float)
        
        if len(blobs) > 0:
            # Extract coordinates and scales
            y, x, scales = blobs[:, 0], blobs[:, 1], blobs[:, 2]
            
            # Create distance-weighted scale map
            for i in range(len(blobs)):
                y_coord, x_coord, scale = int(y[i]), int(x[i]), scales[i]
                
                # Create distance mask
                Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                dist = np.sqrt((Y - y_coord)**2 + (X - x_coord)**2)
                
                # Weight by distance
                weight = np.exp(-dist / (2 * scale**2))
                
                # Update scale and confidence maps
                scale_map += scale * weight
                confidence_map += weight
                
            # Normalize
            mask = confidence_map > 0
            scale_map[mask] /= confidence_map[mask]
            confidence_map = confidence_map / confidence_map.max()
            
            # Fill in gaps using interpolation
            if not np.all(mask):
                scale_map = ndimage.interpolation.gaussian_filter(scale_map, sigma=3)
        
        return scale_map, confidence_map
    
    def compute_optimal_params(self, scale_map: np.ndarray) -> Dict[str, int]:
        """
        Compute optimal StarDist parameters based on local scale.
        
        Args:
            scale_map: 2D array of local scale estimates
            
        Returns:
            Dictionary containing optimal parameters (n_rays, grid)
        """
        # Convert scale to StarDist parameters
        mean_scale = np.mean(scale_map[scale_map > 0])
        
        # Adjust n_rays based on scale
        # Larger objects need more rays for better boundary representation
        n_rays = int(np.clip(32 * (mean_scale / 10), 32, 128))
        n_rays = n_rays + (n_rays % 8)  # Make divisible by 8
        
        # Adjust grid based on scale
        # Larger objects allow for coarser grid
        grid_size = int(np.clip(8 / (mean_scale / 10), 1, 4))
        
        return {
            'n_rays': n_rays,
            'grid': (grid_size, grid_size)
        }
