"""
Utility functions for the adaptive StarDist framework.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Tuple

def compute_local_statistics(image: np.ndarray, 
                           window_size: int = 32) -> Dict[str, np.ndarray]:
    """
    Compute local image statistics using sliding windows.
    
    Args:
        image: Input image
        window_size: Size of the sliding window
        
    Returns:
        Dictionary containing local statistics maps
    """
    # Pad image for sliding window
    pad_size = window_size // 2
    padded = np.pad(image, pad_size, mode='reflect')
    
    # Initialize output maps
    mean_map = np.zeros_like(image, dtype=float)
    std_map = np.zeros_like(image, dtype=float)
    gradient_map = np.zeros_like(image, dtype=float)
    
    # Compute gradients
    gx = ndimage.sobel(image, axis=0)
    gy = ndimage.sobel(image, axis=1)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # Compute local statistics
    mean_map = ndimage.uniform_filter(image, size=window_size)
    
    # Local standard deviation
    mean_sq = ndimage.uniform_filter(image**2, size=window_size)
    std_map = np.sqrt(mean_sq - mean_map**2)
    
    # Local gradient statistics
    gradient_map = ndimage.uniform_filter(gradient_magnitude, size=window_size)
    
    return {
        'mean': mean_map,
        'std': std_map,
        'gradient': gradient_map
    }

def estimate_object_sizes(mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate object sizes from a labeled mask.
    
    Args:
        mask: Labeled mask image
        
    Returns:
        min_size, mean_size, max_size
    """
    if mask.max() == 0:
        return 0, 0, 0
        
    # Compute areas for each label
    areas = ndimage.sum(np.ones_like(mask), mask, 
                       index=np.arange(1, mask.max() + 1))
    
    # Convert areas to approximate diameters
    diameters = 2 * np.sqrt(areas / np.pi)
    
    return diameters.min(), diameters.mean(), diameters.max()

def compute_scale_metrics(true_mask: np.ndarray, 
                         pred_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute scale-aware evaluation metrics.
    
    Args:
        true_mask: Ground truth labeled mask
        pred_mask: Predicted labeled mask
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get size statistics
    true_min, true_mean, true_max = estimate_object_sizes(true_mask)
    pred_min, pred_mean, pred_max = estimate_object_sizes(pred_mask)
    
    metrics = {
        'size_bias': (pred_mean - true_mean) / true_mean,
        'size_variance_ratio': np.var(pred_mask) / np.var(true_mask),
        'min_size_error': abs(pred_min - true_min) / true_min,
        'max_size_error': abs(pred_max - true_max) / true_max
    }
    
    return metrics
