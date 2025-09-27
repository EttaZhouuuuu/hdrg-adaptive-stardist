"""
Parameter Optimizer for Adaptive StarDist
Implements sophisticated methods for selecting optimal StarDist parameters
based on image characteristics and object properties.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology, feature, filters
from typing import Dict, Tuple, Optional
from .utils import compute_local_statistics

class ParamOptimizer:
    def __init__(self,
                 min_n_rays: int = 32,
                 max_n_rays: int = 128,
                 min_grid: int = 1,
                 max_grid: int = 4,
                 boundary_weight: float = 0.7,
                 size_weight: float = 0.3):
        """
        Initialize the parameter optimizer.
        
        Args:
            min_n_rays: Minimum number of rays
            max_n_rays: Maximum number of rays
            min_grid: Minimum grid size
            max_grid: Maximum grid size
            boundary_weight: Weight for boundary complexity in ray calculation
            size_weight: Weight for object size in ray calculation
        """
        self.min_n_rays = min_n_rays
        self.max_n_rays = max_n_rays
        self.min_grid = min_grid
        self.max_grid = max_grid
        self.boundary_weight = boundary_weight
        self.size_weight = size_weight
        
    def compute_boundary_complexity(self, mask: np.ndarray) -> float:
        """
        Compute boundary complexity using perimeter-area ratio and curvature.
        
        Args:
            mask: Binary mask of objects
            
        Returns:
            Complexity score (0-1)
        """
        if mask.sum() == 0:
            return 0.0
            
        # Get boundaries
        boundaries = measure.find_contours(mask)
        if not boundaries:
            return 0.0
            
        complexity_scores = []
        for boundary in boundaries:
            # Compute perimeter and area
            perimeter = len(boundary)
            area = cv2.contourArea(boundary.astype(np.float32))
            
            if area == 0:
                continue
                
            # Circularity (1 for circle, less for complex shapes)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Compute local curvature
            dx = np.gradient(boundary[:, 0])
            dy = np.gradient(boundary[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
            mean_curvature = np.mean(curvature[~np.isnan(curvature)])
            
            # Combine metrics
            complexity = (1 - circularity) * 0.6 + np.clip(mean_curvature, 0, 1) * 0.4
            complexity_scores.append(complexity)
            
        return np.mean(complexity_scores) if complexity_scores else 0.0
    
    def estimate_optimal_rays(self, 
                            mask: np.ndarray,
                            scale_map: np.ndarray) -> int:
        """
        Estimate optimal number of rays based on object complexity and size.
        
        Args:
            mask: Binary mask of objects
            scale_map: Map of local scales
            
        Returns:
            Optimal number of rays
        """
        # Compute boundary complexity
        complexity = self.compute_boundary_complexity(mask)
        
        # Get size-based factor
        mean_scale = np.mean(scale_map[scale_map > 0])
        size_factor = np.clip(mean_scale / 20.0, 0, 1)  # Normalize to [0,1]
        
        # Combine factors with weights
        combined_factor = (complexity * self.boundary_weight + 
                         size_factor * self.size_weight)
        
        # Map to ray range
        ray_range = self.max_n_rays - self.min_n_rays
        n_rays = self.min_n_rays + int(ray_range * combined_factor)
        
        # Ensure number is divisible by 8
        n_rays = ((n_rays + 7) // 8) * 8
        
        return np.clip(n_rays, self.min_n_rays, self.max_n_rays)
    
    def compute_optimal_grid(self,
                           scale_map: np.ndarray,
                           density_map: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Compute optimal grid size based on object scales and density.
        
        Args:
            scale_map: Map of local scales
            density_map: Optional map of object density
            
        Returns:
            Tuple of (grid_y, grid_x)
        """
        mean_scale = np.mean(scale_map[scale_map > 0])
        
        if density_map is not None:
            # Consider object density in grid selection
            mean_density = np.mean(density_map)
            # Higher density → finer grid
            density_factor = np.clip(1 - mean_density, 0, 1)
        else:
            density_factor = 0.5
            
        # Combine scale and density factors
        combined_factor = (mean_scale / 20.0 * 0.7 + density_factor * 0.3)
        
        # Map to grid range
        grid_range = self.max_grid - self.min_grid
        grid_size = self.min_grid + int(grid_range * combined_factor)
        
        return (grid_size, grid_size)
    
    def optimize_parameters(self,
                          image: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          scale_map: Optional[np.ndarray] = None) -> Dict:
        """
        Optimize StarDist parameters based on image characteristics.
        
        Args:
            image: Input image
            mask: Optional binary mask of objects
            scale_map: Optional map of local scales
            
        Returns:
            Dictionary of optimal parameters
        """
        # Compute local statistics if not provided
        if scale_map is None:
            stats = compute_local_statistics(image)
            scale_map = stats['gradient']
            
        # Create rough mask if not provided
        if mask is None:
            # Use simple thresholding to get approximate objects
            threshold = filters.threshold_otsu(image)
            mask = image > threshold
            mask = morphology.remove_small_objects(mask)
        
        # Compute density map
        density_map = ndimage.gaussian_filter(mask.astype(float), sigma=10)
        
        # Get optimal parameters
        n_rays = self.estimate_optimal_rays(mask, scale_map)
        grid = self.compute_optimal_grid(scale_map, density_map)
        
        return {
            'n_rays': n_rays,
            'grid': grid,
            'complexity_score': self.compute_boundary_complexity(mask),
            'mean_scale': np.mean(scale_map[scale_map > 0]),
            'mean_density': np.mean(density_map)
        }
