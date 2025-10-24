import numpy as np
from scipy import ndimage
from typing import Tuple, List, Optional
import torch
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import numba
from numba import jit

@jit(nopython=True)
def _compute_rays(mask: np.ndarray, 
                 center_y: float, 
                 center_x: float, 
                 n_rays: int) -> np.ndarray:
    """
    Compute distances from center point to object boundary along n_rays directions.
    
    Args:
        mask: Binary mask for a single object
        center_y, center_x: Center coordinates
        n_rays: Number of rays to compute
        
    Returns:
        Array of distances for each ray direction
    """
    H, W = mask.shape
    distances = np.zeros(n_rays, dtype=np.float32)
    
    # Compute angles for each ray
    angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    
    for i, angle in enumerate(angles):
        # Ray direction vector
        dy = np.sin(angle)
        dx = np.cos(angle)
        
        # Initialize ray tracing
        y, x = center_y, center_x
        step = 0
        
        # Trace ray until hitting boundary or image edge
        while True:
            # Get next position
            y_new = y + dy
            x_new = x + dx
            
            # Check if we're out of bounds
            if (y_new < 0 or y_new >= H or x_new < 0 or x_new >= W):
                break
                
            # Round coordinates for array indexing
            yi, xi = int(round(y_new)), int(round(x_new))
            
            # Check if we've hit the boundary
            if mask[yi, xi] == 0:
                break
                
            # Update position and step count
            y, x = y_new, x_new
            step += 1
        
        # Compute Euclidean distance
        dy = y - center_y
        dx = x - center_x
        distances[i] = np.sqrt(dy*dy + dx*dx)
    
    return distances

def compute_stardist_targets(mask: np.ndarray, 
                           n_rays: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute StarDist targets from instance segmentation mask.
    
    Args:
        mask: Instance segmentation mask (H, W)
        n_rays: Number of rays to compute
        
    Returns:
        tuple: (distances, probabilities)
            - distances: Ray distances for each pixel (n_rays, H, W)
            - probabilities: Object probability map (H, W)
    """
    H, W = mask.shape
    distances = np.zeros((n_rays, H, W), dtype=np.float32)
    probabilities = np.zeros((H, W), dtype=np.float32)
    
    # Get unique object IDs (excluding background)
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0]
    
    # Process each object
    for obj_id in object_ids:
        # Create binary mask for current object
        obj_mask = (mask == obj_id)
        
        # Get object properties
        props = regionprops(obj_mask.astype(np.int32))[0]
        center_y, center_x = props.centroid
        
        # Compute ray distances for current object
        obj_distances = _compute_rays(obj_mask, center_y, center_x, n_rays)
        
        # Get object pixels
        y_coords, x_coords = np.where(obj_mask)
        
        # Assign distances and probabilities
        for y, x in zip(y_coords, x_coords):
            # Compute relative position to center
            dy = y - center_y
            dx = x - center_x
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2*np.pi
                
            # Find nearest ray index
            ray_idx = int(np.round(angle / (2*np.pi) * n_rays)) % n_rays
            
            # Assign distance and probability
            distances[ray_idx, y, x] = obj_distances[ray_idx]
            probabilities[y, x] = 1.0
            
    return distances, probabilities

class StarDistTargetGenerator:
    """
    Class for generating and managing StarDist targets.
    
    Args:
        n_rays: Number of rays to compute
        min_object_size: Minimum object size to consider
        max_object_size: Maximum object size to consider
    """
    def __init__(self,
                n_rays: int = 32,
                min_object_size: int = 10,
                max_object_size: Optional[int] = None):
        self.n_rays = n_rays
        self.min_object_size = min_object_size
        self.max_object_size = max_object_size
        
    def generate_targets(self, 
                        mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate StarDist targets from instance mask.
        
        Args:
            mask: Instance segmentation mask
            
        Returns:
            tuple: (distances, probabilities) as torch tensors
        """
        # Filter objects by size if needed
        if self.min_object_size > 0 or self.max_object_size is not None:
            mask = self._filter_objects_by_size(mask)
        
        # Compute targets
        distances, probabilities = compute_stardist_targets(
            mask, self.n_rays
        )
        
        # Convert to torch tensors
        distances = torch.from_numpy(distances)
        probabilities = torch.from_numpy(probabilities)
        
        return distances, probabilities
    
    def _filter_objects_by_size(self, mask: np.ndarray) -> np.ndarray:
        """Filter objects based on size constraints."""
        filtered_mask = np.zeros_like(mask)
        
        for obj_id in np.unique(mask)[1:]:  # Skip background
            obj_mask = (mask == obj_id)
            obj_size = np.sum(obj_mask)
            
            if obj_size < self.min_object_size:
                continue
            if self.max_object_size and obj_size > self.max_object_size:
                continue
                
            filtered_mask[obj_mask] = obj_id
            
        return filtered_mask
    
    @staticmethod
    def visualize_targets(image: np.ndarray,
                         distances: np.ndarray,
                         probabilities: np.ndarray,
                         ray_idx: Optional[int] = None) -> np.ndarray:
        """
        Visualize StarDist targets.
        
        Args:
            image: Input image
            distances: Ray distances
            probabilities: Object probabilities
            ray_idx: Optional specific ray index to visualize
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        if ray_idx is not None:
            # Visualize specific ray direction
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[1].imshow(distances[ray_idx])
            axes[1].set_title(f'Distances (Ray {ray_idx})')
            axes[2].imshow(probabilities)
            axes[2].set_title('Probabilities')
        else:
            # Visualize average distances
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[1].imshow(np.mean(distances, axis=0))
            axes[1].set_title('Average Distances')
            axes[2].imshow(probabilities)
            axes[2].set_title('Probabilities')
            
        plt.close()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return vis_image
