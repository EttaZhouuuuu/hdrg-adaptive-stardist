"""
Data Augmentation for Cell Segmentation
========================================

Strong augmentation strategies to prevent overfitting.
Includes spatial and intensity augmentations that preserve
segmentation masks.

Based on best practices for medical image segmentation:
- Albumentations library compatible
- Preserves spatial transformations for masks
- Intensity transformations applied to images only
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Callable
from scipy import ndimage


class CellSegmentationAugmenter:
    """
    Comprehensive augmenter for cell segmentation training.
    
    Applies spatial and intensity augmentations while preserving
    the relationship between images and segmentation masks.
    
    Augmentations:
    --------------
    - Spatial: horizontal flip, vertical flip, random rotation,
               elastic deformation, random scaling
    - Intensity: brightness, contrast, gamma correction
    - Noise: gaussian noise
    
    Usage:
    ------
    augmenter = CellSegmentationAugmenter(
        horizontal_flip=True,
        rotation=True,
        elastic_deformation=True,
        brightness=True,
        contrast=True,
        noise=True
    )
    
    augmented_img, augmented_mask = augmenter(img, mask)
    """
    
    def __init__(
        self,
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.5,
        rotation: bool = True,
        rotation_range: float = 30.0,
        elastic_deformation: bool = True,
        elastic_sigma: float = 3.0,
        elastic_alpha: float = 0.1,
        scaling: bool = True,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        brightness: bool = True,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast: bool = True,
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        gamma: bool = False,
        gamma_range: Tuple[float, float] = (0.9, 1.1),
        gaussian_noise: bool = False,
        noise_std: float = 0.02,
        seed: int = 42
    ):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.rotation_range = rotation_range
        self.elastic_deformation = elastic_deformation
        self.elastic_sigma = elastic_sigma
        self.elastic_alpha = elastic_alpha
        self.scaling = scaling
        self.scale_range = scale_range
        self.brightness = brightness
        self.brightness_range = brightness_range
        self.contrast = contrast
        self.contrast_range = contrast_range
        self.gamma = gamma
        self.gamma_range = gamma_range
        self.gaussian_noise = gaussian_noise
        self.noise_std = noise_std
        self.seed = seed
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
    
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random augmentations to image and mask.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image [H, W, C] or [H, W]
        mask : np.ndarray, optional
            Segmentation mask [H, W] or [H, W, 1]
            
        Returns:
        --------
        augmented_image : np.ndarray
            Augmented image
        augmented_mask : np.ndarray or None
            Augmented mask (same spatial transforms applied)
        """
        # Reset random state for reproducibility
        self.rng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
        
        # Ensure proper array types
        image = image.astype(np.float32)
        if mask is not None:
            mask = mask.astype(np.float32)
        
        # Apply spatial augmentations (must affect both image and mask)
        image, mask = self._spatial_augment(image, mask)
        
        # Apply intensity augmentations (image only)
        image = self._intensity_augment(image)
        
        # Add noise if enabled
        if self.gaussian_noise:
            image = self._add_gaussian_noise(image)
        
        # Ensure proper dtype
        if mask is not None:
            mask = mask.astype(np.float32)
        
        return image, mask
    
    def _spatial_augment(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply spatial transformations that affect both image and mask."""
        
        h, w = image.shape[:2]
        
        # Horizontal flip
        if self.rng.random() < self.horizontal_flip:
            image = np.fliplr(image).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        
        # Vertical flip
        if self.rng.random() < self.vertical_flip:
            image = np.flipud(image).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()
        
        # Random rotation
        if self.rotation:
            angle = self.rng.uniform(-self.rotation_range, self.rotation_range)
            angle_rad = np.deg2rad(angle)
            
            # Rotate image
            image = ndimage.rotate(image, angle, reshape=False, order=1, mode='reflect')
            if mask is not None:
                mask = ndimage.rotate(mask, angle, reshape=False, order=0, mode='reflect')
        
        # Elastic deformation
        if self.elastic_deformation and mask is not None:
            image, mask = self._elastic_deform(image, mask)
            # mask is already updated, no need to set to None
        
        # Random scaling
        if self.scaling:
            scale = self.rng.uniform(*self.scale_range)
            if scale != 1.0:
                image = self._scale_image(image, scale)
                if mask is not None:
                    mask = self._scale_image(mask, scale)
        
        return image, mask
    
    def _elastic_deform(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation using grid distortion."""
        
        h, w = image.shape[:2]
        
        # Create random displacement field
        sigma = self.elastic_sigma
        alpha = self.elastic_alpha * min(h, w)
        
        # Generate random noise fields
        dx = ndimage.gaussian_filter(self.rng.randn(h, w), sigma) * alpha
        dy = ndimage.gaussian_filter(self.rng.randn(h, w), sigma) * alpha
        
        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        x_new = np.clip(x_grid + dx, 0, w - 1)
        y_new = np.clip(y_grid + dy, 0, h - 1)
        
        # Apply to image and mask using interpolation
        image = self._warp_image(image, x_new, y_new)
        mask = self._warp_image(mask, x_new, y_new)
        
        return image, mask
    
    def _warp_image(
        self,
        image: np.ndarray,
        x_new: np.ndarray,
        y_new: np.ndarray
    ) -> np.ndarray:
        """Warp image using new coordinate grids."""
        
        if image.ndim == 3:
            warped = np.zeros_like(image)
            for c in range(image.shape[2]):
                warped[:, :, c] = ndimage.map_coordinates(
                    image[:, :, c],
                    [y_new.flatten(), x_new.flatten()],
                    order=1, mode='reflect'
                ).reshape(image.shape[:2])
        else:
            warped = ndimage.map_coordinates(
                image,
                [y_new.flatten(), x_new.flatten()],
                order=1, mode='reflect'
            ).reshape(image.shape)
        
        return warped
    
    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Apply random scaling using scipy.ndimage with correct dimension handling."""
        from scipy.ndimage import zoom
        
        h, w = image.shape[:2]
        
        # Calculate zoom factors to scale and return to original size
        # First zoom by scale, then we need to zoom back by 1/scale to get original size
        zoom_factors = (h / (h * scale), w / (w * scale))
        scaled = zoom(image, zoom_factors, order=1)
        
        # Ensure exact shape
        if scaled.shape[:2] != (h, w):
            # Crop if needed
            if scaled.shape[0] > h:
                scaled = scaled[:h]
            if scaled.shape[1] > w:
                scaled = scaled[:, :w]
            # Pad if needed
            if scaled.shape[0] < h or scaled.shape[1] < w:
                padded = np.zeros((h, w) + image.shape[2:] if image.ndim == 3 else (h, w), dtype=image.dtype)
                padded[:scaled.shape[0], :scaled.shape[1]] = scaled
                scaled = padded
        
        return scaled.astype(image.dtype)
    
    def _intensity_augment(self, image: np.ndarray) -> np.ndarray:
        """Apply intensity augmentations to image only."""
        
        # Ensure single channel for intensity operations
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        
        # Brightness adjustment
        if self.brightness:
            factor = self.rng.uniform(*self.brightness_range)
            image = image * factor
        
        # Contrast adjustment
        if self.contrast:
            factor = self.rng.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * factor + mean
        
        # Gamma correction
        if self.gamma:
            gamma = self.rng.uniform(*self.gamma_range)
            image = np.power(np.clip(image, 0, 1), gamma)
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
        
        return image
    
    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        
        noise_std = self.rng.uniform(0, self.noise_std)
        noise = self.rng.randn(*image.shape).astype(np.float32) * noise_std
        return np.clip(image + noise, 0, 1)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)


def get_he_augmenter(
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotation: bool = True,
    brightness: bool = True,
    contrast: bool = True,
    elastic: bool = True,
    noise: bool = True,
    strength: str = 'medium'
) -> CellSegmentationAugmenter:
    """
    Create an augmenter optimized for H&E stained tissue images.
    
    Parameters:
    -----------
    strength : str
        'light', 'medium', or 'strong' augmentation
    
    Returns:
    --------
    CellSegmentationAugmenter instance
    """
    
    configs = {
        'light': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,
            'rotation_range': 15.0,
            'elastic_alpha': 0.05,
            'scale_range': (0.95, 1.05),
            'brightness_range': (0.9, 1.1),
            'contrast_range': (0.95, 1.05),
            'noise_std': 0.01,
        },
        'medium': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,
            'rotation_range': 30.0,
            'elastic_alpha': 0.1,
            'scale_range': (0.9, 1.1),
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.9, 1.1),
            'noise_std': 0.02,
        },
        'strong': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.5,
            'rotation_range': 45.0,
            'elastic_alpha': 0.15,
            'scale_range': (0.85, 1.15),
            'brightness_range': (0.7, 1.3),
            'contrast_range': (0.8, 1.2),
            'noise_std': 0.03,
        }
    }
    
    config = configs.get(strength, configs['medium'])
    
    return CellSegmentationAugmenter(
        horizontal_flip=config['horizontal_flip'],
        vertical_flip=config['vertical_flip'],
        rotation=True,
        rotation_range=config['rotation_range'],
        elastic_deformation=elastic,
        elastic_alpha=config['elastic_alpha'],
        scaling=True,
        scale_range=config['scale_range'],
        brightness=brightness,
        brightness_range=config['brightness_range'],
        contrast=contrast,
        contrast_range=config['contrast_range'],
        gamma=False,
        gaussian_noise=noise,
        noise_std=config['noise_std']
    )


# Alias for easier imports
def create_augmenter(
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotation: bool = True,
    brightness: bool = True,
    contrast: bool = True,
    elastic: bool = True,
    noise: bool = True,
    strength: str = 'medium'
) -> CellSegmentationAugmenter:
    """
    Create a CellSegmentationAugmenter with specified parameters.
    
    This is the recommended function for creating augmenters.
    """
    return get_he_augmenter(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rotation=rotation,
        brightness=brightness,
        contrast=contrast,
        elastic=elastic,
        noise=noise,
        strength=strength
    )

