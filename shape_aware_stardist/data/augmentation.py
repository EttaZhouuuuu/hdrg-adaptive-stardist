import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2
from typing import Dict, Optional, Tuple, Union, List
import torch
import cv2

class InstanceAwareRandomCrop(DualTransform):
    """
    Random crop that ensures at least one instance is included in the crop.
    
    Args:
        height (int): Height of crop
        width (int): Width of crop
        min_instance_area (float): Minimum area of instance that must be preserved
        p (float): Probability of applying the transform
    """
    def __init__(
        self,
        height: int,
        width: int,
        min_instance_area: float = 0.5,
        always_apply: bool = False,
        p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.min_instance_area = min_instance_area
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply the transform to the image."""
        if 'crop_coords' not in params:
            raise ValueError("Crop coordinates not found in params")
        y1, y2, x1, x2 = params['crop_coords']
        return img[y1:y2, x1:x2]
    
    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """Apply the transform to the mask."""
        if 'crop_coords' not in params:
            raise ValueError("Crop coordinates not found in params")
        y1, y2, x1, x2 = params['crop_coords']
        return mask[y1:y2, x1:x2]
    
    def get_params_dependent_on_targets(self, params: Dict) -> Dict:
        """Get parameters based on mask content."""
        mask = params['mask']
        img_h, img_w = mask.shape[:2]
        
        # Get instance centroids
        instance_ids = np.unique(mask)[1:]  # Exclude background
        if len(instance_ids) == 0:
            # If no instances, do random crop
            y1 = np.random.randint(0, img_h - self.height + 1)
            x1 = np.random.randint(0, img_w - self.width + 1)
        else:
            # Get random instance
            instance_id = np.random.choice(instance_ids)
            instance_mask = (mask == instance_id)
            
            # Get instance bounds
            y_coords, x_coords = np.where(instance_mask)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            
            # Ensure instance is included in crop
            y1 = np.random.randint(
                max(0, y_max - self.height),
                min(img_h - self.height + 1, y_min + 1)
            )
            x1 = np.random.randint(
                max(0, x_max - self.width),
                min(img_w - self.width + 1, x_min + 1)
            )
        
        return {'crop_coords': (y1, y1 + self.height, x1, x1 + self.width)}

class InstanceAwareRotation(DualTransform):
    """
    Rotation augmentation that preserves instance properties.
    
    Args:
        limit (float): Maximum rotation angle
        p (float): Probability of applying the transform
    """
    def __init__(
        self,
        limit: float = 45,
        always_apply: bool = False,
        p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.limit = limit
    
    def apply(self, img: np.ndarray, angle: float = 0, **params) -> np.ndarray:
        """Apply rotation to image."""
        height, width = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(img, matrix, (width, height))
    
    def apply_to_mask(self, mask: np.ndarray, angle: float = 0, **params) -> np.ndarray:
        """Apply rotation to mask, preserving instance IDs."""
        height, width = mask.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    
    def get_params(self) -> Dict:
        """Get random parameters."""
        return {'angle': np.random.uniform(-self.limit, self.limit)}

def get_training_augmentation(
    input_size: Tuple[int, int] = (512, 512),
    min_instance_area: float = 0.5
) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        input_size: Target size (height, width)
        min_instance_area: Minimum instance area to preserve in crop
        
    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        # Spatial transforms
        InstanceAwareRandomCrop(
            height=input_size[0],
            width=input_size[1],
            min_instance_area=min_instance_area,
            p=1.0
        ),
        InstanceAwareRotation(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        
        # Elastic transforms
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=1.0
            ),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(
                distort_limit=1,
                shift_limit=0.5,
                p=1.0
            ),
        ], p=0.3),
        
        # Normalization and conversion to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def get_validation_augmentation(
    input_size: Tuple[int, int] = (512, 512)
) -> A.Compose:
    """
    Get validation augmentation pipeline.
    
    Args:
        input_size: Target size (height, width)
        
    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        A.Resize(
            height=input_size[0],
            width=input_size[1]
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

class AugmentationVisualizer:
    """Utility class for visualizing augmentations."""
    
    @staticmethod
    def visualize_augmentations(
        image: np.ndarray,
        mask: np.ndarray,
        transform: A.Compose,
        n_examples: int = 5
    ) -> np.ndarray:
        """
        Visualize multiple applications of augmentation pipeline.
        
        Args:
            image: Input image
            mask: Input mask
            transform: Augmentation pipeline
            n_examples: Number of examples to generate
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, n_examples + 1, figsize=(3 * (n_examples + 1), 6))
        
        # Original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[1, 0].imshow(mask)
        axes[1, 0].set_title('Original Mask')
        
        # Augmented examples
        for i in range(n_examples):
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            if isinstance(aug_image, torch.Tensor):
                aug_image = aug_image.numpy().transpose(1, 2, 0)
                aug_image = (aug_image * [0.229, 0.224, 0.225] + 
                           [0.485, 0.456, 0.406])
                aug_image = np.clip(aug_image, 0, 1)
            
            axes[0, i + 1].imshow(aug_image)
            axes[0, i + 1].set_title(f'Aug {i + 1} Image')
            axes[1, i + 1].imshow(aug_mask)
            axes[1, i + 1].set_title(f'Aug {i + 1} Mask')
        
        plt.tight_layout()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_image
