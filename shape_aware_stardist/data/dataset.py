import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from skimage import io, transform
from .augmentation import get_training_augmentation, get_validation_augmentation

class ShapeAwareDataset(Dataset):
    """
    Dataset class for Shape-aware StarDist model.
    
    Args:
        image_dir (str): Directory containing input images
        mask_dir (str): Directory containing ground truth masks
        transform (Optional[A.Compose]): Albumentations transformations
        input_size (Tuple[int, int]): Input image size
        n_rays (int): Number of rays for StarDist
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[A.Compose] = None,
        input_size: Tuple[int, int] = (512, 512),
        n_rays: int = 32
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.input_size = input_size
        self.n_rays = n_rays
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.tiff")))
        if not self.image_files:
            self.image_files = sorted(list(self.image_dir.glob("*.png")))
        
        # Validate dataset
        self._validate_dataset()
        
        # Create default transform if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()
    
    def _validate_dataset(self):
        """Validate dataset structure and files"""
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # Check corresponding mask files exist
        for img_file in self.image_files:
            mask_file = self.mask_dir / img_file.name
            if not mask_file.exists():
                raise ValueError(f"Missing mask file for {img_file}")
    
    def _get_default_transform(self) -> A.Compose:
        """Get default data transformations"""
        return A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from path"""
        image = io.imread(str(path))
        if image.ndim == 2:  # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        return image
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask from path"""
        return io.imread(str(path))
    
    def _compute_stardist_targets(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute StarDist targets (distances and probabilities) from instance mask.
        
        Args:
            mask (np.ndarray): Instance segmentation mask
            
        Returns:
            tuple: (distances, probabilities)
        """
        if not hasattr(self, 'target_generator'):
            from .stardist_utils import StarDistTargetGenerator
            self.target_generator = StarDistTargetGenerator(
                n_rays=self.n_rays,
                min_object_size=10  # Minimum object size to consider
            )
        
        return self.target_generator.generate_targets(mask)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Dictionary containing:
                - image: Input image tensor
                - distances: StarDist distance targets
                - probabilities: Object probability targets
                - mask: Original instance mask
        """
        # Load image and mask
        image = self._load_image(self.image_files[idx])
        mask = self._load_mask(self.mask_dir / self.image_files[idx].name)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Compute StarDist targets
        distances, probabilities = self._compute_stardist_targets(mask)
        
        return {
            'image': image,
            'distances': torch.from_numpy(distances),
            'probabilities': torch.from_numpy(probabilities),
            'mask': torch.from_numpy(mask)
        }

def get_data_loaders(
    image_dir: str,
    mask_dir: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    input_size: Tuple[int, int] = (512, 512),
    n_rays: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        image_dir (str): Directory containing input images
        mask_dir (str): Directory containing ground truth masks
        batch_size (int): Batch size
        train_split (float): Proportion of data for training
        input_size (tuple): Input image size
        n_rays (int): Number of rays for StarDist
        num_workers (int): Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets with different transforms
    train_transform = get_training_augmentation(input_size=input_size)
    val_transform = get_validation_augmentation(input_size=input_size)
    
    # Create full dataset
    full_dataset = ShapeAwareDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None,  # Will be set after split
        input_size=input_size,
        n_rays=n_rays
    )
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Set transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
