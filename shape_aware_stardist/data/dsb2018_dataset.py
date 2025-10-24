"""DSB2018 dataset loader for Shape-aware StarDist"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from skimage import io, transform
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..configs.dsb2018_config import (
    DATASET_CONFIG,
    PROCESSING_CONFIG,
    INSTANCE_CONFIG
)
from .stardist_utils import StarDistTargetGenerator

class DSB2018Dataset(Dataset):
    """
    Dataset class for DSB2018 nuclei segmentation.
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        transform: Optional albumentations transformations
        input_size: Input image size
        n_rays: Number of rays for StarDist
    """
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        input_size: Tuple[int, int] = (256, 256),
        n_rays: int = 32
    ):
        self.split = split
        self.input_size = input_size
        self.n_rays = n_rays
        
        # Get data paths
        self.image_dir = DATASET_CONFIG[split]['images']
        self.mask_dir = DATASET_CONFIG[split]['masks']
        
        # Verify paths exist
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                "Please run: python -m shape_aware_stardist.data.download_dsb2018 --download"
            )
        
        # Get file lists
        self.image_files = sorted(list(self.image_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # Initialize target generator
        self.target_generator = StarDistTargetGenerator(
            n_rays=n_rays,
            min_object_size=INSTANCE_CONFIG['min_object_size']
        )
        
        print(f"Loaded {len(self.image_files)} images for {split} split")
    
    def _get_default_transform(self) -> A.Compose:
        """Get default data transformations based on split."""
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.RandomGamma(gamma_limit=(80, 120)),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                ], p=0.5),
                A.Normalize(
                    mean=PROCESSING_CONFIG['normalize']['mean'],
                    std=PROCESSING_CONFIG['normalize']['std']
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(
                    mean=PROCESSING_CONFIG['normalize']['mean'],
                    std=PROCESSING_CONFIG['normalize']['std']
                ),
                ToTensorV2()
            ])
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image."""
        image = io.imread(str(path))
        
        # Convert grayscale to RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        return image
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load instance segmentation mask."""
        mask = io.imread(str(path))
        
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        return mask.astype(np.uint16)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            dict: Dictionary containing:
                - image: Input image tensor
                - distances: StarDist distance targets
                - probabilities: Object probability targets
                - mask: Original instance mask
                - image_id: Image identifier
        """
        # Load image and mask
        image_file = self.image_files[idx]
        mask_file = self.mask_dir / image_file.name
        
        image = self._load_image(image_file)
        mask = self._load_mask(mask_file)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Compute StarDist targets
        distances, probabilities = self.target_generator.generate_targets(mask)
        
        return {
            'image': image,
            'distances': distances,
            'probabilities': probabilities,
            'mask': torch.from_numpy(mask),
            'image_id': image_file.stem
        }

def get_dsb2018_loaders(
    batch_size: int = 8,
    input_size: Tuple[int, int] = (256, 256),
    n_rays: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for DSB2018.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        n_rays: Number of rays for StarDist
        num_workers: Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = DSB2018Dataset(
        split='train',
        input_size=input_size,
        n_rays=n_rays
    )
    
    val_dataset = DSB2018Dataset(
        split='val',
        input_size=input_size,
        n_rays=n_rays
    )
    
    test_dataset = DSB2018Dataset(
        split='test',
        input_size=input_size,
        n_rays=n_rays
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test one image at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test dataset loading
    print("Testing DSB2018 dataset loading...")
    
    try:
        train_loader, val_loader, test_loader = get_dsb2018_loaders(
            batch_size=4,
            num_workers=0
        )
        
        print(f"\nDataset sizes:")
        print(f"Train: {len(train_loader.dataset)}")
        print(f"Val: {len(val_loader.dataset)}")
        print(f"Test: {len(test_loader.dataset)}")
        
        print("\nTesting batch loading...")
        batch = next(iter(train_loader))
        print(f"Image shape: {batch['image'].shape}")
        print(f"Distances shape: {batch['distances'].shape}")
        print(f"Probabilities shape: {batch['probabilities'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        
        print("\nDataset loading test passed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease run the following command to download and prepare the dataset:")
        print("python -m shape_aware_stardist.data.download_dsb2018 --download")
