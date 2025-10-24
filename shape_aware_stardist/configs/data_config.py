"""Data configuration for Shape-aware StarDist"""

from pathlib import Path

# Base paths
DATA_ROOT = Path("/Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/data/shape_data")

# Dataset structure
DATASET_CONFIG = {
    'train': {
        'images': DATA_ROOT / '240819_Ji_N1_H_EScan/patches_2048',
        'masks': DATA_ROOT / '240819_Ji_N1_H_EScan/pseudo_gt_masks_2048',
    },
    'val': {
        'images': DATA_ROOT / 'test_slide_1/patches_2048',
        'masks': DATA_ROOT / 'test_slide_1/pseudo_gt_masks_2048',
    },
    'test': {
        'images': DATA_ROOT / 'test_slide_2/patches_2048',
        'masks': DATA_ROOT / 'test_slide_2/pseudo_gt_masks_2048',
    },
    # Additional test sets
    'scale_variation': {
        'images': DATA_ROOT / 'scale_variation_slide/patches_2048',
        'masks': DATA_ROOT / 'scale_variation_slide/pseudo_gt_masks_2048',
    }
}

# Data processing settings
PROCESSING_CONFIG = {
    'input_size': (2048, 2048),  # Using original patch size
    'normalize': {
        'mean': [0.485, 0.456, 0.406],  # ImageNet normalization
        'std': [0.229, 0.224, 0.225]
    },
    'augmentation': {
        'enabled': True,
        'rotation_range': (-180, 180),
        'scale_range': (0.8, 1.2),
        'flip_probability': 0.5,
        'elastic_transform': {
            'alpha': 120,
            'sigma': 8
        },
        'random_crop': {
            'enabled': True,
            'size': (1024, 1024),  # Crop size for training
            'min_instance_area': 0.5  # Minimum instance area after cropping
        }
    }
}

# Instance segmentation settings
INSTANCE_CONFIG = {
    'min_object_size': 100,  # Minimum object size in pixels
    'max_object_size': None,  # Maximum object size (None for no limit)
    'boundary_thickness': 2,  # Boundary thickness for visualization
    'n_rays': 32  # Number of rays for StarDist
}

def verify_data_paths():
    """Verify that all data paths exist."""
    missing_paths = []
    
    for split, paths in DATASET_CONFIG.items():
        for key, path in paths.items():
            if not path.exists():
                missing_paths.append(f"{split}/{key}: {path}")
    
    if missing_paths:
        raise FileNotFoundError(
            "The following data paths are missing:\n" +
            "\n".join(missing_paths)
        )
    
    return True

def get_dataset_path(split: str, data_type: str) -> Path:
    """Get path for specific dataset split and type."""
    if split not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset split: {split}")
    if data_type not in ['images', 'masks']:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return DATASET_CONFIG[split][data_type]
