"""Configuration for DSB2018 dataset"""

from pathlib import Path

# Base paths
DATA_ROOT = Path("/Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/data/dsb2018")

# Dataset structure for DSB2018
DATASET_CONFIG = {
    'train': {
        'images': DATA_ROOT / 'train/images',
        'masks': DATA_ROOT / 'train/masks',
    },
    'val': {
        'images': DATA_ROOT / 'val/images',
        'masks': DATA_ROOT / 'val/masks',
    },
    'test': {
        'images': DATA_ROOT / 'test/images',
        'masks': DATA_ROOT / 'test/masks',
    }
}

# Data processing settings for DSB2018
PROCESSING_CONFIG = {
    'input_size': (256, 256),  # DSB2018 images vary in size, resize to 256x256
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
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }
    }
}

# Instance segmentation settings for DSB2018
INSTANCE_CONFIG = {
    'min_object_size': 20,  # Minimum nucleus size in pixels
    'max_object_size': None,  # No maximum limit
    'boundary_thickness': 2,
    'n_rays': 32  # Number of rays for StarDist
}

# Training configuration for DSB2018
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 200,
    'early_stopping_patience': 30,
    'validation_interval': 1,
    'save_interval': 10,
    
    # Optimizer settings
    'optimizer': 'adam',
    'weight_decay': 1e-5,
    'momentum': 0.9,
    
    # Learning rate scheduler
    'scheduler': {
        'type': 'reduce_on_plateau',
        'mode': 'min',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-6
    },
    
    # Loss weights
    'loss_weights': {
        'distance': 1.0,
        'probability': 1.0,
        'consistency': 1.0,
        'smoothness': 0.5,
        'adversarial': 0.1
    }
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
            "\n".join(missing_paths) +
            "\n\nPlease run: python -m shape_aware_stardist.data.download_dsb2018 --download"
        )
    
    return True

def get_dataset_path(split: str, data_type: str) -> Path:
    """Get path for specific dataset split and type."""
    if split not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset split: {split}")
    if data_type not in ['images', 'masks']:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return DATASET_CONFIG[split][data_type]
