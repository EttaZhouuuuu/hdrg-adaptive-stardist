"""
Training Script: Xenium Mouse H&E Data
=======================================

This script demonstrates how to train Adaptive Shape StarDist on 
Xenium mouse brain data with H&E images and cell polygon annotations.

Usage:
------
    python scripts/train_xenium_data.py --patches 1000 --epochs 50

Data Format:
------------
Input:
    - cells.zarr: Xenium polygon-based cell segmentation
      * 162,033 cells with polygon vertices
      * Format: interleaved [y0, x0, y1, x1, ...]
      
    - morphology.ome.tif: H&E morphology image
      * Large format image (7065 x 9985 pixels)
      * Currently using synthetic data (see note below)

Output:
    - Trained model in models/xenium_he_*/
    - Training history plots
"""

import numpy as np
import tensorflow as tf

# Configure TensorFlow for offline mode - prevents ALL network operations
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable oneDNN telemetry and network operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Block any potential network operations at Python level
import urllib.request
_original_urlopen = urllib.request.urlopen

def _blocked_urlopen(url, data=None, timeout=None, headers=None, *, context=None):
    """Block all network requests"""
    raise urllib.error.URLError("Network access disabled in training mode")
urllib.request.urlopen = _blocked_urlopen

# Configure TF
tf.config.set_visible_devices([], 'GPU')  # Use CPU only to avoid GPU/network issues
# tf.config.set_visible_devices([], 'GPU')  # Uncomment for CPU-only mode

from pathlib import Path
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

# Import Adaptive Shape StarDist
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.models.adaptive_shape_model import AdaptiveShapeConfig
from adaptive_shape_stardist.utils import plot_training_history
from adaptive_shape_stardist.utils.xenium_preprocessing import (
    XeniumDataLoader,
    create_training_dataset
)
from adaptive_shape_stardist.data.augmentation import create_augmenter


def load_xenium_data(
    cells_zarr_path: str,
    image_path: str,
    patch_size: int = 256,
    num_train_patches: int = 1000,
    num_val_patches: int = 100,
    train_split: float = 0.9,
    seed: int = 42
):
    """
    Load and preprocess Xenium data for training.
    
    Args:
        cells_zarr_path: Path to cells.zarr directory
        image_path: Path to morphology image (H&E)
        patch_size: Size of training patches
        num_train_patches: Number of training patches
        num_val_patches: Number of validation patches
        train_split: Fraction of data for training
        seed: Random seed
        
    Returns:
        X_train, Y_train, X_val, Y_val
    """
    np.random.seed(seed)
    
    print("=" * 60)
    print("LOADING XENIUM MOUSE BRAIN DATA")
    print("=" * 60)
    
    # Load Xenium data
    loader = XeniumDataLoader(
        cells_zarr_path=cells_zarr_path,
        image_path=image_path,
        load_image=False
    )
    
    # Try to load real image, fall back to synthetic
    try:
        import tifffile
        loader.image = tifffile.imread(image_path)
        print(f"✓ Loaded H&E image: {loader.image.shape}")
    except Exception as e:
        print(f"⚠️  Could not load morphology image: {e}")
        print("  Creating synthetic grayscale image for testing...")
        loader.image = np.random.randint(0, 256, (loader.image_height, loader.image_width), dtype=np.uint8)
    
    # Create training patches
    total_patches = num_train_patches + num_val_patches
    
    print(f"\nExtracting {total_patches} patches (patch_size={patch_size})...")
    images, labels = create_training_dataset(
        cells_zarr_path=cells_zarr_path,
        image_path=image_path,
        output_path="",  # Don't save to file
        patch_size=patch_size,
        num_patches=total_patches,
        seed=seed
    )
    
    # Split into train/val
    split_idx = int(len(images) * train_split)
    X_train, X_val = images[:split_idx], images[split_idx:]
    Y_train, Y_val = labels[:split_idx], labels[split_idx:]
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Image shape: {X_train[0].shape}")
    print(f"  Label range: {int(Y_train.min())} - {int(Y_train.max())}")
    
    return X_train, Y_train, X_val, Y_val


def create_HE_normalizer():
    """
    Create a simple H&E normalizer.
    
    For H&E stained tissues, typical preprocessing includes:
    1. Intensity normalization
    2. Color deconvolution (optional)
    3. Illumination correction (optional)
    
    Returns:
        Function that normalizes H&E images
    """
    def normalize(img):
        """Normalize H&E image to [0, 1] range."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        return img
    
    return normalize


def train_xenium_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    backbone: str = "resnet50",
    epochs: int = 50,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    model_name: str = "xenium_he_model",
    model_dir: str = "./models",
    resume_from: int = None
):
    """
    Train Adaptive Shape StarDist model on Xenium H&E data.
    
    Args:
        X_train: Training images [N, H, W] or [N, H, W, C]
        Y_train: Training labels [N, H, W] with instance IDs
        X_val: Validation images
        Y_val: Validation labels
        backbone: Backbone network (resnet50, resnet101, efficientnetb0)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_name: Name for the model
        model_dir: Directory to save model
        resume_from: Epoch number to resume from (None = start fresh)
    """
    print("\n" + "=" * 60)
    print("TRAINING ADAPTIVE SHAPE Stardist")
    print("=" * 60)
    
    # Check for existing checkpoint
    model_path = Path(model_dir) / model_name
    checkpoint_path = model_path / 'checkpoint_epoch.txt'
    
    if resume_from is not None:
        print(f"\n>>> RESUMING TRAINING FROM EPOCH {resume_from}")
    elif checkpoint_path.exists():
        # Auto-detect last saved epoch
        with open(checkpoint_path, 'r') as f:
            last_epoch = int(f.read().strip())
            if last_epoch > 0:
                print(f"\n>>> FOUND CHECKPOINT: Resuming from epoch {last_epoch}")
                resume_from = last_epoch
    
    # 1. Create configuration
    print("\n1. Creating configuration...")
    config = AdaptiveShapeConfig()
    
    # Customize for Xenium data
    config.backbone = backbone
    config.train_epochs = epochs
    config.train_steps_per_epoch = len(X_train) // batch_size
    config.train_batch_size = batch_size
    config.train_learning_rate = learning_rate
    
    # Enable FPN and multi-scale prediction
    config.use_fpn = True
    config.multiscale_prediction = True
    config.fpn_channels = 256
    config.fpn_levels = ['p2', 'p3', 'p4', 'p5']
    
    # Adaptive Shape StarDist specific settings
    # Shape prior with downsampling to avoid OOM
    config.min_sampling_points = 16
    config.max_sampling_points = 32
    config.use_shape_prior = True  # Enabled with downsampling
    config.shape_prior_layers = 2  # Reduced layers for efficiency
    
    # Regularization to prevent overfitting
    config.weight_decay = 1e-4
    config.dropout_rate = 0.3
    config.label_smoothing = 0.1
    
    print(f"   Backbone: {config.backbone}")
    print(f"   Use FPN: {config.use_fpn}")
    print(f"   Multi-scale prediction: {config.multiscale_prediction}")
    print(f"   Epochs: {config.train_epochs}")
    print(f"   Batch size: {config.train_batch_size}")
    print(f"   Learning rate: {config.train_learning_rate}")
    print(f"   Sampling points: {config.min_sampling_points}-{config.max_sampling_points}")
    
    # 2. Create model
    print("\n2. Creating Adaptive Shape StarDist model...")
    model = AdaptiveShapeStarDist(
        config=config,
        name=model_name,
        basedir=model_dir
    )
    
    # Build model by running a dummy forward pass
    print("   Building model...")
    dummy_input = tf.zeros((1, X_train.shape[1], X_train.shape[2], config.n_channel_in))
    _ = model(dummy_input, training=False)
    print(f"   Model created at: {model.model_dir}")
    
    # 3. Normalize images
    print("\n3. Normalizing images...")
    normalizer = create_HE_normalizer()
    X_train_norm = normalizer(X_train)
    X_val_norm = normalizer(X_val)
    print(f"   Image range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    
    # 4. Create augmenter and train model
    print("\n4. Creating data augmenter...")
    augmenter = create_augmenter(
        horizontal_flip=True,
        vertical_flip=True,
        rotation=True,
        brightness=True,
        contrast=True,
        elastic=True,
        noise=True,
        strength='medium'
    )
    print("   - Random horizontal/vertical flips")
    print("   - Random rotations (±30°)")
    print("   - Elastic deformations")
    print("   - Brightness/contrast adjustments")
    print("   - Gaussian noise")
    
    print("\n5. Starting training...")
    print(f"   Training samples: {len(X_train_norm)}")
    print(f"   Validation samples: {len(X_val_norm)}")
    
    history = model.train_model(
        X_train=X_train_norm,
        Y_train=Y_train,
        validation_data=(X_val_norm, Y_val),
        augmenter=augmenter
    )
    
    # 6. Plot training history
    print("\n7. Saving training history...")
    fig = plot_training_history(history, save_path=f'{model_name}_history.png')
    print(f"   Saved to: {model_name}_history.png")
    
    # 7. Save model
    print("\n8. Saving model...")
    model.save_model()
    print(f"   Model saved to: {model.model_dir}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {model.model_dir}")
    print(f"To run inference, use:")
    print(f"  model = AdaptiveShapeStarDist.load('{model.model_dir}')")
    
    return model, history


def main():
    """Main entry point"""
    parser = ArgumentParser(description="Train Adaptive Shape StarDist on Xenium data")
    
    parser.add_argument("--cells-zarr", type=str, 
                        default="data/cells.zarr",
                        help="Path to cells.zarr directory")
    parser.add_argument("--image", type=str,
                        default="data/morphology_focus.ome.tif",
                        help="Path to morphology image")
    parser.add_argument("--patches", type=int, default=1000,
                        help="Number of training patches")
    parser.add_argument("--val-patches", type=int, default=100,
                        help="Number of validation patches")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--backbone", type=str, default="unet",
                        choices=["unet", "resnet"],
                        help="Backbone network")
    parser.add_argument("--model-name", type=str, default="xenium_he_model",
                        help="Model name")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Directory to save model")
    parser.add_argument("--resume", type=int, default=None,
                        help="Resume training from this epoch number")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-detect and resume from last checkpoint")
    
    args = parser.parse_args()
    
    # Get absolute paths (relative to script location if needed)
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    
    # Convert relative paths to absolute paths
    cells_zarr_path = Path(args.cells_zarr)
    if not cells_zarr_path.is_absolute():
        cells_zarr_path = project_dir / args.cells_zarr
    
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = project_dir / args.image
    
    print(f"\nData paths:")
    print(f"  cells.zarr: {cells_zarr_path}")
    print(f"  image: {image_path}")
    
    # Load data
    X_train, Y_train, X_val, Y_val = load_xenium_data(
        cells_zarr_path=str(cells_zarr_path),
        image_path=str(image_path),
        patch_size=args.patch_size,
        num_train_patches=args.patches,
        num_val_patches=args.val_patches
    )
    
    # Train model
    model, history = train_xenium_model(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model_name,
        model_dir=args.model_dir,
        resume_from=args.resume if not args.auto_resume else None
    )


if __name__ == '__main__':
    main()

