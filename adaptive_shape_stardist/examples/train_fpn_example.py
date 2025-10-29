"""
Training Example with FPN Backbone
====================================

Example script for training Adaptive Shape StarDist with FPN backbone.

Features:
- FPN-based multi-scale feature extraction
- Multi-scale loss function
- Data augmentation
- Model checkpointing
- TensorBoard logging

Usage:
    python train_fpn_example.py --data_path /path/to/data --epochs 100

Author: Shape-Aware StarDist Team
Date: 2025-10-29
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from adaptive_shape_stardist.models.adaptive_shape_model import AdaptiveShapeStarDist
from adaptive_shape_stardist.configs.config import get_fpn_config, get_fpn_multiscale_config
from adaptive_shape_stardist.training.multiscale_loss import create_multiscale_loss_fn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Adaptive Shape StarDist with FPN')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='models/fpn_model',
                       help='Output directory for model')
    parser.add_argument('--config', type=str, default='fpn',
                       choices=['fpn', 'fpn_multiscale', 'fpn_fast'],
                       help='Configuration preset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--validate', action='store_true',
                       help='Use validation set')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    
    return parser.parse_args()


def load_data(data_path):
    """
    Load training data
    
    Expected format:
        data_path/
            images/
                img001.tif
                img002.tif
                ...
            masks/
                img001_mask.tif
                img002_mask.tif
                ...
    
    Returns:
        X_train: numpy array of images [N, H, W, C]
        Y_train: numpy array of masks [N, H, W]
    """
    print("\n" + "=" * 60)
    print("Loading Training Data")
    print("=" * 60)
    
    data_path = Path(data_path)
    
    # Load images
    image_dir = data_path / 'images'
    mask_dir = data_path / 'masks'
    
    if not image_dir.exists() or not mask_dir.exists():
        raise ValueError(f"Data directory structure incorrect. Expected {data_path}/images and {data_path}/masks")
    
    # Get image files
    image_files = sorted(list(image_dir.glob('*.tif')) + list(image_dir.glob('*.tiff')) + 
                        list(image_dir.glob('*.png')))
    
    print(f"Found {len(image_files)} images")
    
    # Load images and masks
    images = []
    masks = []
    
    for img_file in image_files:
        # Load image
        from tifffile import imread
        img = imread(img_file)
        
        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-7)
        
        # Add channel dimension if needed
        if img.ndim == 2:
            img = img[..., np.newaxis]
        
        images.append(img)
        
        # Load corresponding mask
        mask_file = mask_dir / f"{img_file.stem}_mask{img_file.suffix}"
        if not mask_file.exists():
            mask_file = mask_dir / f"{img_file.stem}{img_file.suffix}"
        
        if mask_file.exists():
            mask = imread(mask_file)
            masks.append(mask)
        else:
            print(f"Warning: Mask not found for {img_file.name}")
    
    X_train = np.array(images, dtype=np.float32)
    Y_train = np.array(masks, dtype=np.int32)
    
    print(f"Loaded {len(X_train)} training samples")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Mask shape: {Y_train[0].shape}")
    
    return X_train, Y_train


def prepare_targets(masks, n_rays=32):
    """
    Prepare training targets from instance masks
    
    Converts instance masks to:
    - Probability maps (object centers)
    - Distance maps (radial distances)
    
    Parameters:
    -----------
    masks : np.ndarray
        Instance masks [N, H, W]
    n_rays : int
        Number of rays for distance prediction
    
    Returns:
    --------
    targets : dict
        Dictionary containing 'prob' and 'dist' targets
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import label, regionprops
    
    N, H, W = masks.shape
    
    # Initialize targets
    prob_maps = np.zeros((N, H, W, 1), dtype=np.float32)
    dist_maps = np.zeros((N, H, W, n_rays), dtype=np.float32)
    
    # Process each image
    for i in range(N):
        mask = masks[i]
        
        # Get instance regions
        labeled = label(mask)
        props = regionprops(labeled)
        
        # For each instance
        for prop in props:
            cy, cx = prop.centroid
            cy, cx = int(cy), int(cx)
            
            # Set probability at center
            prob_maps[i, cy, cx, 0] = 1.0
            
            # Compute distances (simplified - should use proper StarDist distance computation)
            angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
            for j, angle in enumerate(angles):
                # Simplified distance computation
                # In practice, should use ray-casting to find boundary intersection
                dist_maps[i, cy, cx, j] = 10.0  # Placeholder
    
    return {
        'prob': prob_maps,
        'dist': dist_maps,
    }


def main():
    """Main training function"""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print(" " * 20 + "TRAINING ADAPTIVE SHAPE STARDIST WITH FPN")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    
    if args.config == 'fpn':
        config = get_fpn_config()
    elif args.config == 'fpn_multiscale':
        config = get_fpn_multiscale_config()
    else:
        from adaptive_shape_stardist.configs.config import get_fpn_fast_config
        config = get_fpn_fast_config()
    
    # Override config with command line args
    config.train_epochs = args.epochs
    config.train_batch_size = args.batch_size
    config.train_learning_rate = args.learning_rate
    
    print(f"Backbone: FPN with ResNet-34")
    print(f"FPN Channels: {config.fpn_channels}")
    print(f"Multi-scale Prediction: {config.multiscale_prediction}")
    print(f"Shape Prior: {config.use_shape_prior}")
    print(f"Batch Size: {config.train_batch_size}")
    print(f"Learning Rate: {config.train_learning_rate}")
    print(f"Epochs: {config.train_epochs}")
    
    # Load data
    X_train, Y_train = load_data(args.data_path)
    
    # Prepare targets
    print("\n" + "=" * 60)
    print("Preparing Targets")
    print("=" * 60)
    targets = prepare_targets(Y_train, n_rays=config.max_sampling_points)
    print("✓ Targets prepared")
    
    # Create model
    print("\n" + "=" * 60)
    print("Building Model")
    print("=" * 60)
    
    model = AdaptiveShapeStarDist(
        config=config,
        name='fpn_stardist',
        basedir=str(output_dir.parent)
    )
    
    # Build model (run dummy forward pass)
    dummy_input = tf.zeros((1, 512, 512, config.n_channel_in))
    _ = model(dummy_input, training=False)
    
    print(f"✓ Model built")
    print(f"✓ Total parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
    
    # Setup training
    print("\n" + "=" * 60)
    print("Setup Training")
    print("=" * 60)
    
    # Create loss function
    if config.multiscale_prediction:
        loss_fn = create_multiscale_loss_fn(
            level_weights={'p2': 1.0, 'p3': 0.5, 'p4': 0.25, 'p5': 0.125},
            loss_weights={'focal': 1.0, 'dist': 1.0}
        )
    else:
        # Use standard loss for single-scale prediction
        from adaptive_shape_stardist.training.loss import focal_loss, smooth_l1_loss
        loss_fn = lambda y_true, y_pred: focal_loss(y_pred['prob'], y_true['prob']) + \
                                         smooth_l1_loss(y_pred['dist'], y_true['dist'])
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.train_learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    print("✓ Model compiled")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'model_checkpoint_{epoch:03d}.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=20,
            verbose=1,
            restore_best_weights=True
        ),
    ]
    
    if args.tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(output_dir / 'logs'),
                histogram_freq=1,
                write_graph=True
            )
        )
    
    # Train model
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    history = model.fit(
        X_train,
        targets,
        batch_size=config.train_batch_size,
        epochs=config.train_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    model.save_model(str(output_dir))
    print(f"✓ Model saved to {output_dir}")
    
    # Print training summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

