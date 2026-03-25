"""
GPU-Accelerated Training Script for Xenium Data with Shape Prior
================================================================

Uses GPU for ~10x faster training
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os
import json

# Setup paths
PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

# Configure TensorFlow for GPU
print("=" * 70)
print("CONFIGURING GPU")
print("=" * 70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs found: {len(gpus)}")

if gpus:
    # Enable GPU memory growth to avoid OOM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Use all GPUs with MirroredStrategy
    print(f"✅ Using {len(gpus)} GPUs with TensorFlow")
    
    # Configure for optimal GPU performance
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "arithmetic_optimization": True,
        "disable_meta_optimizer": False,
        "function_optimization": True
    })
else:
    print("⚠️ No GPUs found, using CPU")

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.models.adaptive_shape_model import AdaptiveShapeConfig
from adaptive_shape_stardist.training.trainer import AdaptiveShapeTrainer

tf.get_logger().setLevel('ERROR')

def create_HE_normalizer():
    """Create H&E normalizer"""
    def normalize(img):
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        return img
    return normalize

def main():
    parser = argparse.ArgumentParser(description="Train Adaptive Shape StarDist on Xenium data")
    parser.add_argument("--patches", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)  # Larger batch for GPU
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    print("=" * 70)
    print("ADAPTIVE SHAPE Stardist - GPU-ACCELERATED TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Patches: {args.patches}")
    print(f"  - Batch size: {args.batch_size} (GPU-optimized)")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - GPUs: {len(gpus)}")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    print(f"  Total samples: {len(X)}")
    
    # Split
    train_split = 0.9
    split_idx = int(len(X) * train_split)
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_val = X[split_idx:]
    Y_val = Y[split_idx:]
    
    # Use subset
    X_train = X_train[:args.patches]
    Y_train = Y_train[:args.patches]
    X_val = X_val[:args.patches//10]
    Y_val = Y_val[:args.patches//10]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Create complete config with Shape Prior
    print("\n" + "=" * 70)
    print("CREATING MODEL CONFIGURATION")
    print("=" * 70)
    
    config = AdaptiveShapeConfig(
        backbone='unet',
        n_depth=3,
        n_filter_base=32,
        min_sampling_points=16,
        max_sampling_points=32,
        use_shape_prior=True,
        num_shape_prototypes=16,
        deformable_groups=4,
        use_fpn=False,
        multiscale_prediction=False,
        train_epochs=args.epochs,
        train_batch_size=args.batch_size,
        train_steps_per_epoch=len(X_train) // args.batch_size,
        train_learning_rate=args.lr,
        grid=(1, 1),
        weight_decay=1e-4,
        dropout_rate=0.2,
        label_smoothing=0.1,
    )
    
    print(f"\n  Model Architecture:")
    print(f"    - Backbone: {config.backbone}")
    print(f"    - Shape Prior: {config.use_shape_prior}")
    print(f"    - Steps per epoch: {config.train_steps_per_epoch}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model_dir = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_xenium_gpu')
    model_dir.mkdir(exist_ok=True)
    
    model = AdaptiveShapeStarDist(
        config=config,
        name='xenium_gpu_model',
        basedir=str(model_dir)
    )
    
    # Build model
    print("  Building model on GPU...")
    dummy_input = tf.zeros((1, X_train.shape[1], X_train.shape[2], 1))
    output = model(dummy_input, training=False)
    
    print(f"  Output keys: {list(output.keys())}")
    if 'prototype_weights' in output:
        print(f"  ✅ Shape Prior Encoder active!")
    
    # Normalize images
    print("\n" + "=" * 70)
    print("NORMALIZING IMAGES")
    print("=" * 70)
    
    normalizer = create_HE_normalizer()
    X_train_norm = normalizer(X_train)
    X_val_norm = normalizer(X_val)
    
    print(f"  Image range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    
    # Create trainer
    print("\n" + "=" * 70)
    print("CREATING TRAINER")
    print("=" * 70)
    
    trainer = AdaptiveShapeTrainer(model, config)
    print(f"  Optimizer: Adam")
    print(f"  Loss: AdaptiveShapeLoss")
    print(f"  Steps per epoch: {config.train_steps_per_epoch}")
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'lr': [],
    }
    
    # Save initial checkpoint
    checkpoint_path = model_dir / 'checkpoint_epoch.txt'
    with open(checkpoint_path, 'w') as f:
        f.write('0')
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING GPU-ACCELERATED TRAINING")
    print("=" * 70)
    
    start_time = np.datetime64('now')
    
    for epoch in range(args.epochs):
        epoch_start = np.datetime64('now')
        
        print(f"\nEpoch {epoch + 1:3d}/{args.epochs}")
        print("-" * 50)
        
        # Train epoch
        train_generator = trainer._create_generator(
            X_train_norm, Y_train, config.train_batch_size, 
            augment=True,
            augmenter=None
        )
        train_loss = trainer._train_epoch(train_generator)
        
        # Validate
        val_generator = trainer._create_generator(
            X_val_norm, Y_val, config.train_batch_size, 
            augment=False
        )
        val_loss = trainer._validate_epoch(val_generator)
        
        # Update history
        history['loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['lr'].append(float(config.train_learning_rate))
        
        # Calculate epoch time
        epoch_time = (np.datetime64('now') - epoch_start) / np.timedelta64(1, 's')
        
        # Print progress
        print(f"  Train Loss: {train_loss:10.4f} | Val Loss: {val_loss:10.4f} | Time: {epoch_time:6.1f}s")
        
        # Save checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            print(f"\n  💾 Saving checkpoint at epoch {epoch + 1}...")
            
            # Save model
            model.save_model()
            
            # Save checkpoint epoch
            with open(checkpoint_path, 'w') as f:
                f.write(str(epoch + 1))
            
            # Save training history
            history_path = model_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"  ✅ Checkpoint saved!")
    
    # Save final model
    print("\n" + "=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)
    
    model.save_model()
    
    # Save final history
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Calculate total training time
    total_time = (np.datetime64('now') - start_time) / np.timedelta64(1, 'm')
    
    # Final summary
    print("\n" + "=" * 70)
    print("GPU TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  📊 Final Results:")
    print(f"    - Initial Train Loss: {history['loss'][0]:.4f}")
    print(f"    - Final Train Loss:   {history['loss'][-1]:.4f}")
    print(f"    - Best Val Loss:       {min(history['val_loss']):.4f}")
    print(f"    - Total Training Time: {total_time:.1f} minutes")
    print(f"\n  💾 Model saved to: {model.model_dir}")
    
    return model, history

if __name__ == '__main__':
    import argparse
    main()

