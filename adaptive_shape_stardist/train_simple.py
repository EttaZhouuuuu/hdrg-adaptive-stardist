"""
Simplified Training Script for Xenium Data - Fixed Version
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os

# Setup paths
PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.models.adaptive_shape_model import AdaptiveShapeConfig
from adaptive_shape_stardist.training.trainer import AdaptiveShapeTrainer

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLIFIED TRAINING: XENIUM DATA")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    print(f"   Total samples: {len(X)}")
    
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
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Create simplified config
    print("\n2. Creating simplified config...")
    config = AdaptiveShapeConfig(
        backbone='unet',
        n_depth=2,
        n_filter_base=16,
        min_sampling_points=8,
        max_sampling_points=16,
        use_shape_prior=False,
        use_fpn=False,
        multiscale_prediction=False,
        train_epochs=args.epochs,
        train_batch_size=args.batch_size,
        train_steps_per_epoch=len(X_train) // args.batch_size,
        train_learning_rate=args.lr,
        grid=(1, 1),
        weight_decay=1e-5,
        dropout_rate=0.1,
    )
    print(f"   Backbone: {config.backbone}, Epochs: {config.train_epochs}")
    print(f"   Batch size: {config.train_batch_size}, LR: {config.train_learning_rate}")
    
    # Create model
    print("\n3. Creating model...")
    model_dir = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_xenium')
    model_dir.mkdir(exist_ok=True)
    
    model = AdaptiveShapeStarDist(
        config=config,
        name='xenium_debug',
        basedir=str(model_dir)
    )
    
    # Build model
    print("4. Building model...")
    dummy_input = tf.zeros((1, X_train.shape[1], X_train.shape[2], 1))
    output = model(dummy_input, training=False)
    print(f"   Output keys: {list(output.keys())}")
    
    # Normalize images
    print("\n5. Normalizing images...")
    normalizer = create_HE_normalizer()
    X_train_norm = normalizer(X_train)
    X_val_norm = normalizer(X_val)
    
    # Create trainer
    print("\n6. Creating trainer...")
    trainer = AdaptiveShapeTrainer(model, config)
    
    # Train
    print("\n7. Starting training...")
    history = {
        'loss': [],
        'val_loss': [],
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train epoch
        train_loss = trainer._train_epoch(
            trainer._create_generator(X_train_norm, Y_train, config.train_batch_size, augment=False)
        )
        
        # Validate
        val_generator = trainer._create_generator(X_val_norm, Y_val, config.train_batch_size, augment=False)
        val_loss = trainer._validate_epoch(val_generator)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, history

if __name__ == '__main__':
    import argparse
    main()

