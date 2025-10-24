"""
Example: Training Adaptive Shape StarDist
==========================================

This script demonstrates how to train the Adaptive Shape StarDist model
on your custom dataset.
"""

import numpy as np
from pathlib import Path

# Import Adaptive Shape StarDist
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.configs import get_default_config
from adaptive_shape_stardist.utils import plot_training_history


def load_data():
    """
    Load your training data
    
    Returns:
    --------
    X_train : np.ndarray
        Training images [N, H, W] or [N, H, W, C]
    Y_train : np.ndarray
        Training labels [N, H, W] with instance IDs
    X_val : np.ndarray
        Validation images
    Y_val : np.ndarray
        Validation labels
    """
    
    # TODO: Replace with your data loading code
    # This is a placeholder example
    
    # Example: Load from directory
    # from skimage.io import imread
    # X_train = [imread(f) for f in train_image_files]
    # Y_train = [imread(f) for f in train_label_files]
    
    # For demonstration, create dummy data
    print("Creating dummy data for demonstration...")
    N_train = 20
    N_val = 5
    H, W = 256, 256
    
    X_train = np.random.rand(N_train, H, W).astype(np.float32)
    Y_train = np.zeros((N_train, H, W), dtype=np.int32)
    
    X_val = np.random.rand(N_val, H, W).astype(np.float32)
    Y_val = np.zeros((N_val, H, W), dtype=np.int32)
    
    # Create some dummy instances
    for i in range(N_train):
        # Add 5-10 random circular instances
        n_instances = np.random.randint(5, 10)
        for j in range(1, n_instances + 1):
            cy, cx = np.random.randint(50, H-50), np.random.randint(50, W-50)
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:H, :W]
            mask = (y - cy)**2 + (x - cx)**2 <= radius**2
            Y_train[i][mask] = j
    
    for i in range(N_val):
        n_instances = np.random.randint(5, 10)
        for j in range(1, n_instances + 1):
            cy, cx = np.random.randint(50, H-50), np.random.randint(50, W-50)
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:H, :W]
            mask = (y - cy)**2 + (x - cx)**2 <= radius**2
            Y_val[i][mask] = j
    
    return X_train, Y_train, X_val, Y_val


def main():
    """Main training script"""
    
    print("=" * 60)
    print("Adaptive Shape StarDist - Training Example")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading data...")
    X_train, Y_train, X_val, Y_val = load_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Image shape: {X_train[0].shape}")
    
    # 2. Create configuration
    print("\n2. Creating model configuration...")
    config = get_default_config()
    
    # Customize configuration if needed
    config.train_epochs = 20  # Reduce for quick demo
    config.train_steps_per_epoch = 10
    config.train_batch_size = 2
    
    print(f"   Backbone: {config.backbone}")
    print(f"   Sampling points: {config.min_sampling_points}-{config.max_sampling_points}")
    print(f"   Shape prior: {config.use_shape_prior}")
    
    # 3. Create model
    print("\n3. Creating model...")
    model = AdaptiveShapeStarDist(
        config=config,
        name='my_adaptive_model',
        basedir='./models'
    )
    print("   Model created successfully!")
    
    # 4. Train model
    print("\n4. Starting training...")
    history = model.train_model(
        X_train=X_train,
        Y_train=Y_train,
        validation_data=(X_val, Y_val),
        augmenter=None  # Add data augmentation if needed
    )
    
    # 5. Plot training history
    print("\n5. Plotting training history...")
    fig = plot_training_history(history, save_path='./training_history.png')
    print("   Training history saved to: training_history.png")
    
    # 6. Save model
    print("\n6. Saving model...")
    model.save_model()
    print(f"   Model saved to: {model.model_dir}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

