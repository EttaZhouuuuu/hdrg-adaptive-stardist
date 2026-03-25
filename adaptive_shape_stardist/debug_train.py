"""
Simplified Training Script for Debugging
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Setup paths
sys.path.insert(0, '/data/yitongzhou/workspace/hdrg-adaptive-stardist')

from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.models.adaptive_shape_model import AdaptiveShapeConfig
from adaptive_shape_stardist.training.trainer import AdaptiveShapeTrainer

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

tf.get_logger().setLevel('ERROR')

def main():
    print("=" * 60)
    print("SIMPLIFIED TRAINING DEBUG")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = np.load('/data/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    # Use very small subset for debugging
    X_train = X[:10]
    Y_train = Y[:10]
    X_val = X[10:12]
    Y_val = Y[10:12]
    
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
        use_shape_prior=False,  # Disable for simplicity
        use_fpn=False,
        multiscale_prediction=False,
        train_epochs=1,
        train_batch_size=2,
        train_steps_per_epoch=len(X_train) // 2,
        train_learning_rate=1e-3,
        grid=(1, 1),
        weight_decay=1e-5,
        dropout_rate=0.1,
    )
    print(f"   Config created: backbone={config.backbone}, depth={config.n_depth}")
    
    # Create model
    print("\n3. Creating model...")
    model_dir = Path('/data/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/models_debug')
    model_dir.mkdir(exist_ok=True)
    
    model = AdaptiveShapeStarDist(
        config=config,
        name='debug_model',
        basedir=str(model_dir)
    )
    
    # Build model
    print("4. Building model...")
    dummy_input = tf.zeros((1, X_train.shape[1], X_train.shape[2], 1))
    try:
        output = model(dummy_input, training=False)
        print(f"   Model output keys: {list(output.keys())}")
    except Exception as e:
        print(f"   ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create trainer
    print("\n5. Creating trainer...")
    trainer = AdaptiveShapeTrainer(model, config)
    
    # Normalize images
    print("\n6. Normalizing images...")
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_val_norm = X_val.astype(np.float32) / 255.0
    
    # Train for one step
    print("\n7. Testing training step...")
    try:
        # Create a single batch
        X_batch = X_train_norm[:2]
        Y_batch_sample = Y_train[:2]
        
        # Create label batches
        Y_prob_batch = []
        Y_dist_batch = []
        
        for i in range(2):
            label = Y_batch_sample[i]
            # Create probability map - should be 2D then add channel
            prob_map = (label > 0).astype(np.float32)
            if prob_map.ndim == 2:
                prob_map = prob_map[..., np.newaxis]  # (H, W, 1)
            Y_prob_batch.append(prob_map)
            
            # Create distance map (simplified)
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(label > 0)
            dist = np.stack([dist] * config.max_sampling_points, axis=-1)
            Y_dist_batch.append(dist)
        
        Y_prob_batch = np.stack(Y_prob_batch, axis=0)  # (2, H, W, 1)
        Y_dist_batch = np.stack(Y_dist_batch, axis=0)  # (2, H, W, N_points)
        
        Y_batch = {
            'prob': Y_prob_batch,
            'dist': Y_dist_batch,
            'mask': np.ones((2, Y_prob_batch.shape[1], Y_prob_batch.shape[2], 1)),
        }
        
        print(f"   X_batch shape: {X_batch.shape}")
        print(f"   Y_batch['prob'] shape: {Y_prob_batch.shape}")
        print(f"   Y_batch['dist'] shape: {Y_dist_batch.shape}")
        
        # Test training step
        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            print(f"   Predictions keys: {list(predictions.keys())}")
            
            # Check predictions
            for k, v in predictions.items():
                if hasattr(v, 'shape'):
                    print(f"   - {k}: {v.shape}")
            
            # Compute loss
            loss = trainer.loss_fn(Y_batch, predictions)
            print(f"   Loss: {float(loss)}")
        
        # Backward pass
        gradients = tape.gradient(loss, model.trainable_variables)
        print(f"   Gradients computed: {len(gradients)} variables")
        
        print("\n✅ Training step successful!")
        
    except Exception as e:
        print(f"   ERROR during training step: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE - NO ERRORS!")
    print("=" * 60)

if __name__ == '__main__':
    main()

