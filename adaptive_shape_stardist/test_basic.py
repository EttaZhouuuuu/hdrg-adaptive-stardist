"""
Basic test to verify the project works correctly
"""

import sys
import numpy as np
import tensorflow as tf

print("=" * 60)
print("Testing Adaptive Shape StarDist")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from adaptive_shape_stardist import (
        AdaptiveShapeStarDist,
        DeformableConv2D,
        AdaptiveShapeEncoder,
        ShapePriorEncoder
    )
    print("   ✓ Main imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

try:
    from adaptive_shape_stardist.configs import get_default_config
    print("   ✓ Config imports successful")
except Exception as e:
    print(f"   ✗ Config import error: {e}")
    sys.exit(1)

# Test 2: Create configuration
print("\n2. Testing configuration...")
try:
    config = get_default_config()
    print(f"   ✓ Config created")
    print(f"     - Backbone: {config.backbone}")
    print(f"     - Sampling points: {config.min_sampling_points}-{config.max_sampling_points}")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test 3: Create model
print("\n3. Testing model creation...")
try:
    # Use smaller settings for testing
    config.n_depth = 2
    config.n_filter_base = 16
    config.max_sampling_points = 32
    config.num_shape_prototypes = 4
    
    model = AdaptiveShapeStarDist(config, name='test_model', basedir='./test_models')
    print("   ✓ Model created successfully")
except Exception as e:
    print(f"   ✗ Model creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Forward pass with dummy data
print("\n4. Testing forward pass...")
try:
    # Create dummy input
    dummy_input = tf.random.normal((2, 64, 64, 1))
    print(f"   - Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input, training=False)
    
    print("   ✓ Forward pass successful")
    print(f"     - Prob shape: {output['prob'].shape}")
    print(f"     - Dist shape: {output['dist'].shape}")
    print(f"     - Complexity shape: {output['complexity'].shape}")
    print(f"     - Sampling points shape: {output['sampling_points'].shape}")
    
except Exception as e:
    print(f"   ✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test loss function
print("\n5. Testing loss function...")
try:
    from adaptive_shape_stardist.training import AdaptiveShapeLoss
    
    loss_fn = AdaptiveShapeLoss()
    
    # Create dummy ground truth
    y_true = {
        'prob': tf.random.uniform((2, 64, 64, 1)),
        'dist': tf.random.uniform((2, 64, 64, 32)),
        'mask': tf.ones((2, 64, 64, 1)),
    }
    
    # Compute loss
    loss = loss_fn(y_true, output)
    
    print("   ✓ Loss computation successful")
    print(f"     - Total loss: {loss.numpy():.4f}")
    
except Exception as e:
    print(f"   ✗ Loss error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test deformable convolution
print("\n6. Testing deformable convolution...")
try:
    deform_conv = DeformableConv2D(filters=32, kernel_size=3)
    
    test_input = tf.random.normal((2, 32, 32, 16))
    output = deform_conv(test_input)
    
    print("   ✓ Deformable convolution successful")
    print(f"     - Output shape: {output.shape}")
    
except Exception as e:
    print(f"   ✗ Deformable convolution error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All basic tests passed!")
print("=" * 60)

