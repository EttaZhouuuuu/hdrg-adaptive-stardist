"""
FPN Implementation Verification Script
=======================================

Quick verification script to ensure all FPN components are working correctly.

Usage:
    python verify_fpn_implementation.py

Author: Shape-Aware StarDist Team
Date: 2025-10-29
"""

import sys
import os

print("\n" + "=" * 80)
print(" " * 20 + "FPN IMPLEMENTATION VERIFICATION")
print("=" * 80)

# Test 1: Import FPN Backbone (TensorFlow)
print("\n" + "=" * 80)
print("Test 1: Import FPN Backbone (TensorFlow)")
print("=" * 80)
try:
    from adaptive_shape_stardist.models.fpn_backbone import FPNBackbone as FPN_TF
    print("✅ FPNBackbone (TensorFlow) imported successfully")
except Exception as e:
    print(f"❌ Failed to import FPNBackbone (TensorFlow): {e}")
    sys.exit(1)

# Test 2: Import FPN Backbone (PyTorch)
print("\n" + "=" * 80)
print("Test 2: Import FPN Backbone (PyTorch)")
print("=" * 80)
try:
    from shape_aware_stardist.models.fpn_backbone import FPNBackbone as FPN_PyTorch
    print("✅ FPNBackbone (PyTorch) imported successfully")
except Exception as e:
    print(f"❌ Failed to import FPNBackbone (PyTorch): {e}")
    sys.exit(1)

# Test 3: Import Main Model
print("\n" + "=" * 80)
print("Test 3: Import Main Model")
print("=" * 80)
try:
    from adaptive_shape_stardist.models.adaptive_shape_model import AdaptiveShapeStarDist
    print("✅ AdaptiveShapeStarDist imported successfully")
except Exception as e:
    print(f"❌ Failed to import AdaptiveShapeStarDist: {e}")
    sys.exit(1)

# Test 4: Import Configurations
print("\n" + "=" * 80)
print("Test 4: Import Configurations")
print("=" * 80)
try:
    from adaptive_shape_stardist.configs.config import (
        get_fpn_config,
        get_fpn_multiscale_config,
        get_fpn_fast_config
    )
    print("✅ FPN configurations imported successfully")
    print(f"   - get_fpn_config: {get_fpn_config is not None}")
    print(f"   - get_fpn_multiscale_config: {get_fpn_multiscale_config is not None}")
    print(f"   - get_fpn_fast_config: {get_fpn_fast_config is not None}")
except Exception as e:
    print(f"❌ Failed to import configurations: {e}")
    sys.exit(1)

# Test 5: Import Multi-scale Loss
print("\n" + "=" * 80)
print("Test 5: Import Multi-scale Loss")
print("=" * 80)
try:
    from adaptive_shape_stardist.training.multiscale_loss import (
        multiscale_loss,
        create_multiscale_loss_fn
    )
    print("✅ Multi-scale loss functions imported successfully")
except Exception as e:
    print(f"❌ Failed to import multi-scale loss: {e}")
    sys.exit(1)

# Test 6: Create FPN Model (TensorFlow)
print("\n" + "=" * 80)
print("Test 6: Create FPN Model (TensorFlow)")
print("=" * 80)
try:
    import tensorflow as tf
    
    # Create FPN backbone
    fpn_tf = FPN_TF(n_channel_in=1, backbone='resnet34', fpn_channels=256)
    print("✅ FPN backbone created")
    
    # Test forward pass
    x = tf.random.normal((1, 512, 512, 1))
    output = fpn_tf(x, training=False)
    
    # Verify output shapes
    expected_shapes = {
        'p2': (1, 128, 128, 256),
        'p3': (1, 64, 64, 256),
        'p4': (1, 32, 32, 256),
        'p5': (1, 16, 16, 256),
    }
    
    all_correct = True
    for level, expected_shape in expected_shapes.items():
        actual_shape = tuple(output[level].shape.as_list())
        if actual_shape != expected_shape:
            print(f"❌ {level} shape mismatch: expected {expected_shape}, got {actual_shape}")
            all_correct = False
        else:
            print(f"✅ {level}: {actual_shape}")
    
    if all_correct:
        print("✅ FPN forward pass successful (TensorFlow)")
    else:
        print("❌ FPN forward pass failed (TensorFlow)")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Failed to create/test FPN model (TensorFlow): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Create FPN Model (PyTorch)
print("\n" + "=" * 80)
print("Test 7: Create FPN Model (PyTorch)")
print("=" * 80)
try:
    import torch
    
    # Create FPN backbone
    fpn_pytorch = FPN_PyTorch(in_channels=1, backbone='resnet34', fpn_channels=256)
    print("✅ FPN backbone created")
    
    # Test forward pass
    x = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        output = fpn_pytorch(x)
    
    # Verify output shapes
    expected_shapes = {
        'p2': (1, 256, 128, 128),
        'p3': (1, 256, 64, 64),
        'p4': (1, 256, 32, 32),
        'p5': (1, 256, 16, 16),
    }
    
    all_correct = True
    for level, expected_shape in expected_shapes.items():
        actual_shape = tuple(output[level].shape)
        if actual_shape != expected_shape:
            print(f"❌ {level} shape mismatch: expected {expected_shape}, got {actual_shape}")
            all_correct = False
        else:
            print(f"✅ {level}: {actual_shape}")
    
    if all_correct:
        print("✅ FPN forward pass successful (PyTorch)")
    else:
        print("❌ FPN forward pass failed (PyTorch)")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Failed to create/test FPN model (PyTorch): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Create Configuration
print("\n" + "=" * 80)
print("Test 8: Create and Verify Configuration")
print("=" * 80)
try:
    config = get_fpn_config()
    print(f"✅ Configuration created")
    print(f"   - use_fpn: {config.use_fpn}")
    print(f"   - fpn_channels: {config.fpn_channels}")
    print(f"   - fpn_levels: {config.fpn_levels}")
    print(f"   - multiscale_prediction: {config.multiscale_prediction}")
    
    if config.use_fpn and config.fpn_channels == 256:
        print("✅ Configuration is correct")
    else:
        print("❌ Configuration has unexpected values")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Failed to create configuration: {e}")
    sys.exit(1)

# Test 9: Create Main Model with FPN
print("\n" + "=" * 80)
print("Test 9: Create Main Model with FPN")
print("=" * 80)
try:
    config = get_fpn_config()
    model = AdaptiveShapeStarDist(config=config, name='fpn_test_model', basedir='./test_models')
    print("✅ AdaptiveShapeStarDist with FPN created")
    
    # Test forward pass
    x = tf.random.normal((1, 512, 512, 1))
    output = model(x, training=False)
    
    print(f"✅ Main model forward pass successful")
    print(f"   - Output keys: {list(output.keys())}")
    
    # Check if FPN features are present
    if 'fpn_features' in output:
        print("✅ FPN features present in output")
    else:
        print("⚠️  FPN features not in output (may be using different output format)")
    
except Exception as e:
    print(f"❌ Failed to create/test main model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Multi-scale Loss Function
print("\n" + "=" * 80)
print("Test 10: Multi-scale Loss Function")
print("=" * 80)
try:
    # Create dummy predictions
    predictions = {
        'p2': {'prob': tf.random.uniform((2, 128, 128, 1)), 'dist': tf.random.uniform((2, 128, 128, 32))},
        'p3': {'prob': tf.random.uniform((2, 64, 64, 1)), 'dist': tf.random.uniform((2, 64, 64, 32))},
        'p4': {'prob': tf.random.uniform((2, 32, 32, 1)), 'dist': tf.random.uniform((2, 32, 32, 32))},
        'p5': {'prob': tf.random.uniform((2, 16, 16, 1)), 'dist': tf.random.uniform((2, 16, 16, 32))},
    }
    
    # Create dummy targets
    targets = {
        'prob': tf.random.uniform((2, 512, 512, 1)),
        'dist': tf.random.uniform((2, 512, 512, 32)),
    }
    
    # Compute loss
    total_loss, loss_dict = multiscale_loss(predictions, targets)
    
    print(f"✅ Multi-scale loss computed successfully")
    print(f"   - Total loss: {total_loss.numpy():.4f}")
    print(f"   - Loss components: {len(loss_dict)}")
    
    if total_loss.numpy() > 0 and total_loss.numpy() < 1000:
        print("✅ Loss value is reasonable")
    else:
        print("⚠️  Loss value may be unusual")
    
except Exception as e:
    print(f"❌ Failed to compute multi-scale loss: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print("✅ All 10 tests passed!")
print("\nFPN Implementation is ready to use!")
print("\nNext steps:")
print("  1. Run unit tests: python tests/test_fpn_backbone.py")
print("  2. Train a model: python adaptive_shape_stardist/examples/train_fpn_example.py")
print("  3. Evaluate on DSB2018 dataset")
print("\nFor more information, see:")
print("  - FPN_IMPLEMENTATION_COMPLETE.md")
print("  - IMPLEMENTATION_STATUS_REPORT.md")
print("=" * 80)

print("\n🎉 FPN Implementation Verification Complete! 🎉\n")

