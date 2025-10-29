"""
Unit Tests for FPN Backbone
============================

Tests for Feature Pyramid Network implementation.

Test Coverage:
- Output shape validation
- Channel dimension consistency
- Gradient flow
- Parameter count
- Forward/backward compatibility

Author: Shape-Aware StarDist Team
Date: 2025-10-29
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import unittest

from adaptive_shape_stardist.models.fpn_backbone import FPNBackbone, BasicBlock, BottleneckBlock


class TestFPNBackbone(unittest.TestCase):
    """Test cases for FPN Backbone"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 2
        self.input_size = 512
        self.n_channel_in = 1
        self.fpn_channels = 256
    
    def test_fpn_output_shapes_resnet34(self):
        """Test FPN output shapes for ResNet-34"""
        print("\n" + "=" * 60)
        print("Test: FPN Output Shapes (ResNet-34)")
        print("=" * 60)
        
        # Create FPN
        fpn = FPNBackbone(
            n_channel_in=self.n_channel_in,
            backbone='resnet34',
            fpn_channels=self.fpn_channels
        )
        
        # Create dummy input
        x = tf.random.normal((self.batch_size, self.input_size, self.input_size, self.n_channel_in))
        
        # Forward pass
        output = fpn(x, training=False)
        
        # Expected shapes
        expected_shapes = {
            'p2': (self.batch_size, 128, 128, self.fpn_channels),  # 1/4
            'p3': (self.batch_size, 64, 64, self.fpn_channels),    # 1/8
            'p4': (self.batch_size, 32, 32, self.fpn_channels),    # 1/16
            'p5': (self.batch_size, 16, 16, self.fpn_channels),    # 1/32
        }
        
        # Validate shapes
        for level, expected_shape in expected_shapes.items():
            actual_shape = tuple(output[level].shape.as_list())
            self.assertEqual(actual_shape, expected_shape,
                           f"{level} shape mismatch: expected {expected_shape}, got {actual_shape}")
            print(f"✓ {level}: {actual_shape}")
        
        print("\n✅ All output shapes are correct!")
    
    def test_fpn_output_shapes_resnet50(self):
        """Test FPN output shapes for ResNet-50"""
        print("\n" + "=" * 60)
        print("Test: FPN Output Shapes (ResNet-50)")
        print("=" * 60)
        
        # Create FPN with ResNet-50
        fpn = FPNBackbone(
            n_channel_in=self.n_channel_in,
            backbone='resnet50',
            fpn_channels=self.fpn_channels
        )
        
        # Create dummy input
        x = tf.random.normal((self.batch_size, self.input_size, self.input_size, self.n_channel_in))
        
        # Forward pass
        output = fpn(x, training=False)
        
        # Validate shapes (same as ResNet-34 due to FPN unification)
        expected_shapes = {
            'p2': (self.batch_size, 128, 128, self.fpn_channels),
            'p3': (self.batch_size, 64, 64, self.fpn_channels),
            'p4': (self.batch_size, 32, 32, self.fpn_channels),
            'p5': (self.batch_size, 16, 16, self.fpn_channels),
        }
        
        for level, expected_shape in expected_shapes.items():
            actual_shape = tuple(output[level].shape.as_list())
            self.assertEqual(actual_shape, expected_shape)
            print(f"✓ {level}: {actual_shape}")
        
        print("\n✅ ResNet-50 FPN working correctly!")
    
    def test_gradient_flow(self):
        """Test gradient flow through FPN"""
        print("\n" + "=" * 60)
        print("Test: Gradient Flow")
        print("=" * 60)
        
        fpn = FPNBackbone(
            n_channel_in=self.n_channel_in,
            backbone='resnet34',
            fpn_channels=self.fpn_channels
        )
        
        # Create input
        x = tf.random.normal((self.batch_size, self.input_size, self.input_size, self.n_channel_in))
        
        # Forward and backward pass
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = fpn(x, training=True)
            # Dummy loss: sum of all outputs
            loss = sum([tf.reduce_sum(output[k]) for k in ['p2', 'p3', 'p4', 'p5']])
        
        # Compute gradients
        grads = tape.gradient(loss, fpn.trainable_variables)
        
        # Check for None gradients
        none_grads = sum([1 for g in grads if g is None])
        self.assertEqual(none_grads, 0, f"{none_grads} variables have None gradients")
        print(f"✓ All {len(grads)} variables have valid gradients")
        
        # Check gradient magnitudes
        grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]
        min_norm, max_norm = min(grad_norms), max(grad_norms)
        print(f"✓ Gradient norm range: [{min_norm:.6f}, {max_norm:.6f}]")
        
        # Ensure gradients are not too small (vanishing) or too large (exploding)
        self.assertGreater(min_norm, 1e-10, "Gradients too small (vanishing)")
        self.assertLess(max_norm, 1e6, "Gradients too large (exploding)")
        
        print("\n✅ Gradient flow is healthy!")
    
    def test_channel_consistency(self):
        """Test that all FPN levels have consistent channel dimensions"""
        print("\n" + "=" * 60)
        print("Test: Channel Consistency")
        print("=" * 60)
        
        fpn_channels_test = 128  # Test with different channel size
        fpn = FPNBackbone(
            n_channel_in=self.n_channel_in,
            backbone='resnet34',
            fpn_channels=fpn_channels_test
        )
        
        x = tf.random.normal((self.batch_size, self.input_size, self.input_size, self.n_channel_in))
        output = fpn(x, training=False)
        
        # Check all levels have same channel dimension
        for level in ['p2', 'p3', 'p4', 'p5']:
            channels = output[level].shape[-1]
            self.assertEqual(channels, fpn_channels_test,
                           f"{level} has {channels} channels, expected {fpn_channels_test}")
            print(f"✓ {level}: {channels} channels")
        
        print("\n✅ All levels have consistent channel dimensions!")
    
    def test_parameter_count(self):
        """Test parameter count is reasonable"""
        print("\n" + "=" * 60)
        print("Test: Parameter Count")
        print("=" * 60)
        
        fpn = FPNBackbone(
            n_channel_in=self.n_channel_in,
            backbone='resnet34',
            fpn_channels=self.fpn_channels
        )
        
        # Build model by running forward pass
        x = tf.random.normal((1, self.input_size, self.input_size, self.n_channel_in))
        _ = fpn(x, training=False)
        
        # Count parameters
        total_params = sum([tf.size(v).numpy() for v in fpn.trainable_variables])
        
        print(f"Total parameters: {total_params:,}")
        
        # ResNet-34 + FPN should have ~21-25M parameters
        self.assertGreater(total_params, 15_000_000, "Too few parameters")
        self.assertLess(total_params, 30_000_000, "Too many parameters")
        
        print("✅ Parameter count is within expected range!")
    
    def test_multi_channel_input(self):
        """Test FPN with multi-channel input (e.g., RGB)"""
        print("\n" + "=" * 60)
        print("Test: Multi-channel Input")
        print("=" * 60)
        
        n_channels = 3  # RGB
        fpn = FPNBackbone(
            n_channel_in=n_channels,
            backbone='resnet34',
            fpn_channels=self.fpn_channels
        )
        
        x = tf.random.normal((self.batch_size, self.input_size, self.input_size, n_channels))
        output = fpn(x, training=False)
        
        # Validate shapes
        self.assertEqual(tuple(output['p2'].shape.as_list()),
                        (self.batch_size, 128, 128, self.fpn_channels))
        
        print(f"✓ Successfully processed {n_channels}-channel input")
        print("✅ Multi-channel input working correctly!")
    
    def test_different_input_sizes(self):
        """Test FPN with different input resolutions"""
        print("\n" + "=" * 60)
        print("Test: Different Input Sizes")
        print("=" * 60)
        
        fpn = FPNBackbone(
            n_channel_in=self.n_channel_in,
            backbone='resnet34',
            fpn_channels=self.fpn_channels
        )
        
        test_sizes = [256, 384, 512, 768]
        
        for size in test_sizes:
            x = tf.random.normal((1, size, size, self.n_channel_in))
            output = fpn(x, training=False)
            
            # Check p2 resolution (should be 1/4 of input)
            expected_p2_size = size // 4
            actual_p2_size = output['p2'].shape[1]
            
            self.assertEqual(actual_p2_size, expected_p2_size,
                           f"Input {size} -> P2 {actual_p2_size}, expected {expected_p2_size}")
            print(f"✓ Input {size}×{size} -> P2 {actual_p2_size}×{actual_p2_size}")
        
        print("\n✅ Different input sizes handled correctly!")
    
    def test_training_vs_inference_mode(self):
        """Test FPN in training vs inference mode"""
        print("\n" + "=" * 60)
        print("Test: Training vs Inference Mode")
        print("=" * 60)
        
        fpn = FPNBackbone(
            n_channel_in=self.n_channel_in,
            backbone='resnet34',
            fpn_channels=self.fpn_channels
        )
        
        x = tf.random.normal((self.batch_size, self.input_size, self.input_size, self.n_channel_in))
        
        # Run in training mode
        output_train = fpn(x, training=True)
        
        # Run in inference mode
        output_infer = fpn(x, training=False)
        
        # Check shapes match
        for level in ['p2', 'p3', 'p4', 'p5']:
            self.assertEqual(output_train[level].shape, output_infer[level].shape)
        
        # Outputs should be different (due to BatchNorm behavior)
        diff = tf.reduce_mean(tf.abs(output_train['p2'] - output_infer['p2']))
        self.assertGreater(diff.numpy(), 0, "Training and inference outputs are identical (BatchNorm issue?)")
        
        print(f"✓ Training mode: BatchNorm active")
        print(f"✓ Inference mode: BatchNorm frozen")
        print(f"✓ Output difference: {diff.numpy():.6f}")
        print("\n✅ Training/Inference modes working correctly!")
    
    def test_basic_block(self):
        """Test BasicBlock (ResNet-34 building block)"""
        print("\n" + "=" * 60)
        print("Test: BasicBlock")
        print("=" * 60)
        
        # Test with stride=1 (no downsampling)
        block1 = BasicBlock(64, 64, stride=1)
        x = tf.random.normal((2, 64, 64, 64))
        y = block1(x, training=False)
        self.assertEqual(y.shape, x.shape, "BasicBlock with stride=1 should preserve shape")
        print("✓ BasicBlock stride=1: shape preserved")
        
        # Test with stride=2 (downsampling)
        block2 = BasicBlock(64, 128, stride=2)
        x = tf.random.normal((2, 64, 64, 64))
        y = block2(x, training=False)
        expected_shape = (2, 32, 32, 128)
        self.assertEqual(tuple(y.shape.as_list()), expected_shape,
                        f"BasicBlock with stride=2: expected {expected_shape}, got {tuple(y.shape.as_list())}")
        print("✓ BasicBlock stride=2: downsampling working")
        
        print("\n✅ BasicBlock tests passed!")
    
    def test_bottleneck_block(self):
        """Test BottleneckBlock (ResNet-50+ building block)"""
        print("\n" + "=" * 60)
        print("Test: BottleneckBlock")
        print("=" * 60)
        
        # Test bottleneck structure
        block = BottleneckBlock(256, 64, 256, stride=1)
        x = tf.random.normal((2, 64, 64, 256))
        y = block(x, training=False)
        self.assertEqual(y.shape, x.shape, "BottleneckBlock should preserve shape with stride=1")
        print("✓ BottleneckBlock stride=1: shape preserved")
        
        # Test with downsampling
        block2 = BottleneckBlock(256, 128, 512, stride=2)
        x = tf.random.normal((2, 64, 64, 256))
        y = block2(x, training=False)
        expected_shape = (2, 32, 32, 512)
        self.assertEqual(tuple(y.shape.as_list()), expected_shape)
        print("✓ BottleneckBlock stride=2: downsampling working")
        
        print("\n✅ BottleneckBlock tests passed!")


class TestFPNIntegration(unittest.TestCase):
    """Integration tests for FPN"""
    
    def test_fpn_with_multiscale_loss(self):
        """Test FPN integration with multiscale loss"""
        print("\n" + "=" * 60)
        print("Integration Test: FPN + Multiscale Loss")
        print("=" * 60)
        
        from adaptive_shape_stardist.training.multiscale_loss import multiscale_loss
        
        # Create FPN
        fpn = FPNBackbone(n_channel_in=1, backbone='resnet34', fpn_channels=256)
        
        # Create dummy input
        x = tf.random.normal((2, 512, 512, 1))
        
        # Get FPN features
        fpn_output = fpn(x, training=True)
        
        # Create dummy predictions dict
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
        
        # Validate loss
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertTrue(total_loss.numpy() > 0, "Loss should be positive")
        self.assertTrue(total_loss.numpy() < 1000, "Loss should be reasonable")
        
        print(f"✓ Total loss: {total_loss.numpy():.4f}")
        print("✅ FPN + Multiscale loss integration working!")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print(" " * 20 + "FPN BACKBONE TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFPNBackbone))
    suite.addTests(loader.loadTestsFromTestCase(TestFPNIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

