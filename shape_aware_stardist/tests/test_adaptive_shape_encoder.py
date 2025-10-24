import unittest
import torch
from ..models.adaptive_shape_encoder import (
    AdaptiveShapeEncoder,
    DeformableConv2d,
    DeformableEncoderBlock,
    ShapePriorEncoder,
    AdaptiveSampler,
    RayAttention,
    CrossAttention
)

class TestAdaptiveShapeEncoder(unittest.TestCase):
    """Test cases for adaptive shape encoder"""
    
    def setUp(self):
        """Set up test cases"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = 3
        self.n_rays = 32
        self.height = 64
        self.width = 64
        
        # Create encoder
        self.encoder = AdaptiveShapeEncoder(
            in_channels=self.in_channels,
            n_rays=self.n_rays,
            base_channels=64,
            n_blocks=4,
            use_shape_prior=True
        ).to(self.device)
        
        # Create test input
        self.test_input = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width
        ).to(self.device)
        
        self.shape_prior = torch.randn(
            self.batch_size, self.n_rays, self.height // 8, self.width // 8
        ).to(self.device)
    
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        # Without shape prior
        features, points, intermediates = self.encoder(self.test_input)
        
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(points, torch.Tensor)
        self.assertIsInstance(intermediates, list)
        
        # With shape prior
        features, points, intermediates = self.encoder(
            self.test_input,
            self.shape_prior
        )
        
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(points, torch.Tensor)
        self.assertIsInstance(intermediates, list)
        
        # Check output shapes
        expected_feature_size = self.height // 16  # After 4 encoder blocks
        self.assertEqual(
            features.shape,
            (self.batch_size, 512, expected_feature_size, expected_feature_size)
        )
        
        self.assertEqual(
            points.shape,
            (self.batch_size, self.n_rays, 2, expected_feature_size, expected_feature_size)
        )
    
    def test_deformable_conv(self):
        """Test deformable convolution"""
        conv = DeformableConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        ).to(self.device)
        
        x = torch.randn(self.batch_size, 64, 32, 32).to(self.device)
        output = conv(x)
        
        self.assertEqual(output.shape, (self.batch_size, 128, 32, 32))
    
    def test_encoder_block(self):
        """Test encoder block"""
        block = DeformableEncoderBlock(
            in_channels=64,
            out_channels=128,
            n_rays=self.n_rays
        ).to(self.device)
        
        x = torch.randn(self.batch_size, 64, 32, 32).to(self.device)
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, 128, 16, 16))
    
    def test_shape_prior_encoder(self):
        """Test shape prior encoder"""
        encoder = ShapePriorEncoder(
            in_channels=512,
            n_rays=self.n_rays
        ).to(self.device)
        
        features = torch.randn(self.batch_size, 512, 8, 8).to(self.device)
        prior = torch.randn(self.batch_size, self.n_rays, 8, 8).to(self.device)
        
        output = encoder(features, prior)
        
        self.assertEqual(output.shape, features.shape)
    
    def test_adaptive_sampler(self):
        """Test adaptive sampler"""
        sampler = AdaptiveSampler(
            in_channels=512,
            n_rays=self.n_rays
        ).to(self.device)
        
        features = torch.randn(self.batch_size, 512, 8, 8).to(self.device)
        features, points = sampler(features)
        
        self.assertEqual(features.shape, (self.batch_size, 512, 8, 8))
        self.assertEqual(points.shape, (self.batch_size, self.n_rays, 2, 8, 8))
    
    def test_ray_attention(self):
        """Test ray attention"""
        attention = RayAttention(
            channels=128,
            n_rays=self.n_rays
        ).to(self.device)
        
        x = torch.randn(self.batch_size, 128, 16, 16).to(self.device)
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_cross_attention(self):
        """Test cross attention"""
        attention = CrossAttention(channels=512).to(self.device)
        
        features = torch.randn(self.batch_size, 512, 8, 8).to(self.device)
        prior = torch.randn(self.batch_size, 512, 8, 8).to(self.device)
        
        output = attention(features, prior)
        
        self.assertEqual(output.shape, features.shape)
    
    def test_gradient_flow(self):
        """Test gradient flow through the entire model"""
        # Forward pass
        features, points, _ = self.encoder(self.test_input, self.shape_prior)
        
        # Compute loss
        loss = features.mean() + points.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in self.encoder.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
    
    def test_shape_adaptation(self):
        """Test shape adaptation with different inputs"""
        # Create inputs with different shapes
        input1 = torch.randn(
            self.batch_size, self.in_channels, 32, 32
        ).to(self.device)
        
        input2 = torch.randn(
            self.batch_size, self.in_channels, 128, 128
        ).to(self.device)
        
        # Test with different input sizes
        features1, points1, _ = self.encoder(input1)
        features2, points2, _ = self.encoder(input2)
        
        # Check output shapes are proportional to input
        self.assertEqual(features1.shape[-2:], (2, 2))  # 32 -> 2
        self.assertEqual(features2.shape[-2:], (8, 8))  # 128 -> 8
        
        self.assertEqual(points1.shape[-2:], (2, 2))
        self.assertEqual(points2.shape[-2:], (8, 8))

if __name__ == '__main__':
    unittest.main()
