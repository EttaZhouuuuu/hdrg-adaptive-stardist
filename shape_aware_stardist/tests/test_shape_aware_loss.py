import unittest
import torch
import numpy as np
from ..training.shape_aware_loss import ShapeAwareHybridLoss, ShapeDiscriminator

class TestShapeAwareHybridLoss(unittest.TestCase):
    """Test cases for shape-aware hybrid loss"""
    
    def setUp(self):
        """Set up test cases"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.n_rays = 32
        self.height = 64
        self.width = 64
        
        # Create loss function
        self.loss_fn = ShapeAwareHybridLoss(
            n_rays=self.n_rays,
            consistency_weight=1.0,
            smoothness_weight=0.5,
            adversarial_weight=0.1,
            use_focal_loss=True,
            focal_gamma=2.0,
            use_adaptive_weights=True
        ).to(self.device)
        
        # Create test data
        self.pred_distances = torch.randn(
            self.batch_size, self.n_rays, self.height, self.width
        ).to(self.device)
        
        self.true_distances = torch.abs(torch.randn(
            self.batch_size, self.n_rays, self.height, self.width
        )).to(self.device)
        
        self.pred_probabilities = torch.sigmoid(torch.randn(
            self.batch_size, 1, self.height, self.width
        )).to(self.device)
        
        self.true_probabilities = torch.randint(
            0, 2, (self.batch_size, self.height, self.width)
        ).float().to(self.device)
    
    def test_distance_loss(self):
        """Test distance loss computation"""
        loss = self.loss_fn.compute_distance_loss(
            self.pred_distances,
            self.true_distances,
            self.true_probabilities
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)
    
    def test_probability_loss(self):
        """Test probability loss computation"""
        # Test with focal loss
        loss_focal = self.loss_fn.compute_probability_loss(
            self.pred_probabilities,
            self.true_probabilities
        )
        
        self.assertIsInstance(loss_focal, torch.Tensor)
        self.assertTrue(torch.isfinite(loss_focal))
        self.assertTrue(loss_focal >= 0)
        
        # Test without focal loss
        self.loss_fn.use_focal_loss = False
        loss_bce = self.loss_fn.compute_probability_loss(
            self.pred_probabilities,
            self.true_probabilities
        )
        
        self.assertIsInstance(loss_bce, torch.Tensor)
        self.assertTrue(torch.isfinite(loss_bce))
        self.assertTrue(loss_bce >= 0)
    
    def test_shape_consistency_loss(self):
        """Test shape consistency loss computation"""
        loss = self.loss_fn.compute_shape_consistency_loss(
            self.pred_distances,
            self.true_distances,
            self.true_probabilities
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)
        
        # Test with perfect prediction
        loss_perfect = self.loss_fn.compute_shape_consistency_loss(
            self.true_distances,
            self.true_distances,
            self.true_probabilities
        )
        
        self.assertLess(loss_perfect.item(), loss.item())
    
    def test_boundary_smoothness_loss(self):
        """Test boundary smoothness loss computation"""
        loss = self.loss_fn.compute_boundary_smoothness_loss(
            self.pred_distances,
            self.true_probabilities
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)
        
        # Test with smooth prediction
        smooth_distances = F.avg_pool2d(
            self.pred_distances,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        loss_smooth = self.loss_fn.compute_boundary_smoothness_loss(
            smooth_distances,
            self.true_probabilities
        )
        
        self.assertLess(loss_smooth.item(), loss.item())
    
    def test_adversarial_loss(self):
        """Test adversarial loss computation"""
        # Test generator loss
        g_loss, _ = self.loss_fn.compute_adversarial_loss(
            self.pred_distances,
            self.true_distances,
            is_training_discriminator=False
        )
        
        self.assertIsInstance(g_loss, torch.Tensor)
        self.assertTrue(torch.isfinite(g_loss))
        self.assertTrue(g_loss >= 0)
        
        # Test discriminator loss
        _, d_loss = self.loss_fn.compute_adversarial_loss(
            self.pred_distances,
            self.true_distances,
            is_training_discriminator=True
        )
        
        self.assertIsInstance(d_loss, torch.Tensor)
        self.assertTrue(torch.isfinite(d_loss))
        self.assertTrue(d_loss >= 0)
    
    def test_adaptive_weights(self):
        """Test adaptive weight computation"""
        # Initial weights should be uniform
        weights = self.loss_fn.compute_adaptive_weights()
        expected_weight = 1.0 / 5  # 5 loss components
        
        self.assertEqual(weights.shape, torch.Size([5]))
        self.assertTrue(torch.allclose(weights, torch.tensor([expected_weight] * 5).to(self.device)))
        
        # Update weights
        losses = [torch.tensor(i + 1).to(self.device) for i in range(5)]
        self.loss_fn.update_adaptive_weights(losses)
        
        # Weights should be inversely proportional to losses
        weights = self.loss_fn.compute_adaptive_weights()
        self.assertTrue((weights[:-1] > weights[1:]).all())
    
    def test_forward_pass(self):
        """Test complete forward pass"""
        # Test generator forward pass
        total_loss, components = self.loss_fn(
            self.pred_distances,
            self.pred_probabilities,
            self.true_distances,
            self.true_probabilities,
            is_training_discriminator=False
        )
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(total_loss >= 0)
        
        expected_components = {
            'distance_loss', 'probability_loss', 'consistency_loss',
            'smoothness_loss', 'adversarial_loss', 'distance_weight',
            'probability_weight', 'consistency_weight', 'smoothness_weight',
            'adversarial_weight'
        }
        
        self.assertEqual(set(components.keys()), expected_components)
        
        # Test discriminator forward pass
        total_loss, components = self.loss_fn(
            self.pred_distances,
            self.pred_probabilities,
            self.true_distances,
            self.true_probabilities,
            is_training_discriminator=True
        )
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(total_loss >= 0)


class TestShapeDiscriminator(unittest.TestCase):
    """Test cases for shape discriminator"""
    
    def setUp(self):
        """Set up test cases"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.n_rays = 32
        self.height = 64
        self.width = 64
        
        self.discriminator = ShapeDiscriminator(self.n_rays).to(self.device)
        
        # Create test input
        self.test_input = torch.randn(
            self.batch_size, self.n_rays, self.height, self.width
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test discriminator forward pass"""
        output = self.discriminator(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_feature_extraction(self):
        """Test feature extraction layers"""
        # Get intermediate features
        features = self.discriminator.conv_layers(self.test_input)
        
        expected_size = self.height // 16  # After 4 stride-2 convolutions
        self.assertEqual(
            features.shape,
            (self.batch_size, 512, expected_size, expected_size)
        )
    
    def test_classification(self):
        """Test classification layer"""
        features = self.discriminator.conv_layers(self.test_input)
        output = self.discriminator.classifier(features)
        
        self.assertEqual(output.shape, (self.batch_size, 1, 1, 1))

if __name__ == '__main__':
    unittest.main()
