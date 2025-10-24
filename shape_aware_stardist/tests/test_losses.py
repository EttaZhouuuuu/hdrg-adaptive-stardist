import unittest
import torch
import numpy as np
from ..training.losses import StarDistLoss

class TestStarDistLoss(unittest.TestCase):
    """Test cases for StarDist loss functions"""
    
    def setUp(self):
        """Set up test cases"""
        self.batch_size = 2
        self.n_rays = 32
        self.height = 64
        self.width = 64
        
        # Create loss function
        self.loss_fn = StarDistLoss(
            dist_loss_weight=1.0,
            prob_loss_weight=1.0,
            shape_loss_weight=0.5,
            reg_loss_weight=0.1,
            focal_gamma=2.0,
            adaptive_weight=True
        )
        
        # Create dummy predictions and targets
        self.pred_distances = torch.randn(
            self.batch_size, self.n_rays, self.height, self.width
        )
        self.pred_probabilities = torch.sigmoid(torch.randn(
            self.batch_size, 1, self.height, self.width
        ))
        
        # Create ground truth
        self.true_distances = torch.abs(torch.randn(
            self.batch_size, self.n_rays, self.height, self.width
        ))
        self.true_probabilities = torch.randint(
            0, 2, (self.batch_size, self.height, self.width)
        ).float()
        
        # Create attention maps
        self.attention_maps = torch.randn(
            self.batch_size, self.n_rays, self.height, self.width
        )
    
    def test_loss_computation(self):
        """Test basic loss computation"""
        # Compute loss
        total_loss, components = self.loss_fn(
            self.pred_distances,
            self.pred_probabilities,
            self.true_distances,
            self.true_probabilities,
            self.attention_maps
        )
        
        # Check loss values
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0)
        
        # Check components
        expected_components = {
            'distance_loss', 'probability_loss', 'shape_loss',
            'regularization_loss', 'dist_weight', 'prob_weight',
            'shape_weight', 'reg_weight'
        }
        self.assertEqual(set(components.keys()), expected_components)
        
        # Check component values
        for name, value in components.items():
            self.assertTrue(torch.isfinite(value))
            if 'loss' in name:
                self.assertGreaterEqual(value.item(), 0)
    
    def test_perfect_prediction(self):
        """Test loss computation with perfect predictions"""
        # Set predictions equal to targets
        total_loss, components = self.loss_fn(
            self.true_distances,
            self.true_probabilities.unsqueeze(1),
            self.true_distances,
            self.true_probabilities,
            self.attention_maps
        )
        
        # Distance and probability losses should be close to zero
        self.assertLess(components['distance_loss'].item(), 1e-5)
        self.assertLess(components['probability_loss'].item(), 1e-5)
    
    def test_adaptive_weighting(self):
        """Test adaptive loss weighting"""
        # Compute loss multiple times to update running averages
        for _ in range(10):
            total_loss, components = self.loss_fn(
                self.pred_distances,
                self.pred_probabilities,
                self.true_distances,
                self.true_probabilities,
                self.attention_maps
            )
        
        # Check that weights sum to approximately original total weight
        weights = [
            components['dist_weight'],
            components['prob_weight'],
            components['shape_weight'],
            components['reg_weight']
        ]
        total_weight = sum(weights)
        original_total = (
            self.loss_fn.dist_loss_weight +
            self.loss_fn.prob_loss_weight +
            self.loss_fn.shape_loss_weight +
            self.loss_fn.reg_loss_weight
        )
        self.assertAlmostEqual(total_weight, original_total, places=5)
    
    def test_shape_consistency(self):
        """Test shape consistency loss"""
        # Create predictions with inconsistent shapes
        inconsistent_pred = self.pred_distances.clone()
        inconsistent_pred[:, 1:] = inconsistent_pred[:, :-1] + torch.randn_like(
            inconsistent_pred[:, :-1]
        )
        
        # Compare losses
        _, consistent_components = self.loss_fn(
            self.pred_distances,
            self.pred_probabilities,
            self.true_distances,
            self.true_probabilities,
            self.attention_maps
        )
        
        _, inconsistent_components = self.loss_fn(
            inconsistent_pred,
            self.pred_probabilities,
            self.true_distances,
            self.true_probabilities,
            self.attention_maps
        )
        
        # Shape loss should be higher for inconsistent predictions
        self.assertGreater(
            inconsistent_components['shape_loss'].item(),
            consistent_components['shape_loss'].item()
        )
    
    def test_regularization(self):
        """Test shape regularization loss"""
        # Create predictions with high frequency noise
        noisy_pred = self.pred_distances.clone()
        noise = torch.randn_like(noisy_pred) * 0.1
        noisy_pred = noisy_pred + noise
        
        # Compare losses
        _, smooth_components = self.loss_fn(
            self.pred_distances,
            self.pred_probabilities,
            self.true_distances,
            self.true_probabilities,
            self.attention_maps
        )
        
        _, noisy_components = self.loss_fn(
            noisy_pred,
            self.pred_probabilities,
            self.true_distances,
            self.true_probabilities,
            self.attention_maps
        )
        
        # Regularization loss should be higher for noisy predictions
        self.assertGreater(
            noisy_components['regularization_loss'].item(),
            smooth_components['regularization_loss'].item()
        )

if __name__ == '__main__':
    unittest.main()
