import unittest
import torch
import numpy as np
from ..evaluation.metrics import SegmentationMetrics, MetricsLogger

class TestSegmentationMetrics(unittest.TestCase):
    """Test cases for segmentation metrics"""
    
    def setUp(self):
        """Set up test cases"""
        self.metrics = SegmentationMetrics(n_rays=8)
        
        # Create simple test data
        self.size = 32
        self.n_rays = 8
        
        # Create binary masks
        self.pred_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
        self.true_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
        
        # Create circle in both masks with slight difference
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        center = self.size // 2
        radius = self.size // 4
        
        self.true_mask[(y - center)**2 + (x - center)**2 <= radius**2] = True
        self.pred_mask[(y - center)**2 + (x - center)**2 <= (radius + 1)**2] = True
        
        # Create distance maps
        self.true_distances = torch.zeros((self.n_rays, self.size, self.size))
        self.pred_distances = torch.zeros((self.n_rays, self.size, self.size))
        
        # Fill with simple radial distances
        for i in range(self.n_rays):
            angle = 2 * np.pi * i / self.n_rays
            self.true_distances[i] = radius
            self.pred_distances[i] = radius + 1
    
    def test_iou_computation(self):
        """Test IoU computation"""
        iou = self.metrics.compute_iou(self.pred_mask, self.true_mask)
        
        self.assertIsInstance(iou, torch.Tensor)
        self.assertTrue(0 <= iou <= 1)
        
        # Perfect overlap should give IoU of 1
        perfect_iou = self.metrics.compute_iou(self.true_mask, self.true_mask)
        self.assertAlmostEqual(perfect_iou.item(), 1.0)
        
        # No overlap should give IoU of 0
        no_overlap_mask = torch.zeros_like(self.true_mask)
        no_overlap_mask[:, :self.size//2] = True
        zero_iou = self.metrics.compute_iou(no_overlap_mask, self.true_mask)
        self.assertAlmostEqual(zero_iou.item(), 0.0)
    
    def test_dice_computation(self):
        """Test Dice coefficient computation"""
        dice = self.metrics.compute_dice(self.pred_mask, self.true_mask)
        
        self.assertIsInstance(dice, torch.Tensor)
        self.assertTrue(0 <= dice <= 1)
        
        # Perfect overlap should give Dice of 1
        perfect_dice = self.metrics.compute_dice(self.true_mask, self.true_mask)
        self.assertAlmostEqual(perfect_dice.item(), 1.0)
        
        # No overlap should give Dice of 0
        no_overlap_mask = torch.zeros_like(self.true_mask)
        no_overlap_mask[:, :self.size//2] = True
        zero_dice = self.metrics.compute_dice(no_overlap_mask, self.true_mask)
        self.assertAlmostEqual(zero_dice.item(), 0.0)
    
    def test_shape_metrics(self):
        """Test shape metrics computation"""
        metrics = self.metrics.compute_shape_metrics(
            self.pred_distances,
            self.true_distances,
            self.true_mask
        )
        
        self.assertIn('relative_distance_error', metrics)
        self.assertIn('angular_error', metrics)
        
        # Check value ranges
        self.assertTrue(0 <= metrics['relative_distance_error'])
        self.assertTrue(0 <= metrics['angular_error'] <= np.pi)
        
        # Perfect prediction should give zero error
        perfect_metrics = self.metrics.compute_shape_metrics(
            self.true_distances,
            self.true_distances,
            self.true_mask
        )
        self.assertAlmostEqual(perfect_metrics['relative_distance_error'], 0.0)
        self.assertAlmostEqual(perfect_metrics['angular_error'], 0.0)
    
    def test_instance_metrics(self):
        """Test instance metrics computation"""
        # Create instance segmentation
        pred_instances = torch.zeros((self.size, self.size), dtype=torch.long)
        true_instances = torch.zeros((self.size, self.size), dtype=torch.long)
        
        # Add two instances
        true_instances[10:20, 10:20] = 1
        true_instances[20:30, 20:30] = 2
        
        pred_instances[11:21, 11:21] = 1
        pred_instances[21:31, 21:31] = 2
        
        metrics = self.metrics.compute_instance_metrics(
            pred_instances,
            true_instances
        )
        
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('mean_iou', metrics)
        
        # Check value ranges
        for value in metrics.values():
            self.assertTrue(0 <= value <= 1)
        
        # Perfect prediction should give perfect scores
        perfect_metrics = self.metrics.compute_instance_metrics(
            true_instances,
            true_instances
        )
        for value in perfect_metrics.values():
            self.assertAlmostEqual(value, 1.0)
    
    def test_compute_all_metrics(self):
        """Test computation of all metrics"""
        metrics = self.metrics.compute_all_metrics(
            self.pred_distances,
            self.pred_mask.float(),
            self.true_distances,
            self.true_mask.float()
        )
        
        expected_metrics = {
            'iou', 'dice', 'relative_distance_error', 'angular_error'
        }
        self.assertTrue(expected_metrics.issubset(set(metrics.keys())))
        
        # Check value ranges
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))

class TestMetricsLogger(unittest.TestCase):
    """Test cases for metrics logger"""
    
    def setUp(self):
        """Set up test cases"""
        self.logger = MetricsLogger()
        
        # Sample metrics
        self.metrics1 = {'loss': 1.0, 'accuracy': 0.8}
        self.metrics2 = {'loss': 2.0, 'accuracy': 0.6}
    
    def test_update_and_average(self):
        """Test metrics updating and averaging"""
        # Update with first batch
        self.logger.update(self.metrics1)
        avg_metrics = self.logger.get_average_metrics()
        
        self.assertEqual(avg_metrics['loss'], 1.0)
        self.assertEqual(avg_metrics['accuracy'], 0.8)
        
        # Update with second batch
        self.logger.update(self.metrics2)
        avg_metrics = self.logger.get_average_metrics()
        
        self.assertEqual(avg_metrics['loss'], 1.5)
        self.assertEqual(avg_metrics['accuracy'], 0.7)
    
    def test_reset(self):
        """Test metrics reset"""
        self.logger.update(self.metrics1)
        self.logger.reset()
        
        self.assertEqual(len(self.logger.metrics), 0)
        self.assertEqual(len(self.logger.counts), 0)

if __name__ == '__main__':
    unittest.main()
