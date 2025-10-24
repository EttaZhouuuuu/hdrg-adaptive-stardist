import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from ..visualization.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    """Test cases for visualization tools"""
    
    def setUp(self):
        """Set up test cases"""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = Visualizer(save_dir=self.temp_dir)
        
        # Create test data
        self.size = 64
        self.n_rays = 8
        
        # Create image
        self.image = torch.randn(3, self.size, self.size)
        
        # Create masks
        self.pred_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
        self.true_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
        
        # Create circle in masks
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        center = self.size // 2
        radius = self.size // 4
        
        self.pred_mask[(y - center)**2 + (x - center)**2 <= radius**2] = True
        self.true_mask[(y - center)**2 + (x - center)**2 <= (radius - 2)**2] = True
        
        # Create attention maps
        self.attention_maps = torch.randn(4, self.size, self.size)
        
        # Create distances
        self.pred_distances = torch.randn(self.n_rays, self.size, self.size)
        self.true_distances = torch.randn(self.n_rays, self.size, self.size)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_prediction_visualization(self):
        """Test prediction visualization"""
        # Test basic visualization
        vis_image = self.visualizer.visualize_prediction(
            self.image,
            self.pred_mask
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        self.assertEqual(vis_image.ndim, 3)
        self.assertEqual(vis_image.dtype, np.uint8)
        
        # Test with ground truth
        vis_image = self.visualizer.visualize_prediction(
            self.image,
            self.pred_mask,
            self.true_mask
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        
        # Test with attention maps
        vis_image = self.visualizer.visualize_prediction(
            self.image,
            self.pred_mask,
            self.true_mask,
            self.attention_maps
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        
        # Test saving
        save_path = "test_prediction.png"
        self.visualizer.visualize_prediction(
            self.image,
            self.pred_mask,
            save_path=save_path
        )
        
        self.assertTrue((Path(self.temp_dir) / save_path).exists())
    
    def test_training_progress_visualization(self):
        """Test training progress visualization"""
        metrics = {
            'loss': [1.0, 0.8, 0.6, 0.4],
            'accuracy': [0.5, 0.6, 0.7, 0.8]
        }
        
        # Test visualization
        vis_image = self.visualizer.visualize_training_progress(metrics)
        
        self.assertIsInstance(vis_image, np.ndarray)
        self.assertEqual(vis_image.ndim, 3)
        self.assertEqual(vis_image.dtype, np.uint8)
        
        # Test saving
        save_path = "test_progress.png"
        self.visualizer.visualize_training_progress(
            metrics,
            save_path=save_path
        )
        
        self.assertTrue((Path(self.temp_dir) / save_path).exists())
    
    def test_attention_analysis_visualization(self):
        """Test attention analysis visualization"""
        # Test visualization
        vis_image = self.visualizer.visualize_attention_analysis(
            self.attention_maps
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        self.assertEqual(vis_image.ndim, 3)
        self.assertEqual(vis_image.dtype, np.uint8)
        
        # Test saving
        save_path = "test_attention.png"
        self.visualizer.visualize_attention_analysis(
            self.attention_maps,
            save_path=save_path
        )
        
        self.assertTrue((Path(self.temp_dir) / save_path).exists())
    
    def test_shape_analysis_visualization(self):
        """Test shape analysis visualization"""
        # Test visualization
        vis_image = self.visualizer.visualize_shape_analysis(
            self.pred_distances
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        self.assertEqual(vis_image.ndim, 3)
        self.assertEqual(vis_image.dtype, np.uint8)
        
        # Test with ground truth
        vis_image = self.visualizer.visualize_shape_analysis(
            self.pred_distances,
            self.true_distances
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        
        # Test saving
        save_path = "test_shape.png"
        self.visualizer.visualize_shape_analysis(
            self.pred_distances,
            save_path=save_path
        )
        
        self.assertTrue((Path(self.temp_dir) / save_path).exists())

if __name__ == '__main__':
    unittest.main()
