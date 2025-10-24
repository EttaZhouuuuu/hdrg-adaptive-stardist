import unittest
import numpy as np
import torch
from ..data.stardist_utils import StarDistTargetGenerator, compute_stardist_targets

class TestStarDistTargetGenerator(unittest.TestCase):
    """Test cases for StarDist target generation"""
    
    def setUp(self):
        """Set up test cases"""
        self.generator = StarDistTargetGenerator(n_rays=32)
        
        # Create a simple test mask with two objects
        self.test_mask = np.zeros((100, 100), dtype=np.int32)
        # First object: circle
        y, x = np.ogrid[0:100, 0:100]
        circle = (x - 30)**2 + (y - 30)**2 <= 15**2
        self.test_mask[circle] = 1
        # Second object: rectangle
        self.test_mask[60:80, 60:80] = 2
    
    def test_target_generation(self):
        """Test basic target generation"""
        distances, probabilities = self.generator.generate_targets(self.test_mask)
        
        # Check output types and shapes
        self.assertIsInstance(distances, torch.Tensor)
        self.assertIsInstance(probabilities, torch.Tensor)
        self.assertEqual(distances.shape[0], 32)  # n_rays
        self.assertEqual(distances.shape[1:], self.test_mask.shape)
        self.assertEqual(probabilities.shape, self.test_mask.shape)
        
        # Check probability map
        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.all(probabilities <= 1))
        self.assertTrue(torch.any(probabilities > 0))  # Some objects detected
    
    def test_object_filtering(self):
        """Test object size filtering"""
        # Create generator with size constraints
        generator = StarDistTargetGenerator(
            n_rays=32,
            min_object_size=300,  # Should filter out small objects
            max_object_size=1000
        )
        
        # Create test mask with different sized objects
        mask = np.zeros((100, 100), dtype=np.int32)
        # Small object (should be filtered out)
        mask[10:15, 10:15] = 1
        # Medium object (should be kept)
        mask[30:50, 30:50] = 2
        
        distances, probabilities = generator.generate_targets(mask)
        
        # Check that small object was filtered out
        small_object_region = probabilities[10:15, 10:15]
        self.assertTrue(torch.all(small_object_region == 0))
        
        # Check that medium object was kept
        medium_object_region = probabilities[30:50, 30:50]
        self.assertTrue(torch.any(medium_object_region > 0))
    
    def test_ray_computation(self):
        """Test ray distance computation"""
        # Create a simple circular object
        mask = np.zeros((50, 50), dtype=np.int32)
        y, x = np.ogrid[0:50, 0:50]
        circle = (x - 25)**2 + (y - 25)**2 <= 10**2
        mask[circle] = 1
        
        distances, _ = compute_stardist_targets(mask, n_rays=8)
        
        # Check that distances are roughly equal for a circle
        center_y, center_x = 25, 25
        mask_points = np.where(mask == 1)
        
        # Get distances for points near the center
        center_distances = distances[:, center_y, center_x]
        
        # For a circle, all rays should have similar distances
        max_diff = np.max(center_distances) - np.min(center_distances)
        self.assertLess(max_diff, 2.0)  # Allow small difference due to discretization
    
    def test_visualization(self):
        """Test visualization function"""
        # Generate targets
        distances, probabilities = self.generator.generate_targets(self.test_mask)
        
        # Create dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test visualization with specific ray
        vis_image = self.generator.visualize_targets(
            image, 
            distances.numpy(), 
            probabilities.numpy(),
            ray_idx=0
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        self.assertEqual(vis_image.ndim, 3)
        self.assertEqual(vis_image.dtype, np.uint8)
        
        # Test visualization with average distances
        vis_image = self.generator.visualize_targets(
            image, 
            distances.numpy(), 
            probabilities.numpy()
        )
        
        self.assertIsInstance(vis_image, np.ndarray)
        self.assertEqual(vis_image.ndim, 3)
        self.assertEqual(vis_image.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
