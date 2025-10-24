import unittest
import numpy as np
import torch
from ..data.augmentation import (
    InstanceAwareRandomCrop,
    InstanceAwareRotation,
    get_training_augmentation,
    get_validation_augmentation,
    AugmentationVisualizer
)

class TestAugmentations(unittest.TestCase):
    """Test cases for augmentation module"""
    
    def setUp(self):
        """Set up test cases"""
        # Create test image and mask
        self.image_size = (256, 256)
        self.image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        self.mask = np.zeros(self.image_size, dtype=np.int32)
        
        # Add two instances to mask
        # Instance 1: circle
        y, x = np.ogrid[0:256, 0:256]
        circle = (x - 64)**2 + (y - 64)**2 <= 30**2
        self.mask[circle] = 1
        
        # Instance 2: rectangle
        self.mask[160:200, 160:200] = 2
    
    def test_instance_aware_crop(self):
        """Test instance-aware random crop"""
        crop_size = (128, 128)
        transform = InstanceAwareRandomCrop(
            height=crop_size[0],
            width=crop_size[1],
            min_instance_area=0.5
        )
        
        # Apply transform
        transformed = transform(image=self.image, mask=self.mask)
        cropped_image = transformed['image']
        cropped_mask = transformed['mask']
        
        # Check output shapes
        self.assertEqual(cropped_image.shape, (*crop_size, 3))
        self.assertEqual(cropped_mask.shape, crop_size)
        
        # Check that at least one instance is preserved
        unique_ids = np.unique(cropped_mask)
        self.assertTrue(len(unique_ids) > 1)  # At least one instance + background
    
    def test_instance_aware_rotation(self):
        """Test instance-aware rotation"""
        transform = InstanceAwareRotation(limit=45)
        
        # Apply transform
        transformed = transform(image=self.image, mask=self.mask)
        rotated_image = transformed['image']
        rotated_mask = transformed['mask']
        
        # Check output shapes
        self.assertEqual(rotated_image.shape, self.image.shape)
        self.assertEqual(rotated_mask.shape, self.mask.shape)
        
        # Check that instance IDs are preserved
        original_ids = np.unique(self.mask)
        rotated_ids = np.unique(rotated_mask)
        self.assertTrue(np.array_equal(original_ids, rotated_ids))
    
    def test_training_augmentation(self):
        """Test complete training augmentation pipeline"""
        transform = get_training_augmentation(input_size=(224, 224))
        
        # Apply transform
        transformed = transform(image=self.image, mask=self.mask)
        aug_image = transformed['image']
        aug_mask = transformed['mask']
        
        # Check output types and shapes
        self.assertIsInstance(aug_image, torch.Tensor)
        self.assertEqual(aug_image.shape, (3, 224, 224))
        self.assertEqual(aug_mask.shape, (224, 224))
    
    def test_validation_augmentation(self):
        """Test validation augmentation pipeline"""
        transform = get_validation_augmentation(input_size=(224, 224))
        
        # Apply transform
        transformed = transform(image=self.image, mask=self.mask)
        aug_image = transformed['image']
        aug_mask = transformed['mask']
        
        # Check output types and shapes
        self.assertIsInstance(aug_image, torch.Tensor)
        self.assertEqual(aug_image.shape, (3, 224, 224))
        self.assertEqual(aug_mask.shape, (224, 224))
        
        # Validation transform should be deterministic
        transformed2 = transform(image=self.image, mask=self.mask)
        self.assertTrue(torch.equal(aug_image, transformed2['image']))
    
    def test_augmentation_visualizer(self):
        """Test augmentation visualization"""
        transform = get_training_augmentation(input_size=(224, 224))
        
        # Generate visualization
        vis_image = AugmentationVisualizer.visualize_augmentations(
            self.image,
            self.mask,
            transform,
            n_examples=3
        )
        
        # Check output
        self.assertIsInstance(vis_image, np.ndarray)
        self.assertEqual(vis_image.ndim, 3)
        self.assertEqual(vis_image.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
