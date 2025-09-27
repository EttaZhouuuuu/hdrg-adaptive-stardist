"""
Setup test environment and sample data for testing the StarDist pipeline.
"""

import os
import numpy as np
from pathlib import Path
from skimage import io, draw
import shutil

def create_test_image(size=(512, 512), n_objects=10, min_radius=10, max_radius=50):
    """Create a synthetic test image with objects of varying sizes."""
    image = np.zeros(size)
    mask = np.zeros(size, dtype=np.uint16)
    
    # Create random objects
    for i in range(n_objects):
        # Random center and radius
        center_y = np.random.randint(max_radius, size[0]-max_radius)
        center_x = np.random.randint(max_radius, size[1]-max_radius)
        radius = np.random.randint(min_radius, max_radius)
        
        # Create circle
        rr, cc = draw.disk((center_y, center_x), radius)
        image[rr, cc] = 0.8 + 0.2 * np.random.random()
        mask[rr, cc] = i + 1
    
    # Add noise
    image += 0.1 * np.random.random(size)
    image = np.clip(image, 0, 1)
    
    return image, mask

def setup_test_data(base_dir: Path):
    """Create test data directory structure with synthetic images."""
    # Create directories following the original structure: DATA_PATH/SLIDE/patches_2048/
    data_dirs = {
        '240819_Ji_N1_H_EScan': base_dir / 'data/240819_Ji_N1_H_EScan',
        'test_slide_1': base_dir / 'data/test_slide_1',
        'test_slide_2': base_dir / 'data/test_slide_2',
        'scale_variation_slide': base_dir / 'data/scale_variation_slide'
    }
    
    for dir_path in data_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / 'patches_2048').mkdir(exist_ok=True)
        (dir_path / 'pseudo_gt_masks_2048').mkdir(exist_ok=True)
    
    # Create test images for each slide following original naming convention
    datasets = {
        '240819_Ji_N1_H_EScan': {'n_images': 8, 'size': (2048, 2048), 'n_objects': 15},
        'test_slide_1': {'n_images': 5, 'size': (2048, 2048), 'n_objects': 10},
        'test_slide_2': {'n_images': 5, 'size': (2048, 2048), 'n_objects': 12,
                         'min_radius': 5, 'max_radius': 100},
        'scale_variation_slide': {'n_images': 6, 'size': (2048, 2048),
                                 'n_objects': [5, 8, 12, 15, 20, 25]}
    }
    
    for slide_name, params in datasets.items():
        dir_path = data_dirs[slide_name]
        n_images = params['n_images']
        
        for i in range(n_images):
            if slide_name == 'scale_variation_slide':
                n_objects = params['n_objects'][i]
            else:
                n_objects = params['n_objects']
                
            image, mask = create_test_image(
                size=params['size'],
                n_objects=n_objects,
                min_radius=params.get('min_radius', 20),
                max_radius=params.get('max_radius', 80)
            )
            
            # Save images with original naming pattern: patch_col_X_row_Y.png
            patch_name = f'patch_col_{i//3:02d}_row_{i%3:02d}.png'
            
            # Convert to uint8 for PNG format (original uses PNG)
            image_uint8 = (image * 255).astype(np.uint8)
            # Ensure mask is strictly binary: 0 or 255
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            
            io.imsave(dir_path / 'patches_2048' / patch_name, image_uint8)
            io.imsave(dir_path / 'pseudo_gt_masks_2048' / patch_name, mask_uint8)

def setup_test_environment(base_dir: Path):
    """Set up complete test environment."""
    # Create test data
    setup_test_data(base_dir)
    
    # Create output directories
    output_dirs = [
        base_dir / 'results',
        base_dir / 'models',
        base_dir / 'evaluation'
    ]
    
    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Test environment setup completed!")

if __name__ == "__main__":
    base_dir = Path("/Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev")
    setup_test_environment(base_dir)
