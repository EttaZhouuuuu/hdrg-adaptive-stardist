"""
Example: Inference with Adaptive Shape StarDist
================================================

This script demonstrates how to use a trained model for inference.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import Adaptive Shape StarDist
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.inference import AdaptiveShapePredictor
from adaptive_shape_stardist.utils import visualize_predictions


def load_test_image():
    """
    Load a test image for inference
    
    Returns:
    --------
    image : np.ndarray
        Test image [H, W] or [H, W, C]
    """
    
    # TODO: Replace with your image loading code
    # Example: from skimage.io import imread
    # image = imread('path/to/test/image.tif')
    
    # For demonstration, create a dummy image
    print("Creating dummy test image...")
    H, W = 512, 512
    image = np.random.rand(H, W).astype(np.float32)
    
    # Add some circular objects
    n_objects = 15
    for i in range(n_objects):
        cy, cx = np.random.randint(100, H-100), np.random.randint(100, W-100)
        radius = np.random.randint(20, 50)
        y, x = np.ogrid[:H, :W]
        mask = (y - cy)**2 + (x - cx)**2 <= radius**2
        image[mask] += np.random.rand() * 0.5 + 0.5
    
    return image


def main():
    """Main inference script"""
    
    print("=" * 60)
    print("Adaptive Shape StarDist - Inference Example")
    print("=" * 60)
    
    # 1. Load trained model
    print("\n1. Loading trained model...")
    model_path = './models/my_adaptive_model'
    
    try:
        model = AdaptiveShapeStarDist.load_model(model_path)
        print(f"   Model loaded from: {model_path}")
    except:
        print(f"   Model not found at: {model_path}")
        print("   Please train a model first using train_example.py")
        print("   Using a newly initialized model for demonstration...")
        
        from adaptive_shape_stardist.configs import get_default_config
        config = get_default_config()
        model = AdaptiveShapeStarDist(
            config=config,
            name='demo_model',
            basedir='./models'
        )
    
    # 2. Create predictor
    print("\n2. Creating predictor...")
    predictor = AdaptiveShapePredictor(
        model=model,
        prob_thresh=0.5,
        nms_thresh=0.3
    )
    print("   Predictor ready!")
    
    # 3. Load test image
    print("\n3. Loading test image...")
    image = load_test_image()
    print(f"   Image shape: {image.shape}")
    
    # 4. Run prediction
    print("\n4. Running prediction...")
    labels, details = predictor.predict(
        image,
        normalize=True,
        return_details=True
    )
    
    num_instances = len(np.unique(labels)) - 1  # Exclude background
    print(f"   Detected {num_instances} instances")
    print(f"   Average complexity: {details['complexity']:.3f}")
    
    # 5. Visualize results
    print("\n5. Visualizing results...")
    fig = visualize_predictions(
        image=image,
        labels=labels,
        prob=details['prob'],
        sampling_points=details['sampling_points'],
        title='Adaptive Shape StarDist Prediction'
    )
    
    # Save figure
    output_path = './prediction_result.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Results saved to: {output_path}")
    
    # 6. Print detailed statistics
    print("\n6. Prediction statistics:")
    print(f"   - Number of instances: {num_instances}")
    print(f"   - Shape complexity: {details['complexity']:.3f}")
    print(f"   - Sampling points shape: {details['sampling_points'].shape}")
    print(f"   - Probability map range: [{details['prob'].min():.3f}, {details['prob'].max():.3f}]")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

