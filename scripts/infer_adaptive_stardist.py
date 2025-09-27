"""
Inference script for the Adaptive StarDist model.
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
from skimage import io
from csbdeep.utils import normalize_mi_ma
from adaptive_stardist import AdaptiveStarDist2D
from adaptive_stardist.utils import compute_scale_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Adaptive StarDist")
    parser.add_argument("--model_ckpt_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--slide_fp", type=str, required=True,
                       help="Path to input slide or directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--patch_size", type=int, default=2048,
                       help="Size of image patches")
    parser.add_argument("--prob_threshold", type=float, default=0.5,
                       help="Probability threshold for prediction")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/gpu)")
    return parser.parse_args()

def process_image(model, image_path, output_dir, prob_threshold):
    """Process a single image with the adaptive model."""
    # Load and normalize image
    img = io.imread(image_path)
    img = normalize_mi_ma(img)
    
    # Get filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create output directory for this image
    img_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Predict with adaptive parameters
    labels, details = model.predict_instances(img)
    
    # Save results
    # - Segmentation mask
    io.imsave(os.path.join(img_output_dir, 'pred_labels.png'), 
              labels.astype(np.uint16))
    
    # - Scale map
    scale_map = details['scale_map']
    plt.imsave(os.path.join(img_output_dir, 'scale_map.png'),
               scale_map, cmap='viridis')
    
    # - Confidence map
    conf_map = details['confidence_map']
    plt.imsave(os.path.join(img_output_dir, 'confidence_map.png'),
               conf_map, cmap='viridis')
    
    # Save metadata
    metadata = {
        'adapted_params': details['adapted_params'],
        'n_objects': len(np.unique(labels)) - 1,
        'image_shape': img.shape,
        'prob_threshold': prob_threshold
    }
    
    with open(os.path.join(img_output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return labels, details

def main():
    args = parse_args()
    
    # Load model
    model = AdaptiveStarDist2D(None, name='adaptive_stardist', 
                              basedir=args.model_ckpt_path)
    model.thresholds = dict(prob=args.prob_threshold, nms=0.4)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save run configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Process input
    if os.path.isfile(args.slide_fp):
        # Single image
        process_image(model, args.slide_fp, args.output_dir, args.prob_threshold)
    else:
        # Directory of images
        for filename in os.listdir(args.slide_fp):
            if filename.endswith(('.png', '.jpg', '.tif')):
                image_path = os.path.join(args.slide_fp, filename)
                process_image(model, image_path, args.output_dir, args.prob_threshold)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
