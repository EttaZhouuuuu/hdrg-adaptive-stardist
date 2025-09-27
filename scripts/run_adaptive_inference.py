"""
Script to run inference using the adaptive StarDist pipeline.
"""

import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List
import numpy as np
from skimage import io
from adaptive_stardist.multi_scale_model import MultiScaleStarDist2D
from adaptive_stardist.inference import AdaptiveInferencePipeline

def setup_logging(log_file: Path = None):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run adaptive StarDist inference")
    
    # Input/output arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input image or directory (e.g., DATA_PATH/SLIDE/patches_2048)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save results")
    
    # Processing parameters
    parser.add_argument("--tile_size", type=int, default=2048,
                       help="Size of tiles for processing")
    parser.add_argument("--overlap", type=float, default=0.2,
                       help="Overlap between tiles (0-1)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--prob_thresh", type=float, default=0.5,
                       help="Probability threshold for detection")
    parser.add_argument("--nms_thresh", type=float, default=0.4,
                       help="Non-maximum suppression threshold")
    parser.add_argument("--min_object_size", type=int, default=10,
                       help="Minimum object size in pixels")
    
    # Scale sampling strategy
    parser.add_argument("--scale_sampling", type=str, default="adaptive",
                       choices=["fixed", "adaptive", "dynamic"],
                       help="How to sample scales during inference")
    
    # Additional options
    parser.add_argument("--save_confidence", action="store_true",
                       help="Save confidence maps")
    parser.add_argument("--save_parameters", action="store_true",
                       help="Save processing parameters")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/gpu)")
    
    return parser.parse_args()

def process_single_image(pipeline: AdaptiveInferencePipeline,
                        image_path: Path,
                        output_dir: Path,
                        save_confidence: bool,
                        save_parameters: bool) -> Dict:
    """
    Process a single image and save results.
    
    Args:
        pipeline: Inference pipeline
        image_path: Path to image
        output_dir: Output directory
        save_confidence: Whether to save confidence maps
        save_parameters: Whether to save processing parameters
        
    Returns:
        Processing statistics
    """
    # Create output directory
    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process image
    image = io.imread(image_path)
    labels, details = pipeline.process_image(image)
    
    # Save segmentation mask
    io.imsave(
        image_output_dir / 'segmentation.tiff',
        labels.astype(np.uint16)
    )
    
    # Save additional outputs if requested
    if save_confidence:
        io.imsave(
            image_output_dir / 'confidence_map.tiff',
            details['confidence_map']
        )
    
    if save_parameters:
        with open(image_output_dir / 'processing_details.json', 'w') as f:
            json.dump(details['details'], f, indent=2)
    
    # Compute statistics
    stats = {
        'image_name': image_path.name,
        'image_size': image.shape,
        'object_count': len(np.unique(labels)) - 1,
        'processing_time': details.get('processing_time', None),
        'scales_used': details.get('scales_used', [])
    }
    
    return stats

def main():
    args = parse_args()
    
    # Set up paths
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir / 'inference.log')
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = MultiScaleStarDist2D(None, name='stardist',
                                basedir=args.model_path)
    
    # Create inference pipeline
    pipeline = AdaptiveInferencePipeline(
        model=model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        min_object_size=args.min_object_size,
        scale_sampling=args.scale_sampling
    )
    
    # Process images
    all_stats = []
    if input_path.is_file():
        # Single image
        logger.info(f"Processing single image: {input_path}")
        stats = process_single_image(
            pipeline,
            input_path,
            output_dir,
            args.save_confidence,
            args.save_parameters
        )
        all_stats.append(stats)
    else:
        # Directory of images
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.tif*'))  # Support both .png and .tiff
        logger.info(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            logger.info(f"Processing image: {image_path}")
            try:
                stats = process_single_image(
                    pipeline,
                    image_path,
                    output_dir,
                    args.save_confidence,
                    args.save_parameters
                )
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
    
    # Save summary statistics
    with open(output_dir / 'processing_summary.json', 'w') as f:
        json.dump({
            'processing_parameters': vars(args),
            'image_statistics': all_stats
        }, f, indent=2)
    
    logger.info("Processing completed successfully!")

if __name__ == "__main__":
    main()
