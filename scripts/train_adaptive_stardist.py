"""
Training script for the Adaptive StarDist model.
"""

import argparse
import os
import numpy as np
from skimage import io
from csbdeep.utils import normalize_mi_ma
from stardist.sample_patches import sample_patches
from adaptive_stardist import AdaptiveStarDist2D, AdaptiveConfig2D
from adaptive_stardist.multi_scale_model import MultiScaleStarDist2D, MultiScaleConfig2D
from adaptive_stardist.utils import compute_local_statistics
from adaptive_stardist.training import ScaleAwareLoss, MultiScaleStarDistData2D

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Scale Adaptive StarDist model")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data directory")
    parser.add_argument("--slides", type=str, required=True,
                       help="Comma-separated list of slide names")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to save model checkpoints")
    parser.add_argument("--n_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--min_n_rays", type=int, default=32,
                       help="Minimum number of rays")
    parser.add_argument("--max_n_rays", type=int, default=128,
                       help="Maximum number of rays")
    parser.add_argument("--min_grid", type=int, default=1,
                       help="Minimum grid size")
    parser.add_argument("--max_grid", type=int, default=4,
                       help="Maximum grid size")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU for training")
    parser.add_argument("--scale_factors", type=str, default="1.0,0.5,0.25",
                       help="Comma-separated list of scale factors")
    parser.add_argument("--fusion_mode", type=str, default="attention",
                       choices=["weighted", "attention"],
                       help="Method to fuse multi-scale predictions")
    parser.add_argument("--scale_weights", type=str, default=None,
                       help="Comma-separated weights for each scale (for weighted fusion)")
    parser.add_argument("--attention_channels", type=int, default=64,
                       help="Number of channels in attention module")
    # Scale-aware loss parameters
    parser.add_argument("--scale_loss_weight", type=float, default=0.3,
                       help="Weight for scale matching loss")
    parser.add_argument("--boundary_loss_weight", type=float, default=0.3,
                       help="Weight for boundary accuracy loss")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.4,
                       help="Weight for cross-scale consistency loss")
    return parser.parse_args()

def load_training_data(data_path, slides):
    """Load and prepare training data."""
    X, Y = [], []  # images and masks
    
    # Handle both single slide and multiple slides
    if isinstance(slides, str):
        slides_list = slides.split(',')
    else:
        slides_list = slides
    
    # Get available slides from data directory
    available_slides = [name for name in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, name))]
    
    for slide in slides_list:
        slide = slide.strip()  # Remove whitespace
        if slide not in available_slides:
            print(f"Warning: Slide '{slide}' not found in {data_path}")
            continue
            
        patches_dir = os.path.join(data_path, slide, 'patches_2048')
        masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
        
        if not os.path.exists(patches_dir):
            print(f"Warning: Patches directory not found: {patches_dir}")
            continue
        if not os.path.exists(masks_dir):
            print(f"Warning: Masks directory not found: {masks_dir}")
            continue
        
        # Load all .png files
        patch_files = [f for f in os.listdir(patches_dir) if f.endswith('.png')]
        
        for patch_file in patch_files:
            # Load image and mask
            img_path = os.path.join(patches_dir, patch_file)
            mask_path = os.path.join(masks_dir, patch_file)
            
            if not os.path.exists(mask_path):
                continue
                
            img = io.imread(img_path)
            mask = io.imread(mask_path)
            
            # Normalize image (following original approach)
            img = normalize_mi_ma(img, 0, 255)
            
            # Convert mask to binary (following original: mask // 255)
            mask = (mask // 255).astype(np.uint8)
            
            X.append(img)
            Y.append(mask)
    
    print(f"Loaded {len(X)} images from {len(slides_list)} slides")
    return np.array(X), np.array(Y)

def main():
    args = parse_args()
    
    # Load training data
    X, Y = load_training_data(args.data_path, args.slides)
    print(f"Loaded {len(X)} training samples")
    
    # Split into train/validation
    n_val = max(1, int(0.1 * len(X)))
    indices = np.random.permutation(len(X))
    X_train, Y_train = X[indices[:-n_val]], Y[indices[:-n_val]]
    X_val, Y_val = X[indices[-n_val:]], Y[indices[-n_val:]]
    
    # Parse scale-related arguments
    scale_factors = [float(s) for s in args.scale_factors.split(',')]
    scale_weights = None
    if args.scale_weights:
        scale_weights = [float(w) for w in args.scale_weights.split(',')]
        assert len(scale_weights) == len(scale_factors), \
            "Number of scale weights must match number of scales"
    
    # Configure model
    scale_detector_params = {
        'min_sigma': 1.0,
        'max_sigma': 30.0,
        'sigma_ratio': 1.6,
        'threshold': 0.1
    }
    
    conf = MultiScaleConfig2D(
        min_n_rays=args.min_n_rays,
        max_n_rays=args.max_n_rays,
        min_grid=args.min_grid,
        max_grid=args.max_grid,
        use_gpu=args.use_gpu,
        n_channel_in=1,
        train_patch_size=(256, 256),
        train_batch_size=args.batch_size,
        train_learning_rate=0.0003,
        train_epochs=args.n_epochs,
        train_steps_per_epoch=100,
        scale_detector_params=scale_detector_params,
        # Multi-scale specific parameters
        scale_factors=scale_factors,
        fusion_mode=args.fusion_mode,
        scale_weights=scale_weights,
        attention_channels=args.attention_channels
    )
    
    model = MultiScaleStarDist2D(conf, name='multi_scale_stardist', basedir=args.ckpt_path)
    
    # Create scale-aware loss
    scale_aware_loss = ScaleAwareLoss(
        scale_weight=args.scale_loss_weight,
        boundary_weight=args.boundary_loss_weight,
        consistency_weight=args.consistency_loss_weight
    )
    
    # Create multi-scale data generators
    train_data = MultiScaleStarDistData2D(
        X_train, Y_train,
        batch_size=args.batch_size,
        scale_factors=scale_factors,
        n_rays=conf.n_rays,
        grid=conf.grid[0],  # Assuming square grid
        patch_size=(256, 256)
    )
    
    val_data = MultiScaleStarDistData2D(
        X_val, Y_val,
        batch_size=args.batch_size,
        scale_factors=scale_factors,
        n_rays=conf.n_rays,
        grid=conf.grid[0],
        patch_size=(256, 256)
    )
    
    # Train model with custom loss
    model.train(
        train_data,
        validation_data=val_data,
        epochs=args.n_epochs,
        custom_loss=scale_aware_loss
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
