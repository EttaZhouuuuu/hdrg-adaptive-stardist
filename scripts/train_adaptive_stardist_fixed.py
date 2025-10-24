"""
Training script for the Adaptive StarDist model - Fixed Version
修复版本的训练脚本，集成了验证过的数据处理逻辑
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
from adaptive_stardist.training_final import ScaleAwareLoss, MultiScaleStarDistData2D

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Scale Adaptive StarDist model - Fixed")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data directory")
    parser.add_argument("--slides", type=str, required=True,
                       help="Comma-separated list of slide names")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to save model checkpoints")
    parser.add_argument("--n_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--min_n_rays", type=int, default=32,
                       help="Minimum number of rays")
    parser.add_argument("--max_n_rays", type=int, default=96,
                       help="Maximum number of rays")
    parser.add_argument("--min_grid", type=int, default=1,
                       help="Minimum grid size")
    parser.add_argument("--max_grid", type=int, default=2,
                       help="Maximum grid size")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU for training")
    parser.add_argument("--scale_factors", type=str, default="1.0,0.5",
                       help="Comma-separated list of scale factors")
    parser.add_argument("--fusion_mode", type=str, default="attention",
                       choices=["weighted", "attention"],
                       help="Method to fuse multi-scale predictions")
    parser.add_argument("--scale_weights", type=str, default=None,
                       help="Comma-separated weights for each scale")
    parser.add_argument("--attention_channels", type=int, default=64,
                       help="Number of channels in attention module")
    parser.add_argument("--scale_loss_weight", type=float, default=0.3,
                       help="Weight for scale matching loss")
    parser.add_argument("--boundary_loss_weight", type=float, default=0.3,
                       help="Weight for boundary accuracy loss")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.4,
                       help="Weight for cross-scale consistency loss")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples to load (for testing)")
    return parser.parse_args()

def load_training_data(data_path, slides, max_samples=None):
    """Load and prepare training data with detailed logging."""
    X, Y = [], []
    
    print(f"Loading data from: {data_path}")
    print(f"Slides to process: {slides}")
    print(f"Max samples: {max_samples}")
    
    if isinstance(slides, str):
        slides_list = slides.split(',')
    else:
        slides_list = slides
    
    # 检查数据目录
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_path}")
    
    available_slides = [name for name in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, name))]
    print(f"Available slides: {available_slides}")
    
    sample_count = 0
    for slide in slides_list:
        slide = slide.strip()
        if slide not in available_slides:
            print(f"Warning: Slide '{slide}' not found in {data_path}")
            continue
            
        patches_dir = os.path.join(data_path, slide, 'patches_2048')
        masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
        
        print(f"Processing slide {slide}:")
        print(f"  Patches dir: {patches_dir}")
        print(f"  Masks dir: {masks_dir}")
        
        if not os.path.exists(patches_dir):
            print(f"Warning: Patches directory not found: {patches_dir}")
            continue
        if not os.path.exists(masks_dir):
            print(f"Warning: Masks directory not found: {masks_dir}")
            continue
        
        patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
        print(f"  Found {len(patch_files)} patch files")
        
        # 限制样本数量
        if max_samples:
            remaining_samples = max_samples - sample_count
            patch_files = patch_files[:remaining_samples]
        
        for patch_file in patch_files:
            img_path = os.path.join(patches_dir, patch_file)
            mask_path = os.path.join(masks_dir, patch_file)
            
            if not os.path.exists(mask_path):
                print(f"  Warning: Missing mask for {patch_file}")
                continue
                
            try:
                img = io.imread(img_path)
                mask = io.imread(mask_path)
                
                # 标准化处理
                img = normalize_mi_ma(img, 0, 255).astype(np.float32)
                mask = (mask // 255).astype(np.uint8)
                
                X.append(img)
                Y.append(mask)
                sample_count += 1
                
                if sample_count % 10 == 0:
                    print(f"  已加载 {sample_count} 个样本...")
                
                if max_samples and sample_count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"  Error loading {patch_file}: {e}")
                continue
        
        if max_samples and sample_count >= max_samples:
            print(f"达到最大样本数 {max_samples}，停止加载")
            break
    
    if not X:
        raise ValueError("没有加载到任何数据")
    
    X = np.array(X)
    Y = np.array(Y)
    print(f"\nFinal data shapes:")
    print(f"X: {X.shape} {X.dtype}")
    print(f"Y: {Y.shape} {Y.dtype}")
    print(f"Total samples loaded: {len(X)}")
    
    return X, Y

def main():
    args = parse_args()
    
    print("=== Multi-Scale Adaptive StarDist Training - Fixed Version ===")
    print(f"Parameters: epochs={args.n_epochs}, batch_size={args.batch_size}")
    print(f"Max samples: {args.max_samples}")
    
    # Load training data
    X, Y = load_training_data(args.data_path, args.slides, args.max_samples)
    print(f"Loaded {len(X)} training samples")
    
    # Split into train/validation
    n_val = max(1, min(5, int(0.1 * len(X))))  # 最多5个验证样本
    indices = np.random.permutation(len(X))
    X_train, Y_train = X[indices[:-n_val]], Y[indices[:-n_val]]
    X_val, Y_val = X[indices[-n_val:]], Y[indices[-n_val:]]
    
    print(f"\nData split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    
    # Parse scale-related arguments
    scale_factors = [float(s) for s in args.scale_factors.split(',')]
    scale_weights = None
    if args.scale_weights:
        scale_weights = [float(w) for w in args.scale_weights.split(',')]
        assert len(scale_weights) == len(scale_factors)
    
    print(f"Scale factors: {scale_factors}")
    
    # Configure model
    scale_detector_params = {
        'min_sigma': 1.0,
        'max_sigma': 30.0,
        'sigma_ratio': 1.6,
        'threshold': 0.1
    }
    
    conf = MultiScaleConfig2D(
        n_rays=args.max_n_rays,
        grid=(args.max_grid, args.max_grid),
        use_gpu=args.use_gpu,
        n_channel_in=1,
        train_patch_size=(256, 256),
        train_batch_size=args.batch_size,
        train_learning_rate=0.0003,
        train_epochs=args.n_epochs,
        train_steps_per_epoch=min(50, len(X_train) // args.batch_size),  # 限制步数
        # 多尺度特定参数
        scale_factors=scale_factors,
        fusion_mode=args.fusion_mode,
        scale_weights=scale_weights,
        attention_channels=args.attention_channels,
        # 自适应参数
        min_n_rays=args.min_n_rays,
        max_n_rays=args.max_n_rays,
        min_grid=args.min_grid,
        max_grid=args.max_grid,
        # 尺度检测器参数
        scale_detector_params=scale_detector_params,
        # 权重参数
        boundary_weight=args.boundary_loss_weight,
        size_weight=0.3
    )
    
    print(f"Model configuration: n_rays={conf.n_rays}, grid={conf.grid}")
    
    # Create model
    model = MultiScaleStarDist2D(conf, name='multi_scale_stardist', basedir=args.ckpt_path)
    
    # Create scale-aware loss
    scale_aware_loss = ScaleAwareLoss(
        scale_weight=args.scale_loss_weight,
        boundary_weight=args.boundary_loss_weight,
        consistency_weight=args.consistency_loss_weight
    )
    
    # Create data generators
    print("\n创建数据生成器...")
    train_data = MultiScaleStarDistData2D(
        X=X_train,
        Y=Y_train,
        batch_size=args.batch_size,
        length=len(X_train),
        scale_factors=scale_factors,
        n_rays=conf.n_rays,
        grid=conf.grid,
        patch_size=(256, 256)
    )
    
    val_data = MultiScaleStarDistData2D(
        X=X_val,
        Y=Y_val,
        batch_size=args.batch_size,
        length=len(X_val),
        scale_factors=scale_factors,
        n_rays=conf.n_rays,
        grid=conf.grid,
        patch_size=(256, 256)
    )
    
    print("数据生成器创建成功")
    
    # Train with custom training loop
    print("\n开始训练...")
    model.train_with_generator(
        train_data=iter(train_data),
        validation_data=iter(val_data) if len(X_val) > 0 else None,
        epochs=args.n_epochs,
        steps_per_epoch=conf.train_steps_per_epoch,
        custom_loss=scale_aware_loss
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
