# """
# Training script for the Adaptive StarDist model.
# """

# import argparse
# import os
# import numpy as np
# from skimage import io
# from csbdeep.utils import normalize_mi_ma
# from stardist.sample_patches import sample_patches
# from adaptive_stardist import AdaptiveStarDist2D, AdaptiveConfig2D
# from adaptive_stardist.multi_scale_model import MultiScaleStarDist2D, MultiScaleConfig2D
# from adaptive_stardist.utils import compute_local_statistics
# from adaptive_stardist.training import ScaleAwareLoss, MultiScaleStarDistData2D

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train Multi-Scale Adaptive StarDist model")
#     parser.add_argument("--data_path", type=str, required=True,
#                        help="Path to training data directory")
#     parser.add_argument("--slides", type=str, required=True,
#                        help="Comma-separated list of slide names")
#     parser.add_argument("--ckpt_path", type=str, required=True,
#                        help="Path to save model checkpoints")
#     parser.add_argument("--n_epochs", type=int, default=100,
#                        help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=8,
#                        help="Training batch size")
#     parser.add_argument("--min_n_rays", type=int, default=32,
#                        help="Minimum number of rays")
#     parser.add_argument("--max_n_rays", type=int, default=128,
#                        help="Maximum number of rays")
#     parser.add_argument("--min_grid", type=int, default=1,
#                        help="Minimum grid size")
#     parser.add_argument("--max_grid", type=int, default=4,
#                        help="Maximum grid size")
#     parser.add_argument("--use_gpu", action="store_true",
#                        help="Use GPU for training")
#     parser.add_argument("--scale_factors", type=str, default="1.0,0.5,0.25",
#                        help="Comma-separated list of scale factors")
#     parser.add_argument("--fusion_mode", type=str, default="attention",
#                        choices=["weighted", "attention"],
#                        help="Method to fuse multi-scale predictions")
#     parser.add_argument("--scale_weights", type=str, default=None,
#                        help="Comma-separated weights for each scale (for weighted fusion)")
#     parser.add_argument("--attention_channels", type=int, default=64,
#                        help="Number of channels in attention module")
#     parser.add_argument("--scale_loss_weight", type=float, default=0.3,
#                        help="Weight for scale matching loss")
#     parser.add_argument("--boundary_loss_weight", type=float, default=0.3,
#                        help="Weight for boundary accuracy loss")
#     parser.add_argument("--consistency_loss_weight", type=float, default=0.4,
#                        help="Weight for cross-scale consistency loss")
#     return parser.parse_args()

# def load_training_data(data_path, slides):
#     """Load and prepare training data."""
#     X, Y = [], []
    
#     if isinstance(slides, str):
#         slides_list = slides.split(',')
#     else:
#         slides_list = slides
    
#     available_slides = [name for name in os.listdir(data_path) 
#                        if os.path.isdir(os.path.join(data_path, name))]
    
#     for slide in slides_list:
#         slide = slide.strip()
#         if slide not in available_slides:
#             print(f"Warning: Slide '{slide}' not found in {data_path}")
#             continue
            
#         patches_dir = os.path.join(data_path, slide, 'patches_2048')
#         masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
        
#         if not os.path.exists(patches_dir):
#             print(f"Warning: Patches directory not found: {patches_dir}")
#             continue
#         if not os.path.exists(masks_dir):
#             print(f"Warning: Masks directory not found: {masks_dir}")
#             continue
        
#         patch_files = [f for f in os.listdir(patches_dir) if f.endswith('.png')]
        
#         for patch_file in patch_files:
#             img_path = os.path.join(patches_dir, patch_file)
#             mask_path = os.path.join(masks_dir, patch_file)
            
#             if not os.path.exists(mask_path):
#                 continue
                
#             img = io.imread(img_path)
#             mask = io.imread(mask_path)
            
#             img = normalize_mi_ma(img, 0, 255)
#             mask = (mask // 255).astype(np.uint8)
            
#             X.append(img)
#             Y.append(mask)
    
#     print(f"Loaded {len(X)} images from {len(slides_list)} slides")
#     return np.array(X), np.array(Y)

# def main():
#     args = parse_args()
    
#     X, Y = load_training_data(args.data_path, args.slides)
#     print(f"Loaded {len(X)} training samples")
    
#     n_val = max(1, int(0.1 * len(X)))
#     indices = np.random.permutation(len(X))
#     X_train, Y_train = X[indices[:-n_val]], Y[indices[:-n_val]]
#     X_val, Y_val = X[indices[-n_val:]], Y[indices[-n_val:]]
    
#     scale_factors = [float(s) for s in args.scale_factors.split(',')]
#     scale_weights = None
#     if args.scale_weights:
#         scale_weights = [float(w) for w in args.scale_weights.split(',')]
#         assert len(scale_weights) == len(scale_factors)
    
#     # Configure model
#     scale_detector_params = {
#         'min_sigma': 1.0,
#         'max_sigma': 30.0,
#         'sigma_ratio': 1.6,
#         'threshold': 0.1
#     }
    
#     conf = MultiScaleConfig2D(
#         n_rays=96,
#         grid=(2,2),
#         use_gpu=args.use_gpu,
#         n_channel_in=1,
#         train_patch_size=(256, 256),
#         train_batch_size=args.batch_size,
#         train_learning_rate=0.0003,
#         train_epochs=args.n_epochs,
#         train_steps_per_epoch=100,
#         # 多尺度特定参数
#         scale_factors=scale_factors,
#         fusion_mode=args.fusion_mode,
#         scale_weights=scale_weights,
#         attention_channels=args.attention_channels,
#         # 自适应参数
#         min_n_rays=args.min_n_rays,
#         max_n_rays=args.max_n_rays,
#         min_grid=args.min_grid,
#         max_grid=args.max_grid,
#         # 尺度检测器参数
#         scale_detector_params=scale_detector_params,
#         # 权重参数
#         boundary_weight=args.boundary_loss_weight,
#         size_weight=0.3  # 添加默认的size_weight
#     )
    
#     model = MultiScaleStarDist2D(conf, name='multi_scale_stardist', basedir=args.ckpt_path)
    
#     scale_aware_loss = ScaleAwareLoss(
#         scale_weight=args.scale_loss_weight,
#         boundary_weight=args.boundary_loss_weight,
#         consistency_weight=args.consistency_loss_weight
#     )
    
#     train_data = MultiScaleStarDistData2D(
#         X=X_train,
#         Y=Y_train,
#         batch_size=args.batch_size,
#         length=len(X_train),
#         scale_factors=scale_factors,
#         n_rays=conf.n_rays,
#         grid=(conf.grid[0], conf.grid[0]),
#         patch_size=(256, 256)
#     )
    
#     val_data = MultiScaleStarDistData2D(
#         X=X_val,
#         Y=Y_val,
#         batch_size=args.batch_size,
#         length=len(X_val),
#         scale_factors=scale_factors,
#         n_rays=conf.n_rays,
#         grid=(conf.grid[0], conf.grid[0]),
#         patch_size=(256, 256)
#     )
    
#     model.train_with_generator(
#         train_data=iter(train_data),
#         validation_data=iter(val_data) if val_data is not None else None,
#         epochs=args.n_epochs,
#         steps_per_epoch=100,
#         custom_loss=scale_aware_loss
#     )
    
#     print("Training completed successfully!")

# if __name__ == "__main__":
#     main()


# """
# Training utilities and scale-aware loss functions for Multi-Scale StarDist.
# 修复版本 - 处理数据形状不一致问题
# """

# import numpy as np
# import tensorflow as tf
# from typing import Dict, List, Tuple, Optional
# from csbdeep.utils import normalize_mi_ma
# from stardist.models import StarDistData2D
# from .utils import compute_local_statistics, estimate_object_sizes

# class ScaleAwareLoss:
#     """
#     Scale-aware loss functions for multi-scale StarDist training.
#     """
    
#     def __init__(self,
#                  scale_weight: float = 0.3,
#                  boundary_weight: float = 0.3,
#                  consistency_weight: float = 0.4):
#         """
#         Initialize scale-aware loss.
        
#         Args:
#             scale_weight: Weight for scale-matching loss
#             boundary_weight: Weight for boundary accuracy loss
#             consistency_weight: Weight for cross-scale consistency loss
#         """
#         self.scale_weight = scale_weight
#         self.boundary_weight = boundary_weight
#         self.consistency_weight = consistency_weight
        
#     def scale_matching_loss(self,
#                           pred_dist: tf.Tensor,
#                           true_dist: tf.Tensor,
#                           scale_map: tf.Tensor) -> tf.Tensor:
#         """
#         Compute loss that penalizes scale mismatches.
#         """
#         pred_normalized = pred_dist / (scale_map + 1e-6)
#         true_normalized = true_dist / (scale_map + 1e-6)
#         loss = tf.abs(pred_normalized - true_normalized)
#         scale_importance = tf.nn.sigmoid(scale_map)
#         weighted_loss = loss * scale_importance
#         return tf.reduce_mean(weighted_loss)
    
#     def boundary_accuracy_loss(self,
#                              pred_prob: tf.Tensor,
#                              true_mask: tf.Tensor,
#                              scale_map: tf.Tensor) -> tf.Tensor:
#         """
#         Compute boundary-aware loss that focuses on object edges.
#         """
#         # 简化的边界损失实现
#         bce = tf.keras.losses.binary_crossentropy(true_mask, pred_prob)
#         return tf.reduce_mean(bce)
    
#     def cross_scale_consistency_loss(self,
#                                    pred_list: List[Dict[str, tf.Tensor]],
#                                    scale_factors: List[float]) -> tf.Tensor:
#         """
#         Enforce consistency across predictions at different scales.
#         """
#         n_scales = len(pred_list)
#         if n_scales < 2:
#             return tf.constant(0.0)
        
#         # 简化的一致性损失
#         return tf.constant(0.0)
    
#     def __call__(self,
#                 pred_list: List[Dict[str, tf.Tensor]],
#                 true_dist: tf.Tensor,
#                 true_mask: tf.Tensor,
#                 scale_map: tf.Tensor,
#                 scale_factors: List[float]) -> Dict[str, tf.Tensor]:
#         """
#         Compute total scale-aware loss.
#         """
#         total_loss = 0
#         loss_components = {}
        
#         for i, preds in enumerate(pred_list):
#             # 简化的损失计算
#             prob_loss = tf.keras.losses.binary_crossentropy(
#                 true_mask, preds['prob']
#             )
#             loss_components[f'prob_loss_{i}'] = tf.reduce_mean(prob_loss)
#             total_loss += tf.reduce_mean(prob_loss)
        
#         loss_components['total_loss'] = total_loss
#         return loss_components

# class MultiScaleStarDistData2D(StarDistData2D):
#     """
#     Data generator for multi-scale StarDist training.
#     修复版本 - 安全处理数据形状
#     """
    
#     def __init__(self,
#                  X: np.ndarray,
#                  Y: np.ndarray,
#                  batch_size: int,
#                  length: int,
#                  scale_factors: List[float],
#                  **kwargs):
#         """
#         Initialize multi-scale data generator.
#         """
#         super().__init__(X, Y, batch_size, length=length, **kwargs)
#         self.scale_factors = scale_factors
        
#     def _safe_convert_to_array(self, data):
#         """安全地将数据转换为numpy数组"""
#         if isinstance(data, tuple):
#             # 检查所有元素的形状是否一致
#             shapes = [item.shape for item in data]
#             if len(set(shapes)) == 1:
#                 # 形状一致，可以stack
#                 return np.stack(data)
#             else:
#                 print(f"Warning: 形状不一致 {shapes}, 返回第一个元素")
#                 # 形状不一致，返回第一个作为示例
#                 return np.expand_dims(data[0], axis=0)
#         elif isinstance(data, np.ndarray):
#             return data
#         else:
#             try:
#                 return np.array(data)
#             except Exception as e:
#                 print(f"数据转换失败: {e}")
#                 return data
    
#     def _ensure_4d(self, array):
#         """确保数组是4D [batch, height, width, channels]"""
#         shape = array.shape
        
#         if len(shape) == 2:
#             # [H, W] -> [1, H, W, 1]
#             return array[np.newaxis, ..., np.newaxis]
#         elif len(shape) == 3:
#             # [B, H, W] -> [B, H, W, 1]
#             return array[..., np.newaxis]
#         elif len(shape) == 4:
#             return array
#         else:
#             raise ValueError(f"不支持的形状: {shape}")
    
#     def _resize_image(self, image, scale):
#         """使用numpy resize图像"""
#         if scale == 1.0:
#             return image
        
#         from skimage.transform import resize
        
#         if len(image.shape) == 4:
#             # [B, H, W, C]
#             batch_size, h, w, c = image.shape
#             new_h, new_w = int(h * scale), int(w * scale)
            
#             resized_batch = np.zeros((batch_size, new_h, new_w, c), dtype=image.dtype)
            
#             for i in range(batch_size):
#                 for j in range(c):
#                     resized_batch[i, :, :, j] = resize(
#                         image[i, :, :, j],
#                         (new_h, new_w),
#                         preserve_range=True,
#                         anti_aliasing=True
#                     )
#             return resized_batch
#         else:
#             raise ValueError(f"不支持的图像形状: {image.shape}")
        
#     def __getitem__(self, i: int) -> Tuple[List[np.ndarray], np.ndarray]:
#         """
#         Get batch of training data at multiple scales.
#         """
#         # Get base batch
#         batch = super().__getitem__(i)
#         batch_x, batch_y = batch[0], batch[1]
        
#         print(f"原始数据类型 - X: {type(batch_x)}, Y: {type(batch_y)}")
        
#         # 安全转换为numpy数组
#         batch_x = self._safe_convert_to_array(batch_x)
#         batch_y = self._safe_convert_to_array(batch_y)
        
#         print(f"转换后形状 - X: {batch_x.shape}, Y: {batch_y.shape}")
        
#         # 确保是4D
#         batch_x = self._ensure_4d(batch_x)
        
#         print(f"4D后形状 - X: {batch_x.shape}")
        
#         # 处理多尺度
#         multi_scale_x = []
#         for scale in self.scale_factors:
#             scaled_x = self._resize_image(batch_x, scale)
#             print(f"尺度 {scale}: {scaled_x.shape}")
#             multi_scale_x.append(scaled_x)
        
#         return multi_scale_x, batch_y












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
from adaptive_stardist.training import ScaleAwareLoss, MultiScaleStarDistData2D

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
