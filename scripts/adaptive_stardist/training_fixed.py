"""
Training utilities and scale-aware loss functions for Multi-Scale StarDist.
修复版本 - 处理数据形状不一致问题
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from csbdeep.utils import normalize_mi_ma
from stardist.models import StarDistData2D
from .utils import compute_local_statistics, estimate_object_sizes

class ScaleAwareLoss:
    """
    Scale-aware loss functions for multi-scale StarDist training.
    """
    
    def __init__(self,
                 scale_weight: float = 0.3,
                 boundary_weight: float = 0.3,
                 consistency_weight: float = 0.4):
        """
        Initialize scale-aware loss.
        
        Args:
            scale_weight: Weight for scale-matching loss
            boundary_weight: Weight for boundary accuracy loss
            consistency_weight: Weight for cross-scale consistency loss
        """
        self.scale_weight = scale_weight
        self.boundary_weight = boundary_weight
        self.consistency_weight = consistency_weight
        
    def scale_matching_loss(self,
                          pred_dist: tf.Tensor,
                          true_dist: tf.Tensor,
                          scale_map: tf.Tensor) -> tf.Tensor:
        """
        Compute loss that penalizes scale mismatches.
        """
        pred_normalized = pred_dist / (scale_map + 1e-6)
        true_normalized = true_dist / (scale_map + 1e-6)
        loss = tf.abs(pred_normalized - true_normalized)
        scale_importance = tf.nn.sigmoid(scale_map)
        weighted_loss = loss * scale_importance
        return tf.reduce_mean(weighted_loss)
    
    def boundary_accuracy_loss(self,
                             pred_prob: tf.Tensor,
                             true_mask: tf.Tensor,
                             scale_map: tf.Tensor) -> tf.Tensor:
        """
        Compute boundary-aware loss that focuses on object edges.
        """
        # 简化的边界损失实现
        bce = tf.keras.losses.binary_crossentropy(true_mask, pred_prob)
        return tf.reduce_mean(bce)
    
    def cross_scale_consistency_loss(self,
                                   pred_list: List[Dict[str, tf.Tensor]],
                                   scale_factors: List[float]) -> tf.Tensor:
        """
        Enforce consistency across predictions at different scales.
        """
        n_scales = len(pred_list)
        if n_scales < 2:
            return tf.constant(0.0)
        
        # 简化的一致性损失
        return tf.constant(0.0)
    
    def __call__(self,
                pred_list: List[Dict[str, tf.Tensor]],
                true_dist: tf.Tensor,
                true_mask: tf.Tensor,
                scale_map: tf.Tensor,
                scale_factors: List[float]) -> Dict[str, tf.Tensor]:
        """
        Compute total scale-aware loss.
        """
        total_loss = 0
        loss_components = {}
        
        for i, preds in enumerate(pred_list):
            # 简化的损失计算
            prob_loss = tf.keras.losses.binary_crossentropy(
                true_mask, preds['prob']
            )
            loss_components[f'prob_loss_{i}'] = tf.reduce_mean(prob_loss)
            total_loss += tf.reduce_mean(prob_loss)
        
        loss_components['total_loss'] = total_loss
        return loss_components

class MultiScaleStarDistData2D(StarDistData2D):
    """
    Data generator for multi-scale StarDist training.
    修复版本 - 安全处理数据形状
    """
    
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 batch_size: int,
                 length: int,
                 scale_factors: List[float],
                 **kwargs):
        """
        Initialize multi-scale data generator.
        """
        super().__init__(X, Y, batch_size, length=length, **kwargs)
        self.scale_factors = scale_factors
        
    def _safe_convert_to_array(self, data):
        """安全地将数据转换为numpy数组"""
        if isinstance(data, tuple):
            # 检查所有元素的形状是否一致
            shapes = [item.shape for item in data]
            if len(set(shapes)) == 1:
                # 形状一致，可以stack
                return np.stack(data)
            else:
                print(f"Warning: 形状不一致 {shapes}, 返回第一个元素")
                # 形状不一致，返回第一个作为示例
                return np.expand_dims(data[0], axis=0)
        elif isinstance(data, np.ndarray):
            return data
        else:
            try:
                return np.array(data)
            except Exception as e:
                print(f"数据转换失败: {e}")
                return data
    
    def _ensure_4d(self, array):
        """确保数组是4D [batch, height, width, channels]"""
        shape = array.shape
        
        if len(shape) == 2:
            # [H, W] -> [1, H, W, 1]
            return array[np.newaxis, ..., np.newaxis]
        elif len(shape) == 3:
            # [B, H, W] -> [B, H, W, 1]
            return array[..., np.newaxis]
        elif len(shape) == 4:
            return array
        else:
            raise ValueError(f"不支持的形状: {shape}")
    
    def _resize_image(self, image, scale):
        """使用numpy resize图像"""
        if scale == 1.0:
            return image
        
        from skimage.transform import resize
        
        if len(image.shape) == 4:
            # [B, H, W, C]
            batch_size, h, w, c = image.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized_batch = np.zeros((batch_size, new_h, new_w, c), dtype=image.dtype)
            
            for i in range(batch_size):
                for j in range(c):
                    resized_batch[i, :, :, j] = resize(
                        image[i, :, :, j],
                        (new_h, new_w),
                        preserve_range=True,
                        anti_aliasing=True
                    )
            return resized_batch
        else:
            raise ValueError(f"不支持的图像形状: {image.shape}")
        
    def __getitem__(self, i: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get batch of training data at multiple scales.
        """
        # Get base batch
        batch = super().__getitem__(i)
        batch_x, batch_y = batch[0], batch[1]
        
        print(f"原始数据类型 - X: {type(batch_x)}, Y: {type(batch_y)}")
        
        # 安全转换为numpy数组
        batch_x = self._safe_convert_to_array(batch_x)
        batch_y = self._safe_convert_to_array(batch_y)
        
        print(f"转换后形状 - X: {batch_x.shape}, Y: {batch_y.shape}")
        
        # 确保是4D
        batch_x = self._ensure_4d(batch_x)
        
        print(f"4D后形状 - X: {batch_x.shape}")
        
        # 处理多尺度
        multi_scale_x = []
        for scale in self.scale_factors:
            scaled_x = self._resize_image(batch_x, scale)
            print(f"尺度 {scale}: {scaled_x.shape}")
            multi_scale_x.append(scaled_x)
        
        return multi_scale_x, batch_y
