"""
Training utilities and scale-aware loss functions for Multi-Scale StarDist.
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
        
        Args:
            pred_dist: Predicted distance map
            true_dist: Ground truth distance map
            scale_map: Local scale map
            
        Returns:
            Scale matching loss
        """
        # Normalize distances by local scale
        pred_normalized = pred_dist / (scale_map + 1e-6)
        true_normalized = true_dist / (scale_map + 1e-6)
        
        # Compute scale-normalized loss
        loss = tf.abs(pred_normalized - true_normalized)
        
        # Weight loss by scale importance
        scale_importance = tf.nn.sigmoid(scale_map)
        weighted_loss = loss * scale_importance
        
        return tf.reduce_mean(weighted_loss)
    
    def boundary_accuracy_loss(self,
                             pred_prob: tf.Tensor,
                             true_mask: tf.Tensor,
                             scale_map: tf.Tensor) -> tf.Tensor:
        """
        Compute boundary-aware loss that focuses on object edges.
        
        Args:
            pred_prob: Predicted probability map
            true_mask: Ground truth mask
            scale_map: Local scale map
            
        Returns:
            Boundary accuracy loss
        """
        # Compute boundary mask using gradient
        true_boundary = tf.abs(tf.image.sobel_edges(tf.cast(true_mask, tf.float32)))
        true_boundary = tf.reduce_max(true_boundary, axis=-1)
        
        # Create boundary region with scale-dependent width
        kernel_size = tf.cast(tf.maximum(3, scale_map * 2), tf.int32)
        boundary_region = tf.nn.max_pool2d(
            true_boundary[..., tf.newaxis],
            kernel_size,
            strides=1,
            padding='SAME'
        )[..., 0]
        
        # Compute weighted binary cross entropy
        bce = tf.keras.losses.binary_crossentropy(true_mask, pred_prob)
        weighted_bce = bce * (1 + boundary_region * 2)  # Higher weight at boundaries
        
        return tf.reduce_mean(weighted_bce)
    
    def cross_scale_consistency_loss(self,
                                   pred_list: List[Dict[str, tf.Tensor]],
                                   scale_factors: List[float]) -> tf.Tensor:
        """
        Enforce consistency across predictions at different scales.
        
        Args:
            pred_list: List of predictions at different scales
            scale_factors: List of scale factors used
            
        Returns:
            Consistency loss
        """
        n_scales = len(pred_list)
        if n_scales < 2:
            return 0.0
            
        consistency_losses = []
        
        # Compare predictions across scales
        for i in range(n_scales):
            for j in range(i + 1, n_scales):
                # Resize predictions to same size
                prob_i = pred_list[i]['prob']
                prob_j = tf.image.resize(
                    pred_list[j]['prob'],
                    tf.shape(prob_i)[1:3]
                )
                
                # Compute consistency loss
                diff = tf.abs(prob_i - prob_j)
                
                # Weight by scale difference
                scale_diff = abs(scale_factors[i] - scale_factors[j])
                weight = tf.exp(-scale_diff)
                
                consistency_losses.append(tf.reduce_mean(diff) * weight)
        
        return tf.add_n(consistency_losses) / len(consistency_losses)
    
    def __call__(self,
                pred_list: List[Dict[str, tf.Tensor]],
                true_dist: tf.Tensor,
                true_mask: tf.Tensor,
                scale_map: tf.Tensor,
                scale_factors: List[float]) -> Dict[str, tf.Tensor]:
        """
        Compute total scale-aware loss.
        
        Args:
            pred_list: List of predictions at different scales
            true_dist: Ground truth distance map
            true_mask: Ground truth mask
            scale_map: Local scale map
            scale_factors: List of scale factors used
            
        Returns:
            Dictionary of loss components and total loss
        """
        total_loss = 0
        loss_components = {}
        
        # Compute losses for each scale
        for i, preds in enumerate(pred_list):
            # Scale matching loss
            scale_loss = self.scale_matching_loss(
                preds['dist'],
                true_dist,
                scale_map
            )
            loss_components[f'scale_loss_{i}'] = scale_loss
            
            # Boundary accuracy loss
            boundary_loss = self.boundary_accuracy_loss(
                preds['prob'],
                true_mask,
                scale_map
            )
            loss_components[f'boundary_loss_{i}'] = boundary_loss
            
            # Add to total loss
            total_loss += (
                self.scale_weight * scale_loss +
                self.boundary_weight * boundary_loss
            )
        
        # Cross-scale consistency loss
        consistency_loss = self.cross_scale_consistency_loss(
            pred_list,
            scale_factors
        )
        loss_components['consistency_loss'] = consistency_loss
        
        total_loss += self.consistency_weight * consistency_loss
        loss_components['total_loss'] = total_loss
        
        return loss_components

class MultiScaleStarDistData2D(StarDistData2D):
    """
    Data generator for multi-scale StarDist training.
    """
    
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 batch_size: int,
                 scale_factors: List[float],
                 **kwargs):
        """
        Initialize multi-scale data generator.
        
        Args:
            X: Input images
            Y: Ground truth masks
            batch_size: Batch size
            scale_factors: List of scale factors to use
            **kwargs: Additional arguments for StarDistData2D
        """
        super().__init__(X, Y, batch_size, **kwargs)
        self.scale_factors = scale_factors
        
    def __getitem__(self, i: int) -> Tuple[List[np.ndarray], Dict]:
        """
        Get batch of training data at multiple scales.
        
        Args:
            i: Batch index
            
        Returns:
            Tuple of (multi-scale inputs, targets)
        """
        # Get base batch
        batch_x, batch_y = super().__getitem__(i)
        
        # Process at multiple scales
        multi_scale_x = []
        for scale in self.scale_factors:
            if scale == 1.0:
                scaled_x = batch_x
            else:
                # Resize each image in batch
                scaled_x = tf.image.resize(
                    batch_x,
                    tf.cast(tf.shape(batch_x)[1:3] * scale, tf.int32)
                )
            multi_scale_x.append(scaled_x)
        
        return multi_scale_x, batch_y
