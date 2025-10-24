"""
Loss Functions for Adaptive Shape StarDist
==========================================

Specialized loss functions for training the adaptive shape model.

Key Components:
- Adaptive distance loss: Handles variable number of sampling points
- Complexity regularization: Encourages appropriate complexity
- Shape consistency loss: Ensures smooth boundaries
- Prior alignment loss: Aligns predictions with shape priors
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


class AdaptiveShapeLoss(keras.losses.Loss):
    """
    Combined loss function for Adaptive Shape StarDist
    
    Components:
    -----------
    1. Binary cross-entropy for instance probability
    2. Adaptive distance loss for boundary regression
    3. Complexity regularization
    4. Shape smoothness loss
    5. Prior alignment loss (optional)
    
    Parameters:
    -----------
    prob_weight : float
        Weight for probability loss
    dist_weight : float
        Weight for distance loss
    complexity_weight : float
        Weight for complexity regularization
    smoothness_weight : float
        Weight for shape smoothness loss
    prior_weight : float
        Weight for prior alignment loss
    """
    
    def __init__(
        self,
        prob_weight=1.0,
        dist_weight=1.0,
        complexity_weight=0.1,
        smoothness_weight=0.5,
        prior_weight=0.2,
        name='adaptive_shape_loss',
        **kwargs
    ):
        super(AdaptiveShapeLoss, self).__init__(name=name, **kwargs)
        
        self.prob_weight = prob_weight
        self.dist_weight = dist_weight
        self.complexity_weight = complexity_weight
        self.smoothness_weight = smoothness_weight
        self.prior_weight = prior_weight
    
    def call(self, y_true, y_pred):
        """
        Compute total loss
        
        Parameters:
        -----------
        y_true : dict
            Ground truth containing:
            - 'prob': Probability map [B, H, W, 1]
            - 'dist': Distance map [B, H, W, N_points]
            - 'mask': Valid region mask [B, H, W, 1]
        y_pred : dict
            Predictions from model
            
        Returns:
        --------
        total_loss : tf.Tensor
            Scalar loss value
        """
        
        # Extract ground truth
        true_prob = y_true.get('prob', None)
        true_dist = y_true.get('dist', None)
        valid_mask = y_true.get('mask', None)
        
        # Extract predictions
        pred_prob = y_pred['prob']
        pred_dist = y_pred['dist']
        complexity = y_pred['complexity']
        sampling_points = y_pred['sampling_points']
        
        losses = {}
        
        # 1. Probability loss (Binary Cross-Entropy)
        if true_prob is not None:
            prob_loss = self._probability_loss(true_prob, pred_prob, valid_mask)
            losses['prob_loss'] = self.prob_weight * prob_loss
        
        # 2. Distance loss (Adaptive L1/L2)
        if true_dist is not None:
            dist_loss = self._distance_loss(true_dist, pred_dist, true_prob, valid_mask)
            losses['dist_loss'] = self.dist_weight * dist_loss
        
        # 3. Complexity regularization
        complexity_loss = self._complexity_loss(complexity, true_prob)
        losses['complexity_loss'] = self.complexity_weight * complexity_loss
        
        # 4. Shape smoothness loss
        smoothness_loss = self._smoothness_loss(pred_dist, sampling_points)
        losses['smoothness_loss'] = self.smoothness_weight * smoothness_loss
        
        # 5. Prior alignment loss (if prototypes available)
        if 'prototype_weights' in y_pred:
            prior_loss = self._prior_alignment_loss(y_pred['prototype_weights'])
            losses['prior_loss'] = self.prior_weight * prior_loss
        
        # Total loss
        total_loss = sum(losses.values())
        
        # Store individual losses for monitoring
        self.add_metric(total_loss, name='total_loss')
        for loss_name, loss_value in losses.items():
            self.add_metric(loss_value, name=loss_name)
        
        return total_loss
    
    def _probability_loss(self, y_true, y_pred, mask=None):
        """
        Binary cross-entropy loss for instance probability
        
        Uses focal loss variant to handle class imbalance
        (background vs foreground)
        """
        # Focal loss parameters
        alpha = 0.25
        gamma = 2.0
        
        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross-entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal loss modulation
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = alpha * tf.pow(1 - pt, gamma)
        
        focal_loss = focal_weight * bce
        
        # Apply mask if provided
        if mask is not None:
            focal_loss = focal_loss * mask
            return tf.reduce_sum(focal_loss) / (tf.reduce_sum(mask) + 1e-7)
        else:
            return tf.reduce_mean(focal_loss)
    
    def _distance_loss(self, y_true, y_pred, prob_mask=None, valid_mask=None):
        """
        Adaptive distance loss for boundary regression
        
        Uses smooth L1 loss (Huber loss) which is more robust
        to outliers than L2 loss.
        """
        # Match dimensions if needed
        if tf.shape(y_true)[-1] != tf.shape(y_pred)[-1]:
            # Resize true distances to match predicted points
            y_true = tf.image.resize(
                y_true,
                tf.shape(y_pred)[1:3]
            )
        
        # Smooth L1 loss (Huber loss)
        delta = 1.0
        diff = tf.abs(y_true - y_pred)
        
        smooth_l1 = tf.where(
            diff < delta,
            0.5 * tf.square(diff),
            delta * (diff - 0.5 * delta)
        )
        
        # Weight by probability (focus on foreground regions)
        if prob_mask is not None:
            prob_mask = tf.image.resize(prob_mask, tf.shape(y_pred)[1:3])
            smooth_l1 = smooth_l1 * prob_mask
            
            if valid_mask is not None:
                valid_mask = tf.image.resize(valid_mask, tf.shape(y_pred)[1:3])
                smooth_l1 = smooth_l1 * valid_mask
                return tf.reduce_sum(smooth_l1) / (tf.reduce_sum(prob_mask * valid_mask) + 1e-7)
            else:
                return tf.reduce_sum(smooth_l1) / (tf.reduce_sum(prob_mask) + 1e-7)
        else:
            return tf.reduce_mean(smooth_l1)
    
    def _complexity_loss(self, complexity, true_prob):
        """
        Complexity regularization loss
        
        Encourages the model to use appropriate complexity based on
        the actual shape complexity in the ground truth.
        """
        # Estimate target complexity from ground truth
        # (Higher probability regions should have higher complexity)
        target_complexity = tf.reduce_mean(true_prob, axis=[1, 2, 3])  # [B, 1]
        
        # L2 loss
        complexity_loss = tf.square(complexity - target_complexity)
        
        return tf.reduce_mean(complexity_loss)
    
    def _smoothness_loss(self, distances, sampling_points):
        """
        Shape smoothness loss
        
        Encourages smooth boundaries by penalizing large differences
        between adjacent sampling points.
        """
        # Compute differences between adjacent points
        # Assuming points are ordered around the boundary
        
        # Compute spatial distance gradient
        # [B, H, W, N_points] -> [B, H, W, N_points-1]
        dist_diff = distances[..., 1:] - distances[..., :-1]
        
        # L2 smoothness penalty
        smoothness = tf.square(dist_diff)
        
        return tf.reduce_mean(smoothness)
    
    def _prior_alignment_loss(self, prototype_weights):
        """
        Prior alignment loss
        
        Encourages the use of learned shape prototypes
        by penalizing overly uniform or overly sparse usage.
        """
        # Entropy regularization: encourage diverse use of prototypes
        # but not too uniform
        
        # Add small epsilon for numerical stability
        eps = 1e-7
        weights_norm = prototype_weights + eps
        weights_norm = weights_norm / tf.reduce_sum(weights_norm, axis=-1, keepdims=True)
        
        # Compute entropy
        entropy = -tf.reduce_sum(weights_norm * tf.math.log(weights_norm), axis=-1)
        
        # Target entropy (encourage some diversity but not maximum)
        num_prototypes = tf.cast(tf.shape(prototype_weights)[-1], tf.float32)
        target_entropy = 0.7 * tf.math.log(num_prototypes)
        
        # L2 loss from target entropy
        prior_loss = tf.square(entropy - target_entropy)
        
        return tf.reduce_mean(prior_loss)
    
    def get_config(self):
        """Get configuration"""
        config = super(AdaptiveShapeLoss, self).get_config()
        config.update({
            'prob_weight': self.prob_weight,
            'dist_weight': self.dist_weight,
            'complexity_weight': self.complexity_weight,
            'smoothness_weight': self.smoothness_weight,
            'prior_weight': self.prior_weight,
        })
        return config


class DiceLoss(keras.losses.Loss):
    """
    Dice loss for segmentation
    
    Useful for handling class imbalance in segmentation tasks.
    """
    
    def __init__(self, smooth=1.0, name='dice_loss', **kwargs):
        super(DiceLoss, self).__init__(name=name, **kwargs)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        """Compute Dice loss"""
        
        # Flatten predictions and targets
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Compute intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss (1 - dice)
        return 1.0 - dice
    
    def get_config(self):
        config = super(DiceLoss, self).get_config()
        config.update({'smooth': self.smooth})
        return config


class BoundaryLoss(keras.losses.Loss):
    """
    Boundary-focused loss
    
    Puts more weight on boundary regions to improve
    boundary accuracy.
    """
    
    def __init__(self, boundary_weight=5.0, name='boundary_loss', **kwargs):
        super(BoundaryLoss, self).__init__(name=name, **kwargs)
        self.boundary_weight = boundary_weight
    
    def call(self, y_true, y_pred):
        """Compute boundary loss"""
        
        # Compute gradient magnitude to find boundaries
        true_grad_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        true_grad_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        
        true_grad_y = tf.pad(true_grad_y, [[0, 0], [0, 1], [0, 0], [0, 0]])
        true_grad_x = tf.pad(true_grad_x, [[0, 0], [0, 0], [0, 1], [0, 0]])
        
        boundary_mask = tf.abs(true_grad_y) + tf.abs(true_grad_x)
        boundary_mask = tf.clip_by_value(boundary_mask, 0, 1)
        
        # Weight map: higher weight on boundaries
        weight_map = 1.0 + self.boundary_weight * boundary_mask
        
        # Weighted binary cross-entropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce = tf.expand_dims(bce, axis=-1)
        
        weighted_bce = bce * weight_map
        
        return tf.reduce_mean(weighted_bce)
    
    def get_config(self):
        config = super(BoundaryLoss, self).get_config()
        config.update({'boundary_weight': self.boundary_weight})
        return config

