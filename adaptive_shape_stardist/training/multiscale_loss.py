"""
Multi-scale Loss Functions for FPN
====================================

Implements loss functions that operate on multiple scales for FPN-based models.

Key Features:
- Weighted combination of losses across FPN levels
- Automatic target resizing to match prediction scales
- Focal loss for handling class imbalance at all scales
- Smooth L1 loss for distance regression
- Scale-adaptive weighting (emphasize high-resolution predictions)

Author: Shape-Aware StarDist Team
Date: 2025-10-29
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def multiscale_loss(
    predictions,
    targets,
    level_weights=None,
    alpha=0.25,
    gamma=2.0,
    smooth_l1_delta=1.0
):
    """
    Multi-scale loss for FPN predictions
    
    Computes losses at each FPN level and combines them with appropriate weights.
    
    Parameters:
    -----------
    predictions : dict
        Multi-scale predictions from FPN
        {
            'p2': {'prob': [B, H/4, W/4, 1], 'dist': [B, H/4, W/4, N]},
            'p3': {'prob': [B, H/8, W/8, 1], 'dist': [B, H/8, W/8, N]},
            ...
        }
    targets : dict
        Ground truth targets
        {
            'prob': [B, H, W, 1],  # Full resolution
            'dist': [B, H, W, N],
            'mask': [B, H, W, 1],  # Valid region mask
        }
    level_weights : dict, optional
        Weights for each FPN level
        Default: {'p2': 1.0, 'p3': 0.5, 'p4': 0.25, 'p5': 0.125}
        Higher weights for finer scales
    alpha : float
        Focal loss alpha parameter
    gamma : float
        Focal loss gamma parameter
    smooth_l1_delta : float
        Delta for smooth L1 loss
    
    Returns:
    --------
    total_loss : tf.Tensor
        Weighted sum of all level losses
    loss_dict : dict
        Individual losses for each level and component
    """
    
    if level_weights is None:
        # Default: emphasize high-resolution predictions
        level_weights = {
            'p2': 1.0,    # 1/4 resolution - highest weight
            'p3': 0.5,    # 1/8 resolution
            'p4': 0.25,   # 1/16 resolution
            'p5': 0.125,  # 1/32 resolution - lowest weight
        }
    
    total_loss = 0.0
    loss_dict = {}
    
    # Process each FPN level
    for level, weight in level_weights.items():
        if level not in predictions:
            continue
        
        pred_level = predictions[level]
        pred_prob = pred_level['prob']
        pred_dist = pred_level['dist']
        
        # Resize targets to match this level's resolution
        target_resized = _resize_targets(
            targets,
            target_shape=tf.shape(pred_prob)[1:3]
        )
        
        # Compute focal loss for probability
        loss_prob = _focal_loss(
            pred_prob,
            target_resized['prob'],
            alpha=alpha,
            gamma=gamma
        )
        
        # Compute smooth L1 loss for distance
        # Only compute where there are objects (prob > 0)
        mask = tf.cast(target_resized['prob'] > 0.5, tf.float32)
        loss_dist = _smooth_l1_loss(
            pred_dist,
            target_resized['dist'],
            mask=mask,
            delta=smooth_l1_delta
        )
        
        # Combine losses for this level
        level_loss = loss_prob + loss_dist
        
        # Weight and accumulate
        total_loss += weight * level_loss
        
        # Store individual losses
        loss_dict[f'{level}_prob'] = loss_prob
        loss_dict[f'{level}_dist'] = loss_dist
        loss_dict[f'{level}_total'] = level_loss
    
    loss_dict['total'] = total_loss
    
    return total_loss, loss_dict


def _resize_targets(targets, target_shape):
    """
    Resize ground truth targets to match prediction resolution
    
    Parameters:
    -----------
    targets : dict
        Original targets at full resolution
    target_shape : tf.TensorShape or tuple
        Target spatial dimensions [H, W]
    
    Returns:
    --------
    resized_targets : dict
        Targets resized to target_shape
    """
    resized = {}
    
    # Resize probability map (use nearest neighbor to preserve labels)
    resized['prob'] = tf.image.resize(
        targets['prob'],
        size=target_shape,
        method='nearest'
    )
    
    # Resize distance map (use bilinear for smooth interpolation)
    if 'dist' in targets and targets['dist'] is not None:
        resized['dist'] = tf.image.resize(
            targets['dist'],
            size=target_shape,
            method='bilinear'
        )
        
        # Scale distances proportionally to resolution change
        original_h = tf.cast(tf.shape(targets['dist'])[1], tf.float32)
        target_h = tf.cast(target_shape[0], tf.float32)
        scale_factor = target_h / original_h
        resized['dist'] = resized['dist'] * scale_factor
    
    # Resize mask if present
    if 'mask' in targets and targets['mask'] is not None:
        resized['mask'] = tf.image.resize(
            targets['mask'],
            size=target_shape,
            method='nearest'
        )
    
    return resized


def _focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        Predicted probabilities [B, H, W, 1]
    y_true : tf.Tensor
        Ground truth labels [B, H, W, 1]
    alpha : float
        Balancing factor for positive/negative samples
    gamma : float
        Focusing parameter (higher = more focus on hard examples)
    
    Returns:
    --------
    loss : tf.Tensor
        Scalar focal loss
    """
    # Clip predictions to prevent log(0)
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Compute focal loss
    # For positive samples (y_true = 1)
    pos_loss = -alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred) * y_true
    
    # For negative samples (y_true = 0)
    neg_loss = -(1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred) * (1 - y_true)
    
    # Combine
    loss = pos_loss + neg_loss
    
    # Average over all pixels
    return tf.reduce_mean(loss)


def _smooth_l1_loss(y_pred, y_true, mask=None, delta=1.0):
    """
    Smooth L1 Loss (Huber Loss)
    
    L(x) = 0.5 * x^2                  if |x| < delta
           delta * (|x| - 0.5 * delta) otherwise
    
    More robust to outliers than L2, smoother than L1.
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        Predicted distances [B, H, W, N]
    y_true : tf.Tensor
        Ground truth distances [B, H, W, N]
    mask : tf.Tensor, optional
        Binary mask indicating valid pixels [B, H, W, 1]
    delta : float
        Threshold for switching between L2 and L1
    
    Returns:
    --------
    loss : tf.Tensor
        Scalar smooth L1 loss
    """
    # Compute absolute difference
    diff = tf.abs(y_pred - y_true)
    
    # Smooth L1 formula
    loss = tf.where(
        diff < delta,
        0.5 * tf.square(diff),  # L2 for small errors
        delta * (diff - 0.5 * delta)  # L1 for large errors
    )
    
    # Apply mask if provided
    if mask is not None:
        # Expand mask to match distance dimensions
        mask_expanded = tf.expand_dims(mask, axis=-1)  # [B, H, W, 1, 1]
        mask_expanded = tf.tile(mask_expanded, [1, 1, 1, tf.shape(y_pred)[-1]])
        loss = loss * mask_expanded
        
        # Average only over valid pixels
        num_valid = tf.reduce_sum(mask_expanded) + 1e-7
        return tf.reduce_sum(loss) / num_valid
    else:
        return tf.reduce_mean(loss)


def combined_multiscale_loss(
    predictions,
    targets,
    level_weights=None,
    loss_weights=None,
    **kwargs
):
    """
    Combined multi-scale loss with multiple components
    
    Total Loss = w_focal * L_focal + w_dist * L_dist + w_shape * L_shape
    
    Parameters:
    -----------
    predictions : dict
        Multi-scale predictions from FPN
    targets : dict
        Ground truth targets
    level_weights : dict
        Weights for each FPN level
    loss_weights : dict
        Weights for different loss components
        Default: {'focal': 1.0, 'dist': 1.0, 'shape': 0.1}
    **kwargs : dict
        Additional parameters for individual losses
    
    Returns:
    --------
    total_loss : tf.Tensor
        Weighted combination of all losses
    loss_dict : dict
        Detailed breakdown of all loss components
    """
    
    if loss_weights is None:
        loss_weights = {
            'focal': 1.0,   # Probability loss
            'dist': 1.0,    # Distance regression
            'shape': 0.1,   # Shape consistency (optional)
        }
    
    # Compute main multi-scale loss
    ms_loss, ms_loss_dict = multiscale_loss(
        predictions,
        targets,
        level_weights=level_weights,
        **kwargs
    )
    
    # Separate focal and dist losses
    focal_loss = sum([v for k, v in ms_loss_dict.items() if 'prob' in k])
    dist_loss = sum([v for k, v in ms_loss_dict.items() if 'dist' in k])
    
    # Compute combined loss
    total_loss = (
        loss_weights['focal'] * focal_loss +
        loss_weights['dist'] * dist_loss
    )
    
    # Optional: add shape consistency loss
    if 'shape' in loss_weights and 'shape_consistency' in targets:
        shape_loss = _shape_consistency_loss(
            predictions,
            targets['shape_consistency']
        )
        total_loss += loss_weights['shape'] * shape_loss
        ms_loss_dict['shape'] = shape_loss
    
    ms_loss_dict['combined_total'] = total_loss
    
    return total_loss, ms_loss_dict


def _shape_consistency_loss(predictions, shape_targets):
    """
    Shape consistency loss across scales
    
    Encourages predictions at different scales to be consistent.
    
    Parameters:
    -----------
    predictions : dict
        Multi-scale predictions
    shape_targets : dict
        Shape consistency targets
    
    Returns:
    --------
    loss : tf.Tensor
        Shape consistency loss
    """
    # Simple implementation: L2 distance between neighboring scales
    loss = 0.0
    levels = ['p2', 'p3', 'p4', 'p5']
    
    for i in range(len(levels) - 1):
        level_fine = levels[i]
        level_coarse = levels[i + 1]
        
        if level_fine not in predictions or level_coarse not in predictions:
            continue
        
        # Downsample fine prediction to coarse resolution
        pred_fine = predictions[level_fine]['prob']
        pred_coarse = predictions[level_coarse]['prob']
        
        target_shape = tf.shape(pred_coarse)[1:3]
        pred_fine_down = tf.image.resize(pred_fine, target_shape, method='bilinear')
        
        # Compute L2 distance
        loss += tf.reduce_mean(tf.square(pred_fine_down - pred_coarse))
    
    return loss / (len(levels) - 1)


# ==============================================================================
# Helper function to create loss function for Keras model
# ==============================================================================

def create_multiscale_loss_fn(level_weights=None, loss_weights=None, **kwargs):
    """
    Create a loss function compatible with Keras model.compile()
    
    Usage:
        model.compile(
            optimizer='adam',
            loss=create_multiscale_loss_fn(
                level_weights={'p2': 1.0, 'p3': 0.5, 'p4': 0.25, 'p5': 0.125},
                loss_weights={'focal': 1.0, 'dist': 1.0}
            )
        )
    
    Parameters:
    -----------
    level_weights : dict
        Weights for each FPN level
    loss_weights : dict
        Weights for different loss components
    **kwargs : dict
        Additional parameters
    
    Returns:
    --------
    loss_fn : callable
        Loss function with signature: loss_fn(y_true, y_pred) -> loss
    """
    
    def loss_fn(y_true, y_pred):
        """
        Keras-compatible loss function
        
        Parameters:
        -----------
        y_true : dict or tf.Tensor
            Ground truth (converted to dict internally)
        y_pred : dict
            Model predictions (multiscale_predictions dict)
        
        Returns:
        --------
        loss : tf.Tensor
            Scalar loss value
        """
        # Convert y_true to dict format if necessary
        if not isinstance(y_true, dict):
            # Assume y_true is a batch of labels
            targets = {
                'prob': y_true[..., 0:1],  # First channel
                'dist': y_true[..., 1:],   # Remaining channels
            }
        else:
            targets = y_true
        
        # Compute loss
        total_loss, _ = combined_multiscale_loss(
            y_pred,
            targets,
            level_weights=level_weights,
            loss_weights=loss_weights,
            **kwargs
        )
        
        return total_loss
    
    return loss_fn


# ==============================================================================
# Testing and Debugging
# ==============================================================================

def test_multiscale_loss():
    """Test multi-scale loss computation"""
    print("=" * 60)
    print("Testing Multi-scale Loss")
    print("=" * 60)
    
    # Create dummy predictions (4 FPN levels)
    predictions = {
        'p2': {
            'prob': tf.random.uniform((2, 128, 128, 1)),
            'dist': tf.random.uniform((2, 128, 128, 32)),
        },
        'p3': {
            'prob': tf.random.uniform((2, 64, 64, 1)),
            'dist': tf.random.uniform((2, 64, 64, 32)),
        },
        'p4': {
            'prob': tf.random.uniform((2, 32, 32, 1)),
            'dist': tf.random.uniform((2, 32, 32, 32)),
        },
        'p5': {
            'prob': tf.random.uniform((2, 16, 16, 1)),
            'dist': tf.random.uniform((2, 16, 16, 32)),
        },
    }
    
    # Create dummy targets (full resolution)
    targets = {
        'prob': tf.random.uniform((2, 512, 512, 1)),
        'dist': tf.random.uniform((2, 512, 512, 32)),
    }
    
    # Compute loss
    total_loss, loss_dict = multiscale_loss(predictions, targets)
    
    print(f"\nTotal Loss: {total_loss.numpy():.4f}")
    print("\nIndividual Losses:")
    print("-" * 60)
    for key, value in sorted(loss_dict.items()):
        if isinstance(value, tf.Tensor):
            print(f"  {key:20s}: {value.numpy():.4f}")
    
    print("\n" + "=" * 60)
    print("Multi-scale Loss Test Complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_multiscale_loss()

