"""
Multi-Scale StarDist Model Architecture
Implements a multi-scale version of StarDist that processes images at multiple scales
and combines the results using scale-aware fusion.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from csbdeep.utils import normalize_mi_ma
from stardist.models import StarDist2D
from .adaptive_model import AdaptiveStarDist2D, AdaptiveConfig2D

class MultiScaleConfig2D(AdaptiveConfig2D):
    """Configuration for multi-scale StarDist model."""
    def __init__(self,
                 scale_factors: List[float] = [1.0, 0.5, 0.25],
                 fusion_mode: str = 'weighted',
                 scale_weights: Optional[List[float]] = None,
                 attention_channels: int = 64,
                 **kwargs):
        """
        Initialize multi-scale configuration.
        
        Args:
            scale_factors: List of scales to process image at
            fusion_mode: How to combine multi-scale predictions ('weighted' or 'attention')
            scale_weights: Optional weights for each scale (if fusion_mode='weighted')
            attention_channels: Number of channels in attention module
            **kwargs: Additional arguments for AdaptiveConfig2D
        """
        super().__init__(**kwargs)
        self.scale_factors = scale_factors
        self.fusion_mode = fusion_mode
        self.scale_weights = scale_weights or [1.0] * len(scale_factors)
        self.attention_channels = attention_channels

class ScaleAttentionModule(tf.keras.layers.Layer):
    """Attention module for scale-aware feature fusion."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Attention layers
        self.conv1 = tf.keras.layers.Conv2D(channels, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(channels, 3, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(1, 1, padding='same')
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute attention weights.
        
        Args:
            x: Input features [batch, height, width, channels]
            
        Returns:
            Attention weights [batch, height, width, 1]
        """
        h = tf.nn.relu(self.conv1(x))
        h = tf.nn.relu(self.conv2(h))
        return tf.nn.sigmoid(self.conv3(h))

class MultiScaleStarDist2D(AdaptiveStarDist2D):
    """
    Multi-scale StarDist model that processes images at multiple scales.
    """
    
    def __init__(self, config, name=None, basedir=None):
        """
        Initialize the multi-scale model.
        
        Args:
            config: MultiScaleConfig2D configuration
            name: Model name
            basedir: Base directory for model files
        """
        super().__init__(config, name=name, basedir=basedir)
        
        # Create attention modules if using attention fusion
        if config.fusion_mode == 'attention':
            self.attention_modules = [
                ScaleAttentionModule(config.attention_channels)
                for _ in config.scale_factors
            ]
    
    def _process_single_scale(self, 
                            img: np.ndarray,
                            scale: float) -> Tuple[np.ndarray, Dict]:
        """
        Process image at a single scale.
        
        Args:
            img: Input image
            scale: Scale factor
            
        Returns:
            Tuple of (prediction, details)
        """
        # Resize image to target scale
        if scale != 1.0:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = tf.image.resize(img[None,...], (new_h, new_w))[0]
        else:
            scaled_img = img
            
        # Get predictions at this scale
        labels, details = AdaptiveStarDist2D._predict_instances(self, scaled_img)
        
        # Resize predictions back to original size if needed
        if scale != 1.0:
            labels = tf.image.resize(labels[None,...], (h, w),
                                   method='nearest')[0]
            
            # Resize probability and distance maps
            details['prob'] = tf.image.resize(details['prob'], (h, w))
            details['dist'] = tf.image.resize(details['dist'], (h, w))
            
        return labels, details
    
    def _fuse_predictions(self,
                         predictions: List[Tuple[np.ndarray, Dict]],
                         fusion_mode: str) -> Tuple[np.ndarray, Dict]:
        """
        Fuse predictions from multiple scales.
        
        Args:
            predictions: List of (labels, details) tuples from each scale
            fusion_mode: How to combine predictions ('weighted' or 'attention')
            
        Returns:
            Fused predictions and details
        """
        labels_list, details_list = zip(*predictions)
        
        if fusion_mode == 'weighted':
            # Weighted average of probability maps
            prob_maps = np.stack([d['prob'] for d in details_list])
            weights = np.array(self.config.scale_weights)[:, None, None, None]
            fused_prob = np.sum(prob_maps * weights, axis=0)
            
            # Use probability-weighted average for distance predictions
            dist_maps = np.stack([d['dist'] for d in details_list])
            fused_dist = np.sum(dist_maps * weights, axis=0)
            
        elif fusion_mode == 'attention':
            # Concatenate features for attention
            features = np.concatenate([d['prob'] for d in details_list], axis=-1)
            
            # Compute attention weights for each scale
            attention_weights = [
                module(features) for module in self.attention_modules
            ]
            attention_weights = tf.nn.softmax(tf.stack(attention_weights, axis=0), axis=0)
            
            # Apply attention weights
            prob_maps = np.stack([d['prob'] for d in details_list])
            dist_maps = np.stack([d['dist'] for d in details_list])
            
            fused_prob = np.sum(prob_maps * attention_weights, axis=0)
            fused_dist = np.sum(dist_maps * attention_weights, axis=0)
        
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
            
        # Generate final instance segmentation
        labels = self._predict_instances_from_maps(fused_prob, fused_dist)
        
        details = {
            'prob': fused_prob,
            'dist': fused_dist,
            'scale_predictions': predictions
        }
        
        return labels, details
    
    def _predict_instances(self, img: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Predict instances using multi-scale processing.
        
        Args:
            img: Input image
            
        Returns:
            Instance labels and details
        """
        # Process image at each scale
        predictions = []
        for scale in self.config.scale_factors:
            pred = self._process_single_scale(img, scale)
            predictions.append(pred)
            
        # Fuse predictions from different scales
        return self._fuse_predictions(predictions, self.config.fusion_mode)
    
    def _predict_instances_from_maps(self,
                                   prob: np.ndarray,
                                   dist: np.ndarray) -> np.ndarray:
        """
        Generate instance segmentation from probability and distance maps.
        
        Args:
            prob: Probability map
            dist: Distance map
            
        Returns:
            Instance segmentation labels
        """
        # Apply non-maximum suppression
        prob = tf.nn.relu(prob)  # Ensure non-negative
        dist = tf.nn.relu(dist)  # Ensure non-negative
        
        # Get points that are local maxima
        coords = tf.where(tf.nn.max_pool2d(
            prob[None,...], 3, 1, 'SAME')[0] == prob)
        
        # Filter by probability threshold
        points = coords[prob[coords[:, 0], coords[:, 1]] > self.thresholds['prob']]
        
        if len(points) == 0:
            return np.zeros(prob.shape[:2], np.uint16)
            
        # Get distances for these points
        points_dist = dist[points[:, 0], points[:, 1]]
        
        # Create polygons and render final labels
        labels = self._render_instances(points, points_dist, prob.shape[:2])
        
        return labels
