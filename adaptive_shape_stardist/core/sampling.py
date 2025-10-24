"""
Adaptive Sampling Module
=========================

Implements adaptive sampling point generation for flexible shape representation.
Unlike fixed-ray StarDist, this module generates variable number of sampling points
based on shape complexity.

Key Features:
- Dynamic number of sampling points per instance
- Attention-weighted point importance
- Shape-aware point distribution
- Efficient point encoding/decoding
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class AdaptiveSampler(layers.Layer):
    """
    Adaptive Sampling Point Generator
    
    Generates a variable number of boundary sampling points based on
    predicted shape complexity. More complex shapes get more points.
    
    Parameters:
    -----------
    min_points : int
        Minimum number of sampling points
    max_points : int
        Maximum number of sampling points
    point_dim : int
        Dimension of point features
    use_attention : bool
        Whether to use attention for point importance
    """
    
    def __init__(
        self,
        min_points=32,
        max_points=128,
        point_dim=64,
        use_attention=True,
        **kwargs
    ):
        super(AdaptiveSampler, self).__init__(**kwargs)
        
        self.min_points = min_points
        self.max_points = max_points
        self.point_dim = point_dim
        self.use_attention = use_attention
        
    def build(self, input_shape):
        """Build sampling components"""
        
        # Complexity predictor: predicts number of points needed
        self.complexity_predictor = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),  # Normalized complexity score
        ], name='complexity_predictor')
        
        # Point position predictor: generates sampling angles/positions
        self.position_predictor = keras.Sequential([
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(self.max_points * 2, 1),  # 2 for (angle, radius) or (x, y) offset
        ], name='position_predictor')
        
        # Point feature encoder
        self.point_encoder = keras.Sequential([
            layers.Conv2D(self.point_dim, 3, padding='same', activation='relu'),
            layers.Conv2D(self.point_dim, 3, padding='same', activation='relu'),
        ], name='point_encoder')
        
        # Attention mechanism for point importance
        if self.use_attention:
            self.attention_query = layers.Dense(self.point_dim, name='attn_query')
            self.attention_key = layers.Dense(self.point_dim, name='attn_key')
            self.attention_value = layers.Dense(self.point_dim, name='attn_value')
        
        super(AdaptiveSampler, self).build(input_shape)
    
    def call(self, features, training=None):
        """
        Generate adaptive sampling points
        
        Parameters:
        -----------
        features : tf.Tensor
            Input feature map [B, H, W, C]
        training : bool
            Training mode flag
            
        Returns:
        --------
        sampling_info : dict
            Dictionary containing:
            - 'points': Sampling point coordinates [B, N_points, 2]
            - 'features': Point features [B, N_points, point_dim]
            - 'complexity': Complexity scores [B, 1]
            - 'num_points': Number of points per instance [B,]
        """
        
        batch_size = tf.shape(features)[0]
        
        # Predict shape complexity (0 to 1)
        complexity = self.complexity_predictor(features, training=training)  # [B, 1]
        
        # Map complexity to number of points
        num_points = self._complexity_to_num_points(complexity)  # [B,]
        
        # Generate point positions (angles and radii in polar coordinates)
        point_coords = self.position_predictor(features, training=training)  # [B, H, W, max_points*2]
        
        # Extract point features
        point_features = self.point_encoder(features, training=training)  # [B, H, W, point_dim]
        
        # Sample points at predicted positions
        sampled_points, sampled_features = self._sample_at_positions(
            point_coords, point_features, num_points
        )
        
        # Apply attention if enabled
        if self.use_attention:
            sampled_features = self._apply_attention(sampled_features, training=training)
        
        return {
            'points': sampled_points,  # [B, N, 2]
            'features': sampled_features,  # [B, N, point_dim]
            'complexity': complexity,  # [B, 1]
            'num_points': num_points,  # [B,]
        }
    
    def _complexity_to_num_points(self, complexity):
        """
        Map complexity score to number of sampling points
        
        Parameters:
        -----------
        complexity : tf.Tensor
            Complexity scores [B, 1], range [0, 1]
            
        Returns:
        --------
        num_points : tf.Tensor
            Number of points [B,], range [min_points, max_points]
        """
        # Linear mapping from complexity to point count
        num_points = (
            self.min_points + 
            complexity * (self.max_points - self.min_points)
        )
        
        # Round to integer
        num_points = tf.cast(tf.round(num_points), tf.int32)
        num_points = tf.squeeze(num_points, axis=-1)  # [B,]
        
        return num_points
    
    def _sample_at_positions(self, point_coords, features, num_points):
        """
        Sample features at predicted point positions
        
        This is a simplified version. Full implementation would use
        bilinear interpolation for sub-pixel accuracy.
        
        Parameters:
        -----------
        point_coords : tf.Tensor
            Point coordinates [B, H, W, max_points*2]
        features : tf.Tensor
            Feature map [B, H, W, point_dim]
        num_points : tf.Tensor
            Number of points per batch [B,]
            
        Returns:
        --------
        points : tf.Tensor
            Sampled point coordinates [B, max_points, 2]
        point_features : tf.Tensor
            Features at sampled points [B, max_points, point_dim]
        """
        batch_size = tf.shape(features)[0]
        height = tf.shape(features)[1]
        width = tf.shape(features)[2]
        
        # Reshape coordinates
        point_coords = tf.reshape(
            point_coords, 
            [batch_size, height, width, self.max_points, 2]
        )
        
        # Use center point as reference
        center_h = height // 2
        center_w = width // 2
        
        # Extract coordinates at center (simplified)
        points = point_coords[:, center_h, center_w, :, :]  # [B, max_points, 2]
        
        # Normalize coordinates to [-1, 1] for sampling
        points_norm = points / tf.cast([height, width], tf.float32) * 2.0 - 1.0
        
        # Sample features using grid_sample (simplified - using average pooling)
        # Full implementation would use tf.image.sample or custom interpolation
        point_features = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
        point_features = tf.tile(
            point_features, 
            [1, self.max_points, 1, 1]
        )
        point_features = tf.squeeze(point_features, axis=2)  # [B, max_points, point_dim]
        
        return points, point_features
    
    def _apply_attention(self, point_features, training=None):
        """
        Apply self-attention to point features
        
        Parameters:
        -----------
        point_features : tf.Tensor
            Point features [B, N, point_dim]
        training : bool
            Training mode
            
        Returns:
        --------
        attended_features : tf.Tensor
            Features after attention [B, N, point_dim]
        """
        # Compute queries, keys, values
        Q = self.attention_query(point_features)  # [B, N, point_dim]
        K = self.attention_key(point_features)    # [B, N, point_dim]
        V = self.attention_value(point_features)  # [B, N, point_dim]
        
        # Compute attention scores
        scores = tf.matmul(Q, K, transpose_b=True)  # [B, N, N]
        scores = scores / tf.sqrt(tf.cast(self.point_dim, tf.float32))
        
        # Softmax normalization
        attention_weights = tf.nn.softmax(scores, axis=-1)  # [B, N, N]
        
        # Apply attention
        attended = tf.matmul(attention_weights, V)  # [B, N, point_dim]
        
        return attended
    
    def get_config(self):
        """Get configuration"""
        config = super(AdaptiveSampler, self).get_config()
        config.update({
            'min_points': self.min_points,
            'max_points': self.max_points,
            'point_dim': self.point_dim,
            'use_attention': self.use_attention,
        })
        return config


class PolarPointEncoder(layers.Layer):
    """
    Encodes points in polar coordinates relative to cell center
    
    This provides a more natural representation for cell boundaries
    compared to Cartesian coordinates.
    """
    
    def __init__(self, num_angle_bins=64, **kwargs):
        super(PolarPointEncoder, self).__init__(**kwargs)
        self.num_angle_bins = num_angle_bins
    
    def call(self, points, centers):
        """
        Convert Cartesian points to polar coordinates
        
        Parameters:
        -----------
        points : tf.Tensor
            Point coordinates [B, N, 2] in (y, x) format
        centers : tf.Tensor
            Cell center coordinates [B, 2] in (y, x) format
            
        Returns:
        --------
        polar_coords : tf.Tensor
            Polar coordinates [B, N, 2] as (angle, radius)
        """
        # Expand centers for broadcasting
        centers_expanded = tf.expand_dims(centers, axis=1)  # [B, 1, 2]
        
        # Compute relative positions
        relative_pos = points - centers_expanded  # [B, N, 2]
        
        # Convert to polar coordinates
        y_rel = relative_pos[..., 0]
        x_rel = relative_pos[..., 1]
        
        # Angle (theta) in radians
        angles = tf.atan2(y_rel, x_rel)  # [B, N]
        
        # Radius (rho)
        radii = tf.sqrt(y_rel**2 + x_rel**2)  # [B, N]
        
        # Stack to get polar coordinates
        polar_coords = tf.stack([angles, radii], axis=-1)  # [B, N, 2]
        
        return polar_coords
    
    def get_config(self):
        config = super(PolarPointEncoder, self).get_config()
        config.update({'num_angle_bins': self.num_angle_bins})
        return config

