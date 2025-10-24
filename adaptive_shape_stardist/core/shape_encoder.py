"""
Adaptive Shape Encoder
=======================

Core module that integrates deformable convolutions and adaptive sampling
to create a flexible shape representation that adapts to irregular cell boundaries.

Key Features:
- Multi-scale feature extraction with deformable convolutions
- Adaptive point-based shape representation
- Dynamic feature aggregation
- Shape complexity estimation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .deformable_conv import DeformableConv2D, DeformableConvBlock
from .sampling import AdaptiveSampler, PolarPointEncoder


class AdaptiveShapeEncoder(layers.Layer):
    """
    Adaptive Shape Encoder
    
    Encodes cell shapes using adaptive sampling points and deformable convolutions.
    This replaces StarDist's fixed-ray representation with a flexible,
    learned representation.
    
    Architecture:
    -------------
    1. Multi-scale feature extraction with deformable convolutions
    2. Shape complexity estimation
    3. Adaptive sampling point generation
    4. Point feature extraction and aggregation
    5. Shape descriptor generation
    
    Parameters:
    -----------
    min_sampling_points : int
        Minimum number of sampling points per cell
    max_sampling_points : int
        Maximum number of sampling points per cell
    feature_dims : list of int
        Feature dimensions at each scale
    deformable_groups : int
        Number of deformable groups
    use_polar : bool
        Whether to use polar coordinate encoding
    """
    
    def __init__(
        self,
        min_sampling_points=32,
        max_sampling_points=128,
        feature_dims=[64, 128, 256],
        deformable_groups=4,
        use_polar=True,
        **kwargs
    ):
        super(AdaptiveShapeEncoder, self).__init__(**kwargs)
        
        self.min_sampling_points = min_sampling_points
        self.max_sampling_points = max_sampling_points
        self.feature_dims = feature_dims
        self.deformable_groups = deformable_groups
        self.use_polar = use_polar
        
    def build(self, input_shape):
        """Build encoder components"""
        
        # Multi-scale deformable convolution blocks
        self.deformable_blocks = []
        for i, dim in enumerate(self.feature_dims):
            block = DeformableConvBlock(
                filters=dim,
                kernel_size=3,
                use_residual=True,
                deformable_groups=self.deformable_groups,
                name=f'deform_block_{i}'
            )
            self.deformable_blocks.append(block)
        
        # Adaptive sampler
        self.sampler = AdaptiveSampler(
            min_points=self.min_sampling_points,
            max_points=self.max_sampling_points,
            point_dim=self.feature_dims[-1],
            use_attention=True,
            name='adaptive_sampler'
        )
        
        # Polar coordinate encoder (optional)
        if self.use_polar:
            self.polar_encoder = PolarPointEncoder(
                num_angle_bins=self.max_sampling_points,
                name='polar_encoder'
            )
        
        # Feature fusion layers
        self.feature_fusion = keras.Sequential([
            layers.Conv2D(self.feature_dims[-1], 1, activation='relu', name='fusion_conv1'),
            layers.Conv2D(self.feature_dims[-1], 3, padding='same', activation='relu', name='fusion_conv2'),
        ], name='feature_fusion')
        
        # Shape descriptor head
        self.shape_descriptor = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
        ], name='shape_descriptor')
        
        # Point-wise refinement network
        self.point_refiner = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(2),  # Output refined (x, y) or (angle, radius)
        ], name='point_refiner')
        
        super(AdaptiveShapeEncoder, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass of adaptive shape encoder
        
        Parameters:
        -----------
        inputs : tf.Tensor or dict
            If tensor: Input feature map [B, H, W, C]
            If dict: Must contain 'features' and optionally 'centers'
        training : bool
            Training mode flag
            
        Returns:
        --------
        output : dict
            Dictionary containing:
            - 'shape_features': Encoded shape features [B, H, W, C]
            - 'sampling_points': Adaptive sampling points [B, N, 2]
            - 'point_features': Features at sampling points [B, N, D]
            - 'shape_descriptor': Global shape descriptor [B, 128]
            - 'complexity': Shape complexity score [B, 1]
        """
        
        # Handle different input formats
        if isinstance(inputs, dict):
            features = inputs['features']
            centers = inputs.get('centers', None)
        else:
            features = inputs
            centers = None
        
        # Multi-scale feature extraction with deformable convolutions
        multi_scale_features = []
        x = features
        
        for block in self.deformable_blocks:
            x = block(x, training=training)
            multi_scale_features.append(x)
        
        # Fuse multi-scale features
        fused_features = self.feature_fusion(x, training=training)
        
        # Generate adaptive sampling points
        sampling_info = self.sampler(fused_features, training=training)
        
        sampling_points = sampling_info['points']  # [B, N, 2]
        point_features = sampling_info['features']  # [B, N, D]
        complexity = sampling_info['complexity']  # [B, 1]
        num_points = sampling_info['num_points']  # [B,]
        
        # Convert to polar coordinates if enabled
        if self.use_polar and centers is not None:
            polar_points = self.polar_encoder(sampling_points, centers)
        else:
            polar_points = sampling_points
        
        # Refine sampling points based on local features
        refined_points = self.point_refiner(point_features, training=training)
        final_points = sampling_points + refined_points  # Residual refinement
        
        # Generate global shape descriptor
        shape_desc = self.shape_descriptor(fused_features, training=training)
        
        return {
            'shape_features': fused_features,  # [B, H, W, C]
            'sampling_points': final_points,  # [B, N, 2]
            'polar_points': polar_points if self.use_polar else None,  # [B, N, 2]
            'point_features': point_features,  # [B, N, D]
            'shape_descriptor': shape_desc,  # [B, 128]
            'complexity': complexity,  # [B, 1]
            'num_points': num_points,  # [B,]
            'multi_scale_features': multi_scale_features,  # List of feature maps
        }
    
    def get_config(self):
        """Get configuration"""
        config = super(AdaptiveShapeEncoder, self).get_config()
        config.update({
            'min_sampling_points': self.min_sampling_points,
            'max_sampling_points': self.max_sampling_points,
            'feature_dims': self.feature_dims,
            'deformable_groups': self.deformable_groups,
            'use_polar': self.use_polar,
        })
        return config


class MultiScaleShapeAggregator(layers.Layer):
    """
    Aggregates shape information across multiple scales
    
    This helps capture both fine-grained boundary details and
    global shape context.
    """
    
    def __init__(self, output_dim=256, num_scales=3, **kwargs):
        super(MultiScaleShapeAggregator, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_scales = num_scales
    
    def build(self, input_shape):
        """Build aggregation components"""
        
        # Scale-specific projection layers
        self.scale_projections = []
        for i in range(self.num_scales):
            proj = layers.Conv2D(
                self.output_dim,
                kernel_size=1,
                activation='relu',
                name=f'scale_proj_{i}'
            )
            self.scale_projections.append(proj)
        
        # Attention weights for scale fusion
        self.scale_attention = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.num_scales, activation='softmax'),
        ], name='scale_attention')
        
        # Final fusion layer
        self.fusion = layers.Conv2D(
            self.output_dim,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='fusion'
        )
        
        super(MultiScaleShapeAggregator, self).build(input_shape)
    
    def call(self, multi_scale_features, training=None):
        """
        Aggregate multi-scale features
        
        Parameters:
        -----------
        multi_scale_features : list of tf.Tensor
            List of feature maps at different scales
        training : bool
            Training mode
            
        Returns:
        --------
        aggregated : tf.Tensor
            Aggregated feature map [B, H, W, output_dim]
        """
        # Get target spatial size from the finest scale
        target_size = tf.shape(multi_scale_features[0])[1:3]
        
        # Project and resize all scales to same size
        projected_features = []
        for i, (feat, proj) in enumerate(zip(multi_scale_features, self.scale_projections)):
            # Project to common dimension
            feat_proj = proj(feat, training=training)
            
            # Resize to target size if needed
            if tf.shape(feat_proj)[1] != target_size[0]:
                feat_proj = tf.image.resize(
                    feat_proj,
                    target_size,
                    method='bilinear'
                )
            
            projected_features.append(feat_proj)
        
        # Stack features
        stacked = tf.stack(projected_features, axis=-1)  # [B, H, W, C, num_scales]
        
        # Compute attention weights for each scale
        # Use the last (coarsest) scale for global context
        attn_weights = self.scale_attention(
            multi_scale_features[-1], 
            training=training
        )  # [B, num_scales]
        
        # Reshape for broadcasting
        attn_weights = tf.reshape(
            attn_weights, 
            [-1, 1, 1, 1, self.num_scales]
        )  # [B, 1, 1, 1, num_scales]
        
        # Weighted aggregation
        aggregated = tf.reduce_sum(stacked * attn_weights, axis=-1)  # [B, H, W, C]
        
        # Final fusion
        output = self.fusion(aggregated, training=training)
        
        return output
    
    def get_config(self):
        config = super(MultiScaleShapeAggregator, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'num_scales': self.num_scales,
        })
        return config

