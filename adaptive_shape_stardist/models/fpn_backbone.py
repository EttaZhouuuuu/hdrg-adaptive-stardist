"""
FPN (Feature Pyramid Network) Backbone
========================================

Implements the standard Feature Pyramid Network for multi-scale feature extraction.

Key Components:
1. Bottom-up pathway: ResNet encoder for hierarchical feature extraction
2. Lateral connections: 1×1 convolutions to unify channel dimensions
3. Top-down pathway: Upsampling and element-wise addition for semantic propagation
4. Smooth convolutions: 3×3 convolutions to reduce upsampling artifacts

Reference:
- Feature Pyramid Networks for Object Detection (Lin et al., 2017)
- https://arxiv.org/abs/1612.03144

Author: Shape-Aware StarDist Team
Date: 2025-10-29
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class FPNBackbone(keras.Model):
    """
    Feature Pyramid Network (FPN) Backbone
    
    Extracts multi-scale features with semantic information propagated from
    deep layers to shallow layers via top-down pathway.
    
    Architecture:
    1. Bottom-up: ResNet encoder (C2, C3, C4, C5)
    2. Lateral: 1×1 conv to 256 channels (P2', P3', P4', P5')
    3. Top-down: P_i = P_i' + upsample(P_{i+1}), then 3×3 smooth
    
    Parameters:
    -----------
    n_channel_in : int
        Number of input channels (default: 1 for grayscale)
    backbone : str
        ResNet variant: 'resnet34' or 'resnet50' (default: 'resnet34')
    fpn_channels : int
        Number of channels in FPN layers (default: 256)
    pretrained : bool
        Whether to use pretrained ResNet weights (default: False)
    trainable_backbone : bool
        Whether backbone is trainable (default: True)
    
    Output:
    -------
    Dictionary with keys:
        'p2': Feature map at 1/4 resolution, 256 channels
        'p3': Feature map at 1/8 resolution, 256 channels
        'p4': Feature map at 1/16 resolution, 256 channels
        'p5': Feature map at 1/32 resolution, 256 channels
        'c_features': Original ResNet features (for debugging)
    """
    
    def __init__(
        self,
        n_channel_in=1,
        backbone='resnet34',
        fpn_channels=256,
        pretrained=False,
        trainable_backbone=True,
        **kwargs
    ):
        super(FPNBackbone, self).__init__(**kwargs)
        
        self.n_channel_in = n_channel_in
        self.backbone_name = backbone
        self.fpn_channels = fpn_channels
        self.pretrained = pretrained
        self.trainable_backbone = trainable_backbone
        
        # ResNet channel configuration
        if backbone == 'resnet34':
            self.resnet_channels = [64, 128, 256, 512]  # C2, C3, C4, C5
        elif backbone == 'resnet50':
            self.resnet_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet34' or 'resnet50'")
        
        # Build components
        self._build_bottom_up_pathway()
        self._build_lateral_connections()
        self._build_top_down_pathway()
    
    def _build_bottom_up_pathway(self):
        """
        Build ResNet encoder for bottom-up pathway
        
        Creates a ResNet that outputs features at 4 scales:
        - C2: 1/4 resolution
        - C3: 1/8 resolution
        - C4: 1/16 resolution
        - C5: 1/32 resolution
        """
        # Initial convolution (reduces to 1/2 resolution)
        self.initial_conv = keras.Sequential([
            layers.Conv2D(64, 7, strides=2, padding='same', name='conv1'),
            layers.BatchNormalization(name='bn_conv1'),
            layers.Activation('relu'),
        ], name='initial_conv')
        
        # Max pooling (reduces to 1/4 resolution) - start of C2
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool1')
        
        # Build ResNet stages
        if self.backbone_name == 'resnet34':
            # ResNet-34: [3, 4, 6, 3] blocks
            self.stage2 = self._make_resnet_stage(64, 64, 3, stride=1, name='stage2')    # C2
            self.stage3 = self._make_resnet_stage(64, 128, 4, stride=2, name='stage3')   # C3
            self.stage4 = self._make_resnet_stage(128, 256, 6, stride=2, name='stage4')  # C4
            self.stage5 = self._make_resnet_stage(256, 512, 3, stride=2, name='stage5')  # C5
        elif self.backbone_name == 'resnet50':
            # ResNet-50: [3, 4, 6, 3] bottleneck blocks
            self.stage2 = self._make_resnet_stage(64, 256, 3, stride=1, bottleneck=True, name='stage2')
            self.stage3 = self._make_resnet_stage(256, 512, 4, stride=2, bottleneck=True, name='stage3')
            self.stage4 = self._make_resnet_stage(512, 1024, 6, stride=2, bottleneck=True, name='stage4')
            self.stage5 = self._make_resnet_stage(1024, 2048, 3, stride=2, bottleneck=True, name='stage5')
    
    def _make_resnet_stage(self, in_channels, out_channels, num_blocks, stride=1, bottleneck=False, name='stage'):
        """
        Create a ResNet stage with multiple residual blocks
        
        Parameters:
        -----------
        in_channels : int
            Input channels
        out_channels : int
            Output channels
        num_blocks : int
            Number of residual blocks
        stride : int
            Stride for first block (for downsampling)
        bottleneck : bool
            Use bottleneck blocks (for ResNet-50+)
        name : str
            Stage name
        """
        blocks = []
        
        for i in range(num_blocks):
            # First block may have stride > 1 for downsampling
            block_stride = stride if i == 0 else 1
            block_in_channels = in_channels if i == 0 else out_channels
            
            if bottleneck:
                block = BottleneckBlock(
                    block_in_channels,
                    out_channels // 4,  # Bottleneck reduces channels
                    out_channels,
                    stride=block_stride,
                    name=f'{name}_block{i+1}'
                )
            else:
                block = BasicBlock(
                    block_in_channels,
                    out_channels,
                    stride=block_stride,
                    name=f'{name}_block{i+1}'
                )
            
            blocks.append(block)
        
        return keras.Sequential(blocks, name=name)
    
    def _build_lateral_connections(self):
        """
        Build 1×1 convolutions for lateral connections
        
        Purpose: Unify channel dimensions from ResNet features to FPN channels
        - C2 (64 or 256 channels) → P2' (256 channels)
        - C3 (128 or 512 channels) → P3' (256 channels)
        - C4 (256 or 1024 channels) → P4' (256 channels)
        - C5 (512 or 2048 channels) → P5' (256 channels)
        """
        self.lateral_c2 = layers.Conv2D(
            self.fpn_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            name='lateral_c2'
        )
        
        self.lateral_c3 = layers.Conv2D(
            self.fpn_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            name='lateral_c3'
        )
        
        self.lateral_c4 = layers.Conv2D(
            self.fpn_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            name='lateral_c4'
        )
        
        self.lateral_c5 = layers.Conv2D(
            self.fpn_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            name='lateral_c5'
        )
    
    def _build_top_down_pathway(self):
        """
        Build 3×3 convolutions for smoothing after top-down fusion
        
        Purpose: Remove upsampling artifacts and learn optimal fusion
        - Applied after P_i = P_i' + upsample(P_{i+1})
        - Uses 3×3 kernel for local smoothing
        """
        self.smooth_p2 = layers.Conv2D(
            self.fpn_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            name='smooth_p2'
        )
        
        self.smooth_p3 = layers.Conv2D(
            self.fpn_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            name='smooth_p3'
        )
        
        self.smooth_p4 = layers.Conv2D(
            self.fpn_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            name='smooth_p4'
        )
        
        # Note: P5 doesn't need smoothing (no deeper layer to fuse with)
    
    def call(self, inputs, training=None):
        """
        Forward pass of FPN
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input image [B, H, W, C]
        training : bool
            Training mode flag
        
        Returns:
        --------
        output : dict
            Dictionary containing:
            - 'p2': 1/4 resolution features (256 channels)
            - 'p3': 1/8 resolution features (256 channels)
            - 'p4': 1/16 resolution features (256 channels)
            - 'p5': 1/32 resolution features (256 channels)
            - 'c_features': Original ResNet features (for analysis)
        """
        
        # ============================================
        # Step 1: Bottom-up Pathway (ResNet Encoder)
        # ============================================
        
        # Initial conv: H×W×C → H/2×W/2×64
        x = self.initial_conv(inputs, training=training)
        
        # MaxPool: H/2×W/2×64 → H/4×W/4×64 (start of C2)
        x = self.maxpool(x)
        
        # Stage 2 (C2): H/4×W/4 (no stride)
        c2 = self.stage2(x, training=training)
        
        # Stage 3 (C3): H/4×W/4 → H/8×W/8 (stride 2)
        c3 = self.stage3(c2, training=training)
        
        # Stage 4 (C4): H/8×W/8 → H/16×W/16 (stride 2)
        c4 = self.stage4(c3, training=training)
        
        # Stage 5 (C5): H/16×W/16 → H/32×W/32 (stride 2)
        c5 = self.stage5(c4, training=training)
        
        # ============================================
        # Step 2: Lateral Connections (1×1 Conv)
        # ============================================
        
        # Unify all features to fpn_channels (256)
        p5_lateral = self.lateral_c5(c5, training=training)  # H/32×W/32×256
        p4_lateral = self.lateral_c4(c4, training=training)  # H/16×W/16×256
        p3_lateral = self.lateral_c3(c3, training=training)  # H/8×W/8×256
        p2_lateral = self.lateral_c2(c2, training=training)  # H/4×W/4×256
        
        # ============================================
        # Step 3: Top-down Pathway with Element-wise Addition
        # ============================================
        
        # Initialize with deepest layer (no upsampling needed)
        p5 = p5_lateral  # H/32×W/32×256
        
        # P5 → P4 fusion
        # Upsample P5 to match P4 spatial dimensions
        p5_upsampled = self._upsample_add(p5, p4_lateral, training=training)
        # Add lateral connection
        p4_fused = p4_lateral + p5_upsampled
        # Smooth with 3×3 conv
        p4 = self.smooth_p4(p4_fused, training=training)  # H/16×W/16×256
        
        # P4 → P3 fusion
        p4_upsampled = self._upsample_add(p4, p3_lateral, training=training)
        p3_fused = p3_lateral + p4_upsampled
        p3 = self.smooth_p3(p3_fused, training=training)  # H/8×W/8×256
        
        # P3 → P2 fusion
        p3_upsampled = self._upsample_add(p3, p2_lateral, training=training)
        p2_fused = p2_lateral + p3_upsampled
        p2 = self.smooth_p2(p2_fused, training=training)  # H/4×W/4×256
        
        # ============================================
        # Return FPN Features
        # ============================================
        
        return {
            'p2': p2,  # 1/4 resolution, 256 channels (highest resolution)
            'p3': p3,  # 1/8 resolution, 256 channels
            'p4': p4,  # 1/16 resolution, 256 channels
            'p5': p5,  # 1/32 resolution, 256 channels (lowest resolution)
            'c_features': {  # Original ResNet features (for debugging/analysis)
                'c2': c2,
                'c3': c3,
                'c4': c4,
                'c5': c5,
            }
        }
    
    def _upsample_add(self, feature_deep, feature_shallow, training=None):
        """
        Upsample deep features to match shallow feature dimensions
        
        Uses nearest neighbor interpolation (fast and no learned params)
        
        Parameters:
        -----------
        feature_deep : tf.Tensor
            Features from deeper layer (lower resolution)
        feature_shallow : tf.Tensor
            Features from shallower layer (higher resolution, target size)
        
        Returns:
        --------
        upsampled : tf.Tensor
            Upsampled features matching shallow feature size
        """
        # Get target spatial dimensions from shallow features
        target_shape = tf.shape(feature_shallow)[1:3]
        
        # Upsample using nearest neighbor (preserves values, fast)
        upsampled = tf.image.resize(
            feature_deep,
            size=target_shape,
            method='nearest'  # Nearest neighbor as per FPN paper
        )
        
        return upsampled
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super(FPNBackbone, self).get_config()
        config.update({
            'n_channel_in': self.n_channel_in,
            'backbone': self.backbone_name,
            'fpn_channels': self.fpn_channels,
            'pretrained': self.pretrained,
            'trainable_backbone': self.trainable_backbone,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration"""
        return cls(**config)


class BasicBlock(layers.Layer):
    """
    Basic ResNet block (for ResNet-18, ResNet-34)
    
    Architecture:
        x -> Conv(3×3) -> BN -> ReLU -> Conv(3×3) -> BN -> (+) -> ReLU
        |                                                    ^
        |-----> (optional downsample) ----------------------|
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int
        Stride for first convolution (for downsampling)
    name : str
        Block name
    """
    
    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path
        self.conv1 = layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        
        self.conv2 = layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        
        # Skip connection (identity or projection)
        if stride != 1 or in_channels != out_channels:
            # Need to project/downsample skip connection
            self.downsample = keras.Sequential([
                layers.Conv2D(
                    out_channels,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False
                ),
                layers.BatchNormalization(),
            ])
        else:
            self.downsample = None
        
        self.relu2 = layers.Activation('relu')
    
    def call(self, inputs, training=None):
        """Forward pass with residual connection"""
        identity = inputs
        
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(identity, training=training)
        
        # Add residual
        x = x + identity
        x = self.relu2(x)
        
        return x
    
    def get_config(self):
        config = super(BasicBlock, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'stride': self.stride,
        })
        return config


class BottleneckBlock(layers.Layer):
    """
    Bottleneck ResNet block (for ResNet-50, ResNet-101, ResNet-152)
    
    Architecture:
        x -> Conv(1×1) -> BN -> ReLU -> Conv(3×3) -> BN -> ReLU -> Conv(1×1) -> BN -> (+) -> ReLU
        |                                                                          ^
        |-----> (optional downsample) ---------------------------------------------|
    
    The 1×1 convolutions reduce and then restore dimensions, creating a bottleneck.
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    bottleneck_channels : int
        Number of channels in the bottleneck (middle layer)
    out_channels : int
        Number of output channels
    stride : int
        Stride for 3×3 convolution (for downsampling)
    name : str
        Block name
    """
    
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path: 1×1 -> 3×3 -> 1×1
        self.conv1 = layers.Conv2D(
            bottleneck_channels,
            kernel_size=1,
            strides=1,
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        
        self.conv2 = layers.Conv2D(
            bottleneck_channels,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        
        self.conv3 = layers.Conv2D(
            out_channels,
            kernel_size=1,
            strides=1,
            use_bias=False
        )
        self.bn3 = layers.BatchNormalization()
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.downsample = keras.Sequential([
                layers.Conv2D(
                    out_channels,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False
                ),
                layers.BatchNormalization(),
            ])
        else:
            self.downsample = None
        
        self.relu3 = layers.Activation('relu')
    
    def call(self, inputs, training=None):
        """Forward pass with residual connection"""
        identity = inputs
        
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(identity, training=training)
        
        # Add residual
        x = x + identity
        x = self.relu3(x)
        
        return x
    
    def get_config(self):
        config = super(BottleneckBlock, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'bottleneck_channels': self.bottleneck_channels,
            'out_channels': self.out_channels,
            'stride': self.stride,
        })
        return config


# ============================================
# Helper Functions for Testing and Debugging
# ============================================

def test_fpn_backbone():
    """
    Test FPN backbone with dummy input
    
    Verifies:
    1. Output shapes are correct
    2. Channel dimensions are unified to 256
    3. Spatial dimensions follow 1/4, 1/8, 1/16, 1/32 pattern
    4. Gradient flow is healthy
    """
    print("=" * 60)
    print("Testing FPN Backbone")
    print("=" * 60)
    
    # Create model
    fpn = FPNBackbone(
        n_channel_in=1,
        backbone='resnet34',
        fpn_channels=256
    )
    
    # Dummy input: batch_size=2, 512×512×1
    x = tf.random.normal((2, 512, 512, 1))
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = fpn(x, training=False)
    
    # Check outputs
    print("\n" + "-" * 60)
    print("FPN Output Shapes:")
    print("-" * 60)
    for key in ['p2', 'p3', 'p4', 'p5']:
        shape = output[key].shape
        print(f"{key}: {shape} (expected: [2, H/{2**(int(key[1])+1)}, W/{2**(int(key[1])+1)}, 256])")
    
    # Expected shapes
    expected = {
        'p2': (2, 128, 128, 256),  # 1/4
        'p3': (2, 64, 64, 256),    # 1/8
        'p4': (2, 32, 32, 256),    # 1/16
        'p5': (2, 16, 16, 256),    # 1/32
    }
    
    # Validate
    all_correct = True
    for key, exp_shape in expected.items():
        actual_shape = tuple(output[key].shape.as_list())
        if actual_shape != exp_shape:
            print(f"❌ {key} shape mismatch! Expected {exp_shape}, got {actual_shape}")
            all_correct = False
    
    if all_correct:
        print("\n✅ All output shapes are correct!")
    
    # Test gradient flow
    print("\n" + "-" * 60)
    print("Testing Gradient Flow:")
    print("-" * 60)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        output = fpn(x, training=True)
        # Dummy loss: sum of all outputs
        loss = sum([tf.reduce_sum(output[k]) for k in ['p2', 'p3', 'p4', 'p5']])
    
    # Compute gradients
    grads = tape.gradient(loss, fpn.trainable_variables)
    
    # Check for None gradients
    none_grads = sum([1 for g in grads if g is None])
    if none_grads > 0:
        print(f"⚠️  Warning: {none_grads} variables have None gradients")
    else:
        print("✅ All variables have valid gradients")
    
    # Check gradient magnitudes
    grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]
    print(f"   Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    
    print("\n" + "=" * 60)
    print("FPN Backbone Test Complete!")
    print("=" * 60)
    
    return fpn, output


if __name__ == '__main__':
    # Run test
    fpn_model, fpn_output = test_fpn_backbone()
    
    # Print model summary
    print("\n" + "=" * 60)
    print("Model Summary:")
    print("=" * 60)
    fpn_model.summary()

