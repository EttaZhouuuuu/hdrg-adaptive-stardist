"""
FPN (Feature Pyramid Network) Backbone - PyTorch Implementation
================================================================

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class FPNBackbone(nn.Module):
    """
    Feature Pyramid Network (FPN) Backbone - PyTorch Version
    
    Extracts multi-scale features with semantic information propagated from
    deep layers to shallow layers via top-down pathway.
    
    Architecture:
    1. Bottom-up: ResNet encoder (C2, C3, C4, C5)
    2. Lateral: 1×1 conv to 256 channels (P2', P3', P4', P5')
    3. Top-down: P_i = P_i' + upsample(P_{i+1}), then 3×3 smooth
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        backbone (str): ResNet variant: 'resnet34' or 'resnet50' (default: 'resnet34')
        fpn_channels (int): Number of channels in FPN layers (default: 256)
        pretrained (bool): Whether to use pretrained ResNet weights (default: False)
        trainable_backbone (bool): Whether backbone is trainable (default: True)
    
    Returns:
        Dictionary with keys:
            'p2': Feature map at 1/4 resolution, 256 channels
            'p3': Feature map at 1/8 resolution, 256 channels
            'p4': Feature map at 1/16 resolution, 256 channels
            'p5': Feature map at 1/32 resolution, 256 channels
            'c_features': Original ResNet features (for debugging)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        backbone: str = 'resnet34',
        fpn_channels: int = 256,
        pretrained: bool = False,
        trainable_backbone: bool = True
    ):
        super(FPNBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.backbone_name = backbone
        self.fpn_channels = fpn_channels
        self.pretrained = pretrained
        self.trainable_backbone = trainable_backbone
        
        # ResNet channel configuration
        if backbone == 'resnet34':
            self.resnet_channels = [64, 128, 256, 512]  # C2, C3, C4, C5
            self.num_blocks = [3, 4, 6, 3]  # Number of blocks per stage
            self.block_type = BasicBlock
        elif backbone == 'resnet50':
            self.resnet_channels = [256, 512, 1024, 2048]
            self.num_blocks = [3, 4, 6, 3]
            self.block_type = BottleneckBlock
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet34' or 'resnet50'")
        
        # Build components
        self._build_bottom_up_pathway()
        self._build_lateral_connections()
        self._build_top_down_pathway()
        
        # Initialize weights
        self._initialize_weights()
    
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
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling (reduces to 1/4 resolution) - start of C2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build ResNet stages
        if self.backbone_name == 'resnet34':
            # ResNet-34: [3, 4, 6, 3] basic blocks
            self.stage2 = self._make_resnet_stage(64, 64, self.num_blocks[0], stride=1)    # C2
            self.stage3 = self._make_resnet_stage(64, 128, self.num_blocks[1], stride=2)   # C3
            self.stage4 = self._make_resnet_stage(128, 256, self.num_blocks[2], stride=2)  # C4
            self.stage5 = self._make_resnet_stage(256, 512, self.num_blocks[3], stride=2)  # C5
        elif self.backbone_name == 'resnet50':
            # ResNet-50: [3, 4, 6, 3] bottleneck blocks
            self.stage2 = self._make_resnet_stage(64, 256, self.num_blocks[0], stride=1, bottleneck=True)
            self.stage3 = self._make_resnet_stage(256, 512, self.num_blocks[1], stride=2, bottleneck=True)
            self.stage4 = self._make_resnet_stage(512, 1024, self.num_blocks[2], stride=2, bottleneck=True)
            self.stage5 = self._make_resnet_stage(1024, 2048, self.num_blocks[3], stride=2, bottleneck=True)
    
    def _make_resnet_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        bottleneck: bool = False
    ) -> nn.Sequential:
        """
        Create a ResNet stage with multiple residual blocks
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of residual blocks
            stride: Stride for first block (for downsampling)
            bottleneck: Use bottleneck blocks (for ResNet-50+)
        
        Returns:
            Sequential module containing all blocks
        """
        layers = []
        
        # First block may have stride > 1 for downsampling
        if bottleneck:
            layers.append(BottleneckBlock(in_channels, out_channels // 4, out_channels, stride))
            # Subsequent blocks
            for _ in range(1, num_blocks):
                layers.append(BottleneckBlock(out_channels, out_channels // 4, out_channels, stride=1))
        else:
            layers.append(BasicBlock(in_channels, out_channels, stride))
            # Subsequent blocks
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _build_lateral_connections(self):
        """
        Build 1×1 convolutions for lateral connections
        
        Purpose: Unify channel dimensions from ResNet features to FPN channels
        - C2 (64 or 256 channels) → P2' (256 channels)
        - C3 (128 or 512 channels) → P3' (256 channels)
        - C4 (256 or 1024 channels) → P4' (256 channels)
        - C5 (512 or 2048 channels) → P5' (256 channels)
        """
        self.lateral_c2 = nn.Conv2d(
            self.resnet_channels[0],
            self.fpn_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.lateral_c3 = nn.Conv2d(
            self.resnet_channels[1],
            self.fpn_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.lateral_c4 = nn.Conv2d(
            self.resnet_channels[2],
            self.fpn_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.lateral_c5 = nn.Conv2d(
            self.resnet_channels[3],
            self.fpn_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
    
    def _build_top_down_pathway(self):
        """
        Build 3×3 convolutions for smoothing after top-down fusion
        
        Purpose: Remove upsampling artifacts and learn optimal fusion
        - Applied after P_i = P_i' + upsample(P_{i+1})
        - Uses 3×3 kernel for local smoothing
        """
        self.smooth_p2 = nn.Conv2d(
            self.fpn_channels,
            self.fpn_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.smooth_p3 = nn.Conv2d(
            self.fpn_channels,
            self.fpn_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.smooth_p4 = nn.Conv2d(
            self.fpn_channels,
            self.fpn_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Note: P5 doesn't need smoothing (no deeper layer to fuse with)
    
    def _initialize_weights(self):
        """Initialize weights for lateral and smooth convolutions"""
        # Lateral connections
        for m in [self.lateral_c2, self.lateral_c3, self.lateral_c4, self.lateral_c5]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # Smooth convolutions
        for m in [self.smooth_p2, self.smooth_p3, self.smooth_p4]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of FPN
        
        Args:
            x: Input image [B, C, H, W]
        
        Returns:
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
        x = self.initial_conv(x)
        
        # MaxPool: H/2×W/2×64 → H/4×W/4×64 (start of C2)
        x = self.maxpool(x)
        
        # Stage 2 (C2): H/4×W/4 (no stride)
        c2 = self.stage2(x)
        
        # Stage 3 (C3): H/4×W/4 → H/8×W/8 (stride 2)
        c3 = self.stage3(c2)
        
        # Stage 4 (C4): H/8×W/8 → H/16×W/16 (stride 2)
        c4 = self.stage4(c3)
        
        # Stage 5 (C5): H/16×W/16 → H/32×W/32 (stride 2)
        c5 = self.stage5(c4)
        
        # ============================================
        # Step 2: Lateral Connections (1×1 Conv)
        # ============================================
        
        # Unify all features to fpn_channels (256)
        p5_lateral = self.lateral_c5(c5)  # H/32×W/32×256
        p4_lateral = self.lateral_c4(c4)  # H/16×W/16×256
        p3_lateral = self.lateral_c3(c3)  # H/8×W/8×256
        p2_lateral = self.lateral_c2(c2)  # H/4×W/4×256
        
        # ============================================
        # Step 3: Top-down Pathway with Element-wise Addition
        # ============================================
        
        # Initialize with deepest layer (no upsampling needed)
        p5 = p5_lateral  # H/32×W/32×256
        
        # P5 → P4 fusion
        # Upsample P5 to match P4 spatial dimensions
        p5_upsampled = self._upsample_add(p5, p4_lateral)
        # Add lateral connection
        p4_fused = p4_lateral + p5_upsampled
        # Smooth with 3×3 conv
        p4 = self.smooth_p4(p4_fused)  # H/16×W/16×256
        
        # P4 → P3 fusion
        p4_upsampled = self._upsample_add(p4, p3_lateral)
        p3_fused = p3_lateral + p4_upsampled
        p3 = self.smooth_p3(p3_fused)  # H/8×W/8×256
        
        # P3 → P2 fusion
        p3_upsampled = self._upsample_add(p3, p2_lateral)
        p2_fused = p2_lateral + p3_upsampled
        p2 = self.smooth_p2(p2_fused)  # H/4×W/4×256
        
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
    
    def _upsample_add(
        self,
        feature_deep: torch.Tensor,
        feature_shallow: torch.Tensor
    ) -> torch.Tensor:
        """
        Upsample deep features to match shallow feature dimensions
        
        Uses nearest neighbor interpolation (fast and no learned params)
        
        Args:
            feature_deep: Features from deeper layer (lower resolution)
            feature_shallow: Features from shallower layer (higher resolution, target size)
        
        Returns:
            Upsampled features matching shallow feature size
        """
        # Get target spatial dimensions from shallow features
        _, _, H, W = feature_shallow.shape
        
        # Upsample using nearest neighbor (preserves values, fast)
        upsampled = F.interpolate(
            feature_deep,
            size=(H, W),
            mode='nearest'  # Nearest neighbor as per FPN paper
        )
        
        return upsampled


class BasicBlock(nn.Module):
    """
    Basic ResNet block (for ResNet-18, ResNet-34)
    
    Architecture:
        x -> Conv(3×3) -> BN -> ReLU -> Conv(3×3) -> BN -> (+) -> ReLU
        |                                                    ^
        |-----> (optional downsample) ----------------------|
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution (for downsampling)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
        if stride != 1 or in_channels != out_channels:
            # Need to project/downsample skip connection
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck ResNet block (for ResNet-50, ResNet-101, ResNet-152)
    
    Architecture:
        x -> Conv(1×1) -> BN -> ReLU -> Conv(3×3) -> BN -> ReLU -> Conv(1×1) -> BN -> (+) -> ReLU
        |                                                                          ^
        |-----> (optional downsample) ---------------------------------------------|
    
    The 1×1 convolutions reduce and then restore dimensions, creating a bottleneck.
    
    Args:
        in_channels: Number of input channels
        bottleneck_channels: Number of channels in the bottleneck (middle layer)
        out_channels: Number of output channels
        stride: Stride for 3×3 convolution (for downsampling)
    """
    
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super(BottleneckBlock, self).__init__()
        
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path: 1×1 -> 3×3 -> 1×1
        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual
        out += identity
        out = self.relu(out)
        
        return out


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
    print("Testing FPN Backbone (PyTorch)")
    print("=" * 60)
    
    # Create model
    fpn = FPNBackbone(
        in_channels=1,
        backbone='resnet34',
        fpn_channels=256
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fpn = fpn.to(device)
    print(f"\nDevice: {device}")
    
    # Dummy input: batch_size=2, 1×512×512
    x = torch.randn(2, 1, 512, 512).to(device)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = fpn(x)
    
    # Check outputs
    print("\n" + "-" * 60)
    print("FPN Output Shapes:")
    print("-" * 60)
    for key in ['p2', 'p3', 'p4', 'p5']:
        shape = tuple(output[key].shape)
        resolution = 2 ** (int(key[1]) + 1)
        print(f"{key}: {shape} (expected: [2, 256, {512//resolution}, {512//resolution}])")
    
    # Expected shapes
    expected = {
        'p2': (2, 256, 128, 128),  # 1/4
        'p3': (2, 256, 64, 64),    # 1/8
        'p4': (2, 256, 32, 32),    # 1/16
        'p5': (2, 256, 16, 16),    # 1/32
    }
    
    # Validate
    all_correct = True
    for key, exp_shape in expected.items():
        actual_shape = tuple(output[key].shape)
        if actual_shape != exp_shape:
            print(f"❌ {key} shape mismatch! Expected {exp_shape}, got {actual_shape}")
            all_correct = False
    
    if all_correct:
        print("\n✅ All output shapes are correct!")
    
    # Test gradient flow
    print("\n" + "-" * 60)
    print("Testing Gradient Flow:")
    print("-" * 60)
    
    # Enable gradients
    x_grad = torch.randn(2, 1, 512, 512, requires_grad=True).to(device)
    output = fpn(x_grad)
    
    # Dummy loss: sum of all outputs
    loss = sum([output[k].sum() for k in ['p2', 'p3', 'p4', 'p5']])
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = x_grad.grad is not None
    if has_grad:
        grad_norm = x_grad.grad.norm().item()
        print(f"✅ Input gradient norm: {grad_norm:.6f}")
    else:
        print("❌ No gradients computed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in fpn.parameters())
    trainable_params = sum(p.numel() for p in fpn.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("FPN Backbone Test Complete!")
    print("=" * 60)
    
    return fpn, output


if __name__ == '__main__':
    # Run test
    fpn_model, fpn_output = test_fpn_backbone()
    
    # Print model structure
    print("\n" + "=" * 60)
    print("Model Structure:")
    print("=" * 60)
    print(fpn_model)

