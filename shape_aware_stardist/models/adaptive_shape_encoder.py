import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 implementation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        groups: Number of groups for grouped convolution
        deformable_groups: Number of deformable groups
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        deformable_groups: int = 1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.deformable_groups = deformable_groups
        
        # Regular convolution for feature extraction
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # Offset prediction
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * deformable_groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # Modulation mask prediction
        self.mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # Initialize offset and mask
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        nn.init.constant_(self.mask_conv.weight, 0.0)
        nn.init.constant_(self.mask_conv.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with deformable convolution."""
        # Predict offsets and modulation mask
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        # Apply deformable convolution
        return deformable_conv2d_function(
            x, offset, mask, self.conv.weight,
            self.kernel_size, self.stride, self.padding,
            self.groups, self.deformable_groups
        )

class AdaptiveShapeEncoder(nn.Module):
    """
    Adaptive Shape Encoder with deformable convolutions and shape priors.
    
    Args:
        in_channels: Number of input channels
        n_rays: Initial number of rays (will be adapted)
        base_channels: Number of base channels
        n_blocks: Number of encoder blocks
        use_shape_prior: Whether to use shape prior encoding
    """
    def __init__(
        self,
        in_channels: int = 3,
        n_rays: int = 32,
        base_channels: int = 64,
        n_blocks: int = 4,
        use_shape_prior: bool = True
    ):
        super().__init__()
        
        self.n_rays = n_rays
        self.use_shape_prior = use_shape_prior
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Deformable encoder blocks
        self.encoder_blocks = nn.ModuleList([
            DeformableEncoderBlock(
                in_channels=base_channels * (2**min(i, 3)),
                out_channels=base_channels * (2**min(i+1, 3)),
                n_rays=n_rays
            ) for i in range(n_blocks)
        ])
        
        # Shape prior encoder
        if use_shape_prior:
            self.shape_prior = ShapePriorEncoder(
                in_channels=base_channels * (2**min(n_blocks-1, 3)),
                n_rays=n_rays
            )
        
        # Adaptive sampling module
        self.adaptive_sampler = AdaptiveSampler(
            in_channels=base_channels * (2**min(n_blocks-1, 3)),
            n_rays=n_rays
        )
    
    def forward(
        self,
        x: torch.Tensor,
        shape_prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            shape_prior: Optional shape prior tensor
            
        Returns:
            tuple:
                - Encoded features
                - Sampling points
                - Intermediate features
        """
        # Initial features
        features = self.init_conv(x)
        intermediate_features = [features]
        
        # Encoder blocks
        for block in self.encoder_blocks:
            features = block(features)
            intermediate_features.append(features)
        
        # Shape prior encoding
        if self.use_shape_prior and shape_prior is not None:
            features = self.shape_prior(features, shape_prior)
        
        # Adaptive sampling
        features, sampling_points = self.adaptive_sampler(features)
        
        return features, sampling_points, intermediate_features


class DeformableEncoderBlock(nn.Module):
    """
    Encoder block with deformable convolutions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_rays: Number of rays
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_rays: int
    ):
        super().__init__()
        
        self.deform_conv1 = DeformableConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        self.deform_conv2 = DeformableConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Ray attention
        self.ray_attention = RayAttention(out_channels, n_rays)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Deformable convolutions
        x = self.relu(self.norm1(self.deform_conv1(x)))
        x = self.relu(self.norm2(self.deform_conv2(x)))
        
        # Apply ray attention
        x = self.ray_attention(x)
        
        return x


class ShapePriorEncoder(nn.Module):
    """
    Shape prior encoding module.
    
    Args:
        in_channels: Number of input channels
        n_rays: Number of rays
    """
    def __init__(
        self,
        in_channels: int,
        n_rays: int
    ):
        super().__init__()
        
        self.n_rays = n_rays
        
        # Prior feature extraction
        self.prior_conv = nn.Sequential(
            nn.Conv2d(n_rays, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Cross attention
        self.cross_attention = CrossAttention(in_channels)
    
    def forward(
        self,
        features: torch.Tensor,
        shape_prior: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with shape prior."""
        # Extract prior features
        prior_features = self.prior_conv(shape_prior)
        
        # Apply cross attention
        features = self.cross_attention(features, prior_features)
        
        return features


class AdaptiveSampler(nn.Module):
    """
    Adaptive sampling module for ray points.
    
    Args:
        in_channels: Number of input channels
        n_rays: Initial number of rays
    """
    def __init__(
        self,
        in_channels: int,
        n_rays: int
    ):
        super().__init__()
        
        self.n_rays = n_rays
        
        # Sampling point prediction
        self.point_conv = nn.Conv2d(in_channels, 2 * n_rays, kernel_size=1)
        
        # Importance prediction
        self.importance_conv = nn.Conv2d(in_channels, n_rays, kernel_size=1)
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features
            
        Returns:
            tuple:
                - Updated features
                - Sampling points
        """
        batch_size = features.size(0)
        
        # Predict sampling points
        points = self.point_conv(features)
        points = points.view(batch_size, self.n_rays, 2, *points.shape[-2:])
        
        # Predict importance weights
        importance = torch.sigmoid(self.importance_conv(features))
        importance = importance.view(batch_size, self.n_rays, 1, *importance.shape[-2:])
        
        # Apply importance weighting
        points = points * importance
        
        return features, points


class RayAttention(nn.Module):
    """
    Attention mechanism for ray features.
    
    Args:
        channels: Number of channels
        n_rays: Number of rays
    """
    def __init__(
        self,
        channels: int,
        n_rays: int
    ):
        super().__init__()
        
        self.channels = channels
        self.n_rays = n_rays
        
        # Ray query, key, value projections
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = channels ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ray attention."""
        B, C, H, W = x.shape
        
        # Compute query, key, value
        q = self.query(x).view(B, self.n_rays, -1, H * W)
        k = self.key(x).view(B, self.n_rays, -1, H * W)
        v = self.value(x).view(B, self.n_rays, -1, H * W)
        
        # Compute attention
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        x = torch.matmul(v, attn.transpose(-2, -1))
        x = x.view(B, C, H, W)
        
        return self.out_proj(x)


class CrossAttention(nn.Module):
    """
    Cross attention between features and shape prior.
    
    Args:
        channels: Number of channels
    """
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        
        # Feature projections
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = channels ** -0.5
    
    def forward(
        self,
        features: torch.Tensor,
        prior: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with cross attention."""
        B, C, H, W = features.shape
        
        # Compute query from features
        q = self.query(features).flatten(2).transpose(-2, -1)
        
        # Compute key, value from prior
        k = self.key(prior).flatten(2)
        v = self.value(prior).flatten(2)
        
        # Compute attention
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        x = torch.matmul(attn, v.transpose(-2, -1))
        x = x.transpose(-2, -1).view(B, C, H, W)
        
        return self.out_proj(x)


def deformable_conv2d_function(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int,
    deformable_groups: int
) -> torch.Tensor:
    """
    Deformable convolution function implementation.
    This is a simplified version - in practice, you would use CUDA implementation.
    """
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels = weight.shape[0]
    
    # Output size
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    # Pad input
    padded_input = F.pad(input, [padding] * 4)
    
    # Initialize output
    output = input.new_zeros(
        batch_size, out_channels, out_height, out_width
    )
    
    # For each output position
    for b in range(batch_size):
        for c_out in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    # Get sampling positions
                    h_in = h_out * stride
                    w_in = w_out * stride
                    
                    # Get local offsets
                    local_offset = offset[b, :, h_out, w_out].view(
                        deformable_groups, 2, kernel_size, kernel_size
                    )
                    local_mask = mask[b, :, h_out, w_out].view(
                        deformable_groups, 1, kernel_size, kernel_size
                    )
                    
                    # For each kernel position
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            # Get sampling position
                            h_sample = h_in + i + local_offset[0, 0, i, j]
                            w_sample = w_in + j + local_offset[0, 1, i, j]
                            
                            # Bilinear sampling
                            h_low = torch.floor(h_sample).long()
                            h_high = h_low + 1
                            w_low = torch.floor(w_sample).long()
                            w_high = w_low + 1
                            
                            # Get weights
                            h_weight = (h_sample - h_low).item()
                            w_weight = (w_sample - w_low).item()
                            
                            # Sample values
                            v1 = padded_input[b, :, h_low, w_low]
                            v2 = padded_input[b, :, h_low, w_high]
                            v3 = padded_input[b, :, h_high, w_low]
                            v4 = padded_input[b, :, h_high, w_high]
                            
                            # Interpolate
                            value = (1 - h_weight) * (1 - w_weight) * v1 + \
                                   (1 - h_weight) * w_weight * v2 + \
                                   h_weight * (1 - w_weight) * v3 + \
                                   h_weight * w_weight * v4
                            
                            # Apply mask and weight
                            output[b, c_out, h_out, w_out] += \
                                torch.sum(value * weight[c_out] * local_mask[0, 0, i, j])
    
    return output
