import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ShapeAttentionModule(nn.Module):
    """
    Specialized attention module for capturing shape-related features.
    
    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio for attention computation
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction_ratio
        
        # Channel attention branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, self.inter_channels, kernel_size=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention branch
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, kernel_size=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Shape context branch
        self.shape_context = ShapeContextModule(in_channels, self.inter_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the shape attention module.
        
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Enhanced feature map with shape attention
        """
        # Channel attention
        channel_weight = self.channel_attention(x)
        x_channel = x * channel_weight
        
        # Spatial attention
        spatial_weight = self.spatial_attention(x)
        x_spatial = x * spatial_weight
        
        # Shape context attention
        x_shape = self.shape_context(x)
        
        # Combine all attention mechanisms
        output = x_channel + x_spatial + x_shape
        
        return output

class ShapeContextModule(nn.Module):
    """
    Module for capturing shape context information using dilated convolutions
    and self-attention.
    
    Args:
        in_channels (int): Number of input channels
        inter_channels (int): Number of intermediate channels
    """
    def __init__(self, in_channels: int, inter_channels: int):
        super().__init__()
        
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale shape context
        self.shape_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, 
                         padding=rate, dilation=rate),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            ) for rate in [1, 2, 4, 8]
        ])
        
        # Combine branches
        self.conv_combine = nn.Sequential(
            nn.Conv2d(inter_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Self-attention for global shape context
        self.self_attention = SelfAttention(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the shape context module.
        
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Enhanced feature map with shape context
        """
        # Reduce channels
        feat = self.conv_reduce(x)
        
        # Multi-scale shape context
        shape_feats = []
        for branch in self.shape_branches:
            shape_feats.append(branch(feat))
        
        # Combine multi-scale features
        shape_feat = torch.cat(shape_feats, dim=1)
        shape_feat = self.conv_combine(shape_feat)
        
        # Apply self-attention for global context
        output = self.self_attention(shape_feat)
        
        return output

class SelfAttention(nn.Module):
    """
    Self-attention module for capturing global dependencies.
    
    Args:
        in_channels (int): Number of input channels
    """
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the self-attention module.
        
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Self-attention enhanced feature map
        """
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, H * W)
        value = self.value_conv(x).view(batch_size, -1, H * W)
        
        # Calculate attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out
