import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .transformer_block import TransformerBlock
from .attention_module import ShapeAttentionModule
from .feature_aggregation import FeatureAggregationModule

class ShapeAwareBackbone(nn.Module):
    """
    Shape-aware backbone network incorporating Transformer and specialized attention mechanisms.
    
    Args:
        in_channels (int): Number of input channels
        base_channels (int): Number of base channels (will be doubled in each level)
        num_levels (int): Number of hierarchical levels
        num_transformer_blocks (int): Number of transformer blocks per level
        num_heads (int): Number of attention heads in transformer blocks
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 4,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.shape_attention_modules = nn.ModuleList()
        
        current_channels = base_channels
        for level in range(num_levels):
            # Encoder block (downsampling)
            encoder_block = nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(current_channels * 2),
                nn.ReLU(inplace=True)
            )
            self.encoder_blocks.append(encoder_block)
            
            # Transformer blocks for this level
            transformer_stack = nn.ModuleList([
                TransformerBlock(
                    dim=current_channels * 2,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    dropout=dropout
                ) for _ in range(num_transformer_blocks)
            ])
            self.transformer_blocks.append(transformer_stack)
            
            # Shape attention module
            shape_attention = ShapeAttentionModule(
                in_channels=current_channels * 2,
                reduction_ratio=8
            )
            self.shape_attention_modules.append(shape_attention)
            
            current_channels *= 2
            
        # Feature aggregation
        self.feature_aggregation = FeatureAggregationModule(
            channels_list=[base_channels * 2**i for i in range(num_levels)]
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the backbone network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            tuple: (final_features, intermediate_features)
                - final_features: Aggregated features from all levels
                - intermediate_features: List of features from each level
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Store intermediate features
        intermediate_features = []
        current_feature = x
        
        # Encoder path with transformer and attention
        for level in range(self.num_levels):
            # Downsample
            current_feature = self.encoder_blocks[level](current_feature)
            
            # Apply transformer blocks
            B, C, H, W = current_feature.shape
            feature_sequence = current_feature.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            for transformer in self.transformer_blocks[level]:
                feature_sequence = transformer(feature_sequence)
            
            current_feature = feature_sequence.transpose(1, 2).reshape(B, C, H, W)
            
            # Apply shape attention
            current_feature = self.shape_attention_modules[level](current_feature)
            
            intermediate_features.append(current_feature)
        
        # Feature aggregation
        final_features = self.feature_aggregation(intermediate_features)
        
        return final_features, intermediate_features

    def get_output_channels(self) -> int:
        """Returns the number of output channels from the backbone."""
        return self.base_channels * (2 ** self.num_levels)
