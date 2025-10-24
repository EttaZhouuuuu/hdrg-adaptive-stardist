import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class FeatureAggregationModule(nn.Module):
    """
    Multi-level feature aggregation module with attention-guided fusion.
    
    Args:
        channels_list (List[int]): List of channel dimensions for each level
    """
    def __init__(self, channels_list: List[int]):
        super().__init__()
        
        self.channels_list = channels_list
        self.num_levels = len(channels_list)
        
        # Create attention gates for each level
        self.attention_gates = nn.ModuleList([
            AttentionGate(
                x_channels=channels_list[i],
                g_channels=channels_list[-1],
                inter_channels=channels_list[i] // 2
            ) for i in range(self.num_levels)
        ])
        
        # Feature refinement convolutions
        self.refinement_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_list[i], channels_list[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels_list[i]),
                nn.ReLU(inplace=True)
            ) for i in range(self.num_levels)
        ])
        
        # Final fusion
        total_channels = sum(channels_list)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, channels_list[-1], kernel_size=1),
            nn.BatchNorm2d(channels_list[-1]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the feature aggregation module.
        
        Args:
            features (List[torch.Tensor]): List of feature maps from different levels
            
        Returns:
            torch.Tensor: Aggregated feature map
        """
        assert len(features) == self.num_levels, \
            f"Expected {self.num_levels} feature maps, got {len(features)}"
        
        # Global guidance from deepest level
        global_feature = features[-1]
        
        # Process each level with attention
        refined_features = []
        for i in range(self.num_levels):
            # Apply attention gating
            attended_feat = self.attention_gates[i](features[i], global_feature)
            
            # Refine features
            refined_feat = self.refinement_convs[i](attended_feat)
            
            # Resize to match the size of the deepest feature
            if i < self.num_levels - 1:
                refined_feat = F.interpolate(
                    refined_feat,
                    size=global_feature.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            refined_features.append(refined_feat)
        
        # Concatenate all features
        concat_features = torch.cat(refined_features, dim=1)
        
        # Final fusion
        output = self.fusion_conv(concat_features)
        
        return output

class AttentionGate(nn.Module):
    """
    Attention gate for feature selection and refinement.
    
    Args:
        x_channels (int): Number of channels in the input feature map
        g_channels (int): Number of channels in the gating signal
        inter_channels (int): Number of intermediate channels
    """
    def __init__(self, x_channels: int, g_channels: int, inter_channels: int):
        super().__init__()
        
        # Reduce dimensions for attention computation
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.Wx = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention gate.
        
        Args:
            x (torch.Tensor): Input feature map
            g (torch.Tensor): Gating signal
            
        Returns:
            torch.Tensor: Attended feature map
        """
        # Resize gating signal to match input feature map
        g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Project signals to intermediate space
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        
        # Compute attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi
