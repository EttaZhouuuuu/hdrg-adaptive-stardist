import unittest
import torch
import numpy as np

from ..models.shape_aware_backbone import ShapeAwareBackbone
from ..models.transformer_block import TransformerBlock
from ..models.attention_module import ShapeAttentionModule
from ..models.feature_aggregation import FeatureAggregationModule
from ..configs.config import default_config

class TestShapeAwareBackbone(unittest.TestCase):
    """Test cases for Shape-aware Backbone"""
    
    def setUp(self):
        """Set up test cases"""
        self.config = default_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShapeAwareBackbone(
            in_channels=self.config.backbone.in_channels,
            base_channels=self.config.backbone.base_channels,
            num_levels=self.config.backbone.num_levels,
            num_transformer_blocks=self.config.backbone.num_transformer_blocks,
            num_heads=self.config.backbone.num_heads,
            dropout=self.config.backbone.dropout
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test forward pass with random input"""
        batch_size = 2
        input_size = (256, 256)
        x = torch.randn(batch_size, self.config.backbone.in_channels,
                       *input_size).to(self.device)
        
        # Forward pass
        output, intermediate_features = self.model(x)
        
        # Check output shape
        expected_output_size = (batch_size, self.config.backbone.base_channels * 
                              (2 ** (self.config.backbone.num_levels - 1)),
                              input_size[0] // (2 ** (self.config.backbone.num_levels - 1)),
                              input_size[1] // (2 ** (self.config.backbone.num_levels - 1)))
        
        self.assertEqual(output.shape, expected_output_size)
        self.assertEqual(len(intermediate_features), self.config.backbone.num_levels)
    
    def test_transformer_block(self):
        """Test transformer block"""
        transformer = TransformerBlock(
            dim=64,
            num_heads=8,
            dropout=0.1
        ).to(self.device)
        
        # Test input
        x = torch.randn(2, 100, 64).to(self.device)  # (batch_size, sequence_length, dim)
        
        # Forward pass
        output = transformer(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_attention_module(self):
        """Test shape attention module"""
        attention = ShapeAttentionModule(
            in_channels=64,
            reduction_ratio=8
        ).to(self.device)
        
        # Test input
        x = torch.randn(2, 64, 32, 32).to(self.device)
        
        # Forward pass
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_feature_aggregation(self):
        """Test feature aggregation module"""
        channels_list = [64, 128, 256, 512]
        aggregation = FeatureAggregationModule(
            channels_list=channels_list
        ).to(self.device)
        
        # Create test features
        features = []
        batch_size = 2
        size = 64
        for channels in channels_list:
            features.append(torch.randn(batch_size, channels, size, size).to(self.device))
            size = size // 2
        
        # Forward pass
        output = aggregation(features)
        
        self.assertEqual(output.shape, (batch_size, channels_list[-1], 
                                      features[-1].shape[-2], features[-1].shape[-1]))

if __name__ == '__main__':
    unittest.main()
