import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class MLP(nn.Module):
    """
    Multi-layer perceptron module.
    
    Args:
        dim (int): Input dimension
        mlp_ratio (float): Ratio of hidden dimension to input dimension
        dropout (float): Dropout rate
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of hidden dimension to input dimension in MLP
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # MLP block
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        # Self-attention block
        x = x + self.attn(self.norm1(x))
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
        
        return x
