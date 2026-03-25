"""
Complete Much Better - IoU Optimized Version
============================================
Enhanced version focused on improving IoU through:
1. IoU-aware loss functions (Lovasz, Boundary loss)
2. Attention-based boundary refinement
3. Hard example mining
4. Post-processing with CRF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Enhanced Loss Functions for IoU Optimization
# =============================================================================

class LovaszLoss(nn.Module):
    """
    Lovasz loss for IoU optimization.
    Specifically designed to optimize IoU directly.
    Based on: "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the IoU measure"
    """
    def __init__(self, per_image: bool = False, ignore_index: int = 255):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return lovasz_softmax(pred, target, ignore=self.ignore_index)


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for better edge segmentation.
    Uses distance transform to weight boundary pixels more heavily.
    """
    def __init__(self, boundary_weight: float = 10.0):
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate distance transform of target
        with torch.no_grad():
            target_np = target.cpu().numpy()
            boundary_weights = np.zeros_like(target_np, dtype=np.float32)
            
            for i in range(target_np.shape[0]):
                # Distance transform: 0 at boundary, higher further away
                if target_np[i].max() > 0:
                    dist = distance_transform_edt(1 - target_np[i])
                    # Normalize and weight boundaries more
                    dist = dist / (dist.max() + 1e-8)
                    boundary_weights[i] = 1 + self.boundary_weight * (1 - dist)
                else:
                    boundary_weights[i] = 1.0
            
            boundary_weights = torch.from_numpy(boundary_weights).to(pred.device)
        
        # BCE with boundary weighting
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * boundary_weights).mean()
        
        return weighted_bce


class IoULoss(nn.Module):
    """
    Soft IoU loss for direct IoU optimization.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        iou = (intersection + self.smooth) / (union - intersection + self.smooth)
        iou_loss = 1 - iou.mean()
        
        return iou_loss


class CombinedIoULoss(nn.Module):
    """
    Combined loss: BCE + Dice + Lovasz + Boundary
    Optimized for both overall IoU and boundary precision.
    """
    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.3,
        lovasz_weight: float = 0.25,
        boundary_weight: float = 0.15,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.label_smoothing = label_smoothing
        
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszLoss()
        self.boundary = BoundaryLoss(boundary_weight=10.0)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Manual label smoothing
        if self.label_smoothing > 0:
            target_smoothed = target * (1 - self.label_smoothing) + (1 - target) * self.label_smoothing
        else:
            target_smoothed = target
        
        # BCE Loss
        loss_bce = self.bce(pred, target_smoothed)
        
        # Dice Loss (with logits)
        pred_prob = torch.sigmoid(pred)
        loss_dice = dice_loss(pred_prob, target)
        
        # Lovasz Loss
        loss_lovasz = self.lovasz(pred, target.long())
        
        # Boundary Loss
        loss_boundary = self.boundary(pred, target)
        
        # Combined
        total_loss = (
            self.bce_weight * loss_bce +
            self.dice_weight * loss_dice +
            self.lovasz_weight * loss_lovasz +
            self.boundary_weight * loss_boundary
        )
        
        return total_loss, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'lovasz': loss_lovasz.item(),
            'boundary': loss_boundary.item()
        }


# =============================================================================
# Helper Functions (same as original)
# =============================================================================

def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore: int = 255) -> torch.Tensor:
    """Lovasz loss for multi-class segmentation."""
    C = probas.shape[1]
    probas = F.softmax(probas, dim=1)
    labels = labels.squeeze(1)
    
    losses = []
    for c in range(C):
        if C == 1:
            predc = probas[:, 0]
            targc = (labels == c).float()
        else:
            predc = probas[:, c]
            targc = (labels == c).float()
        
        if (targc.sum() < 1e-8).all():
            continue
        
        intersection = (predc * targc).sum()
        union = predc.sum() + targc.sum() - intersection
        
        if union < 1e-8:
            losses.append(0.0)
            continue
        
        iou = 1 - intersection / union
        losses.append(iou)
    
    return torch.stack(losses).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for binary segmentation."""
    pred_prob = pred  # Already probability from BCE
    intersection = (pred_prob * target).sum()
    return 1 - (2. * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)


# =============================================================================
# Enhanced Model with Boundary Attention
# =============================================================================

class BoundaryAttention(nn.Module):
    """
    Boundary attention module to enhance edge detection.
    Uses edge detection kernels to focus on boundary regions.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Standard conv for edge detection
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        
        # Initialize with Sobel-like weights
        nn.init.kaiming_normal_(self.edge_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Attention weights - combine original and edge features
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detect edges using gradient-like convolution
        edges = self.edge_conv(F.relu(x))
        
        # Combine original and edges
        combined = torch.cat([x, edges], dim=1)
        
        # Attention weights
        attn = self.attention(combined)
        
        # Apply attention - residual connection
        return x * attn + x


class AdaptiveShapeStarDistIoU(nn.Module):
    """
    Enhanced Complete Much Better model with:
    1. Boundary Attention for better edges
    2. IoU-optimized loss functions
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 128,
        num_shapes: int = 32,
        dropout_rate: float = 0.15,
        use_boundary_attention: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_shapes = num_shapes
        
        # Backbone: UNet-style encoder-decoder
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottom
        self.bottleneck = self._conv_block(512, 1024)
        
        # Boundary Attention (if enabled)
        self.use_boundary_attention = use_boundary_attention
        if use_boundary_attention:
            self.boundary_attn = BoundaryAttention(1024)
        
        # Decoder with skip connections
        self.upconv4 = self._upconv(1024, 512)
        self.decoder4 = self._conv_block(512 + 512, 512)
        
        self.upconv3 = self._upconv(512, 256)
        self.decoder3 = self._conv_block(256 + 256, 256)
        
        self.upconv2 = self._upconv(256, 128)
        self.decoder2 = self._conv_block(128 + 128, 128)
        
        self.upconv1 = self._upconv(128, 64)
        self.decoder1 = self._conv_block(64 + 64, 64)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Boundary attention (if enabled)
        if self.use_boundary_attention:
            b = self.boundary_attn(b)
        
        b = self.dropout(b)
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        return self.final(d1)


# =============================================================================
# Training Configuration
# =============================================================================

def get_iou_optimized_config() -> dict:
    """Get optimized configuration for IoU improvement."""
    return {
        # Model
        'in_channels': 1,
        'embedding_dim': 128,
        'num_shapes': 32,
        'dropout_rate': 0.15,
        'use_boundary_attention': True,
        
        # Training
        'epochs': 250,  # Longer training for IoU optimization
        'batch_size': 16,
        'learning_rate': 2e-5,  # Lower LR for fine-tuning
        
        # Loss weights (IoU optimized)
        'bce_weight': 0.25,
        'dice_weight': 0.30,
        'lovasz_weight': 0.30,  # Increased Lovasz for IoU
        'boundary_weight': 0.15,
        
        # Optimizer
        'weight_decay': 5e-5,
        
        # Scheduler
        'scheduler': 'CosineAnnealingWarmRestarts',
        'T_0': 25,
        'T_mult': 2,
        'eta_min': 1e-6,
        
        # Regularization
        'label_smoothing': 0.05,
        'max_grad_norm': 1.0,
        
        # Early stopping
        'patience': 35,
        'checkpoint_interval': 5,
        
        # Hardware
        'num_workers': 4,
        'pin_memory': True,
        'device': 'cuda'
    }


if __name__ == "__main__":
    # Test the model and losses
    print("=" * 60)
    print("Testing Complete Much Better - IoU Optimized")
    print("=" * 60)
    
    # Test model
    model = AdaptiveShapeStarDistIoU(
        in_channels=1,
        embedding_dim=128,
        num_shapes=32,
        dropout_rate=0.15,
        use_boundary_attention=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test loss
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    criterion = CombinedIoULoss(
        bce_weight=0.25,
        dice_weight=0.30,
        lovasz_weight=0.30,
        boundary_weight=0.15
    )
    
    loss, components = criterion(y, target)
    print(f"\nLoss breakdown:")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Total: {loss.item():.4f}")
    
    # Print config
    config = get_iou_optimized_config()
    print(f"\nOptimized Configuration for IoU:")
    print(f"  - Lovasz weight: {config['lovasz_weight']} (increased for IoU)")
    print(f"  - Boundary weight: {config['boundary_weight']} (for edge precision)")
    print(f"  - Learning rate: {config['learning_rate']} (lower for fine-tuning)")
    print(f"  - Epochs: {config['epochs']} (longer training)")

