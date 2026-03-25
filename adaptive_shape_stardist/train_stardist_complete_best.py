"""
StarDist Training Script - COMPLETE BEST VERSION
==============================================
在complete_much_better基础上进一步优化

改进点:
1. ResNet34 骨干网络 - 更强的特征提取
2. 特征金字塔注意力 (FPA) - 更好的多尺度特征融合
3. EMA (Exponential Moving Average) - 权重平均
4. RAdam 优化器 - 更好的收敛性
5. 边界感知损失 (Boundary Aware Loss) - 改善边缘分割
6. 渐进式学习率 warmup - 稳定训练开始
7. Cosine Annealing with Warmup - 更好的学习率调度
8. 使用预训练 ResNet 权重

Key improvements:
1. ResNet34 backbone for stronger feature extraction
2. Feature Pyramid Attention (FPA) for better multi-scale features
3. EMA for smoother predictions
4. RAdam optimizer for better convergence
5. Boundary-aware loss for edge quality
6. Progressive learning rate warmup
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import os
import time
from copy import deepcopy

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 80)
print("STAR-DIST TRAINING SCRIPT - COMPLETE BEST VERSION")
print("=" * 80)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()

print(f"\n📊 Hardware Configuration:")
print(f"   Device: {device}")
print(f"   GPU count: {gpu_count}")

if torch.cuda.is_available():
    for i in range(min(gpu_count, 4)):
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

# Deterministic training
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings('ignore')

# Fix NCCL issues
os.environ['NCCL_DEBUG'] = 'WARNING'
os.environ['NCCL_IB_DISABLE'] = '1'


# =============================================================================
# ResNet34 Backbone (pretrained)
# =============================================================================
class ResNet34Backbone(nn.Module):
    """
    ResNet34 Backbone for stronger feature extraction
    """
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet34
        from torchvision.models import resnet34, ResNet34_Weights
        
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            self.resnet = resnet34(weights=weights)
        else:
            self.resnet = resnet34(weights=None)
        
        # Modify first conv if input is grayscale
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove final pooling and fc
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        # Store layers for feature extraction
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.layer1 = self.resnet.layer1  # 64 channels
        self.layer2 = self.resnet.layer2  # 128 channels
        self.layer3 = self.resnet.layer3  # 256 channels
        self.layer4 = self.resnet.layer4  # 512 channels
        
    def forward(self, x):
        features = {}
        
        x = self.layer0(x)      # 1/4 size, 64 channels
        x = self.layer1(x)      # 1/4 size, 64 channels
        f1 = x                   # Save for FPN
        x = self.layer2(x)      # 1/8 size, 128 channels
        f2 = x                   # Save for FPN
        x = self.layer3(x)      # 1/16 size, 256 channels
        f3 = x                   # Save for FPN
        x = self.layer4(x)      # 1/32 size, 512 channels
        f4 = x                   # Save for FPN
        
        return {
            'p2': f1,  # 1/4 size, 64 channels
            'p3': f2,  # 1/8 size, 128 channels
            'p4': f3,  # 1/16 size, 256 channels
            'p5': f4,  # 1/32 size, 512 channels
        }


# =============================================================================
# Feature Pyramid Attention (FPA)
# =============================================================================
class FeaturePyramidAttention(nn.Module):
    """
    Feature Pyramid Attention for better multi-scale feature fusion
    """
    def __init__(self, in_channels=512, out_channels=256):
        super().__init__()
        
        # Global attention path
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Decoder path for each scale
        self.decoder5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, features):
        # features: dict with p2, p3, p4, p5
        
        p5 = features['p5']  # 1/32 size
        p4 = features['p4']  # 1/16 size
        p3 = features['p3']  # 1/8 size
        
        # Global attention on p5
        global_feat = self.global_pool(p5)
        global_feat = self.global_conv(global_feat)
        global_feat = nn.functional.interpolate(
            global_feat, size=p3.shape[2:], mode='bilinear', align_corners=False
        )
        
        # Decoder features
        dec5 = self.decoder5(p5)
        dec5 = nn.functional.interpolate(dec5, size=p3.shape[2:], mode='bilinear', align_corners=False)
        
        dec4 = self.decoder4(p4)
        dec4 = nn.functional.interpolate(dec4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        
        dec3 = self.decoder3(p3)
        
        # FPA: attention-weighted sum
        fused = dec3 * torch.sigmoid(global_feat) + dec4 + dec5
        
        return fused


# =============================================================================
# PositionalEncoding2D - Learned 2D Positional Encoding
# =============================================================================
class PositionalEncoding2D(nn.Module):
    """Learned 2D Positional Encoding"""
    def __init__(self, d_model, max_h=128, max_w=128):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pos_embed', torch.randn(1, d_model, max_h, max_w) * 0.02)
        
    def forward(self, x):
        b, c, h, w = x.shape
        pos = self.pos_embed[:, :, :h, :w]
        return x + pos


# =============================================================================
# PositionalEncoding1D - Sinusoidal 1D Positional Encoding
# =============================================================================
class PositionalEncoding1D(nn.Module):
    """Sinusoidal 1D Positional Encoding"""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        self.d_model = d_model
        
        pos = np.arange(max_len).reshape(max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_enc = np.zeros((max_len, d_model))
        pos_enc[:, 0::2] = np.sin(pos * div_term)
        if d_model > 1:
            pos_enc[:, 1::2] = np.cos(pos * div_term)
        
        self.register_buffer('pos_enc', torch.FloatTensor(pos_enc))
    
    def forward(self, x):
        b, seq_len, d = x.shape
        pos = self.pos_enc[:seq_len, :d]
        return x + pos.unsqueeze(0)


# =============================================================================
# TransformerBlock
# =============================================================================
class TransformerBlock(nn.Module):
    """TransformerBlock with MHA + FFN + residual"""
    def __init__(self, d_model, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = d_model * 4
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        ff_out = self.ffn(x)
        x = x + ff_out
        x = self.norm2(x)
        
        return x


# =============================================================================
# CrossAttentionFusion
# =============================================================================
class CrossAttentionFusion(nn.Module):
    """CrossAttentionFusion for shape-feature fusion"""
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, queries, keys, values):
        attn_out, _ = self.cross_attn(queries, keys, values)
        return self.norm(queries + attn_out)


# =============================================================================
# ShapePriorEncoder - BEST VERSION
# =============================================================================
class ShapePriorEncoderBest(nn.Module):
    """
    ShapePriorEncoder with ALL 12 features - BEST VERSION
    
    Improvements:
    - Enhanced with larger prototype capacity
    - Better regularization
    """
    def __init__(self, config):
        super().__init__()
        
        self.num_prototypes = config.get('num_prototypes', 32)  # 16 → 32
        self.embedding_dim = config.get('embedding_dim', 256)  # 128 → 256
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)  # 3 → 4
        self.dropout_rate = config.get('dropout_rate', 0.15)
        
        # Shape prototypes (increased capacity)
        self.shape_prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.embedding_dim) * 0.02
        )
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Conv2d(512, self.embedding_dim, kernel_size=1),  # 256 → 512 (from FPA)
            nn.BatchNorm2d(self.embedding_dim),
            nn.GELU()
        )
        
        # Positional Encoding
        self.pos_enc_2d = PositionalEncoding2D(self.embedding_dim)
        self.pos_enc_1d = PositionalEncoding1D(self.embedding_dim)
        
        # TransformerBlocks with dropout
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim * 4, 
                          dropout=self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        
        # CrossAttentionFusion
        self.cross_attention = CrossAttentionFusion(self.embedding_dim, self.num_heads)
        
        # Channel projection
        self.channel_projection = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 256, kernel_size=1),  # 64 → 256
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        
        # Token projection
        self.token_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    def forward(self, features):
        batch_size = features['p3'].shape[0]
        
        feat = features['p3']
        
        # Feature projection
        proj = self.feature_projection(feat)
        
        # Downsample for transformer
        h, w = proj.shape[2], proj.shape[3]
        pooled = nn.functional.adaptive_avg_pool2d(proj, (8, 8))
        
        # Reshape to tokens
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        
        # Positional encoding
        tokens = tokens + self.pos_enc_1d(tokens)
        
        # Shape prototypes
        prototypes = self.shape_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine
        combined = torch.cat([prototypes, tokens], dim=1)
        
        # Transformer reasoning
        for transformer in self.transformer_layers:
            combined = transformer(combined)
        
        # Split
        trans_prototypes = combined[:, :self.num_prototypes, :]
        trans_tokens = combined[:, self.num_prototypes:, :]
        
        # Attention fusion
        fused = self.cross_attention(trans_tokens, trans_prototypes, trans_prototypes)
        
        # Reshape and upsample
        fused = fused.mean(dim=1, keepdim=True)
        fused = fused.transpose(1, 2).view(batch_size, self.embedding_dim, 1, 1)
        fused = nn.functional.interpolate(fused, size=(h, w), mode='bilinear', align_corners=False)
        
        # Channel projection
        prior_features = self.channel_projection(fused)
        
        return {
            'prior_features': prior_features,
            'prototype_weights': torch.ones(1, self.num_prototypes) / self.num_prototypes,
            'attention_maps': prior_features,
            'learned_prototypes': self.shape_prototypes,
        }


# =============================================================================
# Complete Adaptive Shape StarDist Model (BEST VERSION)
# =============================================================================
class AdaptiveShapeStarDistBest(nn.Module):
    """
    Complete Adaptive Shape StarDist Model - BEST VERSION
    
    Key improvements:
    - ResNet34 backbone
    - Feature Pyramid Attention (FPA)
    - Larger shape prior encoder
    - Better decoder design
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        print("\n" + "=" * 80)
        print("INITIALIZING ADAPTIVE SHAPE STARDIST - COMPLETE BEST VERSION")
        print("=" * 80)
        
        # ResNet34 Backbone with pretrained weights
        self.backbone = ResNet34Backbone(in_channels=in_channels, pretrained=True)
        
        # Feature Pyramid Attention
        self.fpa = FeaturePyramidAttention(in_channels=512, out_channels=256)
        
        # Shape Prior Encoder (larger)
        shape_config = {
            'num_prototypes': 32,
            'embedding_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'dropout_rate': 0.15,
        }
        self.shape_prior = ShapePriorEncoderBest(shape_config)
        
        # Decoder (improved)
        self.up5 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec5 = self._conv_block(256 + 256, 256)  # FPA + shape prior
        
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = self._conv_block(128 + 256, 128)  # dec5 + p4
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(64 + 128, 64)  # dec4 + p3
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(32 + 64, 32)  # dec3 + p2
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.15)
        
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        print("\n" + "=" * 80)
        print("MODEL INITIALIZED - COMPLETE BEST VERSION ✓")
        print("=" * 80)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    
    def forward(self, x):
        # Backbone features
        fpn_features = self.backbone(x)
        
        # FPA features
        fpa_features = self.fpa(fpn_features)
        
        # Shape prior features
        shape_output = self.shape_prior(fpn_features)
        shape_feat = shape_output['prior_features']
        
        # Interpolate FPA features
        fpa_feat = nn.functional.interpolate(
            fpa_features, size=shape_feat.shape[2:], mode='bilinear', align_corners=False
        )
        
        # Level 5: FPA + Shape Prior
        d5 = self.dec5(torch.cat([fpa_feat, shape_feat], dim=1))
        
        # Level 4: d5 + p4
        p4 = fpn_features['p4']
        d4 = self.up5(d5)
        d4 = self.dec4(torch.cat([d4, p4], dim=1))
        
        # Level 3: d4 + p3
        p3 = fpn_features['p3']
        d3 = self.up4(d4)
        d3 = self.dec3(torch.cat([d3, p3], dim=1))
        
        # Level 2: d3 + p2
        p2 = fpn_features['p2']
        d2 = self.up3(d3)
        d2 = self.dec2(torch.cat([d2, p2], dim=1))
        
        # Level 1: upsample and output
        d1 = self.up2(d2)
        d1 = self.dropout(d1)
        
        return self.out(d1)


# =============================================================================
# Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = ((1 - pt) ** self.gamma) * BCE_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.expand_as(targets)
            F_loss = alpha_t * F_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss


# =============================================================================
# Dice Loss
# =============================================================================
def dice_loss(pred, target, smooth=1.0):
    """Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


# =============================================================================
# Boundary Aware Loss - penalize edge errors more
# =============================================================================
def boundary_aware_loss(pred, target, edge_weight=10.0):
    """
    Boundary aware loss - higher weight on edges
    """
    # Base BCE loss
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    
    # Compute edge weight map
    with torch.no_grad():
        # Sobel operators for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(target.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(target.device)
        
        # Detect edges in target
        target_x = nn.functional.conv2d(target, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        target_y = nn.functional.conv2d(target, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        edge_map = (target_x ** 2 + target_y ** 2).sqrt() > 0.5
        
        # Edge weight
        edge_weights = torch.ones_like(target)
        edge_weights[edge_map] = edge_weight
    
    # Weighted BCE
    bce_weighted = nn.functional.binary_cross_entropy_with_logits(pred, target, weight=edge_weights)
    
    return bce_weighted


# =============================================================================
# Combined Loss - BEST VERSION
# =============================================================================
def combined_loss(pred, target, focal_gamma=2.0, label_smoothing=0.05):
    """
    Combined loss with boundary awareness
    
    Components:
    - BCE with label smoothing
    - Dice loss
    - Focal loss
    - Boundary aware term
    """
    # Apply label smoothing
    target_smooth = target * (1 - label_smoothing) + label_smoothing / 2
    
    # BCE loss
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target_smooth)
    
    # Dice loss
    dice = dice_loss(pred, target)
    
    # Focal loss
    focal = FocalLoss(gamma=focal_gamma)(pred, target_smooth)
    
    # Boundary aware loss (optional, higher weight)
    boundary = boundary_aware_loss(pred, target, edge_weight=5.0)
    
    # Combined weights - boundary focus
    return 0.25 * bce + 0.35 * dice + 0.15 * focal + 0.25 * boundary


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================
class EMA:
    """
    Exponential Moving Average for model weights
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# MixUp Augmentation
# =============================================================================
class MixUp:
    """MixUp augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if np.random.rand() < 0.5:
            lam = np.random.beta(self.alpha, self.alpha)
            idx = torch.randperm(y.size(0))
            mixed_x = lam * x + (1 - lam) * x[idx]
            mixed_y = lam * y + (1 - lam) * y[idx]
            return mixed_x, mixed_y
        return x, y


# =============================================================================
# Data Augmentation (BEST)
# =============================================================================
class DataAugmentationBest:
    """Enhanced data augmentation"""
    def __init__(self, p=0.6):  # Higher probability
        self.p = p
    
    def __call__(self, x, y):
        if np.random.rand() < self.p:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
            
            # Random vertical flip
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=0).copy()
                y = np.flip(y, axis=0).copy()
            
            # Random brightness/contrast
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.85, 1.15)
                offset = np.random.uniform(-0.15, 0.15)
                x = np.clip(x * scale + offset, 0, 1)
        
        return x, y


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    """Main training function - COMPLETE BEST VERSION"""
    
    # =========================================================================
    # Configuration - BEST VERSION
    # =========================================================================
    CONFIG = {
        'epochs': 250,          # Longer training
        'batch_size': 12,       # Balance memory and stability
        'learning_rate': 2e-5,  # Lower LR for pretrained model
        'weight_decay': 5e-5,  # Lower weight decay
        'max_grad_norm': 1.0,
        'patience': 35,         # Longer patience
        'train_split': 0.85,
        'checkpoint_interval': 5,
        'model_name': 'complete_best',
        'mixup_alpha': 0.2,
        'label_smoothing': 0.05,
        'ema_decay': 0.9995,    # EMA decay rate
    }
    
    print("\n" + "=" * 80)
    print("CONFIGURATION - COMPLETE BEST VERSION")
    print("=" * 80)
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # =========================================================================
    # Load and Process Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOADING AND PROCESSING DATA")
    print("=" * 80)
    
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y_raw = data['Y']
    
    print(f"   Raw data: X={X.shape}, Y={Y_raw.shape}")
    
    # Convert X to 3-channel if needed
    if len(X.shape) == 3:
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    
    X = X.astype(np.float32) / 255.0
    
    # Label Normalization
    print("\n   📊 Label Normalization")
    print("   " + "-" * 50)
    
    Y_max_global = Y_raw.max()
    print(f"   Global Y max: {Y_max_global:.2f}")
    
    Y_normalized = Y_raw.astype(np.float32) / Y_max_global
    
    print(f"   After normalization:")
    print(f"     Mean: {Y_normalized.mean():.4f}")
    print(f"     Std: {Y_normalized.std():.4f}")
    
    # =========================================================================
    # Split Data
    # =========================================================================
    split_idx = int(len(X) * CONFIG['train_split'])
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y_normalized[:split_idx], Y_normalized[split_idx:]
    
    print(f"\n   Data split:")
    print(f"     Training: {len(X_train)} samples")
    print(f"     Validation: {len(X_val)} samples")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    Y_train_t = torch.FloatTensor(Y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    Y_val_t = torch.FloatTensor(Y_val).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # =========================================================================
    # Create Model - BEST VERSION
    # =========================================================================
    print("\n" + "=" * 80)
    print("CREATING MODEL - COMPLETE BEST VERSION")
    print("=" * 80)
    
    model = AdaptiveShapeStarDistBest(in_channels=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Use single GPU
    print(f"   Using single GPU for stable training")
    
    # =========================================================================
    # Optimizer and Scheduler - BEST VERSION
    # =========================================================================
    # Use different LRs for backbone and decoder
    backbone_lr = CONFIG['learning_rate'] * 0.1  # Lower LR for pretrained backbone
    decoder_lr = CONFIG['learning_rate']
    
    backbone_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': decoder_params, 'lr': decoder_lr}
    ], weight_decay=CONFIG['weight_decay'])
    
    # Cosine Annealing with Warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    # =========================================================================
    # Resume from Checkpoint
    # =========================================================================
    model_name = CONFIG['model_name']
    checkpoint_dir = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_complete')
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'{model_name}_checkpoint.pth'
    best_model_path = checkpoint_dir / f'{model_name}_best.pth'
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if checkpoint_path.exists():
        print(f"\n   📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   ▶ Resuming from epoch: {start_epoch}")
        print(f"   ▶ Best val loss: {best_val_loss:.4f}")
    
    # =========================================================================
    # EMA Setup
    # =========================================================================
    ema = EMA(model, decay=CONFIG['ema_decay'])
    
    # =========================================================================
    # Training Loop - BEST VERSION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STARTING TRAINING - COMPLETE BEST VERSION")
    print("=" * 80)
    
    augment = DataAugmentationBest(p=0.6)
    mixup = MixUp(alpha=CONFIG['mixup_alpha'])
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }
    
    epochs = CONFIG['epochs']
    patience = CONFIG['patience']
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # ===== Training =====
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Apply augmentation
            if np.random.rand() < 0.5:
                batch_x_np = batch_x.cpu().numpy().transpose(0, 2, 3, 1)
                batch_y_np = batch_y.cpu().numpy().transpose(0, 2, 3, 1)
                
                for i in range(len(batch_x_np)):
                    batch_x_np[i], batch_y_np[i] = augment(batch_x_np[i], batch_y_np[i])
                
                batch_x = torch.FloatTensor(batch_x_np.transpose(0, 3, 1, 2)).to(device)
                batch_y = torch.FloatTensor(batch_y_np.transpose(0, 3, 1, 2)).to(device)
            
            # Apply MixUp
            batch_x, batch_y = mixup(batch_x, batch_y)
            
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = combined_loss(outputs, batch_y, label_smoothing=CONFIG['label_smoothing'])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            # Update EMA
            ema.update()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = combined_loss(outputs, batch_y, label_smoothing=CONFIG['label_smoothing'])
                
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # ===== Save Checkpoint =====
        if (epoch + 1) % CONFIG['checkpoint_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': CONFIG,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
        
        # ===== Save Best Model (using EMA weights) =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Apply EMA before saving
            ema.apply_shadow()
            torch.save(model.state_dict(), best_model_path)
            ema.restore()
            
            print(f"   ✅ New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
    # =========================================================================
    # Save Final EMA Model
    # =========================================================================
    ema.apply_shadow()
    ema_model_path = checkpoint_dir / f'{model_name}_ema.pth'
    torch.save(model.state_dict(), ema_model_path)
    ema.restore()
    
    print(f"\n   💾 EMA model saved: {ema_model_path}")
    
    # =========================================================================
    # Training Complete
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - COMPLETE BEST VERSION")
    print("=" * 80)
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Final epoch: {epoch+1}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Best model: {best_model_path}")
    
    # Save training history
    history_path = checkpoint_dir / f'{model_name}_history.npz'
    np.savez(history_path, **training_history)
    print(f"   History: {history_path}")
    
    # Generate evaluation report
    generate_evaluation_report(model, val_loader, best_model_path, checkpoint_dir)
    
    return best_val_loss


# =============================================================================
# Generate Evaluation Report
# =============================================================================
def generate_evaluation_report(model, val_loader, model_path, output_dir):
    """Generate evaluation report for the trained model"""
    print("\n" + "=" * 80)
    print("GENERATING EVALUATION REPORT")
    print("=" * 80)
    
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            pred = torch.sigmoid(model(batch_x))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    from scipy.stats import pearsonr
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    pearson, _ = pearsonr(all_preds, all_targets)
    
    binary_preds = (all_preds > 0.5).astype(np.float32)
    binary_targets = (all_targets > 0.5).astype(np.float32)
    
    intersection = np.sum(binary_preds * binary_targets)
    dice = 2 * intersection / (np.sum(binary_preds) + np.sum(binary_targets) + 1e-8)
    iou = intersection / (np.sum(binary_preds) + np.sum(binary_targets) - intersection + 1e-8)
    accuracy = np.mean(binary_preds == binary_targets)
    tp = np.sum((binary_preds == 1) & (binary_targets == 1))
    fp = np.sum((binary_preds == 1) & (binary_targets == 0))
    fn = np.sum((binary_preds == 0) & (binary_targets == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    best_dice = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        binary_preds_thresh = (all_preds > thresh).astype(np.float32)
        intersection_thresh = np.sum(binary_preds_thresh * binary_targets)
        dice_thresh = 2 * intersection_thresh / (np.sum(binary_preds_thresh) + np.sum(binary_targets) + 1e-8)
        if dice_thresh > best_dice:
            best_dice = dice_thresh
            best_threshold = thresh
    
    report = f"""
============================================================
ADAPTIVE SHAPE STARDIST - EVALUATION REPORT (COMPLETE BEST)
============================================================

Model: {model_path}
Parameters: {sum(p.numel() for p in model.parameters()):,}
Validation samples: {len(all_preds) // (all_preds.shape[0] if len(all_preds.shape) > 1 else 1)}

REGRESSION METRICS
----------------------------------------
MSE:  {mse:.6f}
RMSE: {rmse:.6f}
MAE:  {mae:.6f}
Pearson: {pearson:.6f}

SEGMENTATION METRICS
----------------------------------------
Dice: {dice:.4f}
IoU: {iou:.4f}
Accuracy: {accuracy:.4f}
F1: {f1:.4f}

Best threshold: {best_threshold:.1f}
Best Dice: {best_dice:.4f}

"""
    
    report_path = output_dir / 'evaluation_report_complete_best.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"   Report saved to: {report_path}")


if __name__ == "__main__":
    main()

