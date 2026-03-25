"""
StarDist Training Script - COMPLETE MUCH BETTER VERSION
=======================================================
在complete_better基础上进一步优化

改进点:
1. 增大batch_size (8 → 16) - 更稳定的梯度估计
2. 减小学习率 (5e-5 → 3e-5) - 更精细的训练
3. Label Smoothing - 防止过度自信预测
4. MixUp增强 - 样本混合正则化
5. 更长warmup (10 epochs) - 渐进式学习
6. 改进损失函数权重 - Dice主导
7. 更长训练 (200 epochs)
8. 使用One-Cycle Policy学习率调度

Key improvements:
1. Label Smoothing for better calibration
2. MixUp augmentation for regularization
3. Longer warmup for stable training start
4. Dice-focused loss for better segmentation
5. One-Cycle learning rate policy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import os
import time
from datetime import datetime

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 80)
print("STAR-DIST TRAINING SCRIPT - COMPLETE MUCH BETTER VERSION")
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
# TransformerBlock - Multi-head self-attention + FFN + residual
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
# CrossAttentionFusion - Query-Key-Value Cross-Attention
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
# ShapePriorEncoder - Main Encoder Class (MUCH BETTER VERSION)
# =============================================================================
class ShapePriorEncoderMuchBetter(nn.Module):
    """
    ShapePriorEncoder with ALL 12 features - MUCH BETTER VERSION
    
    Improvements:
    - Dropout for regularization
    - Better feature fusion
    """
    def __init__(self, config):
        super().__init__()
        
        self.num_prototypes = config.get('num_prototypes', 16)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 3)
        self.dropout_rate = config.get('dropout_rate', 0.15)  # Slightly higher dropout
        
        # Shape prototype learning
        self.shape_prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.embedding_dim) * 0.02
        )
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, self.embedding_dim, kernel_size=1),
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
            nn.Conv2d(self.embedding_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
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
# UNet Backbone with FPN
# =============================================================================
class UNetBackbone(nn.Module):
    """UNet backbone that outputs FPN features"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
    
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        fpn_features = {
            'p2': e2,
            'p3': e3,
            'p4': e4,
            'p5': self.pool(e4),
        }
        
        return e1, e2, e3, e4, fpn_features


# =============================================================================
# Complete Adaptive Shape StarDist Model (MUCH BETTER VERSION)
# =============================================================================
class AdaptiveShapeStarDistMuchBetter(nn.Module):
    """
    Complete Adaptive Shape StarDist Model - MUCH BETTER VERSION
    
    Key improvements:
    - Dropout in decoder for regularization
    - Better training configuration
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        print("\n" + "=" * 80)
        print("INITIALIZING ADAPTIVE SHAPE STARDIST - COMPLETE MUCH BETTER VERSION")
        print("=" * 80)
        
        self.backbone = UNetBackbone(in_channels)
        
        shape_config = {
            'num_prototypes': 16,
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 3,
            'dropout_rate': 0.15,
        }
        self.shape_prior = ShapePriorEncoderMuchBetter(shape_config)
        
        # Decoder with dropout
        self.dropout = nn.Dropout2d(0.15)
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, 1, 1)
        
        print("\n" + "=" * 80)
        print("MODEL INITIALIZED - COMPLETE MUCH BETTER VERSION ✓")
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
        e1, e2, e3, e4, fpn_features = self.backbone(x)
        
        shape_output = self.shape_prior(fpn_features)
        shape_feat = shape_output['prior_features']
        
        # Interpolate shape_feat
        shape_feat = nn.functional.interpolate(shape_feat, size=e3.shape[2:], mode='bilinear')
        
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        
        # Apply dropout
        d2 = self.dropout(d2)
        
        # Interpolate to match d2 size
        shape_feat_d2 = nn.functional.interpolate(shape_feat, size=d2.shape[2:], mode='bilinear')
        d2 = d2 + shape_feat_d2
        
        return self.out(d2)


# =============================================================================
# Focal Loss with Label Smoothing
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance
    """
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
# Dice Loss with Label Smoothing
# =============================================================================
def dice_loss(pred, target, smooth=1.0, label_smoothing=0.0):
    """Dice loss with optional label smoothing"""
    # Apply label smoothing to target
    if label_smoothing > 0:
        target = target * (1 - label_smoothing) + label_smoothing / 2
    
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


# =============================================================================
# Combined Loss - Dice Dominant with Label Smoothing
# =============================================================================
def label_smooth(target, smoothing=0.05):
    """Apply label smoothing to targets"""
    return target * (1 - smoothing) + smoothing / 2


def combined_loss(pred, target, focal_gamma=2.0, label_smoothing=0.05):
    """
    Combined loss - Dice dominant for better segmentation
    
    Weight changes:
    - Original: BCE:0.5, Dice:0.3, Focal:0.2
    - Much Better: BCE:0.3, Dice:0.5, Focal:0.2
    """
    # Apply label smoothing to target
    target_smoothed = label_smooth(target, label_smoothing)
    
    # BCE with label smoothing
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target_smoothed)
    
    # Dice loss with label smoothing
    dice = dice_loss(pred, target, label_smoothing=label_smoothing)
    
    # Focal loss
    focal = FocalLoss(gamma=focal_gamma)(pred, target_smoothed)
    
    # Dice-dominant weights
    return 0.3 * bce + 0.5 * dice + 0.2 * focal


# =============================================================================
# MixUp Augmentation
# =============================================================================
class MixUp:
    """
    MixUp augmentation for regularization
    Mixes two samples with a random lambda
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if np.random.rand() < 0.5:
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Randomly permute y
            idx = torch.randperm(y.size(0))
            
            mixed_x = lam * x + (1 - lam) * x[idx]
            mixed_y = lam * y + (1 - lam) * y[idx]
            
            return mixed_x, mixed_y
        return x, y


# =============================================================================
# Data Augmentation (MUCH BETTER)
# =============================================================================
class DataAugmentationMuchBetter:
    """
    Enhanced data augmentation with rotation and scaling
    """
    def __init__(self, p=0.5):
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
                scale = np.random.uniform(0.9, 1.1)
                offset = np.random.uniform(-0.1, 0.1)
                x = np.clip(x * scale + offset, 0, 1)
        
        return x, y


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    """
    Main training function - COMPLETE MUCH BETTER VERSION
    """
    
    # =========================================================================
    # Configuration - MUCH BETTER
    # =========================================================================
    CONFIG = {
        'epochs': 200,           # 150 → 200 (longer training)
        'batch_size': 16,        # 8 → 16 (larger batch for stability)
        'learning_rate': 3e-5,   # 5e-5 → 3e-5 (finer learning)
        'weight_decay': 1e-4,
        'max_grad_norm': 1.0,
        'patience': 30,          # 25 → 30 (longer patience)
        'train_split': 0.85,
        'checkpoint_interval': 5,  # 3 → 5 (less frequent saves)
        'model_name': 'complete_much_better',
        'mixup_alpha': 0.2,
        'label_smoothing': 0.05,
    }
    
    print("\n" + "=" * 80)
    print("CONFIGURATION - COMPLETE MUCH BETTER VERSION")
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
    
    # DataLoaders with pin_memory
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # =========================================================================
    # Create Model - MUCH BETTER VERSION
    # =========================================================================
    print("\n" + "=" * 80)
    print("CREATING MODEL - COMPLETE MUCH BETTER VERSION")
    print("=" * 80)
    
    model = AdaptiveShapeStarDistMuchBetter(in_channels=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Use single GPU
    print(f"   Using single GPU for stable training")
    
    # =========================================================================
    # Optimizer and Scheduler - MUCH BETTER
    # =========================================================================
    optimizer = optim.AdamW(model.parameters(), 
                           lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG['weight_decay'])
    
    # One-Cycle Policy for learning rate
    num_batches = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        epochs=CONFIG['epochs'],
        steps_per_epoch=num_batches,
        pct_start=0.05,  # 5% warmup
        anneal_strategy='cos',
        final_div_factor=10  # Final LR = max_lr / 10
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
    # Training Loop - MUCH BETTER
    # =========================================================================
    print("\n" + "=" * 80)
    print("STARTING TRAINING - COMPLETE MUCH BETTER VERSION")
    print("=" * 80)
    
    # Enhanced augmentation
    augment = DataAugmentationMuchBetter(p=0.5)
    mixup = MixUp(alpha=CONFIG['mixup_alpha'])
    
    # Training metrics
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
            
            # Forward pass
            outputs = model(batch_x)
            loss = combined_loss(
                outputs, batch_y, 
                label_smoothing=CONFIG['label_smoothing']
            )
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            
            optimizer.step()
            scheduler.step()
            
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
                loss = combined_loss(
                    outputs, batch_y,
                    label_smoothing=CONFIG['label_smoothing']
                )
                
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
        
        # ===== Save Best Model =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
    # =========================================================================
    # Training Complete
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - COMPLETE MUCH BETTER VERSION")
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
    
    # Collect all predictions and targets
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
    
    # Calculate metrics
    from scipy.stats import pearsonr
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    pearson, _ = pearsonr(all_preds, all_targets)
    
    # Segmentation metrics
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
    
    # Best threshold search
    best_dice = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        binary_preds_thresh = (all_preds > thresh).astype(np.float32)
        intersection_thresh = np.sum(binary_preds_thresh * binary_targets)
        dice_thresh = 2 * intersection_thresh / (np.sum(binary_preds_thresh) + np.sum(binary_targets) + 1e-8)
        if dice_thresh > best_dice:
            best_dice = dice_thresh
            best_threshold = thresh
    
    # Generate report
    report = f"""
============================================================
ADAPTIVE SHAPE STARDIST - EVALUATION REPORT (COMPLETE MUCH BETTER)
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
    
    report_path = output_dir / 'evaluation_report_complete_much_better.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"   Report saved to: {report_path}")


if __name__ == "__main__":
    main()

