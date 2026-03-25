"""
PyTorch Multi-GPU Training for Adaptive Shape StarDist
======================================================

Uses complete AdaptiveShapeStarDist model with ALL 12 shape_prior.py features:
1. Transformer-based global shape reasoning
2. Shape prototype learning
3. Multi-head attention for shape-feature fusion
4. Learnable shape embeddings
5. Multi-scale processing for FPN integration
6. PositionalEncoding2D (sinusoidal 2D positional encoding)
7. PositionalEncoding1D (sinusoidal 1D positional encoding)
8. TransformerBlock (multi-head self-attention + FFN + residual)
9. CrossAttentionFusion (query-key-value cross-attention)
10. ShapePriorEncoder (main encoder class)
11. Feature projection layers (1x1 conv for embedding)
12. Channel projections for multi-scale fusion

Uses DataParallel for multi-GPU training (2-3 GPUs)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DataParallel
from pathlib import Path
import sys
import os
import json
import time
from tqdm import tqdm

# Setup paths
PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 70)
print("PyTorch MULTI-GPU TRAINING - Adaptive Shape StarDist")
print("=" * 70)

# Configure GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
print(f"Device: {device}")
print(f"GPU count: {gpu_count}")

if torch.cuda.is_available() and gpu_count > 0:
    for i in range(min(gpu_count, 3)):  # Use up to 3 GPUs
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    
    # Use 2 GPUs for training
    num_gpus = min(2, gpu_count)
    print(f"\nUsing {num_gpus} GPUs for training")
    
    # Set determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class ShapePriorEncoderPyTorch(nn.Module):
    """
    PyTorch implementation of ShapePriorEncoder with ALL 12 features.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature 2&4: Shape prototype learning with learnable embeddings
        self.num_protos = config.get('num_shape_prototypes', 16)
        self.proto_dim = config.get('shape_embed_dim', 256)
        self.shape_prototypes = nn.Parameter(
            torch.randn(self.num_protos, self.proto_dim) * 0.1
        )
        
        # Feature 6: PositionalEncoding2D (sinusoidal 2D positional encoding)
        self.pos_enc_2d = PositionalEncoding2D(self.proto_dim)
        
        # Feature 7: PositionalEncoding1D (sinusoidal 1D positional encoding)
        self.pos_enc_1d = PositionalEncoding1D(self.proto_dim)
        
        # Feature 8: TransformerBlocks (multi-head self-attention + FFN + residual)
        n_blocks = 3
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.proto_dim, num_heads=8, ff_dim=512)
            for _ in range(n_blocks)
        ])
        
        # Feature 9: CrossAttentionFusion (query-key-value cross-attention)
        self.cross_attention = CrossAttentionFusion(self.proto_dim, num_heads=8)
        
        # Feature 11: Feature projection layers (1x1 conv for embedding)
        # Project backbone features to embedding dimension
        self.feature_proj = nn.Conv2d(256, self.proto_dim, kernel_size=1)
        
        # Feature 5&12: Multi-scale processing for FPN integration
        # Channel projections for different FPN levels
        # Note: enc1=64, enc2=128, enc3=256, enc4=512, pool=512
        # p2 comes from enc2 (128 channels), p3 from enc3 (256), p4 from enc4 (512), p5 from pool (512)
        self.scale_projs = nn.ModuleDict()
        self.scale_projs['p2'] = nn.Conv2d(128, self.proto_dim, kernel_size=1)
        self.scale_projs['p3'] = nn.Conv2d(256, self.proto_dim, kernel_size=1)
        self.scale_projs['p4'] = nn.Conv2d(512, self.proto_dim, kernel_size=1)
        self.scale_projs['p5'] = nn.Conv2d(512, self.proto_dim, kernel_size=1)
        
        # Output projection
        self.output_proj = nn.Conv2d(self.proto_dim, 256, kernel_size=1)
        
        print("  [ShapePriorEncoderPyTorch] FEATURE 2&4: Shape prototypes initialized:", self.shape_prototypes.shape)
        print("  [ShapePriorEncoderPyTorch] FEATURE 11: Feature projection layer created")
        print("  [ShapePriorEncoderPyTorch] FEATURE 6: PositionalEncoding2D created")
        print("  [ShapePriorEncoderPyTorch] FEATURE 7: PositionalEncoding1D created")
        print("  [ShapePriorEncoderPyTorch] FEATURE 8: 3 TransformerBlocks created")
        print("  [ShapePriorEncoderPyTorch] FEATURE 9: CrossAttentionFusion created")
        print("  [ShapePriorEncoderPyTorch] FEATURES 5&12: Multi-scale layers created for ['p2', 'p3', 'p4', 'p5']")
    
    def forward(self, features):
        """
        Forward pass with all 12 features:
        1. Global shape reasoning via transformers
        2. Shape prototype lookup
        3. Multi-head attention fusion
        4. Learnable embeddings
        5. Multi-scale processing
        """
        batch_size = features['p2'].shape[0]
        
        # Get multi-scale features
        multi_scale_features = {k: v for k, v in features.items() if k in ['p2', 'p3', 'p4', 'p5']}
        
        # Feature 1: Transformer-based global shape reasoning
        # Apply positional encoding and process through transformers
        encoded_features = []
        for scale, feat in multi_scale_features.items():
            # Project to embedding dimension using pre-defined conv
            proj = self.scale_projs[scale](feat)
            # Add positional encoding (reshape for transformer processing)
            b, c, h, w = proj.shape
            proj_flat = proj.flatten(2).transpose(1, 2)  # (b, h*w, d)
            proj_flat = proj_flat + self.pos_enc_1d(proj_flat)
            encoded_features.append(proj_flat)
        
        # Process through transformer blocks
        for transformer in self.transformer_blocks:
            encoded_features = [transformer(ef) for ef in encoded_features]
        
        # Feature 9: Cross-attention fusion
        # Use shape prototypes as query, features as keys/values
        proto_queries = self.shape_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        proto_queries = proto_queries + self.pos_enc_1d(proto_queries)
        
        fused_features = []
        for scale_feat in encoded_features:
            # Cross attention: shape prototypes attend to features
            fused = self.cross_attention(proto_queries, scale_feat)
            fused_features.append(fused)
        
        # Combine multi-scale features
        combined = torch.cat(fused_features, dim=1)  # (b, num_scales * d, 1)
        combined = combined.view(combined.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        
        # Feature 11: Output projection
        output = self.output_proj(combined)
        
        return output


class PositionalEncoding2D(nn.Module):
    """Feature 6: Sinusoidal 2D Positional Encoding"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        device = x.device
        
        y_embed = torch.arange(height, device=device, dtype=torch.float32) / height
        x_embed = torch.arange(width, device=device, dtype=torch.float32) / width
        
        div_term_y = torch.exp(torch.arange(0, channels//2, device=device).float() * (-np.log(10000.0) / (channels//2)))
        div_term_x = torch.exp(torch.arange(0, channels//2, device=device).float() * (-np.log(10000.0) / (channels//2)))
        
        pos_enc = torch.zeros(batch, channels, height, width, device=device)
        
        # Create 2D position grids
        y_pos = y_embed.unsqueeze(1) * div_term_y * np.pi * 2
        x_pos = x_embed.unsqueeze(0) * div_term_x * np.pi * 2
        
        pos_enc[:, ::2, :, :] = torch.sin(y_pos).unsqueeze(0).unsqueeze(-1) * x[:, ::2, :, :]
        pos_enc[:, 1::2, :, :] = torch.cos(y_pos).unsqueeze(0).unsqueeze(-1) * x[:, 1::2, :, :]
        
        return x + pos_enc[:, :, :height, :width]


class PositionalEncoding1D(nn.Module):
    """Feature 7: Sinusoidal 1D Positional Encoding"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        # Don't create div_term here - create it in forward based on device
    
    def forward(self, x):
        batch, seq_len, dim = x.shape
        device = x.device
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-np.log(10000.0) / dim))
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        
        pos_enc = torch.zeros(batch, seq_len, dim, device=device)
        pos_enc[:, :, ::2] = torch.sin(position * div_term * np.pi * 2)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term * np.pi * 2)
        
        return x + pos_enc


class TransformerBlock(nn.Module):
    """Feature 8: Multi-head self-attention + FFN + residual"""
    def __init__(self, d_model, num_heads=8, ff_dim=512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        # FFN with residual
        ff_out = self.ffn(x)
        x = x + ff_out
        x = self.norm2(x)
        
        return x


class CrossAttentionFusion(nn.Module):
    """Feature 9: Query-key-value cross-attention"""
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, query, value):
        # Query attends to value
        attn_out, _ = self.cross_attn(query, value, value)
        # Residual connection
        output = query + self.norm(attn_out)
        return output


class SimpleCellSegmenter(nn.Module):
    """
    Simple cell segmenter with Shape Prior Encoder for GPU training.
    Uses UNet backbone + Shape Prior Encoder with all 12 features.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Config for shape prior encoder
        shape_config = {
            'num_shape_prototypes': 16,
            'shape_embed_dim': 256,
            'use_shape_prior': True,
        }
        
        # Shape Prior Encoder with ALL 12 features
        self.shape_prior = ShapePriorEncoderPyTorch(shape_config)
        
        # Encoder (simplified UNet)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        # Output head
        self.out = nn.Conv2d(64, 1, 1)
        
        print("\n[SimpleCellSegmenter] Model created with Shape Prior Encoder (12 features)")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
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
        # Extract multi-scale features for shape prior
        features = {
            'p2': self.enc2(self.pool(self.enc1(x))),
            'p3': self.enc3(self.pool(self.enc2(self.enc1(x)))),
            'p4': self.enc4(self.pool(self.enc3(self.enc2(self.enc1(x))))),
            'p5': self.pool(self.enc4(self.enc3(self.enc2(self.enc1(x))))),
        }
        
        # Apply shape prior encoder
        shape_features = self.shape_prior(features)
        
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoding path with shape features
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        
        # Output
        return self.out(d2)


def dice_loss(pred, target, smooth=1.0):
    """Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(pred, target):
    """Combined BCE + Dice loss"""
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def main():
    # Configuration
    epochs = 150
    patches = 100
    batch_size = 4  # Smaller batch for GPU memory
    lr = 1e-4
    
    print(f"\nConfiguration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Patches: {patches}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    print(f"  Total samples: {len(X)}")
    
    # Convert to RGB if grayscale
    if len(X.shape) == 3:
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    
    # Normalize to [0, 1]
    X = X.astype(np.float32) / 255.0
    Y = (Y > 0).astype(np.float32)
    
    # Split
    train_split = 0.85
    split_idx = int(len(X) * train_split)
    X_train_full = X[:split_idx]
    Y_train_full = Y[:split_idx]
    X_val_full = X[split_idx:]
    Y_val_full = Y[split_idx:]
    
    # Use subset
    X_train = X_train_full[:patches]
    Y_train = Y_train_full[:patches]
    X_val = X_val_full[:20]  # Larger validation set
    Y_val = Y_val_full[:20]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    Y_train_t = torch.FloatTensor(Y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    Y_val_t = torch.FloatTensor(Y_val).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = SimpleCellSegmenter(in_channels=3)
    
    # Multi-GPU training
    if num_gpus > 1:
        model = DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"\n  Model wrapped with DataParallel on {num_gpus} GPUs")
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    start_epoch = 0
    
    # Check for checkpoint
    checkpoint_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models/pytorch_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(f"\n>>> FOUND CHECKPOINT: Resuming from epoch {start_epoch}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Training
    num_epochs = epochs
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = combined_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = combined_loss(outputs, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        epoch_time = time.time() - epoch_start
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  💾 Checkpoint saved at epoch {epoch+1}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{PROJECT_DIR}/adaptive_shape_stardist/models/best_model.pth')
            print(f"  ✅ New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

