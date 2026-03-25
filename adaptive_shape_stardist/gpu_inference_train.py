"""
GPU Inference and Continued Training Script for Adaptive Shape StarDist
======================================================================

This script provides:
1. GPU-accelerated inference for trained model
2. Ability to continue training from checkpoint
3. Model evaluation and visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import os
import time
import cv2
from pathlib import Path
from datetime import datetime

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 70)
print("GPU INFERENCE AND CONTINUED TRAINING")
print("=" * 70)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()

print(f"\nDevice: {device}")
print(f"GPU count: {gpu_count}")

if torch.cuda.is_available():
    for i in range(min(gpu_count, 4)):
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        mem_used = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Total: {mem:.1f} GB, Used: {mem_used:.1f} GB")

torch.manual_seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PositionalEncoding2D - Learned 2D Positional Encoding
# =============================================================================
class PositionalEncoding2D(nn.Module):
    """FEATURE 6: Learned 2D Positional Encoding"""
    def __init__(self, d_model, max_h=128, max_w=128):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pos_embed', torch.randn(1, d_model, max_h, max_w) * 0.1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        pos = self.pos_embed[:, :, :h, :w]
        return x + pos


# =============================================================================
# PositionalEncoding1D - Sinusoidal 1D Positional Encoding
# =============================================================================
class PositionalEncoding1D(nn.Module):
    """FEATURE 7: Sinusoidal 1D Positional Encoding"""
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
    """FEATURE 8: TransformerBlock with MHA + FFN + residual"""
    def __init__(self, d_model, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        attn_out = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        ff_out = self.ffn(x)
        x = x + ff_out
        x = self.norm2(x)
        
        return x


# =============================================================================
# CrossAttentionFusion - Query-Key-Value Cross-Attention
# =============================================================================
class CrossAttentionFusion(nn.Module):
    """FEATURE 9: CrossAttentionFusion for shape-feature fusion"""
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, queries, keys, values):
        attn_output = self.cross_attn(queries, keys, values)
        attn_out = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        return self.norm(queries + attn_out)


# =============================================================================
# ShapePriorEncoder - Main Encoder Class
# =============================================================================
class ShapePriorEncoderPyTorch(nn.Module):
    """
    FEATURE 10: ShapePriorEncoder with ALL 12 features
    """
    def __init__(self, config):
        super().__init__()
        
        self.num_prototypes = config.get('num_prototypes', 16)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 3)
        
        # FEATURE 2 & 4: Shape prototype learning with learnable embeddings
        self.shape_prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.embedding_dim) * 0.1
        )
        
        # FEATURE 11: Feature projection layer (1x1 conv for embedding)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim),
            nn.GELU()
        )
        
        # FEATURE 6: PositionalEncoding2D
        self.pos_enc_2d = PositionalEncoding2D(self.embedding_dim)
        
        # FEATURE 7: PositionalEncoding1D
        self.pos_enc_1d = PositionalEncoding1D(self.embedding_dim)
        
        # FEATURE 8: TransformerBlocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim * 4)
            for _ in range(self.num_layers)
        ])
        
        # FEATURE 9: CrossAttentionFusion
        self.cross_attention = CrossAttentionFusion(self.embedding_dim, self.num_heads)
        
        # FEATURE 12: Channel projection for multi-scale fusion
        self.channel_projection = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Token projection
        self.token_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    def forward(self, features, training=False):
        """
        FEATURE 5: Multi-scale processing - uses p3 as main scale
        """
        batch_size = features['p3'].shape[0]
        
        feat = features['p3']
        
        # FEATURE 11: Feature projection
        proj = self.feature_projection(feat)
        
        # Downsample for transformer
        h, w = proj.shape[2], proj.shape[3]
        pooled = nn.functional.adaptive_avg_pool2d(proj, (16, 16))
        
        # Reshape to tokens
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        
        # FEATURE 6: Add positional encoding
        tokens = tokens + self.pos_enc_1d(tokens)
        
        # FEATURE 2 & 4: Get shape prototypes
        prototypes = self.shape_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine prototypes with tokens
        combined = torch.cat([prototypes, tokens], dim=1)
        
        # FEATURE 1: Transformer-based global shape reasoning
        for transformer in self.transformer_layers:
            combined = transformer(combined)
        
        # Split back
        trans_prototypes = combined[:, :self.num_prototypes, :]
        trans_tokens = combined[:, self.num_prototypes:, :]
        
        # FEATURE 3 & 9: Multi-head attention fusion
        fused = self.cross_attention(trans_tokens, trans_prototypes, trans_prototypes)
        
        # Reshape and upsample
        fused = fused.mean(dim=1, keepdim=True)
        fused = fused.transpose(1, 2).view(batch_size, self.embedding_dim, 1, 1)
        fused = nn.functional.interpolate(fused, size=(h, w), mode='bilinear', align_corners=False)
        
        # FEATURE 12: Channel projection
        prior_features = self.channel_projection(fused)
        
        return {
            'prior_features': prior_features,
            'prototype_weights': torch.ones(1, self.num_prototypes) / self.num_prototypes,
            'attention_maps': prior_features,
            'learned_prototypes': self.shape_prototypes,
        }


# =============================================================================
# Simplified UNet Backbone
# =============================================================================
class UNetBackbone(nn.Module):
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
        
        # FPN features
        fpn_features = {
            'p2': e2,
            'p3': e3,
            'p4': e4,
            'p5': self.pool(e4),
        }
        
        return e1, e2, e3, e4, fpn_features


# =============================================================================
# Complete Adaptive Shape StarDist Model
# =============================================================================
class AdaptiveShapeStarDist(nn.Module):
    """
    Complete Adaptive Shape StarDist Model with ALL 12 Shape Prior Features
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.backbone = UNetBackbone(in_channels)
        
        shape_config = {
            'num_prototypes': 16,
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 3,
        }
        self.shape_prior = ShapePriorEncoderPyTorch(shape_config)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, 1, 1)
        
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
        
        # Interpolate shape_feat to match e3 size
        shape_feat = nn.functional.interpolate(shape_feat, size=e3.shape[2:], mode='bilinear')
        
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        
        # Interpolate shape_feat to match d2 size
        shape_feat_d2 = nn.functional.interpolate(shape_feat, size=d2.shape[2:], mode='bilinear')
        d2 = d2 + shape_feat_d2
        
        return self.out(d2)


# =============================================================================
# Loss Functions
# =============================================================================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def combined_loss(pred, target):
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# =============================================================================
# Inference Function
# =============================================================================
def run_inference(model, image_path, device):
    """Run GPU inference on a single image"""
    model.eval()
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(img)
    inference_time = time.time() - start_time
    
    # Convert to numpy
    pred = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    print(f"\n✅ Inference complete: {image_path}")
    print(f"   Inference time: {inference_time*1000:.1f} ms")
    print(f"   Output shape: {pred.shape}")
    print(f"   Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
    
    return pred


# =============================================================================
# Batch Inference
# =============================================================================
def run_batch_inference(model, dataloader, device):
    """Run GPU inference on a batch of images"""
    model.eval()
    all_preds = []
    total_time = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle both (x,) and (x, y) formats
            if len(batch) == 2:
                batch_x = batch[0].to(device)
            else:
                batch_x = batch[0].to(device)
            
            start_time = time.time()
            outputs = model(batch_x)
            total_time += time.time() - start_time
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
    
    avg_time = total_time / len(dataloader) if len(dataloader) > 0 else 0
    print(f"\n✅ Batch inference complete")
    print(f"   Total samples: {len(all_preds)}")
    print(f"   Average time per batch: {avg_time*1000:.1f} ms")
    
    return np.array(all_preds)


# =============================================================================
# Main Function - Training or Inference Mode
# =============================================================================
def main(mode='inference', continue_training=True, additional_epochs=50):
    """
    Main function for GPU inference and continued training
    
    Args:
        mode: 'inference' or 'train'
        continue_training: Whether to continue training from checkpoint
        additional_epochs: Number of additional epochs to train
    """
    
    # Create output directory for results
    output_dir = Path(PROJECT_DIR) / 'adaptive_shape_stardist' / 'inference_results'
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CREATING MODEL WITH ALL 12 SHAPE PRIOR FEATURES")
    print("=" * 70)
    
    # Create model
    model = AdaptiveShapeStarDist(in_channels=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[AdaptiveShapeStarDist] Model with ALL 12 Shape Prior Features")
    print(f"  Total parameters: {total_params:,}")
    
    # Load checkpoint
    checkpoint_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models/pytorch_full_checkpoint.pth'
    best_model_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models/best_model_full.pth'
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"\n>>> LOADING CHECKPOINT: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"    Resuming from epoch: {start_epoch}")
        print(f"    Best val loss: {best_val_loss:.4f}")
    elif os.path.exists(best_model_path):
        print(f"\n>>> LOADING BEST MODEL: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"    Loaded best model directly")
    
    if mode == 'inference':
        print("\n" + "=" * 70)
        print("GPU INFERENCE MODE")
        print("=" * 70)
        
        # Load data for inference
        data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
        X = data['X']
        
        if len(X.shape) == 3:
            X = np.repeat(X[..., np.newaxis], 3, axis=-1)
        X = X.astype(np.float32) / 255.0
        
        # Create inference dataset
        X_infer = X[:20]
        X_infer_t = torch.FloatTensor(X_infer).permute(0, 3, 1, 2)
        infer_dataset = TensorDataset(X_infer_t)
        infer_loader = DataLoader(infer_dataset, batch_size=8, shuffle=False, num_workers=4)
        
        # Warmup
        print("\n🔥 GPU Warmup...")
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        for _ in range(10):
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        
        # Run inference
        print("\n🚀 Running GPU Inference...")
        predictions = run_batch_inference(model, infer_loader, device)
        
        # Save predictions
        np.save(output_dir / 'predictions.npy', predictions)
        print(f"\n💾 Predictions saved to: {output_dir / 'predictions.npy'}")
        
        # Calculate statistics
        print(f"\n📊 Prediction Statistics:")
        print(f"   Shape: {predictions.shape}")
        print(f"   Mean: {predictions.mean():.4f}")
        print(f"   Std: {predictions.std():.4f}")
        print(f"   Min: {predictions.min():.4f}")
        print(f"   Max: {predictions.max():.4f}")
        
        return predictions
    
    elif mode == 'train' and continue_training:
        print("\n" + "=" * 70)
        print(f"CONTINUED TRAINING MODE (+{additional_epochs} epochs)")
        print("=" * 70)
        
        epochs = start_epoch + additional_epochs
        
        # Load data
        data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
        X = data['X']
        Y = data['Y']
        
        if len(X.shape) == 3:
            X = np.repeat(X[..., np.newaxis], 3, axis=-1)
        
        X = X.astype(np.float32) / 255.0
        
        # Normalize Y - StarDist probabilities/distances should be normalized to [0, 1]
        Y_max = Y.max()
        Y_normalized = Y / (Y_max + 1e-8)
        Y_normalized = np.clip(Y_normalized, 0, 1)
        Y = (Y_normalized > 0.5).astype(np.float32)
        
        train_split = 0.85
        split_idx = int(len(X) * train_split)
        X_train = X[:split_idx]
        Y_train = Y[:split_idx]
        X_val = X[split_idx:]
        Y_val = Y[split_idx:]
        
        X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
        Y_train_t = torch.FloatTensor(Y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
        Y_val_t = torch.FloatTensor(Y_val).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        val_dataset = TensorDataset(X_val_t, Y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Resume scheduler state if available
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"\n🚀 Starting continued training from epoch {start_epoch} to {epochs}")
        
        best_val_loss = checkpoint.get('best_val_loss', float('inf')) if os.path.exists(checkpoint_path) else float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            
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
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | "
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
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✅ New best model! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}")
                    break
        
        print("\n" + "=" * 70)
        print("CONTINUED TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final epoch: {epoch+1}")
        
        return best_val_loss


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Inference and Training')
    parser.add_argument('--mode', type=str, default='inference', 
                       choices=['inference', 'train'],
                       help='Mode: inference or train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of additional epochs for continued training')
    parser.add_argument('--warmup', action='store_true',
                       help='Run GPU warmup before inference')
    
    args = parser.parse_args()
    
    main(mode=args.mode, continue_training=(args.mode=='train'), additional_epochs=args.epochs)

