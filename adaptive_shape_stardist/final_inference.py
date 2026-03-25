"""
Final GPU Inference Script for Adaptive Shape StarDist
======================================================

This script properly handles the StarDist probability map outputs.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import time

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 70)
print("FINAL GPU INFERENCE - ADAPTIVE SHAPE Stardist")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()

print(f"\nDevice: {device}")
print(f"GPU count: {gpu_count}")

if torch.cuda.is_available():
    for i in range(min(gpu_count, 4)):
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

torch.manual_seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Model Definition (same as training)
# =============================================================================
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=128, max_w=128):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pos_embed', torch.randn(1, d_model, max_h, max_w) * 0.1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        pos = self.pos_embed[:, :, :h, :w]
        return x + pos


class PositionalEncoding1D(nn.Module):
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


class TransformerBlock(nn.Module):
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


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, queries, keys, values):
        attn_output = self.cross_attn(queries, keys, values)
        attn_out = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        return self.norm(queries + attn_out)


class ShapePriorEncoderPyTorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_prototypes = config.get('num_prototypes', 16)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 3)
        
        self.shape_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.embedding_dim) * 0.1)
        
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim),
            nn.GELU()
        )
        
        self.pos_enc_2d = PositionalEncoding2D(self.embedding_dim)
        self.pos_enc_1d = PositionalEncoding1D(self.embedding_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim * 4)
            for _ in range(self.num_layers)
        ])
        
        self.cross_attention = CrossAttentionFusion(self.embedding_dim, self.num_heads)
        
        self.channel_projection = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        self.token_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    def forward(self, features):
        batch_size = features['p3'].shape[0]
        feat = features['p3']
        proj = self.feature_projection(feat)
        h, w = proj.shape[2], proj.shape[3]
        pooled = nn.functional.adaptive_avg_pool2d(proj, (16, 16))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        tokens = tokens + self.pos_enc_1d(tokens)
        prototypes = self.shape_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([prototypes, tokens], dim=1)
        
        for transformer in self.transformer_layers:
            combined = transformer(combined)
        
        trans_prototypes = combined[:, :self.num_prototypes, :]
        trans_tokens = combined[:, self.num_prototypes:, :]
        fused = self.cross_attention(trans_tokens, trans_prototypes, trans_prototypes)
        fused = fused.mean(dim=1, keepdim=True)
        fused = fused.transpose(1, 2).view(batch_size, self.embedding_dim, 1, 1)
        fused = nn.functional.interpolate(fused, size=(h, w), mode='bilinear', align_corners=False)
        prior_features = self.channel_projection(fused)
        
        return {
            'prior_features': prior_features,
            'prototype_weights': torch.ones(1, self.num_prototypes) / self.num_prototypes,
            'attention_maps': prior_features,
            'learned_prototypes': self.shape_prototypes,
        }


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
        fpn_features = {'p2': e2, 'p3': e3, 'p4': e4, 'p5': self.pool(e4)}
        return e1, e2, e3, e4, fpn_features


class AdaptiveShapeStarDist(nn.Module):
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
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
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
        shape_feat = nn.functional.interpolate(shape_feat, size=e3.shape[2:], mode='bilinear')
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        shape_feat_d2 = nn.functional.interpolate(shape_feat, size=d2.shape[2:], mode='bilinear')
        d2 = d2 + shape_feat_d2
        return self.out(d2)


# =============================================================================
# Main Inference
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    print(f"  Total samples: {len(X)}")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Y mean: {Y.mean():.2f}, max: {Y.max():.2f}")
    
    # Preprocess
    if len(X.shape) == 3:
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    X = X.astype(np.float32) / 255.0
    
    # Normalize Y to [0, 1] range for probability map
    Y_max = Y.max()
    Y_norm = Y / Y_max
    Y_norm = np.clip(Y_norm, 0, 1)
    
    print(f"  Y normalized mean: {Y_norm.mean():.4f}")
    print(f"  Positive pixels: {(Y_norm > 0.1).sum() / Y_norm.size * 100:.1f}%")
    
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = AdaptiveShapeStarDist(in_channels=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Load checkpoint
    checkpoint_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models/pytorch_full_checkpoint.pth'
    if Path(checkpoint_path).exists():
        print(f"\n>>> Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"    Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"    Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    model.eval()
    
    # GPU warmup
    print("\n🔥 GPU Warmup...")
    dummy = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    torch.cuda.synchronize()
    
    # Run inference on validation set
    print("\n🚀 Running GPU Inference...")
    
    split_idx = int(len(X) * 0.85)
    X_val = torch.FloatTensor(X[split_idx:][:20]).permute(0, 3, 1, 2).to(device)
    Y_val = torch.FloatTensor(Y_norm[split_idx:][:20]).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(X_val)
    inference_time = time.time() - start_time
    
    # Get predictions
    preds = torch.sigmoid(outputs)
    
    print(f"\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Prediction Statistics:")
    print(f"   Shape: {preds.shape}")
    print(f"   Mean: {preds.mean().item():.4f}")
    print(f"   Std: {preds.std().item():.4f}")
    print(f"   Min: {preds.min().item():.4f}")
    print(f"   Max: {preds.max().item():.4f}")
    print(f"   Inference time: {inference_time*1000:.1f} ms")
    
    print(f"\n📊 Ground Truth Statistics:")
    print(f"   Mean: {Y_val.mean().item():.4f}")
    print(f"   Positive pixels: {(Y_val > 0.1).sum() / Y_val.numel() * 100:.1f}%")
    
    # Calculate metrics
    def dice_score(pred, target, threshold=0.5):
        pred_bin = (pred > threshold).float()
        intersection = (pred_bin * target).float().sum()
        return (2. * intersection) / (pred_bin.sum() + target.sum() + 1e-8)
    
    for thresh in [0.1, 0.3, 0.5]:
        dice = dice_score(preds, Y_val, threshold=thresh)
        print(f"\n🎯 Dice Score (threshold={thresh}): {dice.item():.4f}")
    
    # Save predictions
    output_dir = Path(PROJECT_DIR) / 'adaptive_shape_stardist' / 'inference_results'
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'predictions_final.npy', preds.cpu().numpy())
    print(f"\n💾 Predictions saved to: {output_dir / 'predictions_final.npy'}")
    
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

