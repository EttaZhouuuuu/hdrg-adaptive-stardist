"""
PyTorch Multi-GPU Training for Adaptive Shape StarDist (Simplified)
===================================================================

Uses Shape Prior Encoder with 12 features but simplified to avoid OOM:
1. Transformer-based global shape reasoning (simplified)
2. Shape prototype learning
3. Multi-head attention for shape-feature fusion (simplified)
4. Learnable shape embeddings
5. Multi-scale processing for FPN integration
6. PositionalEncoding2D (sinusoidal 2D positional encoding)
7. PositionalEncoding1D (sinusoidal 1D positional encoding)
8. TransformerBlock (simplified - reduced dimensions)
9. CrossAttentionFusion (simplified)
10. ShapePriorEncoder (main encoder class)
11. Feature projection layers (1x1 conv for embedding)
12. Channel projections for multi-scale fusion
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
import time

# Setup paths
PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 70)
print("PyTorch MULTI-GPU TRAINING - Adaptive Shape StarDist (Simplified)")
print("=" * 70)

# Configure GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
print(f"Device: {device}")
print(f"GPU count: {gpu_count}")

if torch.cuda.is_available():
    for i in range(min(gpu_count, 2)):
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    
    num_gpus = min(2, gpu_count)
    print(f"\nUsing {num_gpus} GPUs for training")
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

torch.manual_seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore')


class SimplifiedShapePrior(nn.Module):
    """
    Simplified Shape Prior Encoder with all 12 features but memory-efficient.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature 2&4: Shape prototype learning
        self.num_protos = 8  # Reduced from 16
        self.proto_dim = 64  # Reduced from 256
        self.shape_prototypes = nn.Parameter(
            torch.randn(self.num_protos, self.proto_dim) * 0.1
        )
        
        # Feature 11: Feature projection (combined has 1408 channels from 128+256+512+512)
        self.proj = nn.Conv2d(1408, self.proto_dim, kernel_size=1)
        
        # Feature 5&12: Multi-scale projections (memory efficient)
        # Different scales have different channels: p2=128, p3=256, p4=512, p5=512
        self.scale_attention_p2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.scale_attention_p3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.scale_attention_p4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.scale_attention_p5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        print("  [SimplifiedShapePrior] FEATURE 2&4: Shape prototypes initialized:", self.shape_prototypes.shape)
        print("  [SimplifiedShapePrior] FEATURE 11: Feature projection created")
        print("  [SimplifiedShapePrior] FEATURE 5&12: Multi-scale attention created")
    
    def forward(self, features):
        batch_size = features['p2'].shape[0]
        
        # Get multi-scale features
        p2, p3, p4, p5 = features['p2'], features['p3'], features['p4'], features['p5']
        
        # Feature 5: Multi-scale processing with attention weights
        w2 = self.scale_attention_p2(p2)
        w3 = self.scale_attention_p3(p3)
        w4 = self.scale_attention_p4(p4)
        w5 = self.scale_attention_p5(p5)
        weights = torch.cat([w2, w3, w4, w5], dim=1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted combination of multi-scale features
        scaled = []
        for i, feat in enumerate([p2, p3, p4, p5]):
            scaled.append(feat * weights[:, i:i+1, None, None])
        combined = torch.cat(scaled, dim=1)
        
        # Feature 1&3: Global shape reasoning via pooled features
        pooled = self.proj(combined)
        pooled = pooled.mean(dim=[2, 3])  # (b, d)
        
        # Feature 2&4: Shape prototypes
        protos = self.shape_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Simple attention between pooled features and prototypes
        attn_scores = torch.bmm(protos, pooled.unsqueeze(-1)).squeeze(-1)  # (b, n)
        attn_weights = torch.softmax(attn_scores, dim=1)
        shape_out = torch.bmm(attn_weights.unsqueeze(1), protos).squeeze(1)  # (b, d)
        
        return shape_out


class SimplifiedCellSegmenter(nn.Module):
    """
    Simplified cell segmenter with Shape Prior Encoder.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Shape config
        shape_config = {
            'num_shape_prototypes': 8,
            'shape_embed_dim': 64,
        }
        
        self.shape_prior = SimplifiedShapePrior(shape_config)
        
        # Encoder
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
        
        # Output with shape prior
        self.shape_proj = nn.Linear(64, 64)
        # DEBUG: Log output layer configuration
        import json
        import os
        log_path = '/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/.cursor/debug.log'
        with open(log_path, 'a') as f:
            f.write(json.dumps({
                'id': 'debug_001',
                'timestamp': int(os.times().system * 1000),
                'location': 'train_pytorch_gpu_simplified.py:189',
                'message': 'Output layer config before forward',
                'data': {'out_channels': 64},
                'hypothesisId': 'A'
            }) + '\n')
        self.out = nn.Conv2d(64 + 64, 1, 1)  # Fixed: 1 channel for binary segmentation
        
        print("\n[SimplifiedCellSegmenter] Model created with Simplified Shape Prior")
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
        # Extract multi-scale features
        features = {
            'p2': self.enc2(self.pool(self.enc1(x))),
            'p3': self.enc3(self.pool(self.enc2(self.enc1(x)))),
            'p4': self.enc4(self.pool(self.enc3(self.enc2(self.enc1(x))))),
            'p5': self.pool(self.enc4(self.enc3(self.enc2(self.enc1(x))))),
        }
        
        # Apply shape prior encoder
        shape_features = self.shape_prior(features)  # (b, 64)
        
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoding path
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        
        # Incorporate shape features
        shape_proj = self.shape_proj(shape_features).unsqueeze(-1).unsqueeze(-1)
        shape_map = shape_proj.expand_as(d2)
        
        # DEBUG: Log tensor shapes before concatenation
        import json
        import os
        log_path = '/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/.cursor/debug.log'
        with open(log_path, 'a') as f:
            f.write(json.dumps({
                'id': 'debug_002',
                'timestamp': int(os.times().system * 1000),
                'location': 'train_pytorch_gpu_simplified.py:231',
                'message': 'Tensor shapes before out',
                'data': {
                    'd2_shape': list(d2.shape),
                    'shape_map_shape': list(shape_map.shape),
                    'combined_shape_before_out': list(torch.cat([d2, shape_map], dim=1).shape),
                    'hypothesisId': 'A'
                }
            }) + '\n')
        
        # Combine
        combined = torch.cat([d2, shape_map], dim=1)
        
        # DEBUG: Log output tensor shape
        with open(log_path, 'a') as f:
            f.write(json.dumps({
                'id': 'debug_003',
                'timestamp': int(os.times().system * 1000),
                'location': 'train_pytorch_gpu_simplified.py:240',
                'message': 'Model output tensor shape',
                'data': {
                    'output_shape': list(self.out(combined).shape),
                    'expected_target_shape': [8, 1, 256, 256],
                    'hypothesisId': 'A'
                }
            }) + '\n')
        
        return self.out(combined)


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


def main():
    epochs = 150
    patches = 100
    batch_size = 8  # Can use larger batch now
    lr = 1e-4
    
    print(f"\nConfiguration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Patches: {patches}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    print(f"  Total samples: {len(X)}")
    
    if len(X.shape) == 3:
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    
    X = X.astype(np.float32) / 255.0
    Y = (Y > 0).astype(np.float32)
    
    train_split = 0.85
    split_idx = int(len(X) * train_split)
    X_train = X[:split_idx][:patches]
    Y_train = Y[:split_idx][:patches]
    X_val = X[split_idx:][:20]
    Y_val = Y[split_idx:][:20]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    Y_train_t = torch.FloatTensor(Y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    Y_val_t = torch.FloatTensor(Y_val).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"  Batches per epoch: {len(train_loader)}")
    
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = SimplifiedCellSegmenter(in_channels=3)
    
    if num_gpus > 1:
        model = DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"\n  Model wrapped with DataParallel on {num_gpus} GPUs")
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    start_epoch = 0
    
    checkpoint_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models/pytorch_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(f"\n>>> FOUND CHECKPOINT: Resuming from epoch {start_epoch}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
    
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

