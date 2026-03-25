"""
PyTorch GPU-Accelerated Training Script for Adaptive Shape StarDist
================================================================

Fixed version with:
1. Deterministic validation (no random sampling)
2. ReduceLROnPlateau
3. Early stopping
4. Best checkpoint saving
5. Larger validation set

Uses PyTorch with GPU for ~10x faster training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import sys
import os
import json
from tqdm import tqdm

# Setup paths
PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

# Configure PyTorch for GPU
print("=" * 70)
print("CONFIGURING GPU")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    # Set determinism for reproducibility
    torch.backends.cudnn.benchmark = False  # Disable for determinism
    torch.backends.cudnn.deterministic = True  # Deterministic conv
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

# Set random seeds globally
torch.manual_seed(42)
np.random.seed(42)

# Suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("ADAPTIVE SHAPE Stardist - PyTorch GPU TRAINING (FIXED)")
    print("=" * 70)
    
    # Arguments
    patches = 100
    epochs = 150
    batch_size = 4
    lr = 1e-4
    
    print(f"\nConfiguration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Patches: {patches}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Device: {device}")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    print(f"  Total samples: {len(X)}")
    
    # Split
    train_split = 0.85  # 85% train, 15% val
    split_idx = int(len(X) * train_split)
    X_train_full = X[:split_idx]
    Y_train_full = Y[:split_idx]
    X_val_full = X[split_idx:]
    Y_val_full = Y[split_idx:]
    
    # Use subset for training
    X_train = X_train_full[:patches]
    Y_train = Y_train_full[:patches]
    
    # FIXED: 增大验证集 - 使用20个样本（原来只有10个）
    X_val = X_val_full[:20]
    Y_val = Y_val_full[:20]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)} (FIXED: larger validation set)")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train / 255.0).unsqueeze(1)  # [N, 1, H, W]
    X_val_tensor = torch.FloatTensor(X_val / 255.0).unsqueeze(1)
    
    # Pre-compute validation dataset (FIXED: deterministic, no random)
    print("\n  Pre-computing validation targets...")
    from scipy.ndimage import distance_transform_edt
    
    val_prob_maps = torch.FloatTensor([(y > 0).astype(np.float32) for y in Y_val]).unsqueeze(1)
    val_dist_maps = torch.FloatTensor([
        np.stack([distance_transform_edt(y > 0) for _ in range(16)], axis=0)
        for y in Y_val
    ])
    
    print(f"  Val prob shape: {val_prob_maps.shape}")
    print(f"  Val dist shape: {val_dist_maps.shape}")
    
    # Simple model for testing
    print("\n" + "=" * 70)
    print("CREATING SIMPLE SEGMENTATION MODEL")
    print("=" * 70)
    
    class SimpleCellSegmenter(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = self._conv_block(1, 32)
            self.enc2 = self._conv_block(32, 64)
            self.enc3 = self._conv_block(64, 128)
            
            # Decoder
            self.dec2 = self._conv_block(128 + 64, 64)
            self.dec1 = self._conv_block(64 + 32, 32)
            
            # Output heads
            self.prob_head = nn.Conv2d(32, 1, kernel_size=1)
            self.dist_head = nn.Conv2d(32, 16, kernel_size=1)
            
        def _conv_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(nn.MaxPool2d(2)(e1))
            e3 = self.enc3(nn.MaxPool2d(2)(e2))
            
            # Decoder with skip connections
            d2 = self.dec2(torch.cat([nn.functional.interpolate(e3, e2.shape[2:], mode='bilinear'), e2], dim=1))
            d1 = self.dec1(torch.cat([nn.functional.interpolate(d2, e1.shape[2:], mode='bilinear'), e1], dim=1))
            
            # Outputs
            prob = torch.sigmoid(self.prob_head(d1))
            dist = torch.relu(self.dist_head(d1))
            
            return {'prob': prob, 'dist': dist}
    
    model = SimpleCellSegmenter().to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer with weight decay (regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    # FIXED: Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Monitor validation loss (minimize)
        factor=0.5,           # Reduce lr by half
        patience=5,           # Wait 5 epochs before reducing
        threshold=1e-4,       # Minimum improvement threshold
        threshold_mode='rel', # Relative improvement
        verbose=True
    )
    print(f"  ✅ ReduceLROnPlateau added (factor=0.5, patience=5)")
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING GPU TRAINING (WITH ALL FIXES)")
    print("=" * 70)
    
    history = {'loss': [], 'val_loss': []}
    X_train_tensor = X_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    
    # FIXED: Pre-computed validation targets on GPU
    val_prob_maps = val_prob_maps.to(device)
    val_dist_maps = val_dist_maps.to(device)
    
    steps_per_epoch = len(X_train_tensor) // batch_size
    
    # FIXED: Best checkpoint tracking & Early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 15  # Stop after 15 epochs without improvement
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # FIXED: Fixed seed for reproducibility
        torch.manual_seed(42 + epoch)
        indices = torch.randperm(len(X_train_tensor))
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        
        for step in pbar:
            # Get batch
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(X_train_tensor))
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train_tensor[batch_indices]
            Y_batch = Y_train[batch_indices.numpy()]
            
            # Create probability and distance maps
            prob_maps = torch.FloatTensor([(y > 0).astype(np.float32) for y in Y_batch]).unsqueeze(1).to(device)
            
            # Distance transform
            dist_maps = torch.FloatTensor([
                np.stack([distance_transform_edt(y > 0) for _ in range(16)], axis=0)
                for y in Y_batch
            ]).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Compute loss
            prob_target = prob_maps
            dist_target = dist_maps
            
            loss = bce_loss(outputs['prob'], prob_target) + mse_loss(outputs['dist'], dist_target / 100.0)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # FIXED: Validation with DETERMINISTIC evaluation
        model.eval()
        with torch.no_grad():  # Disable gradient computation
            # FIXED: Use ALL validation samples (not random subset)
            val_outputs = model(X_val_tensor)
            val_loss = bce_loss(val_outputs['prob'], val_prob_maps) + mse_loss(val_outputs['dist'], val_dist_maps / 100.0)
        
        # Update history
        avg_loss = epoch_loss / steps_per_epoch
        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss.item())
        
        print(f"\nEpoch {epoch+1:3d}/{epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss.item():.4f}", end="")
        
        # FIXED: Update learning rate scheduler
        scheduler.step(val_loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            print(f" | LR: {current_lr:.2e}", end="")
        
        # FIXED: Best checkpoint saving & Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f" | ✅ NEW BEST! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f" | ⏳ No improvement: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch + 1}")
                print(f"   Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
                break
        
        print()
        
        # Save checkpoint every 3 epochs (for resuming)
        if (epoch + 1) % 3 == 0:
            checkpoint_path = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/checkpoint_epoch_{epoch+1}.pt')
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
            }, checkpoint_path)
            
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # FIXED: Restore best model and save
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    if best_model_state is not None:
        # Restore best model
        model.load_state_dict(best_model_state)
        print(f"\n✅ Restored best model from epoch {best_epoch}")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/final_model.pt')
    final_model_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
        }
    }, final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    # Save training history
    history_path = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"💾 Training history saved: {history_path}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   - Total epochs trained: {len(history['loss'])}")
    print(f"   - Best epoch: {best_epoch}")
    print(f"   - Best Val Loss: {best_val_loss:.4f}")
    print(f"   - Final Train Loss: {history['loss'][-1]:.4f}")
    
    return model, history


if __name__ == '__main__':
    main()

