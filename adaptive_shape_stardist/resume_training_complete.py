"""
Resume Training Script - Continue training from checkpoint
==========================================================
Resumes training from epoch 30 checkpoint
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

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 80)
print("RESUMING TRAINING FROM CHECKPOINT")
print("=" * 80)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()

print(f"\n📊 Hardware Configuration:")
print(f"   Device: {device}")
print(f"   GPU count: {gpu_count}")

# Deterministic training
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings('ignore')

# Import model components from the complete training script
# We'll redefine them to ensure compatibility
from train_stardist_complete import (
    AdaptiveShapeStarDistComplete, combined_loss, DataAugmentation,
    PositionalEncoding2D, PositionalEncoding1D, TransformerBlock,
    CrossAttentionFusion, ShapePriorEncoderComplete, FocalLoss, dice_loss
)

# Configuration
CONFIG = {
    'epochs': 150,
    'batch_size': 4,
    'learning_rate': 5e-5,
    'weight_decay': 1e-4,
    'max_grad_norm': 1.0,
    'patience': 25,
    'train_split': 0.85,
    'checkpoint_interval': 3,
    'resume_from': 132,  # Resume from epoch 132
}

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# Load and Process Data
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
X = data['X']
Y_raw = data['Y']

print(f"   Raw data: X={X.shape}, Y={Y_raw.shape}")

if len(X.shape) == 3:
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)

X = X.astype(np.float32) / 255.0

# Normalize Y
Y_max_global = Y_raw.max()
print(f"   Global Y max: {Y_max_global:.2f}")
Y_normalized = Y_raw.astype(np.float32) / (Y_max_global + 1e-8)
print(f"   After normalization: min={Y_normalized.min():.4f}, max={Y_normalized.max():.4f}")

# Split data
n_samples = len(X)
indices = np.random.permutation(n_samples)
train_end = int(n_samples * CONFIG['train_split'])
train_idx = indices[:train_end]
val_idx = indices[train_end:]

print(f"\n   Dataset split: {len(train_idx)} train, {len(val_idx)} val")

# Create datasets
X_train = X[train_idx]
Y_train = Y_normalized[train_idx]
X_val = X[val_idx]
Y_val = Y_normalized[val_idx]

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
Y_train_t = torch.FloatTensor(Y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
Y_val_t = torch.FloatTensor(Y_val).unsqueeze(1)

print(f"\n   Tensor shapes:")
print(f"   X_train: {X_train_t.shape}, Y_train: {Y_train_t.shape}")

# Create dataloaders
train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset = TensorDataset(X_val_t, Y_val_t)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

# Augmentation
augment = DataAugmentation(p=0.5)

# Model
print("\n" + "=" * 80)
print("CREATING MODEL (ALL 12 FEATURES)")
print("=" * 80)

model = AdaptiveShapeStarDistComplete(in_channels=3)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\n   Total parameters: {total_params:,}")

# Multi-GPU
if gpu_count > 1:
    print(f"\n   Using {gpu_count} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)

model = model.to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])

# Scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Checkpoints
checkpoint_dir = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_complete')
checkpoint_dir.mkdir(exist_ok=True)

# Resume from checkpoint
resume_path = checkpoint_dir / f'checkpoint_epoch_{CONFIG["resume_from"]}.pth'
print(f"\n📂 Resuming from: {resume_path}")

checkpoint = torch.load(resume_path, map_location=device)

# Handle DataParallel prefix
if gpu_count > 1:
    model.module.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch']
best_val_loss = checkpoint['best_val_loss']
patience_counter = 0  # Initialize patience counter

print(f"   ✓ Resumed from epoch {start_epoch}")
print(f"   ✓ Best val loss so far: {best_val_loss:.4f}")

# Training loop
print("\n" + "=" * 80)
print(f"CONTINUING TRAINING (Epoch {start_epoch + 1}/{CONFIG['epochs']})")
print("=" * 80)

log_file = f'{PROJECT_DIR}/adaptive_shape_stardist/training_complete.log'

for epoch in range(start_epoch, CONFIG['epochs']):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Augmentation
        if np.random.rand() < 0.5:
            batch_x_np = batch_x.cpu().numpy()
            batch_y_np = batch_y.cpu().numpy()
            batch_x_aug, batch_y_aug = augment(batch_x_np, batch_y_np)
            batch_x = torch.FloatTensor(batch_x_aug).to(device)
            batch_y = torch.FloatTensor(batch_y_aug).to(device)
        
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = combined_loss(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        
        train_loss += loss.item()
        num_batches += 1
    
    train_loss /= num_batches
    scheduler.step()
    
    # Validation
    model.eval()
    val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = combined_loss(pred, batch_y)
            val_loss += loss.item()
            num_val_batches += 1
    
    val_loss /= num_val_batches
    
    epoch_time = time.time() - epoch_start
    
    # Log
    current_lr = optimizer.param_groups[0]['lr']
    log_msg = f"Epoch {epoch+1}/{CONFIG['epochs']} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | Train: {train_loss:.4f} | Val: {val_loss:.4f}"
    
    print(log_msg)
    
    with open(log_file, 'a') as f:
        f.write(log_msg + "\n")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        if gpu_count > 1:
            torch.save(model.module.state_dict(), checkpoint_dir / 'best_complete.pth')
        else:
            torch.save(model.state_dict(), checkpoint_dir / 'best_complete.pth')
        
        print(f"   ✓ New best model saved: val_loss={val_loss:.4f}")
    else:
        patience_counter += 1
    
    # Save checkpoint every 3 epochs
    if (epoch + 1) % CONFIG['checkpoint_interval'] == 0:
        ckpt_name = f'checkpoint_epoch_{epoch+1}.pth'
        if gpu_count > 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_dir / ckpt_name)
        else:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_dir / ckpt_name)
        print(f"   ✓ Checkpoint saved: {ckpt_name}")
    
    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\n   ⏹️ Early stopping at epoch {epoch+1}")
        break

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"   Best validation loss: {best_val_loss:.4f}")
print(f"   Model saved to: {checkpoint_dir / 'best_complete.pth'}")

