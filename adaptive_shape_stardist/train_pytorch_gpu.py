"""
PyTorch GPU-Accelerated Training Script for Adaptive Shape StarDist
================================================================

Uses PyTorch with GPU for ~10x faster training
"""

import numpy as np
import torch
import torch.nn as nn
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("ADAPTIVE SHAPE Stardist - PyTorch GPU TRAINING")
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
    train_split = 0.9
    split_idx = int(len(X) * train_split)
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_val = X[split_idx:]
    Y_val = Y[split_idx:]
    
    # Use subset
    X_train = X_train[:patches]
    Y_train = Y_train[:patches]
    X_val = X_val[:patches//10]
    Y_val = Y_val[:patches//10]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train / 255.0).unsqueeze(1)  # [N, 1, H, W]
    X_val_tensor = torch.FloatTensor(X_val / 255.0).unsqueeze(1)
    
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
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING GPU TRAINING")
    print("=" * 70)
    
    history = {'loss': [], 'val_loss': []}
    X_train_tensor = X_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    
    steps_per_epoch = len(X_train_tensor) // batch_size
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle indices
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
            from scipy.ndimage import distance_transform_edt
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_indices = torch.randperm(len(X_val_tensor))[:batch_size * 2]
            X_val_batch = X_val_tensor[val_indices]
            Y_val_batch = Y_val[val_indices.numpy()]
            
            val_prob_maps = torch.FloatTensor([(y > 0).astype(np.float32) for y in Y_val_batch]).unsqueeze(1).to(device)
            val_dist_maps = torch.FloatTensor([
                np.stack([distance_transform_edt(y > 0) for _ in range(16)], axis=0)
                for y in Y_val_batch
            ]).to(device)
            
            val_outputs = model(X_val_batch)
            val_loss = bce_loss(val_outputs['prob'], val_prob_maps) + mse_loss(val_outputs['dist'], val_dist_maps / 100.0)
        
        # Update history
        avg_loss = epoch_loss / steps_per_epoch
        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss.item())
        
        print(f"\nEpoch {epoch+1:3d}/{epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss.item():.4f}")
        
        # Save checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            checkpoint_path = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/checkpoint_epoch_{epoch+1}.pt')
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, checkpoint_path)
            
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    # Save final checkpoint
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/final_model.pt')
    
    # Save history
    with open(f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n📊 Final Results:")
    print(f"  - Initial Train Loss: {history['loss'][0]:.4f}")
    print(f"  - Final Train Loss:   {history['loss'][-1]:.4f}")
    print(f"  - Best Val Loss:      {min(history['val_loss']):.4f}")
    print(f"\n💾 Model saved to: {PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/")
    
    return model, history

if __name__ == '__main__':
    main()

