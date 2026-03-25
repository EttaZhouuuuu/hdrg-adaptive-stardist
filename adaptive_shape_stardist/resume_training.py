"""
Resume PyTorch GPU Training from Checkpoint
============================================
Continues training from epoch 36 with all fixes applied
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import sys
import json
from tqdm import tqdm

# Setup paths
PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

# Configure PyTorch for GPU
print("=" * 70)
print("RESUMING GPU TRAINING FROM CHECKPOINT")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(42)

torch.manual_seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore')


def main():
    # Configuration
    start_epoch = 36  # Resume from epoch 36
    epochs = 150
    batch_size = 4
    lr = 1e-4
    
    print(f"\nResuming from epoch {start_epoch}")
    print(f"Target epochs: {epochs}")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    
    # Split 85/15
    split_idx = int(len(X) * 0.85)
    X_train = X[:split_idx][:100]
    Y_train = Y[:split_idx][:100]
    X_val = X[split_idx:][:20]
    Y_val = Y[split_idx:][:20]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train / 255.0).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val / 255.0).unsqueeze(1)
    
    # Pre-compute validation targets
    from scipy.ndimage import distance_transform_edt
    val_prob_maps = torch.FloatTensor([(y > 0).astype(np.float32) for y in Y_val]).unsqueeze(1)
    val_dist_maps = torch.FloatTensor([
        np.stack([distance_transform_edt(y > 0) for _ in range(16)], axis=0)
        for y in Y_val
    ])
    
    # Model
    class SimpleCellSegmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = self._conv_block(1, 32)
            self.enc2 = self._conv_block(32, 64)
            self.enc3 = self._conv_block(64, 128)
            self.dec2 = self._conv_block(128 + 64, 64)
            self.dec1 = self._conv_block(64 + 32, 32)
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
            e1 = self.enc1(x)
            e2 = self.enc2(nn.MaxPool2d(2)(e1))
            e3 = self.enc3(nn.MaxPool2d(2)(e2))
            d2 = self.dec2(torch.cat([nn.functional.interpolate(e3, e2.shape[2:], mode='bilinear'), e2], dim=1))
            d1 = self.dec1(torch.cat([nn.functional.interpolate(d2, e1.shape[2:], mode='bilinear'), e1], dim=1))
            prob = torch.sigmoid(self.prob_head(d1))
            dist = torch.relu(self.dist_head(d1))
            return {'prob': prob, 'dist': dist}
    
    model = SimpleCellSegmenter().to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Load checkpoint
    checkpoint_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/checkpoint_epoch_{start_epoch}.pt'
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Update scheduler with saved best val loss
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"  Resumed from epoch {start_epoch}")
    print(f"  Previous best Val Loss: {best_val_loss:.4f}")
    
    # Move to device
    X_train_tensor = X_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    val_prob_maps = val_prob_maps.to(device)
    val_dist_maps = val_dist_maps.to(device)
    
    # Training
    print("\n" + "=" * 70)
    print(f"CONTINUING TRAINING (Epoch {start_epoch + 1} to {epochs})")
    print("=" * 70)
    
    history = {'loss': [], 'val_loss': []}
    steps_per_epoch = len(X_train_tensor) // batch_size
    
    patience = 15
    patience_counter = 0
    best_epoch = start_epoch
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        torch.manual_seed(42 + epoch)
        indices = torch.randperm(len(X_train_tensor))
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        
        for step in pbar:
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(X_train_tensor))
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train_tensor[batch_indices]
            Y_batch = Y_train[batch_indices.numpy()]
            
            prob_maps = torch.FloatTensor([(y > 0).astype(np.float32) for y in Y_batch]).unsqueeze(1).to(device)
            dist_maps = torch.FloatTensor([
                np.stack([distance_transform_edt(y > 0) for _ in range(16)], axis=0)
                for y in Y_batch
            ]).to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = bce_loss(outputs['prob'], prob_maps) + mse_loss(outputs['dist'], dist_maps / 100.0)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = bce_loss(val_outputs['prob'], val_prob_maps) + mse_loss(val_outputs['dist'], val_dist_maps / 100.0)
        
        avg_loss = epoch_loss / steps_per_epoch
        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss.item())
        
        print(f"\nEpoch {epoch+1:3d}/{epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss.item():.4f}", end="")
        
        scheduler.step(val_loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < lr:
            print(f" | LR: {current_lr:.2e}", end="")
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
            patience_counter = 0
            print(f" | ✅ NEW BEST! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f" | ⏳ No improvement: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch + 1}")
                break
        
        print()
        
        # Save checkpoint every 3 epochs
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
    
    # Save final model
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    final_model_path = Path(f'{PROJECT_DIR}/adaptive_shape_stardist/models_pytorch/final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
    }, final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   - Best epoch: {best_epoch}")
    print(f"   - Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

