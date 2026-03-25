"""
Complete Much Better - IoU Optimized Training Script
=====================================================
Enhanced training script focused on improving IoU through:
1. IoU-aware loss functions (Lovasz, Boundary loss)
2. Boundary Attention module
3. Hard example mining
4. Optimal threshold tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from typing import Optional, Tuple, Dict
import time
import warnings
warnings.filterwarnings('ignore')

# Import from the IoU optimization module
from train_stardist_complete_much_better_iou import (
    AdaptiveShapeStarDistIoU,
    CombinedIoULoss,
    BoundaryAttention,
    LovaszLoss,
    BoundaryLoss,
    IoULoss,
    get_iou_optimized_config
)

# =============================================================================
# Dataset (Same as original)
# =============================================================================

class CellSegmentationDataset(Dataset):
    """Dataset for cell segmentation with augmentation."""
    
    def __init__(self, data_path: str, augment: bool = True):
        data = np.load(data_path, allow_pickle=True)
        self.X = data['X']
        self.Y_raw = data['Y']
        self.augment = augment
        
        # Normalize X to [0, 1]
        if self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32) / 255.0
        
        # Create binary mask from instance IDs
        self.Y = (self.Y_raw > 0).astype(np.float32)
        
        print(f"Dataset loaded: {len(self.X)} samples")
        print(f"  X shape: {self.X.shape}, Y shape: {self.Y.shape}")
        print(f"  Y mean: {self.Y.mean():.4f} (cell coverage)")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        # Add channel dimension
        x = x[np.newaxis, ...]
        y = y[np.newaxis, ...]
        
        if self.augment:
            x, y = self._augment(x, y)
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def _augment(self, x, y):
        """Data augmentation."""
        # Random flip
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()
        
        # Random brightness/contrast
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            x = np.clip(x * factor, 0, 1)
        
        return x, y


# =============================================================================
# Metrics (Same as original)
# =============================================================================

def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict:
    """Calculate segmentation metrics."""
    pred_binary = (pred > threshold).astype(np.float32)
    
    intersection = np.logical_and(pred_binary, target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    dice = 2 * intersection / (pred_binary.sum() + target.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    accuracy = (pred_binary == target).mean()
    precision = intersection / (pred_binary.sum() + 1e-8)
    recall = intersection / (target.sum() + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


# =============================================================================
# Training Function
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {'bce': 0, 'dice': 0, 'lovasz': 0, 'boundary': 0}
    
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        pred = model(X)
        loss, components = criterion(pred, Y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
    
    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


def evaluate(model, dataloader, criterion, device, thresholds=None):
    """Evaluate model on dataset."""
    if thresholds is None:
        thresholds = np.arange(0.3, 0.8, 0.05)
    
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            prob = torch.sigmoid(pred).cpu().numpy()
            all_preds.append(prob)
            all_targets.append(Y.cpu().numpy())
            loss, _ = criterion(pred, Y)
            total_loss += loss.item()
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Find best threshold
    best_dice = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        metrics = calculate_metrics(all_preds, all_targets, thresh)
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            best_threshold = thresh
            best_metrics = metrics
    
    return {
        'loss': total_loss / len(dataloader),
        'threshold': best_threshold,
        **best_metrics
    }


def train_model():
    """Main training function."""
    # Configuration
    config = get_iou_optimized_config()
    
    # Paths
    PROJECT_DIR = Path('/data/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist')
    DATA_PATH = PROJECT_DIR / 'training_data.npz'
    OUTPUT_DIR = PROJECT_DIR / 'models_complete'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Dataset
    dataset = CellSegmentationDataset(str(DATA_PATH), augment=True)
    
    # Split train/val
    n_samples = len(dataset)
    n_train = int(0.85 * n_samples)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_samples - n_train]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Model
    model = AdaptiveShapeStarDistIoU(
        in_channels=config['in_channels'],
        embedding_dim=config['embedding_dim'],
        num_shapes=config['num_shapes'],
        dropout_rate=config['dropout_rate'],
        use_boundary_attention=config['use_boundary_attention']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: AdaptiveShapeStarDistIoU")
    print(f"Parameters: {total_params:,}")
    
    # Loss
    criterion = CombinedIoULoss(
        bce_weight=config['bce_weight'],
        dice_weight=config['dice_weight'],
        lovasz_weight=config['lovasz_weight'],
        boundary_weight=config['boundary_weight'],
        label_smoothing=config['label_smoothing']
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['T_0'],
        T_mult=config['T_mult'],
        eta_min=config['eta_min']
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting IoU-Optimized Training")
    print(f"{'='*60}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Loss: BCE({config['bce_weight']}) + Dice({config['dice_weight']}) + Lovasz({config['lovasz_weight']}) + Boundary({config['boundary_weight']})")
    print(f"{'='*60}\n")
    
    best_iou = 0
    best_dice = 0
    patience_counter = 0
    history = []
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"Thresh: {val_metrics['threshold']:.2f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_dice': val_metrics['dice'],
            'val_iou': val_metrics['iou'],
            'val_threshold': val_metrics['threshold']
        })
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_dice = val_metrics['dice']
            best_threshold = val_metrics['threshold']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'iou': best_iou,
                'dice': best_dice,
                'threshold': best_threshold
            }
            torch.save(checkpoint, OUTPUT_DIR / 'best_complete_much_better_iou.pth')
            print(f"  → New best model saved! IoU: {best_iou:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Best Threshold: {best_threshold:.2f}")
    
    # Save history
    history_np = np.array(history)
    np.save(OUTPUT_DIR / 'training_history_iou.npy', history_np)
    print(f"\nTraining history saved.")
    
    return model, history


if __name__ == "__main__":
    train_model()

