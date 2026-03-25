"""
IoUComplete Training Script
==========================
Training script for the ultimate IoUComplete model that combines:
- Complete Much Better features (Transformer, PositionalEncoding, etc.)
- IoU Optimization features (Lovasz, Boundary Loss, BAM)

This represents the ULTIMATE combination of both approaches!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from model_ioucomplete import (
    IoUCompleteModel, 
    CombinedIoUCompleteLoss
)
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============================================================================
# DATASET CLASS
# =============================================================================

class CellDataset(Dataset):
    """Dataset for cell segmentation training"""
    def __init__(self, data_path, transform=None, augment=True):
        self.data = np.load(data_path)
        self.images = self.data['X']
        self.labels = self.data['Y']
        
        # Ensure correct data types and shapes
        self.images = self.images.astype(np.float32)
        # Convert instance IDs to binary masks (0=background, 1=cells)
        self.labels = (self.labels > 0).astype(np.float32)
        
        # Add channel dimension if needed
        if self.images.ndim == 3:
            self.images = self.images[..., np.newaxis]
        if self.labels.ndim == 3:
            self.labels = self.labels[..., np.newaxis]
        
        self.transform = transform
        self.augment = augment
        
        print(f"Dataset loaded: {len(self.images)} samples")
        print(f"Image shape: {self.images.shape}, Label shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            
            # Random vertical flip
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0).copy()
                label = np.flip(label, axis=0).copy()
            
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            label = np.rot90(label, k).copy()
            
            # Random brightness/contrast
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.9, 1.1)
                contrast = np.random.uniform(0.9, 1.1)
                image = np.clip(image * brightness * contrast, 0, 1)
            
            # Random noise
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0, 1)
        
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(label.transpose(2, 0, 1))


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_dice(pred, target, threshold=0.5, eps=1e-8):
    """Compute Dice coefficient"""
    pred_binary = (pred > threshold).astype(np.float32)
    intersection = np.sum(pred_binary * target)
    union = np.sum(pred_binary) + np.sum(target)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def compute_iou(pred, target, threshold=0.5, eps=1e-8):
    """Compute IoU (Jaccard index)"""
    pred_binary = (pred > threshold).astype(np.float32)
    intersection = np.sum(pred_binary * target)
    union = np.sum(pred_binary) + np.sum(target) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou


def compute_mse(pred, target):
    """Compute Mean Squared Error"""
    return np.mean((pred - target) ** 2)


def compute_pearson(pred, target):
    """Compute Pearson correlation coefficient"""
    if np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    return np.corrcoef(pred.flatten(), target.flatten())[0, 1]


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=100,
    device='cuda',
    save_dir='IoUComplete/',
    project_name='IoUComplete_Experiment'
):
    """Train the IoUComplete model"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_dice': [], 'val_iou': [],
        'val_mse': [], 'val_pearson': [],
        'learning_rates': [],
        'loss_components': {'bce': [], 'dice': [], 'iou': [], 'lovasz': []}
    }
    
    best_val_iou = 0.0
    patience_counter = 0
    patience = 50  # Early stopping patience
    
    print(f"\n{'='*60}")
    print("Starting IoUComplete Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # =================================================================
        # TRAINING PHASE
        # =================================================================
        model.train()
        train_loss = 0.0
        loss_components_sum = {'bce': 0.0, 'dice': 0.0, 'iou': 0.0, 'lovasz': 0.0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            pred_center, pred_distance = model(images)
            
            # Create combined target for both center and distance
            target_center = labels  # [B, 1, H, W]
            target_distance = labels.expand(-1, pred_distance.size(1), -1, -1)  # [B, 32, H, W]
            target_combined = torch.cat([target_center, target_distance], dim=1)  # [B, 33, H, W]
            
            # Compute loss
            loss, components = criterion(
                torch.cat([pred_center, pred_distance], dim=1),
                target_combined
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            for k, v in components.items():
                loss_components_sum[k] += v
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average training metrics
        train_loss /= len(train_loader)
        for k in loss_components_sum:
            loss_components_sum[k] /= len(train_loader)
        
        # Update scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # =================================================================
        # VALIDATION PHASE
        # =================================================================
        model.eval()
        val_loss = 0.0
        val_metrics = {'dice': [], 'iou': [], 'mse': [], 'pearson': []}
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(device)
                labels = labels.to(device)
                
                pred_center, pred_distance = model(images)
                pred = torch.sigmoid(pred_center)  # Use center channel only
                
                # Create combined target
                target_center = labels
                target_distance = labels.expand(-1, pred_distance.size(1), -1, -1)
                target_combined = torch.cat([target_center, target_distance], dim=1)
                
                # Compute loss
                loss, components = criterion(
                    torch.cat([pred_center, pred_distance], dim=1),
                    target_combined
                )
                val_loss += loss.item()
                
                # Compute metrics for each sample
                for i in range(pred.shape[0]):
                    pred_i = pred[i, 0].cpu().numpy()
                    label_i = labels[i, 0].cpu().numpy()
                    
                    # Compute metrics
                    dice = compute_dice(pred_i, label_i)
                    iou = compute_iou(pred_i, label_i)
                    mse = compute_mse(pred_i, label_i)
                    pearson = compute_pearson(pred_i, label_i)
                    
                    val_metrics['dice'].append(dice)
                    val_metrics['iou'].append(iou)
                    val_metrics['mse'].append(mse)
                    val_metrics['pearson'].append(pearson)
        
        # Average validation metrics
        val_loss /= len(val_loader)
        val_dice = np.mean(val_metrics['dice'])
        val_iou = np.mean(val_metrics['iou'])
        val_mse = np.mean(val_metrics['mse'])
        val_pearson = np.mean(val_metrics['pearson'])
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_mse'].append(val_mse)
        history['val_pearson'].append(val_pearson)
        history['learning_rates'].append(current_lr)
        for k, v in loss_components_sum.items():
            history['loss_components'][k].append(v)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Dice: {val_dice:.4f} (Best: {best_val_iou:.4f})")
        print(f"  Val IoU: {val_iou:.4f}")
        print(f"  Val MSE: {val_mse:.4f}")
        print(f"  Val Pearson: {val_pearson:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_mse': val_mse,
                'val_pearson': val_pearson,
                'history': history
            }
            torch.save(checkpoint, save_dir / 'ioucomplete_best.pth')
            print(f"  ✅ New best model saved! IoU: {val_iou:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n{'='*60}")
            print(f"Early stopping at epoch {epoch+1}")
            print(f"{'='*60}")
            break
        
        # Save training curves periodically
        if (epoch + 1) % 10 == 0:
            save_training_curves(history, save_dir)
    
    # =================================================================
    # SAVE FINAL RESULTS
    # =================================================================
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Best Val Dice: {np.max(history['val_dice']):.4f}")
    
    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final training curves
    save_training_curves(history, save_dir)
    
    return model, history


def save_training_curves(history, save_dir):
    """Save training curves visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice score
    axes[0, 1].plot(history['val_dice'], label='Val Dice', color='blue')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Validation Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU score
    axes[0, 2].plot(history['val_iou'], label='Val IoU', color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].set_title('Validation IoU')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # MSE
    axes[1, 0].plot(history['val_mse'], label='Val MSE', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Validation MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Pearson correlation
    axes[1, 1].plot(history['val_pearson'], label='Val Pearson', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Pearson Correlation')
    axes[1, 1].set_title('Validation Pearson Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning rate
    axes[1, 2].plot(history['learning_rates'], label='Learning Rate', color='orange')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ioucomplete_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save loss components
    fig, ax = plt.subplots(figsize=(10, 6))
    for component, values in history['loss_components'].items():
        ax.plot(values, label=component.capitalize())
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'ioucomplete_loss_components.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train IoUComplete Model')
    parser.add_argument('--data', type=str, default='training_data.npz',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='IoUComplete/',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        data_path = Path(__file__).parent / args.data
    
    dataset = CellDataset(data_path)
    
    # Split dataset (85% train, 15% val)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = IoUCompleteModel(
        in_channels=1,
        out_channels=32,
        base_channels=64,
        num_prototypes=8,
        num_heads=8,
        num_transformer_layers=2,
        dropout_rate=args.dropout,
        label_smoothing=args.label_smoothing,
        use_bam=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,}")
    
    # Initialize loss function
    criterion = CombinedIoUCompleteLoss(
        bce_weight=0.20,
        dice_weight=0.30,
        iou_weight=0.30,
        lovasz_weight=0.20,
        boundary_weight=0.0,
        label_smoothing=args.label_smoothing
    )
    
    # Initialize optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Initialize scheduler (Cosine Annealing with Warm Restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        project_name='IoUComplete_Experiment'
    )
    
    print("\n✅ Training complete! Check the output directory for results.")
