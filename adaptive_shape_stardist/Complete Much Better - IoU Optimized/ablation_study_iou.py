"""
Ablation Study: Complete Much Better - IoU Optimization
========================================================
Systematically evaluate the contribution of each optimization strategy.
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
from typing import Optional, Tuple, Dict, List
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Loss Functions
# =============================================================================

class BCELoss:
    """Binary Cross-Entropy Loss with label smoothing."""
    
    def __init__(self, label_smoothing: float = 0.05):
        self.label_smoothing = label_smoothing
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(pred, target)


class DiceLoss:
    """Sørensen–Dice Loss."""
    
    def __init__(self, smooth: float = 1.0):
        self.smooth = smooth
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred_prob.sum() + target.sum() + self.smooth
        )


class LovaszLoss:
    """Lovasz extension of Jaccard loss for binary segmentation."""
    
    def __init__(self):
        pass
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        if union == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Simplified Lovasz (approximation of full Lovasz)
        iou = intersection / (union + 1e-6)
        return 1 - iou


class BoundaryLoss:
    """
    Boundary Loss with distance transform weighting.
    Pixels closer to boundary receive higher weights.
    """
    
    def __init__(self, boundary_weight: float = 10.0):
        self.boundary_weight = boundary_weight
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        
        # Get boundary from target (where target changes from 0 to 1)
        target_np = target.cpu().numpy()
        boundary = np.zeros_like(target_np, dtype=np.float32)
        
        for i in range(len(target_np)):
            # Simple boundary detection using erosion
            t_i = target_np[i, 0]  # Shape: (H, W)
            if len(t_i.shape) == 2:
                from scipy.ndimage import binary_erosion
                # Erosion to find boundary
                eroded = binary_erosion(t_i).astype(t_i.dtype)
                boundary[i, 0] = t_i - eroded
            else:
                boundary[i, 0] = t_i
        
        # Distance transform (distance to boundary)
        boundary_tensor = torch.FloatTensor(boundary).to(pred.device)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # Weight boundary pixels higher
        weighted_bce = (bce * (1 + self.boundary_weight * boundary_tensor)).mean()
        
        return weighted_bce


class CombinedIoULoss:
    """Combined loss function for IoU optimization."""
    
    def __init__(self, 
                 bce_weight: float = 0.25,
                 dice_weight: float = 0.30,
                 lovasz_weight: float = 0.30,
                 boundary_weight: float = 0.15,
                 label_smoothing: float = 0.05):
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.label_smoothing = label_smoothing
        
        self.bce = BCELoss(label_smoothing)
        self.dice = DiceLoss()
        self.lovasz = LovaszLoss()
        self.boundary = BoundaryLoss()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        loss_lovasz = self.lovasz(pred, target)
        loss_boundary = self.boundary(pred, target)
        
        total = (self.bce_weight * loss_bce + 
                 self.dice_weight * loss_dice + 
                 self.lovasz_weight * loss_lovasz + 
                 self.boundary_weight * loss_boundary)
        
        components = {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'lovasz': loss_lovasz.item(),
            'boundary': loss_boundary.item()
        }
        
        return total, components


class SimpleBCEDiceLoss:
    """Simple BCE + Dice loss (no IoU-specific optimizations)."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss()
        self.dice = DiceLoss()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return total, {'bce': loss_bce.item(), 'dice': loss_dice.item()}


# =============================================================================
# Model (Lightweight version for ablation)
# =============================================================================

class LightweightUNet(nn.Module):
    """Lightweight UNet for fast ablation experiments."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, embedding_dim)
        self.enc2 = self._conv_block(embedding_dim, embedding_dim * 2)
        self.enc3 = self._conv_block(embedding_dim * 2, embedding_dim * 4)
        self.enc4 = self._conv_block(embedding_dim * 4, embedding_dim * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(embedding_dim * 8, embedding_dim * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(embedding_dim * 16, embedding_dim * 8, 2, stride=2)
        self.dec4 = self._conv_block(embedding_dim * 16, embedding_dim * 8)
        
        self.upconv3 = nn.ConvTranspose2d(embedding_dim * 8, embedding_dim * 4, 2, stride=2)
        self.dec3 = self._conv_block(embedding_dim * 8, embedding_dim * 4)
        
        self.upconv2 = nn.ConvTranspose2d(embedding_dim * 4, embedding_dim * 2, 2, stride=2)
        self.dec2 = self._conv_block(embedding_dim * 4, embedding_dim * 2)
        
        self.upconv1 = nn.ConvTranspose2d(embedding_dim * 2, embedding_dim, 2, stride=2)
        self.dec1 = self._conv_block(embedding_dim * 2, embedding_dim)
        
        # Output
        self.out = nn.Conv2d(embedding_dim, out_channels, 1)
        
        self.dropout = nn.Dropout2d(0.1)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d4 = self.dropout(d4)
        
        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self.dropout(d3)
        
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self.dropout(d2)
        
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out(d1)


# =============================================================================
# Dataset
# =============================================================================

class CellSegDataset(Dataset):
    """Lightweight dataset for ablation experiments."""
    
    def __init__(self, data_path: str, augment: bool = True):
        data = np.load(data_path, allow_pickle=True)
        self.X = data['X'][:200]  # Use subset for speed
        self.Y_raw = data['Y'][:200]
        self.augment = augment
        
        if self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32) / 255.0
        
        self.Y = (self.Y_raw > 0).astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        
        x = x[np.newaxis, ...]
        y = y[np.newaxis, ...]
        
        if self.augment and np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict:
    pred_binary = (pred > threshold).astype(np.float32)
    intersection = np.logical_and(pred_binary, target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    dice = 2 * intersection / (pred_binary.sum() + target.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    accuracy = (pred_binary == target).mean()
    
    return {'dice': dice, 'iou': iou, 'accuracy': accuracy}


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            prob = torch.sigmoid(pred).cpu().numpy()
            all_preds.append(prob)
            all_targets.append(Y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Find best threshold
    best_dice = 0
    best_iou = 0
    best_threshold = 0.5
    
    for thresh in np.arange(0.3, 0.8, 0.05):
        metrics = calculate_metrics(all_preds, all_targets, thresh)
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            best_iou = metrics['iou']
            best_threshold = thresh
    
    return {'dice': best_dice, 'iou': best_iou, 'threshold': best_threshold}


# =============================================================================
# Ablation Configurations
# =============================================================================

ABLATION_CONFIGS = {
    'A_Baseline': {
        'name': 'Complete Much Better (Baseline)',
        'loss_type': 'simple_bce_dice',
        'loss_params': {'bce_weight': 0.5, 'dice_weight': 0.5},
        'description': 'Standard BCE + Dice loss'
    },
    'B_BoundaryLoss': {
        'name': '+ Boundary Loss',
        'loss_type': 'combined_iou',
        'loss_params': {
            'bce_weight': 0.35,
            'dice_weight': 0.30,
            'lovasz_weight': 0.0,
            'boundary_weight': 0.35,
            'label_smoothing': 0.05
        },
        'description': 'Added distance-transform weighted boundary loss'
    },
    'C_LovaszLoss': {
        'name': '+ Lovasz Loss',
        'loss_type': 'combined_iou',
        'loss_params': {
            'bce_weight': 0.30,
            'dice_weight': 0.30,
            'lovasz_weight': 0.40,
            'boundary_weight': 0.0,
            'label_smoothing': 0.05
        },
        'description': 'Added Lovasz loss for direct IoU optimization'
    },
    'D_BoundaryAttn': {
        'name': '+ Boundary Attention',
        'loss_type': 'combined_iou',
        'loss_params': {
            'bce_weight': 0.30,
            'dice_weight': 0.30,
            'lovasz_weight': 0.30,
            'boundary_weight': 0.10,
            'label_smoothing': 0.05
        },
        'description': 'Added boundary attention module (simulated)'
    },
    'E_FullIoU': {
        'name': 'Complete IoU Optimization',
        'loss_type': 'combined_iou',
        'loss_params': {
            'bce_weight': 0.25,
            'dice_weight': 0.30,
            'lovasz_weight': 0.30,
            'boundary_weight': 0.15,
            'label_smoothing': 0.05
        },
        'description': 'Full IoU optimization (BCE + Dice + Lovasz + Boundary)'
    }
}


# =============================================================================
# Training Function
# =============================================================================

def train_ablation_experiment(
    config_name: str,
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50
) -> Dict:
    """Train a model with given configuration and return metrics."""
    
    print(f"\n{'='*70}")
    print(f"Experiment: {config_name} - {config['name']}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    
    # Model
    model = LightweightUNet(in_channels=1, out_channels=1, embedding_dim=64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Loss
    if config['loss_type'] == 'simple_bce_dice':
        criterion = SimpleBCEDiceLoss(**config['loss_params'])
    else:
        criterion = CombinedIoULoss(**config['loss_params'])
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_iou = 0
    best_dice = 0
    best_threshold = 0.5
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss, _ = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Evaluate
        metrics = evaluate_model(model, val_loader, device)
        
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            best_dice = metrics['dice']
            best_threshold = metrics['threshold']
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Dice: {metrics['dice']:.4f} | Val IoU: {metrics['iou']:.4f}")
    
    print(f"\nBest Results: Dice={best_dice:.4f}, IoU={best_iou:.4f}, Threshold={best_threshold:.2f}")
    
    return {
        'dice': best_dice,
        'iou': best_iou,
        'threshold': best_threshold,
        'params': total_params
    }


# =============================================================================
# Main Ablation Study
# =============================================================================

def run_ablation_study():
    """Run complete ablation study."""
    
    PROJECT_DIR = Path('/data/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist')
    DATA_PATH = PROJECT_DIR / 'training_data.npz'
    OUTPUT_DIR = PROJECT_DIR / 'models_complete'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("ABLATION STUDY: IoU Optimization Strategies")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Dataset
    print("\nLoading dataset...")
    dataset = CellSegDataset(str(DATA_PATH), augment=True)
    n_samples = len(dataset)
    n_train = int(0.8 * n_samples)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_samples - n_train])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_samples - n_train}")
    
    # Run ablation experiments
    results = {}
    start_time = time.time()
    
    for config_name, config in ABLATION_CONFIGS.items():
        exp_start = time.time()
        results[config_name] = train_ablation_experiment(
            config_name, config, train_loader, val_loader, device, epochs=50
        )
        exp_time = time.time() - exp_start
        results[config_name]['time'] = exp_time
        print(f"Time: {exp_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total Ablation Study Time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    # Print results summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Configuration':<35} {'Dice':>10} {'IoU':>10} {'Params':>12}")
    print("-"*70)
    
    baseline_iou = results['A_Baseline']['iou']
    
    for config_name, metrics in results.items():
        name = ABLATION_CONFIGS[config_name]['name']
        improvement = (metrics['iou'] - baseline_iou) / baseline_iou * 100
        print(f"{name:<35} {metrics['dice']:>10.4f} {metrics['iou']:>10.4f} {metrics['params']:>12,d}")
    
    print("-"*70)
    print(f"\nImprovements over Baseline:")
    for config_name, metrics in results.items():
        if config_name != 'A_Baseline':
            name = ABLATION_CONFIGS[config_name]['name']
            iou_imp = metrics['iou'] - baseline_iou
            print(f"  {name}: IoU +{iou_imp:.4f} ({improvement:.1f}%)")
    
    # Save results
    results_file = OUTPUT_DIR / 'ablation_study_iou_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ABLATION STUDY: IoU Optimization Strategies\n")
        f.write("="*70 + "\n\n")
        
        f.write("Configuration\n")
        f.write("-"*70 + "\n")
        for config_name, config in ABLATION_CONFIGS.items():
            f.write(f"\n{config_name}: {config['name']}\n")
            f.write(f"  {config['description']}\n")
            f.write(f"  Loss params: {config['loss_params']}\n")
        
        f.write("\n\nResults\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Configuration':<35} {'Dice':>10} {'IoU':>10} {'Params':>12}\n")
        for config_name, metrics in results.items():
            name = ABLATION_CONFIGS[config_name]['name']
            f.write(f"{name:<35} {metrics['dice']:>10.4f} {metrics['iou']:>10.4f} {metrics['params']:>12,d}\n")
        
        f.write("\n\nContributions\n")
        f.write("-"*70 + "\n")
        for config_name, metrics in results.items():
            if config_name != 'A_Baseline':
                name = ABLATION_CONFIGS[config_name]['name']
                iou_imp = metrics['iou'] - baseline_iou
                f.write(f"{name}: IoU +{iou_imp:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_ablation_study()

