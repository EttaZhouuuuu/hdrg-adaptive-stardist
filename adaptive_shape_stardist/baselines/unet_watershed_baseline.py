"""
U-Net Semantic Segmentation Baseline
====================================

This script implements U-Net for semantic segmentation of cell images,
followed by watershed post-processing to obtain instance segmentation.

The semantic segmentation baseline is crucial for comparison because:
1. U-Net is the de facto standard for biomedical image segmentation
2. It provides a pixel-level classification baseline without shape priors
3. Comparing against U-Net demonstrates the value of StarDist's shape modeling

Reference: Ronneberger et al. (2015) U-Net: Convolutional Networks
for Biomedical Image Segmentation

Author: Yitong Zhou
Date: 2026-02-10
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import ndimage
from skimage import measure, segmentation
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.xenium_preprocessing import XeniumDataLoader, create_training_dataset


# Configuration
class Config:
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "baselines" / "unet_watershed"
    model_path = output_dir / "checkpoints"
    
    # Data parameters
    patch_size = 256
    num_patches = 1000
    train_split = 0.85
    
    # Model parameters
    in_channels = 1
    out_channels = 1
    features = [64, 128, 256, 512]
    
    # Training parameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100
    weight_decay = 1e-5
    
    # Watershed parameters
    watershed_level = 0.5
    min_distance = 10
    min_area = 10
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation
    
    Reference: Ronneberger et al. (2015) U-Net: Convolutional Networks
    for Biomedical Image Segmentation
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)  # Skip connection
            x = self.ups[idx + 1](x)  # Double convolution
        
        return torch.sigmoid(self.final_conv(x))


class CellDataset(Dataset):
    """Dataset for cell segmentation"""
    
    def __init__(self, images, masks):
        self.images = torch.FloatTensor(images).unsqueeze(1)  # Add channel dim
        self.masks = torch.FloatTensor(masks > 0).unsqueeze(1).float()  # Binary mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def load_data():
    """Load and preprocess data using XeniumDataLoader"""
    print("Loading data from Xenium dataset...")
    
    # Create training dataset
    data_path = Config.data_dir / "cells.zarr"
    image_path = Config.data_dir / "morphology_focus.ome.tif"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Load images and masks
    loader = XeniumDataLoader(
        cells_zarr_path=str(data_path),
        image_path=str(image_path) if image_path.exists() else None,
        load_image=True
    )
    
    # Extract patches
    images, instance_masks = loader.extract_patches(
        patch_size=Config.patch_size,
        num_patches=Config.num_patches,
        use_precomputed=True
    )
    
    # Create binary masks for semantic segmentation
    binary_masks = (instance_masks > 0).astype(np.float32)
    
    print(f"Loaded {len(images)} samples")
    print(f"Image shape: {images.shape}, Mask shape: {binary_masks.shape}")
    
    return images, binary_masks, instance_masks


def watershed_post_processing(pred_mask, level=0.5, min_distance=10, min_area=10):
    """
    Apply watershed algorithm to convert semantic segmentation to instance segmentation
    """
    # Binarize the prediction
    binary = (pred_mask > level).astype(np.uint8)
    
    if binary.sum() == 0:
        return np.zeros_like(pred_mask, dtype=np.float32)
    
    # Find distance transform
    distance = ndimage.distance_transform_edt(binary)
    
    # Find local maxima for markers
    markers = measure.label(distance > min_distance)
    
    # Apply watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    
    # Filter small regions
    for region in measure.regionprops(labels):
        if region.area < min_area:
            labels[labels == region.label] = 0
    
    return labels.astype(np.float32)


def compute_instance_metrics(pred_instances, gt_instances):
    """
    Compute instance-level metrics (IoU, Dice at instance level)
    """
    # Get unique labels (excluding background 0)
    pred_labels = np.unique(pred_instances)[1:]
    gt_labels = np.unique(gt_instances)[1:]
    
    if len(gt_labels) == 0:
        return {'mean_iou': 0, 'mean_dice': 0}
    
    # Compute per-instance IoU
    ious = []
    for gt_label in gt_labels:
        gt_mask = (gt_instances == gt_label)
        best_iou = 0
        for pred_label in pred_labels:
            pred_mask = (pred_instances == pred_label)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou)
        if best_iou > 0:
            ious.append(best_iou)
    
    mean_iou = np.mean(ious) if ious else 0
    mean_dice = 2 * mean_iou / (mean_iou + 1) if mean_iou > 0 else 0
    
    return {'mean_iou': mean_iou, 'mean_dice': mean_dice}


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Convert to numpy for further processing
            preds = outputs.cpu().numpy()
            all_preds.append(preds)
            all_masks.append(masks.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_masks = np.vstack(all_masks)
    
    # Compute pixel-level metrics
    pred_binary = (all_preds > 0.5).astype(np.float32)
    intersection = (pred_binary * all_masks).sum()
    union = pred_binary.sum() + all_masks.sum() - intersection
    pixel_iou = intersection / (union + 1e-8)
    pixel_dice = 2 * intersection / (pred_binary.sum() + all_masks.sum() + 1e-8)
    
    return total_loss / len(dataloader), pixel_iou, pixel_dice, all_preds, all_masks


def main():
    """Main function for training U-Net baseline"""
    
    print("=" * 60)
    print("U-Net Semantic Segmentation Baseline")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.model_path, exist_ok=True)
    
    # Load data
    images, binary_masks, instance_masks = load_data()
    
    # Split data
    n_samples = len(images)
    n_train = int(n_samples * Config.train_split)
    
    train_images, train_masks = images[:n_train], binary_masks[:n_train]
    train_instance_masks = instance_masks[:n_train]
    val_images, val_masks = images[n_train:], binary_masks[n_train:]
    val_instance_masks = instance_masks[n_train:]
    
    # Create datasets
    train_dataset = CellDataset(train_images, train_masks)
    val_dataset = CellDataset(val_images, val_masks)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Create model
    model = UNet(
        in_channels=Config.in_channels,
        out_channels=Config.out_channels,
        features=Config.features
    ).to(Config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function (Dice + BCE)
    bce_loss = nn.BCELoss()
    
    def combined_loss(pred, target):
        bce = bce_loss(pred, target)
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)
        return 0.5 * bce + 0.5 * dice
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs
    )
    
    # Training loop
    best_iou = 0
    train_losses = []
    val_losses = []
    val_ious = []
    
    print("\nStarting training...")
    
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, combined_loss, optimizer, Config.device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, pixel_iou, pixel_dice, val_preds, val_masks_np = validate(
            model, val_loader, combined_loss, Config.device
        )
        val_losses.append(val_loss)
        val_ious.append(pixel_iou)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Pixel IoU: {pixel_iou:.4f}, Pixel Dice: {pixel_dice:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if pixel_iou > best_iou:
            best_iou = pixel_iou
            torch.save(model.state_dict(), str(Config.model_path / "best_unet.pth"))
            print(f"Saved best model with IoU: {best_iou:.4f}")
    
    # Apply watershed post-processing to validation set
    print("\nApplying watershed post-processing...")
    instance_metrics = []
    
    for i in range(len(val_preds)):
        pred_instance = watershed_post_processing(
            val_preds[i, 0],
            level=Config.watershed_level,
            min_distance=Config.min_distance,
            min_area=Config.min_area
        )
        
        metrics = compute_instance_metrics(pred_instance, val_instance_masks[i])
        instance_metrics.append(metrics)
    
    # Compute average instance metrics
    avg_instance_iou = np.mean([m['mean_iou'] for m in instance_metrics])
    avg_instance_dice = np.mean([m['mean_dice'] for m in instance_metrics])
    
    print(f"\nInstance-level metrics (after watershed):")
    print(f"Mean IoU: {avg_instance_iou:.4f}")
    print(f"Mean Dice: {avg_instance_dice:.4f}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'best_pixel_iou': best_iou,
        'instance_iou': avg_instance_iou,
        'instance_dice': avg_instance_dice
    }
    
    np.save(Config.output_dir / "training_results.npy", results)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Validation IoU')
    
    plt.tight_layout()
    plt.savefig(Config.output_dir / "training_curves.png", dpi=150)
    
    print(f"\nResults saved to {Config.output_dir}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
