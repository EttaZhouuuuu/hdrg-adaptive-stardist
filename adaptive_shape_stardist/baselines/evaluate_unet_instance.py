"""
U-Net Instance-Level Evaluation
================================

This script computes INSTANCE-LEVEL metrics for U-Net + Watershed,
which is directly comparable to StarDist's Instance IoU.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import ndimage
from skimage import measure, segmentation
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.xenium_preprocessing import XeniumDataLoader


class Config:
    data_dir = Path(__file__).parent.parent / "data"
    checkpoint_path = Path(__file__).parent / "unet_watershed" / "checkpoints" / "best_unet.pth"
    patch_size = 256
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        return torch.sigmoid(self.final_conv(x))


class CellDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.FloatTensor(images).unsqueeze(1)
        self.masks = torch.FloatTensor(masks > 0).unsqueeze(1).float()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def load_data():
    """Load validation data"""
    print("Loading data...")
    loader = XeniumDataLoader(
        cells_zarr_path=str(Config.data_dir / "cells.zarr"),
        image_path=str(Config.data_dir / "morphology_focus.ome.tif"),
        load_image=True
    )
    
    images, instance_masks = loader.extract_patches(
        patch_size=Config.patch_size,
        num_patches=1000,  # Load more patches
        use_precomputed=True
    )
    
    # Split 85/15
    n_train = int(len(images) * 0.85)
    val_images = images[n_train:]
    val_instance_masks = instance_masks[n_train:]
    
    # Binary masks for U-Net
    binary_masks = (val_instance_masks > 0).astype(np.float32)
    
    print(f"Validation samples: {len(val_images)}")
    return val_images, binary_masks, val_instance_masks


def watershed_post_processing(pred_mask, level=0.5, min_distance=10, min_area=10):
    """Apply watershed for instance segmentation"""
    binary = (pred_mask > level).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(pred_mask, dtype=np.float32)
    
    distance = ndimage.distance_transform_edt(binary)
    markers = measure.label(distance > min_distance)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    
    for region in measure.regionprops(labels):
        if region.area < min_area:
            labels[labels == region.label] = 0
    
    return labels.astype(np.float32)


def compute_instance_metrics(pred_instances, gt_instances):
    """Compute instance-level IoU"""
    pred_labels = np.unique(pred_instances)[1:]
    gt_labels = np.unique(gt_instances)[1:]
    
    if len(gt_labels) == 0:
        return {'mean_iou': 0, 'mean_dice': 0, 'n_pred': 0, 'n_gt': 0}
    
    ious = []
    for gt_label in gt_labels:
        gt_mask = (gt_instances == gt_label)
        best_iou = 0
        best_pred = 0
        for pred_label in pred_labels:
            pred_mask = (pred_instances == pred_label)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_label
        if best_iou > 0:
            ious.append(best_iou)
    
    mean_iou = np.mean(ious) if ious else 0
    mean_dice = 2 * mean_iou / (mean_iou + 1) if mean_iou > 0 else 0
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'n_pred': len(pred_labels),
        'n_gt': len(gt_labels)
    }


def main():
    print("=" * 60)
    print("U-Net Instance-Level Evaluation")
    print("=" * 60)
    
    # Load data
    val_images, val_binary_masks, val_instance_masks = load_data()
    
    # Create dataset and loader
    val_dataset = CellDataset(val_images, val_binary_masks)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Load model
    print("\nLoading U-Net model...")
    model = UNet(in_channels=1, out_channels=1).to(Config.device)
    model.load_state_dict(torch.load(Config.checkpoint_path, map_location=Config.device))
    model.eval()
    print("Model loaded!")
    
    # Evaluate
    print("\nEvaluating on validation set...")
    all_preds = []
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Processing"):
            images = images.to(Config.device)
            outputs = model(images)
            preds = outputs.cpu().numpy()
            all_preds.append(preds)
    
    all_preds = np.vstack(all_preds)
    print(f"Generated {len(all_preds)} predictions")
    
    # Apply watershed and compute instance metrics
    print("\nApplying watershed and computing instance metrics...")
    instance_metrics = []
    
    for i in tqdm(range(len(all_preds)), desc="Instances"):
        pred_instance = watershed_post_processing(
            all_preds[i, 0],
            level=0.5,
            min_distance=10,
            min_area=10
        )
        
        metrics = compute_instance_metrics(pred_instance, val_instance_masks[i])
        instance_metrics.append(metrics)
    
    # Compute average metrics
    avg_iou = np.mean([m['mean_iou'] for m in instance_metrics])
    avg_dice = np.mean([m['mean_dice'] for m in instance_metrics])
    
    print("\n" + "=" * 60)
    print("U-Net + Watershed Instance-Level Results:")
    print("=" * 60)
    print(f"Mean Instance IoU: {avg_iou:.4f}")
    print(f"Mean Instance Dice: {avg_dice:.4f}")
    print("=" * 60)
    
    # Compare with StarDist
    print("\n" + "=" * 60)
    print("COMPARISON WITH STARIST:")
    print("=" * 60)
    print(f"U-Net + Watershed Instance IoU: {avg_iou:.4f}")
    print(f"Adaptive Shape StarDist IoU:     0.9736")
    print("=" * 60)
    
    if avg_iou < 0.9736:
        print(f"\nStarDist beats U-Net by: {0.9736 - avg_iou:.4f} IoU")
    else:
        print(f"\nU-Net beats StarDist by: {avg_iou - 0.9736:.4f} IoU")
    
    return avg_iou, avg_dice


if __name__ == "__main__":
    main()

