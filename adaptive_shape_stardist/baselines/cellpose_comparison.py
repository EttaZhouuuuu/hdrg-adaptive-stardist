"""
Cellpose Comparison Experiment
=============================

This script provides comparison with Cellpose, a generalist cell segmentation method.

Cellpose is a flow-based generalist that has shown excellent performance on
various cell segmentation tasks without requiring task-specific training.

Reference: Stringer et al. (2020) Cellpose: a generalist algorithm for
cellular segmentation
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage
from skimage import morphology, measure, segmentation
import matplotlib.pyplot as plt

# Try to import cellpose
try:
    from cellpose import models, io, plot
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: cellpose not installed. Please install with: pip install cellpose")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_utils import load_zarr_data, normalize_images


class Config:
    # Paths
    data_path = "data/cells.zarr"
    output_dir = "baselines/cellpose_comparison"
    
    # Data parameters
    img_size = 256
    train_split = 0.85
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cellpose parameters
    # Available models: 'cyto', 'cyto2', 'nuclei', 'bact', 'bact_omni'
    cellpose_model = 'cyto2'
    diameter = 30.0  # Expected cell diameter in pixels
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    stitch_threshold = 0.5


def compute_metrics(pred_instances, gt_instances):
    """
    Compute instance-level metrics
    
    Parameters:
    -----------
    pred_instances : np.ndarray
        Predicted instance segmentation
    gt_instances : np.ndarray
        Ground truth instance segmentation
    
    Returns:
    --------
    dict : Dictionary containing instance metrics
    """
    # Get unique labels (excluding background 0)
    pred_labels = np.unique(pred_instances)[1:]
    gt_labels = np.unique(gt_instances)[1:]
    
    # Compute per-instance IoU
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
    
    # Compute precision, recall, F1 at different IoU thresholds
    true_positives = sum(1 for iou in ious if iou >= 0.5)
    precision = true_positives / len(pred_labels) if len(pred_labels) > 0 else 0
    recall = true_positives / len(gt_labels) if len(gt_labels) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'precision_50': precision,
        'recall_50': recall,
        'f1_50': f1,
        'n_pred': len(pred_labels),
        'n_gt': len(gt_labels)
    }


def run_cellpose(images, model, diameter, flow_threshold, cellprob_threshold):
    """
    Run Cellpose segmentation on a batch of images
    
    Parameters:
    -----------
    images : np.ndarray
        Input images (N, H, W)
    model : cellpose model
        Pre-trained Cellpose model
    diameter : float
        Expected cell diameter
    flow_threshold : float
        Flow threshold for segmentation
    cellprob_threshold : float
        Cell probability threshold
    
    Returns:
    --------
    masks : np.ndarray
        Instance segmentation masks
    """
    # Cellpose expects (N, H, W, C) or (N, H, W) with channel dimension
    if len(images.shape) == 3:
        images = np.stack([images] * 3, axis=-1)  # Convert to RGB
    
    # Run Cellpose
    masks, flows, styles = model.eval(
        images,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[1, 0],  # Grayscale input
        do_3D=False
    )
    
    return masks


def create_pseudo_labels_with_watershed(images, diameter=30):
    """
    Create pseudo instance labels using watershed on predicted probabilities
    
    This is an alternative when Cellpose is not available
    """
    from skimage.feature import peak_local_max
    
    masks = []
    
    for img in tqdm(images, desc="Creating pseudo labels"):
        # Simple threshold and watershed
        binary = (img > 0.5).astype(np.uint8)
        
        if binary.sum() == 0:
            masks.append(np.zeros_like(img, dtype=np.float32))
            continue
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima as markers
        markers = peak_local_max(distance, min_distance=int(diameter/3), 
                                  threshold_abs=0.1)
        
        if len(markers) == 0:
            markers = np.zeros_like(img, dtype=np.int32)
        else:
            marker_mask = np.zeros_like(img, dtype=np.int32)
            for i, (y, x) in enumerate(markers):
                marker_mask[y, x] = i + 1
            markers = measure.label(marker_mask)
        
        # Watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)
        masks.append(labels.astype(np.float32))
    
    return np.array(masks)


def main():
    """Main function for Cellpose comparison"""
    
    print("=" * 60)
    print("Cellpose Comparison Experiment")
    print("=" * 60)
    
    if not CELLPOSE_AVAILABLE:
        print("Cellpose is not installed. Using alternative method...")
        use_cellpose = False
    else:
        print("Cellpose is available.")
        use_cellpose = True
    
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    zarr_data = zarr.open(Config.data_path, mode="r")
    images = zarr_data['homogeneous_transform'][:]
    gt_masks = zarr_data['masks'][:]
    
    # Normalize images
    images = normalize_images(images)
    
    # Convert gt_masks to instance format if needed
    # Assuming gt_masks is already in instance format
    
    print(f"Loaded {len(images)} samples")
    
    # Split data
    n_samples = len(images)
    n_train = int(n_samples * Config.train_split)
    
    val_images = images[n_train:]
    val_gt_masks = gt_masks[n_train:]
    
    if use_cellpose:
        # Initialize Cellpose model
        print("\nInitializing Cellpose model...")
        model = models.CellposeModel(
            gpu=True if Config.device.type == 'cuda' else False,
            model_type=Config.cellpose_model
        )
        
        print(f"Using model: {Config.cellpose_model}")
        print(f"Expected diameter: {Config.diameter}")
        
        # Run Cellpose on validation set
        print("\nRunning Cellpose segmentation...")
        pred_masks = run_cellpose(
            val_images,
            model,
            diameter=Config.diameter,
            flow_threshold=Config.flow_threshold,
            cellprob_threshold=Config.cellprob_threshold
        )
    else:
        # Use alternative method
        print("\nUsing watershed-based pseudo labels...")
        pred_masks = create_pseudo_labels_with_watershed(val_images, diameter=Config.diameter)
    
    # Compute metrics
    print("\nComputing metrics...")
    all_metrics = []
    
    for i in tqdm(range(len(pred_masks)), desc="Computing metrics"):
        metrics = compute_metrics(pred_masks[i], val_gt_masks[i])
        all_metrics.append(metrics)
    
    # Compute average metrics
    avg_metrics = {
        'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
        'mean_dice': np.mean([m['mean_dice'] for m in all_metrics]),
        'precision_50': np.mean([m['precision_50'] for m in all_metrics]),
        'recall_50': np.mean([m['recall_50'] for m in all_metrics]),
        'f1_50': np.mean([m['f1_50'] for m in all_metrics])
    }
    
    print("\n" + "=" * 60)
    print("Cellpose Results:")
    print("=" * 60)
    print(f"Mean IoU: {avg_metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {avg_metrics['mean_dice']:.4f}")
    print(f"Precision@0.5: {avg_metrics['precision_50']:.4f}")
    print(f"Recall@0.5: {avg_metrics['recall_50']:.4f}")
    print(f"F1@0.5: {avg_metrics['f1_50']:.4f}")
    
    # Save results
    results = {
        'all_metrics': all_metrics,
        'avg_metrics': avg_metrics,
        'pred_masks': pred_masks
    }
    
    np.save(os.path.join(Config.output_dir, "cellpose_results.npy"), results)
    
    # Save visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        # Original image
        axes[0, i].imshow(val_images[i], cmap='gray')
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Prediction
        axes[1, i].imshow(pred_masks[i], cmap='nipy_spectral')
        axes[1, i].set_title(f'Cellpose Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, "cellpose_visualization.png"), dpi=150)
    
    print(f"\nResults saved to {Config.output_dir}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

