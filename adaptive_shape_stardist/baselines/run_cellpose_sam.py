#!/usr/bin/env python3
"""
Cellpose-SAM Baseline Evaluation Script
=======================================
Based on Cellpose-SAM notebook by Cellpose authors.

Usage:
    python run_cellpose_sam.py --data_dir /path/to/images --output_dir /path/to/output
"""

import numpy as np
import torch
from pathlib import Path
from cellpose import models, io, core
from tqdm import trange
import argparse
import json
from typing import Optional
import sys

# Try to import for visualization (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Config:
    """Configuration for Cellpose-SAM evaluation"""
    
    # Model parameters
    model_type = "cyto2"  # 'cyto2' or 'nuclei'
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0
    
    # Batch processing
    batch_size = 32
    
    # Image settings
    image_ext = ".tif"
    
    # Output settings
    save_masks = True
    save_visualization = False


def compute_instance_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute instance-level IoU."""
    pred_labels = np.unique(pred_mask)[1:]
    gt_labels = np.unique(gt_mask)[1:]
    
    if len(gt_labels) == 0:
        return 0.0
    
    ious = []
    for gt_label in gt_labels:
        gt_region = (gt_mask == gt_label)
        best_iou = 0.0
        
        for pred_label in pred_labels:
            pred_region = (pred_mask == pred_label)
            intersection = np.logical_and(gt_region, pred_region).sum()
            union = np.logical_or(gt_region, pred_region).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
        
        if best_iou > 0:
            ious.append(best_iou)
    
    return np.mean(ious) if ious else 0.0


def compute_pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute pixel-level IoU."""
    pred_binary = (pred_mask > 0).astype(int)
    gt_binary = (gt_mask > 0).astype(int)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    return intersection / union if union > 0 else 0.0


def compute_aji(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Aggregate Jaccard Index."""
    pred_labels = np.unique(pred_mask)[1:]
    gt_labels = np.unique(gt_mask)[1:]
    
    if len(gt_labels) == 0:
        return 0.0
    
    intersections = 0
    unions = 0
    
    for gt_label in gt_labels:
        gt_region = (gt_mask == gt_label)
        best_iou = 0.0
        best_pred = 0
        
        for pred_label in pred_labels:
            pred_region = (pred_mask == pred_label)
            intersection = np.logical_and(gt_region, pred_region).sum()
            union = np.logical_or(gt_region, pred_region).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_label
        
        if best_pred > 0:
            intersections += np.logical_and(gt_region, (pred_mask == best_pred)).sum()
            unions += np.logical_or(gt_region, (pred_mask == best_pred)).sum()
    
    return intersections / unions if unions > 0 else 0.0


def load_test_data(data_dir: Path, max_samples: int = 100):
    """Load test data from Xenium format."""
    try:
        from utils.xenium_preprocessing import XeniumDataLoader
        
        print(f"Loading data from {data_dir}...")
        loader = XeniumDataLoader(
            cells_zarr_path=str(data_dir / "cells.zarr"),
            image_path=str(data_dir / "morphology_focus.ome.tif"),
            load_image=True
        )
        
        images, instance_masks = loader.extract_patches(
            patch_size=256,
            num_patches=min(max_samples, 500),
            use_precomputed=True
        )
        
        print(f"Loaded {len(images)} samples")
        return images, instance_masks
    
    except ImportError:
        print("Error: utils.xenium_preprocessing not found")
        print("Please ensure the project is set up correctly.")
        sys.exit(1)


def run_cellpose_sam(
    data_dir: Path,
    output_dir: Path,
    model_type: str = "cyto2",
    max_samples: int = 50,
    with_ground_truth: bool = True,
    use_sam: bool = True
):
    """
    Run Cellpose or Cellpose-SAM evaluation.

    Args:
        data_dir: Path to data directory
        output_dir: Path to save results
        model_type: Cellpose model type ('cyto2' or 'nuclei')
        max_samples: Maximum number of samples to process
        with_ground_truth: Whether ground truth masks are available for evaluation
        use_sam: Whether to use SAM backbone (requires additional download)
    """

    print("=" * 60)
    if use_sam:
        print("Cellpose-SAM Evaluation")
    else:
        print("Cellpose Evaluation (Standard)")
    print("=" * 60)

    # Check GPU
    print("\nChecking GPU availability...")
    gpu_available = core.use_gpu()
    print(f"GPU available: {gpu_available}")

    # Initialize model
    print(f"\nInitializing Cellpose model ({model_type})...")
    if use_sam:
        try:
            model = models.CellposeModel(gpu=True, model_type=model_type)
            print("Using Cellpose-SAM (with SAM backbone)")
        except Exception as e:
            print(f"Failed to load SAM model: {e}")
            print("Falling back to standard Cellpose...")
            use_sam = False
            model = models.CellposeModel(gpu=gpu_available, model_type=model_type)
            print("Using standard Cellpose")
    else:
        model = models.CellposeModel(gpu=gpu_available, model_type=model_type)
        print("Using standard Cellpose")

    print("Model initialized!")
    
    # Load data
    images, gt_masks = load_test_data(data_dir, max_samples)
    
    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    # Process images
    print("\nRunning Cellpose-SAM segmentation...")
    metrics = {
        "pixel_iou": [],
        "instance_iou": [],
        "aji": [],
        "n_pred": [],
        "n_gt": []
    }
    
    for i in trange(len(images), desc="Processing"):
        img = images[i]
        
        # Handle different image formats
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # Convert to RGB
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        
        # Run Cellpose-SAM
        masks, flows, styles = model.eval(
            img,
            batch_size=Config.batch_size,
            flow_threshold=Config.flow_threshold,
            cellprob_threshold=Config.cellprob_threshold,
            normalize={"tile_norm_blocksize": Config.tile_norm_blocksize}
        )
        
        # Handle masks output
        if isinstance(masks, list):
            masks = masks[0]
        
        # Save masks
        if Config.save_masks:
            mask_path = masks_dir / f"mask_{i:04d}.tif"
            io.imsave(str(mask_path), masks)
        
        # Compute metrics if ground truth available
        if with_ground_truth and gt_masks is not None:
            gt_mask = gt_masks[i]
            
            pixel_iou = compute_pixel_iou(masks, gt_mask)
            instance_iou = compute_instance_iou(masks, gt_mask)
            aji = compute_aji(masks, gt_mask)
            
            metrics["pixel_iou"].append(pixel_iou)
            metrics["instance_iou"].append(instance_iou)
            metrics["aji"].append(aji)
            metrics["n_pred"].append(len(np.unique(masks)) - 1)
            metrics["n_gt"].append(len(np.unique(gt_mask)) - 1)
        
        # Save visualization
        if HAS_MATPLOTLIB and Config.save_visualization:
            fig = plt.figure(figsize=(12, 5))
            io.plot_segmentation(fig, img, masks, flows[0])
            plt.tight_layout()
            plt.savefig(output_dir / f"visualization_{i:04d}.png")
            plt.close()
    
    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    if with_ground_truth and metrics["pixel_iou"]:
        print(f"Pixel IoU:      {np.mean(metrics['pixel_iou']):.4f} +/- {np.std(metrics['pixel_iou']):.4f}")
        print(f"Instance IoU:  {np.mean(metrics['instance_iou']):.4f} +/- {np.std(metrics['instance_iou']):.4f}")
        print(f"AJI:           {np.mean(metrics['aji']):.4f} +/- {np.std(metrics['aji']):.4f}")
        print(f"Avg Pred/Cell: {np.mean(metrics['n_pred']):.1f} (GT: {np.mean(metrics['n_gt']):.1f})")
        
        # Save metrics to JSON
        results = {
            "pixel_iou": float(np.mean(metrics["pixel_iou"])),
            "pixel_iou_std": float(np.std(metrics["pixel_iou"])),
            "instance_iou": float(np.mean(metrics["instance_iou"])),
            "instance_iou_std": float(np.std(metrics["instance_iou"])),
            "aji": float(np.mean(metrics["aji"])),
            "aji_std": float(np.std(metrics["aji"])),
            "model_type": model_type,
            "n_samples": len(images)
        }
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_dir / 'metrics.json'}")
    else:
        print(f"Processed {len(images)} images")
        print(f"Masks saved to {masks_dir}")
    
    print("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Cellpose-SAM evaluation")
    parser.add_argument("--data_dir", type=str,
                        default="data",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str,
                        default="baselines/cellpose_results",
                        help="Path to output directory")
    parser.add_argument("--model_type", type=str,
                        default="cyto2",
                        choices=["cyto2", "nuclei"],
                        help="Cellpose model type")
    parser.add_argument("--max_samples", type=int,
                        default=50,
                        help="Maximum number of samples to process")
    parser.add_argument("--no_gt", action="store_true",
                        help="Run without ground truth masks")
    parser.add_argument("--no_sam", action="store_true",
                        help="Use standard Cellpose instead of Cellpose-SAM (avoids downloading)")

    args = parser.parse_args()

    # Run evaluation
    run_cellpose_sam(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_type=args.model_type,
        max_samples=args.max_samples,
        with_ground_truth=not args.no_gt,
        use_sam=not args.no_sam
    )


if __name__ == "__main__":
    main()
