#!/usr/bin/env python3
"""
Demo script showing the correct usage of data paths in the adaptive StarDist pipeline.
This script demonstrates how to use the data structure from the original hDRG pipeline.
"""

import os
import sys
from pathlib import Path
import numpy as np
from skimage import io

def demo_data_loading():
    """Demonstrate how to load data with the correct structure."""
    print("🔍 Testing Data Loading with Original Structure\n")
    
    # Data paths (from original pipeline.sh)
    DATA_PATH = "./data"  # Local test data
    SLIDE = "240819_Ji_N1_H_EScan"  # From original pipeline
    
    print(f"Data path: {DATA_PATH}")
    print(f"Slide: {SLIDE}")
    
    # Check structure
    slide_path = Path(DATA_PATH) / SLIDE
    patches_dir = slide_path / "patches_2048"
    masks_dir = slide_path / "pseudo_gt_masks_2048"
    
    print(f"\n📁 Directory Structure:")
    print(f"   Slide path: {slide_path} {'✓' if slide_path.exists() else '✗'}")
    print(f"   Patches: {patches_dir} {'✓' if patches_dir.exists() else '✗'}")
    print(f"   Masks: {masks_dir} {'✓' if masks_dir.exists() else '✗'}")
    
    if not patches_dir.exists():
        print("❌ Run 'python scripts/test_setup.py' first to create test data!")
        return False
    
    # Load sample data
    patch_files = list(patches_dir.glob("*.png"))
    mask_files = list(masks_dir.glob("*.png"))
    
    print(f"\n📊 Data Summary:")
    print(f"   Patch files: {len(patch_files)}")
    print(f"   Mask files: {len(mask_files)}")
    
    if patch_files:
        # Load first image and mask
        img_path = patch_files[0]
        mask_path = masks_dir / img_path.name
        
        print(f"\n🖼️  Sample Data:")
        print(f"   Image: {img_path.name}")
        print(f"   Mask: {mask_path.name}")
        
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        
        print(f"   Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"   Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"   Mask values: {np.unique(mask)}")
        
        # Verify mask is binary (0 and 255)
        mask_values = np.unique(mask)
        is_binary = len(mask_values) <= 2 and all(v in [0, 255] for v in mask_values)
        print(f"   Binary mask: {'✓' if is_binary else '✗'}")
        
    return True

def demo_commands():
    """Show example commands with correct paths."""
    print("\n🚀 Example Commands with Correct Paths\n")
    
    print("1️⃣  Training:")
    print("   python scripts/train_adaptive_stardist.py \\")
    print("       --data_path ./data \\")
    print("       --slides 240819_Ji_N1_H_EScan \\")
    print("       --ckpt_path ./models/adaptive \\")
    print("       --n_epochs 10")
    
    print("\n2️⃣  Inference:")
    print("   python scripts/run_adaptive_inference.py \\")
    print("       --model_path ./models/adaptive \\")
    print("       --input_path ./data/240819_Ji_N1_H_EScan/patches_2048 \\")
    print("       --output_dir ./results")
    
    print("\n3️⃣  Evaluation:")
    print("   python scripts/evaluate_results.py \\")
    print("       --pred_dir ./results \\")
    print("       --true_dir ./data/240819_Ji_N1_H_EScan/pseudo_gt_masks_2048 \\")
    print("       --output_dir ./evaluation")
    
    print("\n4️⃣  Experiments:")
    print("   python -m scripts.experiments.experiment_runner \\")
    print("       --experiment_name basic_comparison \\")
    print("       --data_path ./data")

def main():
    print("=" * 60)
    print("  Multi-Scale Adaptive StarDist - Data Path Demo")
    print("=" * 60)
    
    # Test data loading
    success = demo_data_loading()
    
    if success:
        demo_commands()
        
        print("\n" + "=" * 60)
        print("✅ Data structure verification complete!")
        print("\n💡 Key Points:")
        print("   • Uses original hDRG structure: DATA_PATH/SLIDE/patches_2048/")
        print("   • Images are PNG format (not TIFF)")
        print("   • Masks are binary: 0=background, 255=object")
        print("   • Patches are 2048x2048 pixels")
        print("   • Follows exact naming: patch_col_XX_row_YY.png")
        print("=" * 60)
    else:
        print("\n❌ Please create test data first: python scripts/test_setup.py")

if __name__ == "__main__":
    main()
