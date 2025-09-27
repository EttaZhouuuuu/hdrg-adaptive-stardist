#!/usr/bin/env python3
"""
Script to run the adaptive StarDist pipeline with real data paths.
This demonstrates how to use the correct data structure from the original pipeline.
"""

import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run adaptive StarDist with real data paths")
    parser.add_argument("--data_path", type=str, 
                       default="/hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/",
                       help="Base data path (as used in original pipeline.sh)")
    parser.add_argument("--slide", type=str, 
                       default="240819_Ji_N1_H_EScan",
                       help="Slide name (as used in original pipeline.sh)")
    parser.add_argument("--action", type=str, choices=['train', 'infer', 'evaluate', 'experiment'],
                       required=True, help="Action to perform")
    parser.add_argument("--model_path", type=str, default="./models/adaptive_model",
                       help="Path to save/load model")
    parser.add_argument("--output_dir", type=str, default="./results/adaptive_inference",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Verify data structure exists
    data_path = Path(args.data_path)
    slide_path = data_path / args.slide
    patches_dir = slide_path / 'patches_2048'
    masks_dir = slide_path / 'pseudo_gt_masks_2048'
    
    print(f"🔍 Checking data structure:")
    print(f"   Data path: {data_path}")
    print(f"   Slide path: {slide_path}")
    print(f"   Patches: {patches_dir} {'✓' if patches_dir.exists() else '✗'}")
    print(f"   Masks: {masks_dir} {'✓' if masks_dir.exists() else '✗'}")
    
    if not patches_dir.exists() or not masks_dir.exists():
        print("❌ Required data directories not found!")
        print("\n📁 Expected structure:")
        print(f"   {args.data_path}/")
        print(f"   └── {args.slide}/")
        print(f"       ├── patches_2048/          # Input images (.png)")
        print(f"       └── pseudo_gt_masks_2048/  # Ground truth masks (.png)")
        return
    
    # Count available data
    patch_files = list(patches_dir.glob('*.png'))
    mask_files = list(masks_dir.glob('*.png'))
    print(f"   Found {len(patch_files)} patches and {len(mask_files)} masks")
    
    if args.action == 'train':
        print(f"\n🚀 Training adaptive StarDist model...")
        train_cmd = f"""
python scripts/train_adaptive_stardist.py \\
    --data_path {args.data_path} \\
    --slides {args.slide} \\
    --ckpt_path {args.model_path} \\
    --n_epochs 50 \\
    --scale_factors 1.0,0.5,0.25 \\
    --use_gpu
"""
        print("Command to run:")
        print(train_cmd)
        
    elif args.action == 'infer':
        print(f"\n🔮 Running adaptive inference...")
        infer_cmd = f"""
python scripts/run_adaptive_inference.py \\
    --model_path {args.model_path} \\
    --input_path {patches_dir} \\
    --output_dir {args.output_dir} \\
    --save_confidence \\
    --save_parameters
"""
        print("Command to run:")
        print(infer_cmd)
        
    elif args.action == 'evaluate':
        print(f"\n📊 Evaluating results...")
        eval_cmd = f"""
python scripts/evaluate_results.py \\
    --pred_dir {args.output_dir} \\
    --true_dir {masks_dir} \\
    --output_dir ./evaluation \\
    --create_plots
"""
        print("Command to run:")
        print(eval_cmd)
        
    elif args.action == 'experiment':
        print(f"\n🧪 Running comparative experiments...")
        exp_cmd = f"""
python -m scripts.experiments.experiment_runner \\
    --experiment_name basic_comparison \\
    --data_path {args.data_path} \\
    --output_dir ./experiments/results
"""
        print("Command to run:")
        print(exp_cmd)
    
    print(f"\n💡 Tips:")
    print(f"   - Use 'python scripts/test_setup.py' to create synthetic test data")
    print(f"   - Original pipeline uses .png files, not .tiff")
    print(f"   - Masks are binary: 0=background, 255=object")
    print(f"   - Images are 2048x2048 patches from larger slides")

if __name__ == "__main__":
    main()
