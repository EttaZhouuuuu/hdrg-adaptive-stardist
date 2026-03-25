#!/usr/bin/env python
"""
Xenium Data Explorer
====================
Explore the structure of Xenium output files:
- cells.zarr.zip: Cell segmentation results
- morphology.ome.tif: H&E image
"""

import zarr
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path("/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/data")
CELLS_ZARR = DATA_DIR / "cells.zarr"  # Xenium outputs as directory, not .zip
MORPHOLOGY_TIFF = DATA_DIR / "morphology.ome.tif"


def explore_zarr_structure(zarr_path):
    """
    Explore the structure of a zarr archive
    """
    print("=" * 70)
    print("XENIUM CELLS.ZARR STRUCTURE EXPLORATION")
    print("=" * 70)

    print(f"\n📁 File: {zarr_path}")
    print(f"📦 Archive size: {sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / (1024**2):.2f} MB")
    
    # List all entries (zarr directory structure)
    print("\n📋 All entries in zarr:")
    print("-" * 50)

    entry_info = []
    for name in sorted(zarr_path.rglob('*')):
        if name.is_dir():
            continue

        rel_name = str(name.relative_to(zarr_path))
        
        try:
            # Read the array directly
            arr = zarr.open(str(name), mode='r')
            shape = arr.shape if hasattr(arr, 'shape') else 'N/A'
            dtype = arr.dtype if hasattr(arr, 'dtype') else 'N/A'

            entry_info.append({
                'name': rel_name,
                'shape': shape,
                'dtype': dtype,
                'size': arr.size if hasattr(arr, 'size') else 0
            })

            print(f"  {rel_name}")
            print(f"      Shape: {shape}, Dtype: {dtype}")

        except Exception as e:
            print(f"  {rel_name} - ERROR: {e}")
            entry_info.append({'name': rel_name, 'error': str(e)})
    
    return entry_info


def analyze_cell_labels(zarr_path):
    """
    Analyze the cell segmentation labels
    """
    print("\n" + "=" * 70)
    print("CELL SEGMENTATION LABEL ANALYSIS")
    print("=" * 70)

    found_arrays = []

    for name in sorted(zarr_path.rglob('*')):
        if name.is_dir():
            continue

        try:
            arr = zarr.open(str(name), mode='r')

            # Check if this looks like a segmentation mask (2D or 3D integer array)
            if hasattr(arr, 'shape') and hasattr(arr, 'dtype'):
                is_integer_mask = np.issubdtype(arr.dtype, np.integer) if hasattr(np, 'issubdtype') else 'unknown'
                is_2d_or_3d = len(arr.shape) in [2, 3]

                if is_2d_or_3d:
                    found_arrays.append({
                        'name': str(name.relative_to(zarr_path)),
                        'shape': arr.shape,
                        'dtype': arr.dtype,
                        'min': arr[:].min() if arr.size < 1e8 else 'skip',
                        'max': arr[:].max() if arr.size < 1e8 else 'skip',
                        'n_unique': len(np.unique(arr[:])) if arr.size < 1e8 else 'skip'
                    })

        except Exception as e:
            pass
    
    print(f"\n🔍 Found {len(found_arrays)} potential segmentation arrays:")
    print("-" * 70)
    
    for info in found_arrays:
        print(f"\n📊 {info['name']}")
        print(f"   Shape: {info['shape']}")
        print(f"   Dtype: {info['dtype']}")
        print(f"   Min: {info['min']}, Max: {info['max']}")
        print(f"   Unique values (instances): {info['n_unique']}")
    
    return found_arrays


def generate_preprocessing_guide(zarr_path):
    """
    Generate a guide for preprocessing Xenium data
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING GUIDE")
    print("=" * 70)
    
    print("""
Xenium Output Structure (typically):
------------------------------------
1. Cell Boundaries/Segmentations (2D mask):
   - Shape: (height, width) or (z, height, width)
   - Values: 0 = background, 1,2,3... = cell IDs
   - Each cell has a unique integer ID
   
2. Cell Features (optional):
   - cell_x_centroid, cell_y_centroid: Cell center coordinates
   - cell_area, cell_perimeter: Cell geometry features
   - gene_expression: RNA counts per cell

Expected Output Format for StarDist:
-------------------------------------
X_train: np.ndarray [N, H, W] or [N, H, W, C] - training images
Y_train: np.ndarray [N, H, W] - instance labels (0=background, 1,2,3...=instances)

Steps:
------
1. Load morphology.ome.tif (H&E image)
2. Load cell segmentation from zarr
3. Convert to instance mask format
4. Split into train/val sets
5. Normalize images (H&E specific: may need different normalization)
6. Train StarDist model
""")


def main():
    """Main exploration function"""
    
    # Check files exist
    if not CELLS_ZARR.exists():
        print(f"❌ ERROR: {CELLS_ZARR} not found!")
        return
    
    if not MORPHOLOGY_TIFF.exists():
        print(f"⚠️  WARNING: {MORPHOLOGY_TIFF} not found!")
    
    # Explore zarr structure
    entry_info = explore_zarr_structure(CELLS_ZARR)
    
    # Analyze segmentation labels
    seg_arrays = analyze_cell_labels(CELLS_ZARR)
    
    # Generate guide
    generate_preprocessing_guide(CELLS_ZARR)
    
    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

