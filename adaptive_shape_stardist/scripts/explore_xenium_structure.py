#!/usr/bin/env python
"""
Xenium Data Structure Explorer
==============================
Properly explore the structure of Xenium output files:
- cells.zarr: Cell segmentation results (as directory, not zip)
- morphology.ome.tif: H&E image
"""

import zarr
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Configuration
DATA_DIR = Path("/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/data")
CELLS_ZARR = DATA_DIR / "cells.zarr"
MORPHOLOGY_TIFF = DATA_DIR / "morphology.ome.tif"


def explore_zarr_structure(zarr_path):
    """
    Explore the structure of a zarr directory
    """
    print("=" * 80)
    print("XENIUM CELLS.ZARR STRUCTURE EXPLORATION")
    print("=" * 80)
    
    print(f"\n📁 Directory: {zarr_path}")
    
    # Check if directory exists
    if not zarr_path.exists():
        print(f"❌ ERROR: {zarr_path} not found!")
        return {}
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
    print(f"📦 Total size: {total_size / (1024**2):.2f} MB")
    
    # Open zarr root
    root = zarr.open(str(zarr_path), mode='r')
    
    print("\n📋 Root-level groups and arrays:")
    print("-" * 60)
    
    structure = {}
    
    # List root-level items
    for name in sorted(root.keys()):
        try:
            obj = root[name]
            if isinstance(obj, zarr.Group):
                print(f"  📂 {name}/")
                structure[name] = {'type': 'group', 'children': {}}
                
                # Explore subgroup
                if name in ['masks', 'cell_summary', 'polygon_vertices']:
                    structure[name]['children'] = explore_group(obj, name, indent=4)
            else:
                # Array
                arr = obj
                shape_info = f"Shape: {arr.shape}, Dtype: {arr.dtype}"
                print(f"  📊 {name}: {shape_info}")
                structure[name] = {'type': 'array', 'shape': arr.shape, 'dtype': str(arr.dtype)}
        except Exception as e:
            print(f"  ❌ {name}: ERROR - {e}")
            structure[name] = {'type': 'error', 'error': str(e)}
    
    return structure


def explore_group(group, group_name, indent=2):
    """Recursively explore a zarr group"""
    children = {}
    prefix = " " * indent
    
    for name in sorted(group.keys()):
        try:
            obj = group[name]
            if isinstance(obj, zarr.Group):
                print(f"{prefix}📂 {name}/")
                children[name] = {'type': 'group', 'children': {}}
                children[name]['children'] = explore_group(obj, name, indent + 2)
            else:
                arr = obj
                # Sample first element if large
                if arr.size > 1000000:
                    sample = arr[0] if len(arr.shape) > 0 else arr
                    sample_info = f"(sampled first element)"
                else:
                    full_data = arr[:]
                    sample_info = f"Values: min={full_data.min()}, max={full_data.max()}, unique={len(np.unique(full_data))}"
                
                shape_info = f"Shape: {arr.shape}, Dtype: {arr.dtype}"
                print(f"{prefix}📊 {name}: {shape_info} {sample_info}")
                children[name] = {
                    'type': 'array', 
                    'shape': arr.shape, 
                    'dtype': str(arr.dtype),
                    'info': sample_info
                }
        except Exception as e:
            print(f"{prefix}❌ {name}: ERROR - {e}")
            children[name] = {'type': 'error', 'error': str(e)}
    
    return children


def analyze_cell_id():
    """Analyze cell_id array"""
    print("\n" + "=" * 80)
    print("CELL ID ANALYSIS")
    print("=" * 80)
    
    root = zarr.open(str(CELLS_ZARR), mode='r')
    
    if 'cell_id' in root.keys():
        cell_id_arr = root['cell_id'][:]
        print(f"\n📊 cell_id array:")
        print(f"   Shape: {cell_id_arr.shape}")
        print(f"   Dtype: {cell_id_arr.dtype}")
        print(f"   Unique values: {len(np.unique(cell_id_arr))}")
        print(f"   Sample values: {np.unique(cell_id_arr)[:10]}")
        print(f"   Total cells: {len(np.unique(cell_id_arr)) - 1}")  # -1 for background


def analyze_masks():
    """Analyze masks structure"""
    print("\n" + "=" * 80)
    print("MASKS STRUCTURE ANALYSIS")
    print("=" * 80)
    
    root = zarr.open(str(CELLS_ZARR), mode='r')
    
    if 'masks' in root.keys():
        masks_group = root['masks']
        
        # Check if it's a group or array
        if isinstance(masks_group, zarr.Group):
            print(f"\n📂 masks/ group:")
            print(f"   Number of subgroups: {list(masks_group.keys())}")
            
            # Analyze first few masks
            mask_keys = sorted(list(masks_group.keys()))[:3]
            for key in mask_keys:
                try:
                    mask_subgroup = masks_group[key]
                    print(f"\n   📂 {key}/")
                    for arr_name in sorted(mask_subgroup.keys()):
                        arr = mask_subgroup[arr_name]
                        data = arr[:]
                        print(f"      📊 {arr_name}: shape={arr.shape}, dtype={arr.dtype}")
                        if data.size < 100:
                            print(f"         Values: {data}")
                        else:
                            print(f"         Min: {data.min()}, Max: {data.max()}")
                except Exception as e:
                    print(f"   ❌ Error reading {key}: {e}")
            
            # Count total masks
            total_masks = len(list(masks_group.keys()))
            print(f"\n   Total cell masks: {total_masks}")
        else:
            # It's an array - this is the full instance mask!
            arr = masks_group[:]
            print(f"\n📊 masks array (FULL INSTANCE MASK):")
            print(f"   Shape: {arr.shape}")
            print(f"   Dtype: {arr.dtype}")
            print(f"   Value range: [{arr.min()}, {arr.max()}]")
            print(f"   Unique values: {len(np.unique(arr))}")
            print(f"   Total cells: {len(np.unique(arr)) - 1}")  # -1 for background


def analyze_polygons():
    """Analyze polygon vertices"""
    print("\n" + "=" * 80)
    print("POLYGON VERTICES ANALYSIS")
    print("=" * 80)
    
    root = zarr.open(str(CELLS_ZARR), mode='r')
    
    # Check polygon_vertices
    if 'polygon_vertices' in root.keys():
        pv = root['polygon_vertices']
        print(f"\n📊 polygon_vertices:")
        print(f"   Shape: {pv.shape}")
        print(f"   Dtype: {pv.dtype}")
        
        # Try to understand the indexing
        print(f"\n   Structure hints:")
        print(f"   - First dimension (0, 1, ...): Likely cell index")
        print(f"   - Second dimension (0, 1, 2, 3): Likely multiple polygons per cell")
        print(f"   - Remaining dimensions: vertex coordinates (y, x)")
        
        # Sample data
        try:
            sample = pv[0, 0]
            print(f"   Sample [0,0]: shape={sample.shape}, dtype={sample.dtype}")
            print(f"   Sample values: {sample}")
        except:
            print(f"   Sample [0,0]: Unable to read")
    
    # Check polygon_num_vertices
    if 'polygon_num_vertices' in root.keys():
        pnv = root['polygon_num_vertices'][:]
        print(f"\n📊 polygon_num_vertices:")
        print(f"   Shape: {pnv.shape}")
        print(f"   Dtype: {pnv.dtype}")
        print(f"   Unique vertex counts: {np.unique(pnv)}")


def analyze_morphology_image():
    """Analyze morphology image"""
    print("\n" + "=" * 80)
    print("MORPHOLOGY IMAGE ANALYSIS")
    print("=" * 80)
    
    if not MORPHOLOGY_TIFF.exists():
        print(f"⚠️  WARNING: {MORPHOLOGY_TIFF} not found!")
        return
    
    # Try with PIL first (for basic info)
    try:
        img = Image.open(str(MORPHOLOGY_TIFF))
        print(f"\n📊 Image properties:")
        print(f"   Size: {img.size}")
        print(f"   Mode: {img.mode}")
        print(f"   Format: {img.format}")
    except Exception as e:
        print(f"   PIL read error: {e}")
    
    # Try with tifffile for detailed info
    try:
        import tifffile
        with tifffile.TiffFile(str(MORPHOLOGY_TIFF)) as tif:
            print(f"\n📊 Tiff properties:")
            print(f"   Number of pages: {len(tif.pages)}")
            if tif.pages:
                page = tif.pages[0]
                print(f"   Shape: {page.shape}")
                print(f"   Dtype: {page.dtype}")
                print(f"   Compression: {page.compression}")
            
            # Read data if small enough
            if len(tif.pages) <= 10:
                data = tifffile.imread(str(MORPHOLOGY_TIFF))
                print(f"   Full data shape: {data.shape}")
                print(f"   Full data dtype: {data.dtype}")
                print(f"   Value range: [{data.min()}, {data.max()}]")
    except Exception as e:
        print(f"   tifffile read error: {e}")


def analyze_cell_summary():
    """Analyze cell summary data"""
    print("\n" + "=" * 80)
    print("CELL SUMMARY ANALYSIS")
    print("=" * 80)
    
    root = zarr.open(str(CELLS_ZARR), mode='r')
    
    if 'cell_summary' in root.keys():
        cs_arr = root['cell_summary'][:]
        print(f"\n📊 cell_summary array:")
        print(f"   Shape: {cs_arr.shape}")
        print(f"   Dtype: {cs_arr.dtype}")
        
        # Show statistics for each column
        print(f"\n   Column statistics:")
        col_names = ['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']
        for i, name in enumerate(col_names):
            col_data = cs_arr[:, i]
            print(f"   {name}: min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}")
        
        # Sample data
        print(f"\n   Sample rows:")
        print(f"   {cs_arr[:5]}")


def generate_data_loader_guide():
    """Generate guide for creating data loader"""
    print("\n" + "=" * 80)
    print("DATA LOADER IMPLEMENTATION GUIDE")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ XENIUM DATA FORMAT SUMMARY                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  cells.zarr/                                                                │
│  ├── cell_id/                    → Cell IDs (unique per cell)              │
│  │   ├── 0                                                                    │
│  │   └── 1                                                                    │
│  │                                                                            │
│  ├── masks/                      → Cell segmentation masks                 │
│  │   ├── 0/                       → Cell 0                                 │
│  │   │   ├── 0                    → Binary mask for first polygon           │
│  │   │   ├── 1                    → Binary mask for second polygon          │
│  │   │   └── ...                                                             │
│  │   └── 1/                       → Cell 1                                 │
│  │                                                                            │
│  ├── polygon_vertices/           → Cell boundary coordinates               │
│  │   ├── 0.0.0                   → Cell 0, polygon 0, vertex coordinates   │
│  │   ├── 0.0.1                                                               │
│  │   ├── 0.0.2                                                               │
│  │   ├── 0.0.3                                                               │
│  │   ├── 0.1.0                  → Cell 0, polygon 1 (e.g., nucleus)         │
│  │   └── ...                                                                 │
│  │                                                                            │
│  ├── polygon_num_vertices/       → Number of vertices per polygon           │
│  │                                                                            │
│  ├── cell_summary/               → Cell metadata/features                    │
│  │                                                                            │
│  └── homogeneous_transform/     → Coordinate transform matrix               │
│                                                                             │
│  morphology.ome.tif                                                         │
│  └── H&E stained brightfield image                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ RECOMMENDED DATA LOADER STRATEGY                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Approach 1: Use masks (binary masks per cell)                            │
│  ────────────────────────────────────────                                   │
│  1. Load morphology.ome.tif as input image                                 │
│  2. For each cell ID:                                                      │
│     - Load all binary masks from masks/{cell_id}/                          │
│     - Combine (OR operation) into single cell mask                         │
│     - Store in instance mask array (0=bg, 1,2,3...=cells)                  │
│                                                                             │
│  Approach 2: Use polygon vertices (more accurate)                         │
│  ────────────────────────────────────────                                   │
│  1. Load morphology.ome.tif as input image                                │
│  2. For each cell ID:                                                      │
│     - Load polygon vertices from polygon_vertices/                         │
│     - Convert (y,x) coordinates to mask using skimage.draw.polygon         │
│     - Store in instance mask array                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")


def main():
    """Main exploration function"""
    
    print("\n" + "=" * 80)
    print("  🔬 XENIUM DATA STRUCTURE EXPLORER  ")
    print("=" * 80)
    
    # Check files exist
    if not CELLS_ZARR.exists():
        print(f"\n❌ ERROR: {CELLS_ZARR} not found!")
        return
    
    # Explore zarr structure
    structure = explore_zarr_structure(CELLS_ZARR)
    
    # Analyze specific components
    analyze_cell_id()
    analyze_masks()
    analyze_polygons()
    analyze_cell_summary()
    
    # Analyze morphology image
    analyze_morphology_image()
    
    # Generate guide
    generate_data_loader_guide()
    
    print("\n" + "=" * 80)
    print("  ✅ EXPLORATION COMPLETE  ")
    print("=" * 80)


if __name__ == "__main__":
    main()

