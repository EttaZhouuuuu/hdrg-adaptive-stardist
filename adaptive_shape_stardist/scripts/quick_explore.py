#!/usr/bin/env python
"""
Xenium Data Quick Explorer
==========================
Quick exploration of Xenium zarr structure (polygon-based)
"""

import zarr
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/data")
CELLS_ZARR = DATA_DIR / "cells.zarr"

print("=" * 70)
print("XENIUM CELLS.ZARR STRUCTURE (POLYGON-BASED)")
print("=" * 70)

# Open zarr
root = zarr.open(str(CELLS_ZARR), mode='r')

print(f"\n📁 Root arrays:")
for name, arr in root.arrays():
    print(f"   {name}: shape={arr.shape}, dtype={arr.dtype}")

# Key arrays for polygon-based segmentation
print("\n" + "-" * 70)
print("KEY ARRAYS FOR INSTANCE SEGMENTATION:")
print("-" * 70)

cell_id = root['cell_id'][:]
print(f"\n📊 cell_id: {cell_id.shape}")
print(f"   Min ID: {cell_id.min()}, Max ID: {cell_id.max()}")
print(f"   Total cells: {len(cell_id)}")

polygon_vertices = root['polygon_vertices'][:]
print(f"\n📊 polygon_vertices: {polygon_vertices.shape}")
print(f"   Dimensions: (2=xy, cells={polygon_vertices.shape[1]}, max_vertices={polygon_vertices.shape[2]})")

polygon_num_vertices = root['polygon_num_vertices'][:]
print(f"\n📊 polygon_num_vertices: {polygon_num_vertices.shape}")
print(f"   First 10 cells: {polygon_num_vertices[0, :10]}")  # z=0 plane
print(f"   Mean vertices per cell: {polygon_num_vertices[0, :].mean():.1f}")

cell_summary = root['cell_summary'][:]
print(f"\n📊 cell_summary: {cell_summary.shape}")
print(f"   Columns: [x, y, area, eccentricity, major_axis, minor_axis, orientation]")
print(f"   First 3 cells:\n{cell_summary[:3]}")

# Masks group
if 'masks' in root:
    print(f"\n📊 Masks group: {root['masks']}")
else:
    print("\n⚠️  No 'masks' group found - using polygon-based representation")

