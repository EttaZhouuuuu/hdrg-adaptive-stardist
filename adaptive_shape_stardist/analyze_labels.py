"""
Analyze StarDist Label Format
=============================
Check whether Y is probability map, distance map, or binary mask.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 70)
print("STAR-DIST LABEL FORMAT ANALYSIS")
print("=" * 70)

# Load data
data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
X = data['X']
Y = data['Y']

print(f"\n📊 Dataset Statistics:")
print(f"  X shape: {X.shape}")
print(f"  Y shape: {Y.shape}")
print(f"  X dtype: {X.dtype}")
print(f"  Y dtype: {Y.dtype}")

# Detailed analysis of Y
print(f"\n📊 Y Value Distribution:")
print(f"  Min: {Y.min():.4f}")
print(f"  Max: {Y.max():.4f}")
print(f"  Mean: {Y.mean():.4f}")
print(f"  Median: {np.median(Y):.4f}")
print(f"  Std: {Y.std():.4f}")

# Check for different value ranges
unique_vals = np.unique(Y)
print(f"\n📊 Unique Values Analysis:")
print(f"  Total unique values: {len(unique_vals)}")
print(f"  Top 10 most common values:")

# Count value frequencies
from collections import Counter
val_counts = Counter(Y.flatten())
for val, count in val_counts.most_common(10):
    pct = count / Y.size * 100
    print(f"    {val:.2f}: {count:,} ({pct:.1f}%)")

# Analyze value ranges
print(f"\n📊 Value Range Analysis:")
print(f"  Zeros: {(Y == 0).sum():,} ({100*(Y == 0).sum()/Y.size:.1f}%)")
print(f"  Small values (< 1): {((Y > 0) & (Y < 1)).sum():,} ({100*((Y > 0) & (Y < 1)).sum()/Y.size:.1f}%)")
print(f"  Medium values (1-1000): {((Y >= 1) & (Y <= 1000)).sum():,} ({100*((Y >= 1) & (Y <= 1000)).sum()/Y.size:.1f}%)")
print(f"  Large values (1000-10000): {((Y > 1000) & (Y <= 10000)).sum():,} ({100*((Y > 1000) & (Y <= 10000)).sum()/Y.size:.1f}%)")
print(f"  Very large values (> 10000): {(Y > 10000).sum():,} ({100*(Y > 10000).sum()/Y.size:.1f}%)")

# Check if Y looks like distance map (exponential decay) or probability (0-1)
print(f"\n📊 Format Hypothesis:")
if Y.max() > 1000:
    print("  ⚠️ Y has very large values (>1000)")
    print("  → Likely a distance map or unnormalized StarDist output")
    print("  → Should be normalized to [0, 1] or [0, max_distance]")
    
if Y.mean() > 100:
    print(f"  ⚠️ Y mean is high ({Y.mean():.1f})")
    print("  → Not a standard binary mask (would be ~0.5)")
    print("  → Not a standard probability map (would be ~0.1-0.3)")

# Check sample images
print(f"\n📊 Sample Analysis (first 3 images):")
for i in range(min(3, len(Y))):
    y_img = Y[i]
    print(f"\n  Image {i}:")
    print(f"    Min: {y_img.min():.4f}, Max: {y_img.max():.4f}")
    print(f"    Mean: {y_img.mean():.4f}")
    print(f"    Non-zero pixels: {(y_img > 0).sum():,} ({100*(y_img > 0).sum()/y_img.size:.1f}%)")
    print(f"    Unique non-zero values: {len(np.unique(y_img[y_img > 0]))}")

# Percentile analysis
print(f"\n📊 Percentile Analysis (entire dataset):")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(Y, p)
    print(f"    {p}th percentile: {val:.4f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
StarDist Output Format:
- StarDist typically outputs probability maps or distance maps
- Probability maps: values in [0, 1], ~10-30% positive pixels
- Distance maps: values representing distance to cell boundaries
- Raw StarDist output: can have very large unnormalized values

Recommended Normalization:
1. If distance map: Y_normalized = Y / Y.max()
2. If probability map: Y_normalized = Y (already in [0,1])
3. For binary segmentation: Y_binary = (Y > 0.5).astype(float)

For cell segmentation tasks, Y often represents:
- StarDist "prob" map: probability of cell center
- StarDist "dist" map: distances to cell boundaries
""")

