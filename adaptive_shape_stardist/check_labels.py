"""
Check the nature of Y label in training_data.npz
"""
import numpy as np
from pathlib import Path

data_path = Path('/data/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/training_data.npz')
data = np.load(str(data_path), allow_pickle=True)

X = data['X']
Y = data['Y']

print("=" * 60)
print("DATA ANALYSIS: training_data.npz")
print("=" * 60)

print(f"\n📊 X (Images):")
print(f"   Shape: {X.shape}")
print(f"   dtype: {X.dtype}")
print(f"   Value range: [{X.min()}, {X.max()}]")

print(f"\n📊 Y (Labels):")
print(f"   Shape: {Y.shape}")
print(f"   dtype: {Y.dtype}")
print(f"   Value range: [{Y.min()}, {Y.max()}]")

# 分析Y的分布
if len(Y.shape) == 3:
    Y_flat = Y.flatten()
    print(f"\n   Statistics (flattened):")
    print(f"   Mean: {Y_flat.mean():.6f}")
    print(f"   Std: {Y_flat.std():.6f}")
    print(f"   Median: {np.median(Y_flat):.6f}")
    
    # 检查Y是二值还是连续
    unique_vals = len(np.unique(Y))
    print(f"   Unique values: {unique_vals}")
    
    if Y.max() > 1:
        print(f"\n   ⚠️ Y.max() = {Y.max()} > 1")
        print(f"   This is NOT binary mask!")
        print(f"   This appears to be a continuous probability map.")
        print(f"\n   Possible sources:")
        print(f"   1. StarDist output probability map")
        print(f"   2. Distance transform map")
        print(f"   3. Normalized annotation score")
    
    # 检查二值化后的连通区域
    Y_binary = (Y > 0.5).astype(np.float32)
    num_ones = np.sum(Y_binary)
    num_zeros = np.sum(Y_binary == 0)
    print(f"\n   Binary (threshold=0.5):")
    print(f"   Positive pixels: {num_ones} ({num_ones/Y.size*100:.2f}%)")
    print(f"   Negative pixels: {num_zeros} ({num_zeros/Y.size*100:.2f}%)")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("""
Y is likely a StarDist-style probability map (continuous values).
This is NOT a traditional binary segmentation mask!

Using this Y as "Ground Truth" for comparison is problematic because:
1. The model is trained to predict similar probability distributions
2. Comparing against itself doesn't show real improvement

For meaningful qualitative analysis, we need:
1. Binary masks from manual annotation (gold standard)
2. Or separate test set with different annotation source
""")

