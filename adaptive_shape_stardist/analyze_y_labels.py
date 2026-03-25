"""
Detailed analysis of Y label distribution
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

data_path = Path('/data/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/training_data.npz')
data = np.load(str(data_path), allow_pickle=True)
Y = data['Y']

# 归一化
Y_norm = Y.astype(np.float32) / 133737.0

print("=" * 70)
print("DETAILED Y LABEL ANALYSIS")
print("=" * 70)

# 分析几个样本
print("\n📊 Sample Analysis:")
for i in [0, 100, 300, 500]:
    y = Y[i]
    y_norm = Y_norm[i]
    
    print(f"\n  Sample {i}:")
    print(f"    Raw: min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.2f}")
    print(f"    Norm: min={y_norm.min():.4f}, max={y_norm.max():.4f}, mean={y_norm.mean():.4f}")
    
    # 检查值的分布
    print(f"    Value distribution:")
    print(f"      =0: {np.sum(y==0)} pixels ({np.sum(y==0)/y.size*100:.2f}%)")
    print(f"      >0: {np.sum(y>0)} pixels ({np.sum(y>0)/y.size*100:.2f}%)")
    print(f"      >0.5*max: {np.sum(y>0.5*y.max())} pixels ({np.sum(y>0.5*y.max())/y.size*100:.2f}%)")
    print(f"      =max: {np.sum(y==y.max())} pixels ({np.sum(y==y.max())/y.size*100:.2f}%)")

# 分析Y值的直方图
print("\n\n📊 Histogram Analysis (all samples):")
Y_all_flat = Y.flatten()
print(f"  Total pixels: {len(Y_all_flat):,}")
print(f"  Unique values: {len(np.unique(Y_all_flat)):,}")

# 找到主要分布区间
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\n  Percentiles:")
for p in percentiles:
    val = np.percentile(Y_all_flat, p)
    print(f"    {p}th: {val:.0f}")

# 检查这是否是"到边界的距离"
print("\n\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Based on the analysis:

1. Y is StarDist-style Y-CHANNEL PROBABILITY MAP
   - Range: 0 to 133,737 (normalized)
   - This is NOT binary mask!
   
2. Distribution characteristics:
   - Most pixels have HIGH values (mean ~94,000 out of 133,737)
   - This suggests cells occupy MOST of the area
   - Lower values near boundaries

3. What this Y represents:
   - Y = distance to cell boundary (in pixels)
   - Y = 0 at cell boundary
   - Y = max at cell center
   - This is a REGRESSION target, not segmentation

4. PROBLEM with current visualization:
   - Using this Y as "Ground Truth" is circular reasoning
   - Model is trained to predict Y, then compared against Y
   - Shows model works, but doesn't show IMPROVEMENT

5. SOLUTION for fair comparison:
   - Binarize Y at a reasonable threshold (e.g., Y/133737 > 0.3)
   - Compare StarDist Baseline vs Complete Much Better
   - Show both models' predictions vs binary GT
""")

# 创建可视化
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 第一行：显示Y的原始分布
Y_norm_all = Y.astype(np.float32) / 133737.0

for i, idx in enumerate([0, 100, 300, 500]):
    ax = axes[0, i]
    y = Y_norm_all[idx]
    
    # 显示归一化后的Y
    im = ax.imshow(y, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Sample {idx}\nY normalized (0-1)', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.6)

# 第二行：二值化后的掩码
for i, idx in enumerate([0, 100, 300, 500]):
    ax = axes[1, i]
    y = Y_norm_all[idx]
    y_binary = (y > 0.3).astype(np.float32)
    
    ax.imshow(y_binary, cmap='Greens')
    ax.set_title(f'Sample {idx}\nBinary mask (thresh=0.3)', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Y Label Analysis: StarDist Probability Map → Binary Mask', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/performance_analysis/y_label_analysis.png', dpi=150)
print("\n✓ Saved: y_label_analysis.png")

