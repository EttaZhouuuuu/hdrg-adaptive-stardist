"""
Generate Ablation Study Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Ablation study results
configs = [
    'A_Baseline',
    'B_BoundaryLoss',
    'C_LovaszLoss',
    'D_BoundaryAttn',
    'E_FullIoU'
]

config_labels = [
    'Baseline\n(BCE+Dice)',
    '+Boundary\nLoss',
    '+Lovasz\nLoss',
    '+Boundary\nAttention',
    '+Full IoU\nOptimization'
]

dice_scores = [0.9680, 0.9707, 0.9702, 0.9731, 0.9705]
iou_scores = [0.9379, 0.9430, 0.9420, 0.9477, 0.9427]
improvements = [0, 0.51, 0.41, 0.98, 0.48]  # Percentage improvement over baseline

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Ablation Study: IoU Optimization Strategies', fontsize=14, fontweight='bold')

# Color scheme
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']

# Plot 1: Dice Scores
ax1 = axes[0]
bars1 = ax1.bar(config_labels, dice_scores, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Dice Score', fontsize=11)
ax1.set_title('Dice Score Comparison', fontsize=12)
ax1.set_ylim([0.960, 0.980])
ax1.axhline(y=dice_scores[0], color='gray', linestyle='--', alpha=0.7, label='Baseline')
for bar, score in zip(bars1, dice_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{score:.4f}', ha='center', va='bottom', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: IoU Scores
ax2 = axes[1]
bars2 = ax2.bar(config_labels, iou_scores, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('IoU Score', fontsize=11)
ax2.set_title('IoU Score Comparison', fontsize=12)
ax2.set_ylim([0.925, 0.955])
ax2.axhline(y=iou_scores[0], color='gray', linestyle='--', alpha=0.7, label='Baseline')
for bar, score in zip(bars2, iou_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{score:.4f}', ha='center', va='bottom', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: IoU Improvement
ax3 = axes[2]
bars3 = ax3.bar(config_labels[1:], improvements[1:], color=colors[1:], edgecolor='black', linewidth=1.2)
ax3.set_ylabel('IoU Improvement (\%)', fontsize=11)
ax3.set_title('IoU Improvement over Baseline', fontsize=12)
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
for bar, imp in zip(bars3, improvements[1:]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'+{imp:.2f}\%', ha='center', va='bottom', fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='Baseline'),
    Patch(facecolor='#2ecc71', edgecolor='black', label='Boundary Loss'),
    Patch(facecolor='#9b59b6', edgecolor='black', label='Lovasz Loss'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Boundary Attention'),
    Patch(facecolor='#f39c12', edgecolor='black', label='Full IoU Optimization'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.savefig('/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/performance_analysis/ablation_study_iou.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Ablation study visualization saved!")

# Print summary
print("\n" + "="*70)
print("ABLATION STUDY SUMMARY")
print("="*70)
print(f"\n{'Configuration':<30} {'Dice':>10} {'IoU':>10} {'Improvement':>12}")
print("-"*70)
for i, (config, dice, iou, imp) in enumerate(zip(config_labels, dice_scores, iou_scores, improvements)):
    improvement_str = f"+{imp:.2f}\%" if imp > 0 else "-"
    print(f"{configs[i]:<30} {dice:>10.4f} {iou:>10.4f} {improvement_str:>12}")
print("-"*70)
print("\nKey Findings:")
print("1. Boundary Attention provides the BEST single improvement (+0.98%)")
print("2. Boundary Loss contributes +0.51%")
print("3. Lovasz Loss contributes +0.41%")
print("4. Full model (42.6M) achieves +3.81% IoU over lightweight baseline")

