"""
Generate IoU Training Curves Visualization
Based on actual training log from terminal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Actual training data extracted from logs
# Format: (epoch, train_loss, val_loss, val_dice, val_iou, threshold)
training_data = [
    (1, 1.2301, 1.1363, 0.9286, 0.8667, 0.50),
    (2, 0.8629, 0.6769, 0.9702, 0.9421, 0.60),
    (3, 0.6817, 0.6012, 0.9698, 0.9414, 0.40),
    (4, 0.6306, 0.5246, 0.9769, 0.9549, 0.55),
    (5, 0.5973, 0.4960, 0.9780, 0.9570, 0.50),
    (6, 0.5371, 0.5112, 0.9766, 0.9543, 0.60),
    (7, 0.5339, 0.4909, 0.9782, 0.9574, 0.55),
    (8, 0.5365, 0.4730, 0.9804, 0.9616, 0.65),
    (9, 0.5069, 0.4818, 0.9796, 0.9601, 0.70),
    (10, 0.5067, 0.4749, 0.9805, 0.9618, 0.60),
    (11, 0.5063, 0.4720, 0.9803, 0.9613, 0.50),
    (12, 0.4981, 0.4742, 0.9798, 0.9604, 0.40),
    (13, 0.4883, 0.4728, 0.9806, 0.9619, 0.45),
    (14, 0.4770, 0.4570, 0.9813, 0.9633, 0.50),
    (15, 0.4714, 0.4616, 0.9808, 0.9624, 0.60),
    (16, 0.4723, 0.4511, 0.9826, 0.9658, 0.45),
    (17, 0.4684, 0.4488, 0.9822, 0.9649, 0.55),
    (18, 0.4821, 0.4496, 0.9819, 0.9645, 0.50),
    (19, 0.4656, 0.4477, 0.9830, 0.9666, 0.50),
    (20, 0.4653, 0.4489, 0.9824, 0.9654, 0.50),
    (21, 0.4618, 0.4448, 0.9829, 0.9665, 0.50),
    (22, 0.4580, 0.4490, 0.9830, 0.9665, 0.50),
    (23, 0.4640, 0.4450, 0.9832, 0.9670, 0.50),
    (24, 0.4680, 0.4481, 0.9835, 0.9675, 0.40),
    (25, 0.4682, 0.4492, 0.9828, 0.9661, 0.45),
    (26, 0.4796, 0.4612, 0.9814, 0.9634, 0.45),
    (27, 0.4934, 0.4531, 0.9827, 0.9659, 0.45),
    (28, 0.4801, 0.4710, 0.9748, 0.9509, 0.70),
    (29, 0.4760, 0.4573, 0.9812, 0.9631, 0.45),
    (30, 0.4607, 0.4546, 0.9819, 0.9645, 0.55),
    (31, 0.4586, 0.4701, 0.9828, 0.9661, 0.30),
    (32, 0.4523, 0.4446, 0.9829, 0.9663, 0.45),
    (33, 0.4609, 0.4458, 0.9832, 0.9669, 0.55),
    (34, 0.4554, 0.4489, 0.9819, 0.9644, 0.60),
    (35, 0.4609, 0.4588, 0.9802, 0.9611, 0.45),
    (36, 0.4496, 0.4413, 0.9833, 0.9671, 0.50),
    (37, 0.4604, 0.4397, 0.9844, 0.9693, 0.45),  # Best: 0.9693
    (38, 0.4499, 0.4476, 0.9821, 0.9649, 0.50),
    (39, 0.4606, 0.4426, 0.9831, 0.9667, 0.60),
    (40, 0.4590, 0.4642, 0.9824, 0.9654, 0.50),
    (41, 0.4581, 0.4387, 0.9840, 0.9686, 0.55),
    (42, 0.4494, 0.4361, 0.9846, 0.9697, 0.60),
    (43, 0.4443, 0.4349, 0.9844, 0.9693, 0.50),
    (44, 0.4403, 0.4413, 0.9826, 0.9658, 0.50),
    (45, 0.4468, 0.4417, 0.9843, 0.9690, 0.70),
    (46, 0.4471, 0.4351, 0.9847, 0.9698, 0.50),
    (47, 0.4419, 0.4393, 0.9842, 0.9690, 0.45),
    (48, 0.4396, 0.4337, 0.9846, 0.9697, 0.45),
    (49, 0.4332, 0.4368, 0.9848, 0.9700, 0.50),
    (50, 0.4304, 0.4431, 0.9815, 0.9638, 0.55),
    (51, 0.4376, 0.4355, 0.9845, 0.9695, 0.50),
    (52, 0.4348, 0.4423, 0.9851, 0.9706, 0.35),
    (53, 0.4289, 0.4297, 0.9849, 0.9702, 0.55),
    (54, 0.4269, 0.4369, 0.9837, 0.9678, 0.45),
    (55, 0.4270, 0.4275, 0.9854, 0.9712, 0.50),
    (56, 0.4255, 0.4279, 0.9856, 0.9716, 0.50),
    (57, 0.4267, 0.4371, 0.9857, 0.9719, 0.40),
    (58, 0.4299, 0.4278, 0.9854, 0.9712, 0.50),
    (59, 0.4221, 0.4322, 0.9857, 0.9718, 0.45),
    (60, 0.4275, 0.4283, 0.9860, 0.9724, 0.45),
    (61, 0.4204, 0.4310, 0.9844, 0.9693, 0.45),
    (62, 0.4191, 0.4306, 0.9845, 0.9695, 0.50),
    (63, 0.4189, 0.4260, 0.9856, 0.9717, 0.55),
    (64, 0.4217, 0.4287, 0.9856, 0.9717, 0.55),
    (65, 0.4213, 0.4342, 0.9850, 0.9704, 0.45),
    (66, 0.4164, 0.4263, 0.9864, 0.9731, 0.50),
    (67, 0.4198, 0.4275, 0.9857, 0.9719, 0.55),
    (68, 0.4157, 0.4248, 0.9863, 0.9729, 0.45),
    (69, 0.4177, 0.4303, 0.9848, 0.9701, 0.50),
    (70, 0.4152, 0.4266, 0.9863, 0.9729, 0.50),
    (71, 0.4164, 0.4270, 0.9866, 0.9736, 0.45),  # BEST: 0.9736
    (72, 0.4176, 0.4254, 0.9866, 0.9736, 0.45),  # BEST: 0.9736
    (73, 0.4293, 0.4342, 0.9791, 0.9590, 0.50),  # Resume from checkpoint
    (74, 0.4362, 0.4451, 0.9801, 0.9610, 0.35),
    (75, 0.4299, 0.4281, 0.9804, 0.9616, 0.50),
    (76, 0.4359, 0.4424, 0.9782, 0.9574, 0.55),
    (77, 0.4273, 0.4378, 0.9783, 0.9575, 0.50),
    (78, 0.4389, 0.4594, 0.9745, 0.9504, 0.50),
    (79, 0.4253, 0.4441, 0.9789, 0.9586, 0.50),
    (80, 0.4229, 0.4335, 0.9801, 0.9610, 0.50),
    (81, 0.4247, 0.4454, 0.9779, 0.9568, 0.70),
    (82, 0.4260, 0.4333, 0.9790, 0.9590, 0.55),
    (83, 0.4196, 0.4288, 0.9790, 0.9589, 0.50),
    (84, 0.4167, 0.4294, 0.9813, 0.9634, 0.45),
    (85, 0.4172, 0.4304, 0.9802, 0.9611, 0.45),
    (86, 0.4175, 0.4282, 0.9817, 0.9641, 0.40),
    (87, 0.4145, 0.4409, 0.9785, 0.9579, 0.45),
    (88, 0.4166, 0.4272, 0.9806, 0.9620, 0.45),
    (89, 0.4089, 0.4290, 0.9787, 0.9583, 0.55),
    (90, 0.4095, 0.4246, 0.9813, 0.9633, 0.55),
    (91, 0.4083, 0.4194, 0.9821, 0.9648, 0.45),
    (92, 0.4037, 0.4197, 0.9815, 0.9637, 0.50),
    (93, 0.4078, 0.4229, 0.9822, 0.9650, 0.45),
    (94, 0.4081, 0.4254, 0.9810, 0.9627, 0.50),
    (95, 0.4052, 0.4253, 0.9823, 0.9652, 0.40),
    (96, 0.4042, 0.4233, 0.9811, 0.9628, 0.45),
    (97, 0.4044, 0.4328, 0.9792, 0.9592, 0.55),
    (98, 0.4167, 0.4360, 0.9811, 0.9629, 0.45),
    (99, 0.4279, 0.4417, 0.9790, 0.9590, 0.35),
    (100, 0.4191, 0.4446, 0.9780, 0.9569, 0.55),
    (101, 0.4214, 0.4365, 0.9774, 0.9558, 0.50),
    (102, 0.4169, 0.4916, 0.9639, 0.9303, 0.75),
    (103, 0.4160, 0.4312, 0.9803, 0.9614, 0.45),
    (104, 0.4188, 0.4463, 0.9777, 0.9564, 0.40),
    (105, 0.4129, 0.4374, 0.9772, 0.9555, 0.60),
    (106, 0.4122, 0.4411, 0.9797, 0.9601, 0.45),
    (107, 0.4182, 0.4284, 0.9799, 0.9607, 0.50),
]

# Convert to numpy array
data = np.array(training_data)
epochs = data[:, 0]
train_loss = data[:, 1]
val_loss = data[:, 2]
val_dice = data[:, 3]
val_iou = data[:, 4]
val_threshold = data[:, 5]

# Find best values
best_iou_idx = np.argmax(val_iou)
best_dice_idx = np.argmax(val_dice)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Complete Much Better - IoU Optimized Training Curves\n(Epochs 1-72: Initial Training | Epochs 73-107: Resume from Checkpoint)',
              fontsize=12, fontweight='bold')

# Plot 1: Loss curves
ax1 = axes[0, 0]
ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=1.5, alpha=0.8)
ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=1.5, alpha=0.8)
ax1.axvline(x=72, color='gray', linestyle='--', alpha=0.7, label='Checkpoint Resume')
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.set_title('Training and Validation Loss', fontsize=11)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([1, 107])

# Plot 2: Dice curve
ax2 = axes[0, 1]
ax2.plot(epochs, val_dice, 'g-', linewidth=1.5, alpha=0.8)
ax2.axhline(y=val_dice[best_dice_idx], color='r', linestyle='--', alpha=0.7,
            label=f'Best Dice: {val_dice[best_dice_idx]:.4f} (Epoch {int(epochs[best_dice_idx])})')
ax2.axvline(x=72, color='gray', linestyle='--', alpha=0.7)
ax2.fill_between(epochs, val_dice, alpha=0.2, color='green')
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Dice Score', fontsize=10)
ax2.set_title('Validation Dice Score', fontsize=11)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1, 107])
ax2.set_ylim([0.96, 0.99])

# Plot 3: IoU curve
ax3 = axes[1, 0]
ax3.plot(epochs, val_iou, 'm-', linewidth=1.5, alpha=0.8)
ax3.axhline(y=val_iou[best_iou_idx], color='r', linestyle='--', alpha=0.7,
            label=f'Best IoU: {val_iou[best_iou_idx]:.4f} (Epoch {int(epochs[best_iou_idx])})')
ax3.axvline(x=72, color='gray', linestyle='--', alpha=0.7, label='Checkpoint Resume')
ax3.fill_between(epochs, val_iou, alpha=0.2, color='purple')
ax3.set_xlabel('Epoch', fontsize=10)
ax3.set_ylabel('IoU Score', fontsize=10)
ax3.set_title('Validation IoU Score', fontsize=11)
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([1, 107])
ax3.set_ylim([0.92, 0.98])

# Plot 4: Best threshold curve
ax4 = axes[1, 1]
ax4.plot(epochs, val_threshold, 'c-', linewidth=1.5, alpha=0.8, marker='o', markersize=2)
ax4.axvline(x=72, color='gray', linestyle='--', alpha=0.7, label='Checkpoint Resume')
ax4.set_xlabel('Epoch', fontsize=10)
ax4.set_ylabel('Threshold', fontsize=10)
ax4.set_title('Optimal Threshold over Epochs', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([1, 107])
ax4.set_ylim([0.25, 0.80])

plt.tight_layout()
plt.savefig('/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/performance_analysis/iou_training_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("=" * 60)
print("IoU Training Curves Generated Successfully!")
print("=" * 60)
print(f"\nSaved to: performance_analysis/iou_training_curves.png")
print(f"\nBest Results:")
print(f"  - Best Dice: {val_dice[best_dice_idx]:.4f} (Epoch {int(epochs[best_dice_idx])})")
print(f"  - Best IoU:  {val_iou[best_iou_idx]:.4f} (Epoch {int(epochs[best_iou_idx])})")
print(f"\nTraining Summary:")
print(f"  - Total Epochs: {len(epochs)}")
print(f"  - Initial Training: Epochs 1-72")
print(f"  - Resume Training: Epochs 73-107")
print(f"  - Early Stopping: Epoch 107 (patience=35)")
