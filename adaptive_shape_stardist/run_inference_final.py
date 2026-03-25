"""
Final Inference Script for Trained StarDist Model
==================================================
"""

import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

print("=" * 80)
print("FINAL INFERENCE - TRAINED Stardist MODEL")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

import warnings
warnings.filterwarnings('ignore')

# Import model from training script
exec(open('adaptive_shape_stardist/train_stardist_fixed.py').read())

# Load data
data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
X = data['X']
Y_raw = data['Y']

print(f"\n📊 Dataset: X={X.shape}, Y={Y_raw.shape}")

# Preprocess
if len(X.shape) == 3:
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)
X = X.astype(np.float32) / 255.0

# Normalize Y
Y_max = Y_raw.max()
Y_norm = Y_raw.astype(np.float32) / Y_max
Y_norm = np.clip(Y_norm, 0, 1)

# Create model and load best checkpoint
model = AdaptiveShapeStarDist(in_channels=3).to(device)
best_model_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models/stardist_best_fixed.pth'

if Path(best_model_path).exists():
    print(f"\n>>> Loading best model: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("    ✅ Best model loaded!")
else:
    print(f"\n❌ Best model not found: {best_model_path}")
    sys.exit(1)

model.eval()

# Split data
split_idx = int(len(X) * 0.85)
X_val = torch.FloatTensor(X[split_idx:]).permute(0, 3, 1, 2).to(device)
Y_val = torch.FloatTensor(Y_norm[split_idx:]).to(device)

print(f"\n📊 Validation set: {len(X_val)} samples")

# Run inference
print("\n🚀 Running Inference...")
with torch.no_grad():
    outputs = model(X_val)

preds = torch.sigmoid(outputs)

# Metrics
def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2. * intersection) / (pred_bin.sum() + target.sum() + 1e-8).item()

print(f"\n" + "=" * 80)
print("INFERENCE RESULTS")
print("=" * 80)

print(f"\n📊 Prediction Statistics:")
print(f"   Shape: {preds.shape}")
print(f"   Mean: {preds.mean().item():.4f}")
print(f"   Std: {preds.std().item():.4f}")
print(f"   Min: {preds.min().item():.4f}")
print(f"   Max: {preds.max().item():.4f}")

print(f"\n📊 Ground Truth Statistics:")
print(f"   Mean: {Y_val.mean().item():.4f}")
print(f"   Positive pixels: {(Y_val > 0.5).sum() / Y_val.numel() * 100:.1f}%")

print(f"\n🎯 Evaluation Metrics:")
for thresh in [0.1, 0.3, 0.5]:
    dice = dice_score(preds, Y_val, threshold=thresh)
    print(f"   Dice Score (threshold={thresh}): {dice:.4f}")

# Pixel accuracy
pred_binary = (preds > 0.5).float()
accuracy = (pred_binary == Y_val).float().mean()
print(f"   Pixel Accuracy (thresh=0.5): {accuracy.item():.4f}")

# Save predictions
output_dir = Path(PROJECT_DIR) / 'adaptive_shape_stardist' / 'inference_results'
output_dir.mkdir(exist_ok=True)
np.save(output_dir / 'final_predictions.npy', preds.cpu().numpy())
print(f"\n💾 Predictions saved to: {output_dir / 'final_predictions.npy'}")

print("\n" + "=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)

