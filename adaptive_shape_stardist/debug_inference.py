"""
Debug inference script - check raw model outputs
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

PROJECT_DIR = '/data/yitongzhou/workspace/hdrg-adaptive-stardist'
sys.path.insert(0, PROJECT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Import model components
from gpu_inference_train import AdaptiveShapeStarDist

# Load data
data = np.load(f'{PROJECT_DIR}/adaptive_shape_stardist/training_data.npz', allow_pickle=True)
X = data['X']
Y = data['Y']

print(f"X shape: {X.shape}, Y shape: {Y.shape}")
print(f"Y mean: {Y.mean():.4f}, Y max: {Y.max():.4f}")

# Preprocess
if len(X.shape) == 3:
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)
X = X.astype(np.float32) / 255.0

# Normalize Y
Y_max = Y.max()
Y_normalized = Y / (Y_max + 1e-8)
Y_normalized = np.clip(Y_normalized, 0, 1)
Y_binary = (Y_normalized > 0.5).astype(np.float32)

print(f"Y_binary mean: {Y_binary.mean():.4f}")
print(f"Positive ratio: {(Y_binary > 0).sum() / Y_binary.size * 100:.1f}%")

# Create model and load checkpoint
model = AdaptiveShapeStarDist(in_channels=3).to(device)

checkpoint_path = f'{PROJECT_DIR}/adaptive_shape_stardist/models/pytorch_full_checkpoint.pth'
if Path(checkpoint_path).exists():
    print(f"\n>>> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Resume from epoch: {checkpoint.get('epoch', 0)}")

model.eval()

# Test on a few samples
X_test = torch.FloatTensor(X[:5]).permute(0, 3, 1, 2).to(device)

with torch.no_grad():
    outputs = model(X_test)

# Check raw outputs (before sigmoid)
print(f"\n📊 Raw Output Statistics (before sigmoid):")
print(f"   Shape: {outputs.shape}")
print(f"   Mean: {outputs.mean().item():.4f}")
print(f"   Std: {outputs.std().item():.4f}")
print(f"   Min: {outputs.min().item():.4f}")
print(f"   Max: {outputs.max().item():.4f}")

# Apply sigmoid
preds = torch.sigmoid(outputs)
print(f"\n📊 After Sigmoid:")
print(f"   Mean: {preds.mean().item():.4f}")
print(f"   Std: {preds.std().item():.4f}")
print(f"   Min: {preds.min().item():.4f}")
print(f"   Max: {preds.max().item():.4f}")

# Compare with ground truth
Y_test = torch.FloatTensor(Y_binary[:5]).unsqueeze(1).to(device)
print(f"\n📊 Ground Truth Statistics:")
print(f"   Mean: {Y_test.mean().item():.4f}")
print(f"   Positive ratio: {(Y_test > 0).sum() / Y_test.numel() * 100:.1f}%")

# Calculate dice score manually
def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2. * intersection) / (pred_bin.sum() + target.sum() + 1e-8)

dice = dice_score(preds, Y_test)
print(f"\n🎯 Dice Score: {dice.item():.4f}")

# Sample pixel-level accuracy
pred_binary = (preds > 0.5).float()
accuracy = (pred_binary == Y_test).float().mean()
print(f"🎯 Pixel Accuracy: {accuracy.item():.4f}")

