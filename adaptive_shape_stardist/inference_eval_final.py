#!/usr/bin/env python3
"""
严谨的 Inference 和 Evaluation 脚本
对训练完成的 Adaptive Shape StarDist 模型进行评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys
import time

# 添加项目路径
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

from adaptive_shape_stardist.train_stardist_complete import (
    AdaptiveShapeStarDistComplete,
    UNetBackbone,
    ShapePriorEncoderComplete,
    PositionalEncoding2D,
    PositionalEncoding1D,
    TransformerBlock,
    CrossAttentionFusion
)

def calculate_metrics(pred, target, threshold=0.5):
    """计算评估指标"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # 裁剪
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)

    # 二值化
    pred_binary = (pred_np > threshold).astype(np.float32)
    target_binary = (target_np > threshold).astype(np.float32)

    # MSE
    mse = np.mean((pred_np - target_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_np - target_np))

    # Pearson Correlation
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

    # Dice Score
    def dice_score(pred_bin, target_bin, smooth=1e-8):
        intersection = np.sum(pred_bin * target_bin)
        union = np.sum(pred_bin) + np.sum(target_bin)
        return (2. * intersection + smooth) / (union + smooth)

    dice_per_sample = [dice_score(pred_binary[i], target_binary[i]) for i in range(pred_binary.shape[0])]
    dice_mean = np.mean(dice_per_sample)
    dice_std = np.std(dice_per_sample)

    # IoU
    def iou_score(pred_bin, target_bin, smooth=1e-8):
        intersection = np.sum(pred_bin * target_bin)
        union = np.sum(pred_bin) + np.sum(target_bin) - intersection
        return (intersection + smooth) / (union + smooth)

    iou_per_sample = [iou_score(pred_binary[i], target_binary[i]) for i in range(pred_binary.shape[0])]
    iou_mean = np.mean(iou_per_sample)
    iou_std = np.std(iou_per_sample)

    # Accuracy
    accuracy = np.mean(pred_binary == target_binary)

    # Precision and Recall
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson_correlation': correlation,
        'dice_mean': dice_mean,
        'dice_std': dice_std,
        'iou_mean': iou_mean,
        'iou_std': iou_std,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def load_data():
    """加载数据"""
    data_path = PROJECT_DIR / 'adaptive_shape_stardist' / 'training_data.npz'
    print(f"   Loading data from: {data_path}")
    data = np.load(str(data_path), allow_pickle=True)
    X = data['X']
    Y_raw = data['Y']

    # Convert X to 3-channel if needed
    if len(X.shape) == 3:
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)

    X = X.astype(np.float32) / 255.0

    return X, Y_raw


def main():
    print("=" * 80)
    print("ADAPTIVE SHAPE STARDIST - INFERENCE AND EVALUATION")
    print("=" * 80)

    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 路径
    model_path = PROJECT_DIR / 'adaptive_shape_stardist' / 'models_complete' / 'best_complete.pth'

    print(f"\n📂 Paths:")
    print(f"   Model: {model_path}")

    # 加载数据
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    X, Y = load_data()
    print(f"   Raw data: X={X.shape}, Y={Y.shape}")

    # Global Y normalization
    Y_max = Y.max()
    print(f"   Global Y max: {Y_max:.2f}")
    Y_normalized = Y / Y_max
    print(f"   After normalization: min={Y_normalized.min():.4f}, max={Y_normalized.max():.4f}")

    # 划分数据集
    train_split = 0.85
    n_samples = len(X)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train

    X_val = X[n_train:]
    Y_val = Y_normalized[n_train:]

    print(f"\n   Dataset split: {n_train} train, {n_val} val")

    # DataLoader
    X_val_t = torch.FloatTensor(X_val.transpose(0, 3, 1, 2))  # (N, 3, H, W)
    # Y is (N, H, W), need to add channel dimension
    Y_val_t = torch.FloatTensor(Y_val[:, np.newaxis, :, :])  # (N, 1, H, W)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"   Validation batches: {len(val_loader)}")

    # 创建模型
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = AdaptiveShapeStarDistComplete(in_channels=3).to(device)

    # 加载权重
    print(f"\n📂 Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 检查 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # checkpoint 直接是 state_dict

    # 处理 DataParallel 状态字典
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    print("   ✓ Model loaded successfully")

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"\n   Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # 推理
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            # 确保形状一致
            if outputs.shape != batch_y.shape:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=batch_y.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            outputs = torch.clamp(outputs, 0, 1)

            all_preds.append(outputs.cpu())
            all_targets.append(batch_y.cpu())

    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    print(f"\n   Predictions shape: {predictions.shape}")
    print(f"   Targets shape: {targets.shape}")

    # 统计
    print(f"\n   Prediction stats:")
    print(f"      Mean: {predictions.mean():.4f}")
    print(f"      Std: {predictions.std():.4f}")

    print(f"\n   Target stats:")
    print(f"      Mean: {targets.mean():.4f}")
    print(f"      Std: {targets.std():.4f}")

    # 计算指标
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    metrics = calculate_metrics(predictions, targets)

    print(f"\n📊 Regression Metrics:")
    print(f"   MSE:  {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE:  {metrics['mae']:.6f}")
    print(f"   Pearson Correlation: {metrics['pearson_correlation']:.6f}")

    print(f"\n📊 Segmentation Metrics:")
    print(f"   Dice Score: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"   IoU:        {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"   Accuracy:   {metrics['accuracy']:.4f}")
    print(f"   Precision:  {metrics['precision']:.4f}")
    print(f"   Recall:     {metrics['recall']:.4f}")
    print(f"   F1 Score:   {metrics['f1_score']:.4f}")

    # 不同阈值
    print(f"\n📊 Dice Score at Different Thresholds:")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_dice = 0
    best_threshold = 0.5
    for thresh in thresholds:
        m = calculate_metrics(predictions, targets, threshold=thresh)
        print(f"   Threshold {thresh:.1f}: Dice = {m['dice_mean']:.4f}")
        if m['dice_mean'] > best_dice:
            best_dice = m['dice_mean']
            best_threshold = thresh

    print(f"\n   Best Dice: {best_dice:.4f} at threshold {best_threshold}")

    # 保存报告
    checkpoint_dir = PROJECT_DIR / 'adaptive_shape_stardist' / 'models_complete'
    report_path = checkpoint_dir / 'evaluation_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ADAPTIVE SHAPE STARDIST - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Validation samples: {n_val}\n\n")
        f.write("REGRESSION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"MSE:  {metrics['mse']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"MAE:  {metrics['mae']:.6f}\n")
        f.write(f"Pearson: {metrics['pearson_correlation']:.6f}\n\n")
        f.write("SEGMENTATION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}\n")
        f.write(f"IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1: {metrics['f1_score']:.4f}\n\n")
        f.write(f"Best threshold: {best_threshold}\n")
        f.write(f"Best Dice: {best_dice:.4f}\n")

    print(f"\n📂 Report saved to: {report_path}")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

