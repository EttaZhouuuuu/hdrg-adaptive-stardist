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
import os

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入数据工具
from adaptive_shape_stardist.utils.data_utils import load_data, set_seed

# 设置随机种子以确保可重复性
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def calculate_metrics(pred, target, threshold=0.5):
    """
    计算各种评估指标

    Args:
        pred: 预测值 (B, 1, H, W)
        target: 目标值 (B, 1, H, W)
        threshold: 二值化阈值

    Returns:
        dict: 包含各种指标的字典
    """
    # 转换为 numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # 裁剪到 [0, 1] 范围
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)

    # 二值化
    pred_binary = (pred_np > threshold).astype(np.float32)
    target_binary = (target_np > threshold).astype(np.float32)

    # 1. MSE (Mean Squared Error)
    mse = np.mean((pred_np - target_np) ** 2)

    # 2. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)

    # 3. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_np - target_np))

    # 4. Pearson Correlation
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

    # 5. Dice Score
    def dice_score(pred_bin, target_bin, smooth=1e-8):
        intersection = np.sum(pred_bin * target_bin)
        union = np.sum(pred_bin) + np.sum(target_bin)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    dice_per_sample = []
    for i in range(pred_binary.shape[0]):
        dice = dice_score(pred_binary[i], target_binary[i])
        dice_per_sample.append(dice)
    dice_mean = np.mean(dice_per_sample)
    dice_std = np.std(dice_per_sample)

    # 6. IoU (Intersection over Union)
    def iou_score(pred_bin, target_bin, smooth=1e-8):
        intersection = np.sum(pred_bin * target_bin)
        union = np.sum(pred_bin) + np.sum(target_bin) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

    iou_per_sample = []
    for i in range(pred_binary.shape[0]):
        iou = iou_score(pred_binary[i], target_binary[i])
        iou_per_sample.append(iou)
    iou_mean = np.mean(iou_per_sample)
    iou_std = np.std(iou_per_sample)

    # 7. Pixel Accuracy
    accuracy = np.mean(pred_binary == target_binary)

    # 8. Precision and Recall
    true_positives = np.sum((pred_binary == 1) & (target_binary == 1))
    false_positives = np.sum((pred_binary == 1) & (target_binary == 0))
    false_negatives = np.sum((pred_binary == 0) & (target_binary == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    # 9. F1 Score
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


def inference_model(model, dataloader, device):
    """
    对模型进行推理

    Args:
        model: PyTorch 模型
        dataloader: 数据加载器
        device: 设备 (cuda 或 cpu)

    Returns:
        tuple: (predictions, targets)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            # 处理不同的输出格式
            if isinstance(outputs, tuple):
                pred = outputs[0]
            else:
                pred = outputs

            # 确保输出形状与目标一致
            if pred.shape != batch_y.shape:
                if pred.shape[2:] != batch_y.shape[2:]:
                    pred = nn.functional.interpolate(
                        pred,
                        size=batch_y.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

            # 裁剪到 [0, 1] 范围
            pred = torch.clamp(pred, 0, 1)

            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_preds, all_targets


def main():
    print("=" * 80)
    print("ADAPTIVE SHAPE STARDIST - INFERENCE AND EVALUATION")
    print("=" * 80)

    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 路径配置
    model_path = project_root / 'models_complete' / 'best_complete.pth'
    checkpoint_dir = project_root / 'models_complete'
    data_path = project_root.parent / 'data'

    print(f"\n📂 Paths:")
    print(f"   Model: {model_path}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    print(f"   Data: {data_path}")

    # 加载数据
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    X, Y = load_data(str(data_path))
    print(f"   Raw data: X={X.shape}, Y={Y.shape}")

    # Global Y normalization
    Y_max = Y.max()
    print(f"   Global Y max: {Y_max:.2f}")
    Y_normalized = Y / Y_max
    print(f"   After normalization: min={Y_normalized.min():.4f}, max={Y_normalized.max():.4f}")

    # 划分数据集 (使用验证集)
    train_split = 0.85
    n_samples = len(X)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train

    X_val = X[n_train:]
    Y_val = Y_normalized[n_train:]

    print(f"\n   Dataset split: {n_train} train, {n_val} val")

    # 创建验证集 DataLoader
    X_val_t = torch.FloatTensor(X_val.transpose(0, 3, 1, 2))  # (N, 3, H, W)
    Y_val_t = torch.FloatTensor(Y_val.transpose(0, 3, 1, 2))  # (N, 1, H, W)

    val_dataset = TensorDataset(X_val_t, Y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"   Validation batches: {len(val_loader)}")

    # 创建模型
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = AdaptiveShapeStarDistComplete(
        in_channels=3,
        out_channels=1,
        backbone='resnet50',
        pretrained=False,
        spine_channels=[128, 256, 512, 1024],
        fpn_channels=256,
        num_prototypes=16,
        d_model=256,
        nhead=8,
        num_encoder_layers=3
    ).to(device)

    # 加载最佳权重
    print(f"\n📂 Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 处理 DataParallel 状态字典
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    print("   ✓ Model loaded successfully")

    # 使用 DataParallel (如果有多GPU)
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

    start_time = time.time()
    predictions, targets = inference_model(model, val_loader, device)
    inference_time = time.time() - start_time

    print(f"\n   Inference time: {inference_time:.2f}s")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Targets shape: {targets.shape}")

    # 计算统计信息
    print(f"\n   Prediction stats:")
    print(f"      Mean: {predictions.mean():.4f}")
    print(f"      Std: {predictions.std():.4f}")
    print(f"      Min: {predictions.min():.4f}")
    print(f"      Max: {predictions.max():.4f}")

    print(f"\n   Target stats:")
    print(f"      Mean: {targets.mean():.4f}")
    print(f"      Std: {targets.std():.4f}")
    print(f"      Min: {targets.min():.4f}")
    print(f"      Max: {targets.max():.4f}")

    # 计算评估指标
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

    # 额外分析：不同阈值下的 Dice 分数
    print(f"\n📊 Dice Score at Different Thresholds:")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_dice = 0
    best_threshold = 0.5
    for thresh in thresholds:
        metrics_thresh = calculate_metrics(predictions, targets, threshold=thresh)
        print(f"   Threshold {thresh:.1f}: Dice = {metrics_thresh['dice_mean']:.4f}")
        if metrics_thresh['dice_mean'] > best_dice:
            best_dice = metrics_thresh['dice_mean']
            best_threshold = thresh

    print(f"\n   Best Dice: {best_dice:.4f} at threshold {best_threshold}")

    # 保存结果
    results = {
        'model_path': str(model_path),
        'val_samples': n_val,
        'inference_time': inference_time,
        'metrics': metrics,
        'best_threshold': best_threshold,
        'best_dice': best_dice
    }

    results_path = checkpoint_dir / 'evaluation_results.npy'
    np.save(results_path, results)
    print(f"\n📂 Results saved to: {results_path}")

    # 保存详细报告
    report_path = checkpoint_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ADAPTIVE SHAPE STARDIST - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Device: {device}\n\n")

        f.write("DATA INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Validation samples: {n_val}\n")
        f.write(f"Y normalization max: {Y_max:.2f}\n\n")

        f.write("REGRESSION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"MSE:  {metrics['mse']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"MAE:  {metrics['mae']:.6f}\n")
        f.write(f"Pearson Correlation: {metrics['pearson_correlation']:.6f}\n\n")

        f.write("SEGMENTATION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dice Score: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}\n")
        f.write(f"IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")

        f.write("BEST CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best threshold: {best_threshold}\n")
        f.write(f"Best Dice: {best_dice:.4f}\n")

    print(f"📂 Report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()

