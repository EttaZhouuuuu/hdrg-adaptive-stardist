"""
Cellpose-SAM Baseline Evaluation
=================================
直接调用 Cellpose-SAM 预训练模型进行细胞实例分割评估
无需训练，开箱即用

Cellpose-SAM uses SAM (Segment Anything Model) as its backbone
for improved generalization to new cell types.
"""

import numpy as np
import torch
from pathlib import Path
from cellpose import models, io
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional


class Config:
    # 数据路径
    data_dir = Path(__file__).parent.parent / "data"
    
    # Cellpose-SAM 模型配置
    # model_type: 'cyto2', 'nuclei' - Cellpose-SAM 会自动使用 SAM backbone
    cellpose_model_type = "cyto2"  
    
    # 分割参数
    flow_threshold = 0.4  # 最大流量误差
    cellprob_threshold = 0.0  # 细胞概率阈值
    tile_norm_blocksize = 0  # 图像分块归一化大小 (0表示整个图像)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 输出路径
    output_dir = Path(__file__).parent / "cellpose_results"
    checkpoint_dir = Path(__file__).parent / "checkpoints"


def compute_instance_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算实例级 IoU"""
    pred_labels = np.unique(pred_mask)[1:]
    gt_labels = np.unique(gt_mask)[1:]
    
    if len(gt_labels) == 0:
        return 0.0
    
    ious = []
    for gt_label in gt_labels:
        gt_region = (gt_mask == gt_label)
        best_iou = 0.0
        
        for pred_label in pred_labels:
            pred_region = (pred_mask == pred_label)
            intersection = np.logical_and(gt_region, pred_region).sum()
            union = np.logical_or(gt_region, pred_region).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
        
        if best_iou > 0:
            ious.append(best_iou)
    
    return np.mean(ious) if ious else 0.0


def compute_pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算像素级 IoU"""
    pred_binary = (pred_mask > 0).astype(int)
    gt_binary = (gt_mask > 0).astype(int)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    return intersection / union if union > 0 else 0.0


def compute_aji(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算 Aggregate Jaccard Index (AJI)"""
    pred_labels = np.unique(pred_mask)[1:]
    gt_labels = np.unique(gt_mask)[1:]
    
    if len(gt_labels) == 0:
        return 0.0
    
    intersections = 0
    unions = 0
    
    for gt_label in gt_labels:
        gt_region = (gt_mask == gt_label)
        best_iou = 0.0
        best_pred = 0
        
        for pred_label in pred_labels:
            pred_region = (pred_mask == pred_label)
            intersection = np.logical_and(gt_region, pred_region).sum()
            union = np.logical_or(gt_region, pred_region).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_label
        
        if best_pred > 0:
            intersections += np.logical_and(gt_region, (pred_mask == best_pred)).sum()
            unions += np.logical_or(gt_region, (pred_mask == best_pred)).sum()
    
    return intersections / unions if unions > 0 else 0.0


def load_data(data_dir: Path, split: str = "test", max_samples: int = 100):
    """加载测试数据"""
    from utils.xenium_preprocessing import XeniumDataLoader
    
    print(f"Loading {split} data from {data_dir}...")
    loader = XeniumDataLoader(
        cells_zarr_path=str(data_dir / "cells.zarr"),
        image_path=str(data_dir / "morphology_focus.ome.tif"),
        load_image=True
    )
    
    # 提取 patches
    images, instance_masks = loader.extract_patches(
        patch_size=256,
        num_patches=min(max_samples, 500),
        use_precomputed=True
    )
    
    # 二值掩码
    binary_masks = (instance_masks > 0).astype(np.float32)
    
    return images, binary_masks, instance_masks


class CellposeSAMBaseline:
    """Cellpose-SAM 基线模型"""
    
    def __init__(
        self, 
        model_type: str = "cyto2", 
        device: torch.device = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        tile_norm_blocksize: int = 0
    ):
        """
        初始化 Cellpose-SAM 模型
        
        Args:
            model_type: 'cyto2' 或 'nuclei' - Cellpose-SAM 会自动使用 SAM backbone
            device: 计算设备
            flow_threshold: 流量误差阈值
            cellprob_threshold: 细胞概率阈值
            tile_norm_blocksize: 分块归一化大小
        """
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.tile_norm_blocksize = tile_norm_blocksize
        
        print(f"Loading Cellpose-SAM model ({model_type}) on {self.device}...")
        self.model = models.CellposeModel(
            gpu=(self.device.type == "cuda"),
            model_type=model_type
        )
        print("Cellpose-SAM model loaded!")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        对单张图像进行预测
        
        Args:
            image: (H, W) 单通道图像
        
        Returns:
            instance_mask: (H, W) 实例分割掩码
        """
        # 确保图像是正确格式
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # 转为 RGB
        
        # Cellpose-SAM 预测
        masks, flows, styles = self.model.eval(
            image,
            batch_size=1,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            normalize={"tile_norm_blocksize": self.tile_norm_blocksize}
        )
        
        # Cellpose 返回的是列表，取第一个
        if isinstance(masks, list):
            masks = masks[0]
        
        return masks.astype(np.float32)


def evaluate_cellpose_sam():
    """运行 Cellpose-SAM 基线评估"""
    
    print("=" * 60)
    print("Cellpose-SAM Baseline Evaluation")
    print("=" * 60)
    
    # 加载数据 (使用较少样本以加快测试)
    images, binary_masks, instance_masks = load_data(
        Config.data_dir, 
        split="test", 
        max_samples=50  # 先用 50 个样本测试
    )
    
    print(f"Loaded {len(images)} test samples")
    
    # 初始化模型
    baseline = CellposeSAMBaseline(
        model_type=Config.cellpose_model_type,
        device=Config.device,
        flow_threshold=Config.flow_threshold,
        cellprob_threshold=Config.cellprob_threshold,
        tile_norm_blocksize=Config.tile_norm_blocksize
    )
    
    # 评估
    print("\nRunning Cellpose-SAM inference...")
    metrics = {
        "pixel_iou": [],
        "instance_iou": [],
        "aji": [],
        "n_pred": [],
        "n_gt": []
    }
    
    for i in tqdm(range(len(images)), desc="Processing"):
        image = images[i]
        gt_mask = instance_masks[i]
        
        # 预测
        pred_mask = baseline.predict(image)
        
        # 计算指标
        pixel_iou = compute_pixel_iou(pred_mask, gt_mask)
        instance_iou = compute_instance_iou(pred_mask, gt_mask)
        aji = compute_aji(pred_mask, gt_mask)
        
        metrics["pixel_iou"].append(pixel_iou)
        metrics["instance_iou"].append(instance_iou)
        metrics["aji"].append(aji)
        metrics["n_pred"].append(len(np.unique(pred_mask)) - 1)
        metrics["n_gt"].append(len(np.unique(gt_mask)) - 1)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("Cellpose-SAM Results:")
    print("=" * 60)
    print(f"Pixel IoU:      {np.mean(metrics['pixel_iou']):.4f} ± {np.std(metrics['pixel_iou']):.4f}")
    print(f"Instance IoU:  {np.mean(metrics['instance_iou']):.4f} ± {np.std(metrics['instance_iou']):.4f}")
    print(f"AJI:           {np.mean(metrics['aji']):.4f} ± {np.std(metrics['aji']):.4f}")
    print(f"Avg Pred/Cell: {np.mean(metrics['n_pred']):.1f} (GT: {np.mean(metrics['n_gt']):.1f})")
    print("=" * 60)
    
    # 保存结果
    Config.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = Config.output_dir / "cellpose_sam_results.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "pixel_iou": float(np.mean(metrics["pixel_iou"])),
            "pixel_iou_std": float(np.std(metrics["pixel_iou"])),
            "instance_iou": float(np.mean(metrics["instance_iou"])),
            "instance_iou_std": float(np.std(metrics["instance_iou"])),
            "aji": float(np.mean(metrics["aji"])),
            "aji_std": float(np.std(metrics["aji"])),
            "model_type": Config.cellpose_model_type,
            "n_samples": len(images)
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return metrics


def compare_with_baselines(cellpose_metrics: dict):
    """与 U-Net 和 StarDist 结果对比"""
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH ALL BASELINES:")
    print("=" * 60)
    
    # U-Net 结果 (从之前的评估)
    unet_results = {
        "instance_iou": 0.0855,  # 需要从实际结果更新
        "pixel_iou": 0.9749
    }
    
    # 您的模型结果 (假设)
    stardist_results = {
        "instance_iou": 0.88,  # 您的模型结果
        "pixel_iou": None
    }
    
    print(f"{'Model':<20} {'Instance IoU':>15} {'Pixel IoU':>15}")
    print("-" * 50)
    print(f"{'Cellpose-SAM':<20} {np.mean(cellpose_metrics['instance_iou']):>15.4f} {np.mean(cellpose_metrics['pixel_iou']):>15.4f}")
    print(f"{'U-Net + Watershed':<20} {unet_results['instance_iou']:>15.4f} {unet_results['pixel_iou']:>15.4f}")
    print(f"{'Adaptive StarDist':<20} {stardist_results['instance_iou']:>15.4f} {'N/A':>15}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        metrics = evaluate_cellpose_sam()
        compare_with_baselines(metrics)
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install Cellpose-SAM first:")
        print("pip install git+https://www.github.com/mouseland/cellpose.git")
        print("\nOr use the fallback watershed method instead.")

