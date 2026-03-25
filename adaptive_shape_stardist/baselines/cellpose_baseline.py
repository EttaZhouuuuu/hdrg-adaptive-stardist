"""
Cellpose Baseline Evaluation
============================
直接调用 Cellpose 预训练模型进行细胞实例分割评估
无需训练，开箱即用
"""

import numpy as np
import torch
from pathlib import Path
from cellpose import models as cellpose_models
from cellpose import utils as cellpose_utils
from tqdm import tqdm
import json
from typing import Dict, List, Tuple


class Config:
    # 数据路径
    data_dir = Path(__file__).parent.parent / "data"
    
    # Cellpose 模型配置
    cellpose_model_type = "cyto2"  # 'cyto', 'cyto2', 'nuclei'
    cellpose_diameter = None  # None 表示自动估计
    cellpose_flow_threshold = 0.4
    cellpose_cellprob_threshold = 0.0
    
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


def compute_aggregate_jaccard_index(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算 AJI (Aggregate Jaccard Index)"""
    # 这是一个简化的 AJI 计算
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
        
        if best_iou > 0:
            intersections += np.logical_and(gt_region, (pred_mask == best_pred)).sum()
            unions += np.logical_or(gt_region, (pred_mask == best_pred)).sum()
    
    return intersections / unions if unions > 0 else 0.0


def load_data(data_dir: Path, split: str = "test"):
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
        num_patches=500,
        use_precomputed=True
    )
    
    # 二值掩码
    binary_masks = (instance_masks > 0).astype(np.float32)
    
    return images, binary_masks, instance_masks


class CellposeBaseline:
    """Cellpose 基线模型"""
    
    def __init__(self, model_type: str = "cyto2", device: torch.device = None):
        """
        初始化 Cellpose 模型
        
        Args:
            model_type: 'cyto', 'cyto2', 'nuclei'
            device: 计算设备
        """
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Cellpose model ({model_type}) on {self.device}...")
        self.model = cellpose_models.Cellpose(
            gpu=(self.device.type == "cuda"),
            model_type=model_type
        )
        print("Cellpose model loaded!")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        对单张图像进行预测
        
        Args:
            image: (H, W) 单通道图像
        
        Returns:
            instance_mask: (H, W) 实例分割掩码
        """
        # Cellpose 预测
        masks, flows, styles = self.model.eval(
            image,
            diameter=self.model_type == "cyto2" and None,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            do_3D=False
        )
        
        # Cellpose 返回的是列表，取第一个
        if isinstance(masks, list):
            masks = masks[0]
        
        return masks.astype(np.float32)


def evaluate_cellpose():
    """运行 Cellpose 基线评估"""
    
    print("=" * 60)
    print("Cellpose Baseline Evaluation")
    print("=" * 60)
    
    # 加载数据
    images, binary_masks, instance_masks = load_data(Config.data_dir, split="test")
    
    print(f"Loaded {len(images)} test samples")
    
    # 初始化模型
    baseline = CellposeBaseline(
        model_type=Config.cellpose_model_type,
        device=Config.device
    )
    
    # 评估
    print("\nRunning Cellpose inference...")
    metrics = {
        "pixel_iou": [],
        "instance_iou": [],
        "aji": [],
        "n_pred": [],
        "n_gt": []
    }
    
    for i in tqdm(range(min(100, len(images))), desc="Processing"):
        image = images[i]
        gt_mask = instance_masks[i]
        
        # 预测
        pred_mask = baseline.predict(image)
        
        # 计算指标
        pixel_iou = compute_pixel_iou(pred_mask, gt_mask)
        instance_iou = compute_instance_iou(pred_mask, gt_mask)
        aji = compute_aggregate_jaccard_index(pred_mask, gt_mask)
        
        metrics["pixel_iou"].append(pixel_iou)
        metrics["instance_iou"].append(instance_iou)
        metrics["aji"].append(aji)
        metrics["n_pred"].append(len(np.unique(pred_mask)) - 1)
        metrics["n_gt"].append(len(np.unique(gt_mask)) - 1)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("Cellpose Results:")
    print("=" * 60)
    print(f"Pixel IoU:      {np.mean(metrics['pixel_iou']):.4f} ± {np.std(metrics['pixel_iou']):.4f}")
    print(f"Instance IoU:  {np.mean(metrics['instance_iou']):.4f} ± {np.std(metrics['instance_iou']):.4f}")
    print(f"AJI:           {np.mean(metrics['aji']):.4f} ± {np.std(metrics['aji']):.4f}")
    print(f"Avg Pred/Cell: {np.mean(metrics['n_pred']):.1f} (GT: {np.mean(metrics['n_gt']):.1f})")
    print("=" * 60)
    
    # 保存结果
    Config.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = Config.output_dir / "cellpose_results.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "pixel_iou": float(np.mean(metrics["pixel_iou"])),
            "pixel_iou_std": float(np.std(metrics["pixel_iou"])),
            "instance_iou": float(np.mean(metrics["instance_iou"])),
            "instance_iou_std": float(np.std(metrics["instance_iou"])),
            "aji": float(np.mean(metrics["aji"])),
            "aji_std": float(np.std(metrics["aji"])),
            "model_type": Config.cellpose_model_type
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return metrics


def compare_with_stardist(cellpose_metrics: dict, stardist_iou: float = 0.88):
    """与 StarDist 结果对比"""
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH STARSHAPE-ADAPTIVE-STARSDIST:")
    print("=" * 60)
    print(f"Cellpose Instance IoU:     {cellpose_metrics['instance_iou']:.4f}")
    print(f"StarDist Instance IoU:   {stardist_iou:.4f}")
    print("=" * 60)
    
    diff = cellpose_metrics['instance_iou'] - stardist_iou
    if diff < 0:
        print(f"\nStarDist beats Cellpose by: {-diff:.4f} IoU")
    else:
        print(f"\nCellpose beats StarDist by: {diff:.4f} IoU")


if __name__ == "__main__":
    metrics = evaluate_cellpose()
    compare_with_stardist(metrics)

