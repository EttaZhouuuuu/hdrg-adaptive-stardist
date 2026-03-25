"""
Fallback: Watershed-based Instance Segmentation
=============================================
如果 Cellpose-SAM 不可用，使用基于 watershed 的方法进行实例分割
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scipy import ndimage
from skimage import morphology, measure, segmentation
from skimage.feature import peak_local_max
from tqdm import tqdm
import json
try:
    from tifffile import imsave as io_imsave
    def io_imsave_wrapper(path, data):
        io_imsave(path, data)
except ImportError:
    try:
        from imageio import imwrite as io_imsave_wrapper
    except ImportError:
        io_imsave_wrapper = None


class Config:
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "baselines" / "cellpose_results"
    
    # Watershed 参数
    diameter = 30.0  # 预期细胞直径
    threshold = 0.5  # 二值化阈值


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
    """计算 AJI (Aggregate Jaccard Index)"""
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


def watershed_segmentation(
    image: np.ndarray, 
    diameter: float = 30.0,
    threshold: float = 0.5
) -> np.ndarray:
    """
    基于 watershed 的细胞实例分割
    
    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        diameter: 预期细胞直径
        threshold: 二值化阈值
    
    Returns:
        分割掩码 (H, W)
    """
    # 转换为灰度图
    if image.ndim == 3:
        gray = image.mean(axis=-1)
    else:
        gray = image.copy()
    
    # 归一化
    gray = gray.astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    
    # 高斯滤波去噪
    gray = ndimage.gaussian_filter(gray, sigma=1.0)
    
    # 二值化
    binary = (gray > threshold).astype(np.uint8)
    
    # 形态学开操作去除小噪声
    kernel = morphology.disk(int(diameter / 8))
    binary = morphology.opening(binary, kernel)
    
    # 距离变换
    distance = ndimage.distance_transform_edt(binary)
    
    # 找到局部极大值点（细胞中心）
    min_distance = max(1, int(diameter / 4))
    local_max = peak_local_max(
        distance, 
        min_distance=min_distance,
        labels=binary
    )
    
    if len(local_max) == 0:
        # 如果没有检测到细胞中心，返回空掩码
        return np.zeros_like(gray, dtype=np.uint32)
    
    # 创建标记
    markers = np.zeros_like(gray, dtype=np.int32)
    for i, (y, x) in enumerate(local_max, start=1):
        markers[y, x] = i
    
    markers = ndimage.label(markers)[0]
    
    # Watershed 分割
    labels = segmentation.watershed(
        -distance, 
        markers, 
        mask=binary,
        compactness=0.001
    )
    
    return labels


def load_data(data_dir: Path, split: str = "test", max_samples: int = 50):
    """从 Xenium 数据加载测试数据"""
    try:
        from utils.xenium_preprocessing import XeniumDataLoader
        
        print(f"Loading {split} data from {data_dir}...")
        loader = XeniumDataLoader(
            cells_zarr_path=str(data_dir / "cells.zarr"),
            image_path=str(data_dir / "morphology_focus.ome.tif"),
            load_image=True
        )
        
        # 提取测试 patch
        images, instance_masks = loader.extract_patches(
            patch_size=256,
            num_patches=min(max_samples, 500),
            use_precomputed=True
        )
        
        print(f"Loaded {len(images)} {split} samples")
        return images, instance_masks
    
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure the project is set up correctly.")
        raise


def evaluate_watershed():
    """运行 Watershed 基线评估"""
    print("=" * 60)
    print("Watershed Baseline Evaluation (Fallback)")
    print("=" * 60)
    
    # 加载数据
    images, gt_masks = load_data(Config.data_dir, split="test", max_samples=50)
    
    # 创建输出目录
    output_dir = Config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    # 处理图像
    print("\nRunning Watershed segmentation...")
    metrics = {
        "pixel_iou": [],
        "instance_iou": [],
        "aji": [],
        "n_pred": [],
        "n_gt": []
    }
    
    for i in tqdm(range(len(images)), desc="Processing"):
        img = images[i]
        gt_mask = gt_masks[i]
        
        # 运行 watershed 分割
        pred_mask = watershed_segmentation(
            img,
            diameter=Config.diameter,
            threshold=Config.threshold
        )
        
        # 保存预测掩码
        mask_path = masks_dir / f"mask_{i:04d}.tif"
        io_imsave_wrapper(str(mask_path), pred_mask.astype(np.uint32))
        
        # 计算指标
        pixel_iou = compute_pixel_iou(pred_mask, gt_mask)
        instance_iou = compute_instance_iou(pred_mask, gt_mask)
        aji = compute_aji(pred_mask, gt_mask)
        
        metrics["pixel_iou"].append(pixel_iou)
        metrics["instance_iou"].append(instance_iou)
        metrics["aji"].append(aji)
        metrics["n_pred"].append(len(np.unique(pred_mask)) - 1)
        metrics["n_gt"].append(len(np.unique(gt_mask)) - 1)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Watershed Results:")
    print("=" * 60)
    print(f"Pixel IoU:      {np.mean(metrics['pixel_iou']):.4f} +/- {np.std(metrics['pixel_iou']):.4f}")
    print(f"Instance IoU:  {np.mean(metrics['instance_iou']):.4f} +/- {np.std(metrics['instance_iou']):.4f}")
    print(f"AJI:           {np.mean(metrics['aji']):.4f} +/- {np.std(metrics['aji']):.4f}")
    print(f"Avg Pred/Cell: {np.mean(metrics['n_pred']):.1f} (GT: {np.mean(metrics['n_gt']):.1f})")
    
    # 保存指标到 JSON
    results = {
        "pixel_iou": float(np.mean(metrics["pixel_iou"])),
        "pixel_iou_std": float(np.std(metrics["pixel_iou"])),
        "instance_iou": float(np.mean(metrics["instance_iou"])),
        "instance_iou_std": float(np.std(metrics["instance_iou"])),
        "aji": float(np.mean(metrics["aji"])),
        "aji_std": float(np.std(metrics["aji"])),
        "method": "watershed",
        "diameter": Config.diameter,
        "threshold": Config.threshold,
        "n_samples": len(images)
    }
    
    with open(output_dir / "watershed_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'watershed_metrics.json'}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    evaluate_watershed()
