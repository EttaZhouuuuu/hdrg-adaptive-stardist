import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
import cv2

class SegmentationMetrics:
    """
    Class for computing segmentation metrics for StarDist predictions.
    
    Implements various metrics including:
    - IoU (Intersection over Union)
    - Dice coefficient
    - Average Precision
    - Shape accuracy metrics
    """
    
    def __init__(self, n_rays: int = 32):
        self.n_rays = n_rays
    
    @staticmethod
    def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between prediction and ground truth masks.
        
        Args:
            pred_mask: Binary prediction mask
            true_mask: Binary ground truth mask
            
        Returns:
            IoU score
        """
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        union = torch.logical_or(pred_mask, true_mask).sum()
        return intersection.float() / (union + 1e-8)
    
    @staticmethod
    def compute_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice coefficient between prediction and ground truth masks.
        
        Args:
            pred_mask: Binary prediction mask
            true_mask: Binary ground truth mask
            
        Returns:
            Dice coefficient
        """
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        return 2 * intersection.float() / (pred_mask.sum() + true_mask.sum() + 1e-8)
    
    def compute_shape_metrics(
        self,
        pred_distances: torch.Tensor,
        true_distances: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute shape-specific metrics.
        
        Args:
            pred_distances: Predicted distance map (n_rays, H, W)
            true_distances: Ground truth distance map (n_rays, H, W)
            mask: Binary mask indicating valid regions
            
        Returns:
            Dictionary of shape metrics
        """
        # Convert to numpy for computation
        pred_dist = pred_distances.cpu().numpy()
        true_dist = true_distances.cpu().numpy()
        mask = mask.cpu().numpy()
        
        # Compute relative distance error
        rel_error = np.abs(pred_dist - true_dist) / (true_dist + 1e-8)
        rel_error = rel_error * mask[None, :, :]  # Apply mask
        mean_rel_error = np.sum(rel_error) / (np.sum(mask) * self.n_rays + 1e-8)
        
        # Compute angular consistency
        pred_angles = np.arctan2(np.diff(pred_dist, axis=0), 1)
        true_angles = np.arctan2(np.diff(true_dist, axis=0), 1)
        angle_diff = np.abs(pred_angles - true_angles)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        mean_angle_error = np.sum(angle_diff * mask[None, :, :]) / (np.sum(mask) * (self.n_rays-1) + 1e-8)
        
        return {
            'relative_distance_error': float(mean_rel_error),
            'angular_error': float(mean_angle_error)
        }
    
    def compute_instance_metrics(
        self,
        pred_instances: torch.Tensor,
        true_instances: torch.Tensor,
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute instance-level metrics.
        
        Args:
            pred_instances: Predicted instance segmentation
            true_instances: Ground truth instance segmentation
            iou_threshold: IoU threshold for matching instances
            
        Returns:
            Dictionary of instance metrics
        """
        pred_instances = pred_instances.cpu().numpy()
        true_instances = true_instances.cpu().numpy()
        
        # Get unique instances (excluding background)
        pred_ids = np.unique(pred_instances)[1:]
        true_ids = np.unique(true_instances)[1:]
        
        if len(true_ids) == 0:
            if len(pred_ids) == 0:
                return {
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1_score': 1.0,
                    'mean_iou': 1.0
                }
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mean_iou': 0.0
            }
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(true_ids), len(pred_ids)))
        for i, true_id in enumerate(true_ids):
            true_mask = (true_instances == true_id)
            for j, pred_id in enumerate(pred_ids):
                pred_mask = (pred_instances == pred_id)
                iou_matrix[i, j] = self.compute_iou(
                    torch.from_numpy(pred_mask),
                    torch.from_numpy(true_mask)
                ).item()
        
        # Match instances using Hungarian algorithm
        true_idx, pred_idx = linear_sum_assignment(-iou_matrix)
        
        # Compute metrics
        matches = iou_matrix[true_idx, pred_idx] > iou_threshold
        num_matches = np.sum(matches)
        
        precision = num_matches / (len(pred_ids) + 1e-8)
        recall = num_matches / (len(true_ids) + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        mean_iou = np.mean(iou_matrix[true_idx, pred_idx]) if len(pred_ids) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'mean_iou': float(mean_iou)
        }
    
    def compute_all_metrics(
        self,
        pred_distances: torch.Tensor,
        pred_probabilities: torch.Tensor,
        true_distances: torch.Tensor,
        true_probabilities: torch.Tensor,
        pred_instances: Optional[torch.Tensor] = None,
        true_instances: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            pred_distances: Predicted distance map
            pred_probabilities: Predicted probability map
            true_distances: Ground truth distance map
            true_probabilities: Ground truth probability map
            pred_instances: Optional predicted instance segmentation
            true_instances: Optional ground truth instance segmentation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic segmentation metrics
        metrics['iou'] = self.compute_iou(
            pred_probabilities > 0.5,
            true_probabilities > 0.5
        ).item()
        
        metrics['dice'] = self.compute_dice(
            pred_probabilities > 0.5,
            true_probabilities > 0.5
        ).item()
        
        # Shape metrics
        shape_metrics = self.compute_shape_metrics(
            pred_distances,
            true_distances,
            true_probabilities > 0.5
        )
        metrics.update(shape_metrics)
        
        # Instance metrics if available
        if pred_instances is not None and true_instances is not None:
            instance_metrics = self.compute_instance_metrics(
                pred_instances,
                true_instances
            )
            metrics.update(instance_metrics)
        
        return metrics

class MetricsLogger:
    """
    Class for logging and aggregating metrics during training/evaluation.
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update running metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
        }
    
    def reset(self):
        """Reset metrics."""
        self.metrics = {}
        self.counts = {}
