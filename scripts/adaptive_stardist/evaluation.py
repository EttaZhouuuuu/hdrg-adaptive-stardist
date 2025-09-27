"""
Evaluation metrics for multi-scale StarDist performance assessment.
"""

import numpy as np
from scipy import ndimage
from skimage import measure, segmentation
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import json
import logging
from .utils import compute_local_statistics, estimate_object_sizes

class MultiScaleEvaluator:
    """
    Comprehensive evaluator for multi-scale instance segmentation.
    """
    
    def __init__(self,
                 min_iou: float = 0.5,
                 size_bins: List[int] = None,
                 boundary_tolerance: int = 2):
        """
        Initialize evaluator.
        
        Args:
            min_iou: Minimum IoU for matching
            size_bins: Size ranges for stratified evaluation
            boundary_tolerance: Pixel tolerance for boundary evaluation
        """
        self.min_iou = min_iou
        self.size_bins = size_bins or [10, 50, 100, 200, 500]
        self.boundary_tolerance = boundary_tolerance
        self.logger = logging.getLogger(__name__)
        
    def compute_size_based_metrics(self,
                                 true_labels: np.ndarray,
                                 pred_labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics stratified by object size.
        
        Args:
            true_labels: Ground truth instance labels
            pred_labels: Predicted instance labels
            
        Returns:
            Dictionary of metrics per size range
        """
        # Get object sizes
        true_props = measure.regionprops(true_labels)
        pred_props = measure.regionprops(pred_labels)
        
        true_sizes = [prop.area for prop in true_props]
        pred_sizes = [prop.area for prop in pred_props]
        
        # Initialize metrics per size bin
        size_metrics = {}
        for i in range(len(self.size_bins) - 1):
            min_size, max_size = self.size_bins[i], self.size_bins[i+1]
            bin_name = f"{min_size}-{max_size}"
            
            # Filter objects by size
            true_mask = np.logical_and(
                np.array(true_sizes) >= min_size,
                np.array(true_sizes) < max_size
            )
            pred_mask = np.logical_and(
                np.array(pred_sizes) >= min_size,
                np.array(pred_sizes) < max_size
            )
            
            # Compute metrics for this size range
            metrics = self._compute_basic_metrics(
                true_labels,
                pred_labels,
                true_indices=np.array([p.label for p in true_props])[true_mask],
                pred_indices=np.array([p.label for p in pred_props])[pred_mask]
            )
            
            size_metrics[bin_name] = metrics
            
        return size_metrics
    
    def compute_boundary_metrics(self,
                               true_labels: np.ndarray,
                               pred_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute boundary-specific metrics.
        
        Args:
            true_labels: Ground truth instance labels
            pred_labels: Predicted instance labels
            
        Returns:
            Dictionary of boundary metrics
        """
        # Get boundaries
        true_boundaries = segmentation.find_boundaries(true_labels)
        pred_boundaries = segmentation.find_boundaries(pred_labels)
        
        # Dilate boundaries for tolerance
        if self.boundary_tolerance > 0:
            true_boundaries = ndimage.binary_dilation(
                true_boundaries,
                iterations=self.boundary_tolerance
            )
            pred_boundaries = ndimage.binary_dilation(
                pred_boundaries,
                iterations=self.boundary_tolerance
            )
        
        # Compute metrics
        boundary_tp = np.logical_and(true_boundaries, pred_boundaries).sum()
        boundary_fp = np.logical_and(~true_boundaries, pred_boundaries).sum()
        boundary_fn = np.logical_and(true_boundaries, ~pred_boundaries).sum()
        
        precision = boundary_tp / (boundary_tp + boundary_fp + 1e-6)
        recall = boundary_tp / (boundary_tp + boundary_fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'boundary_precision': precision,
            'boundary_recall': recall,
            'boundary_f1': f1
        }
    
    def compute_scale_consistency(self,
                                true_labels: np.ndarray,
                                pred_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate consistency of predictions across scales.
        
        Args:
            true_labels: Ground truth instance labels
            pred_labels: Predicted instance labels
            
        Returns:
            Dictionary of scale consistency metrics
        """
        # Compute size distributions
        true_sizes = [r.area for r in measure.regionprops(true_labels)]
        pred_sizes = [r.area for r in measure.regionprops(pred_labels)]
        
        # Compare size distributions
        true_hist, _ = np.histogram(np.log(true_sizes), bins=20)
        pred_hist, _ = np.histogram(np.log(pred_sizes), bins=20)
        
        # Normalize histograms
        true_hist = true_hist / true_hist.sum()
        pred_hist = pred_hist / pred_hist.sum()
        
        # Compute distribution similarity
        kl_div = np.sum(true_hist * np.log((true_hist + 1e-6) / (pred_hist + 1e-6)))
        js_div = 0.5 * np.sum(
            true_hist * np.log((true_hist + 1e-6) / (0.5 * (true_hist + pred_hist) + 1e-6)) +
            pred_hist * np.log((pred_hist + 1e-6) / (0.5 * (true_hist + pred_hist) + 1e-6))
        )
        
        return {
            'size_kl_divergence': kl_div,
            'size_js_divergence': js_div,
            'mean_size_ratio': np.mean(pred_sizes) / np.mean(true_sizes)
        }
    
    def compute_shape_metrics(self,
                            true_labels: np.ndarray,
                            pred_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate shape preservation quality.
        
        Args:
            true_labels: Ground truth instance labels
            pred_labels: Predicted instance labels
            
        Returns:
            Dictionary of shape metrics
        """
        true_props = measure.regionprops(true_labels)
        pred_props = measure.regionprops(pred_labels)
        
        # Match objects
        matches = self._match_objects(true_labels, pred_labels)
        
        shape_errors = []
        for true_idx, pred_idx in matches:
            true_prop = true_props[true_idx - 1]  # -1 because labels start at 1
            pred_prop = pred_props[pred_idx - 1]
            
            # Compare shape properties
            errors = {
                'eccentricity': abs(true_prop.eccentricity - pred_prop.eccentricity),
                'solidity': abs(true_prop.solidity - pred_prop.solidity),
                'extent': abs(true_prop.extent - pred_prop.extent)
            }
            shape_errors.append(errors)
        
        # Aggregate errors
        if shape_errors:
            mean_errors = {
                k: np.mean([e[k] for e in shape_errors])
                for k in shape_errors[0].keys()
            }
        else:
            mean_errors = {
                'eccentricity': 1.0,
                'solidity': 1.0,
                'extent': 1.0
            }
        
        return {
            'shape_eccentricity_error': mean_errors['eccentricity'],
            'shape_solidity_error': mean_errors['solidity'],
            'shape_extent_error': mean_errors['extent']
        }
    
    def _compute_basic_metrics(self,
                             true_labels: np.ndarray,
                             pred_labels: np.ndarray,
                             true_indices: Optional[np.ndarray] = None,
                             pred_indices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute basic segmentation metrics.
        """
        if true_indices is None:
            true_indices = np.unique(true_labels)[1:]
        if pred_indices is None:
            pred_indices = np.unique(pred_labels)[1:]
            
        # Match objects
        matches = self._match_objects(
            true_labels,
            pred_labels,
            true_indices=true_indices,
            pred_indices=pred_indices
        )
        
        # Compute metrics
        n_true = len(true_indices)
        n_pred = len(pred_indices)
        n_matches = len(matches)
        
        precision = n_matches / (n_pred + 1e-6)
        recall = n_matches / (n_true + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_true': n_true,
            'n_pred': n_pred,
            'n_matches': n_matches
        }
    
    def _match_objects(self,
                      true_labels: np.ndarray,
                      pred_labels: np.ndarray,
                      true_indices: Optional[np.ndarray] = None,
                      pred_indices: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Match predicted objects to ground truth objects.
        """
        if true_indices is None:
            true_indices = np.unique(true_labels)[1:]
        if pred_indices is None:
            pred_indices = np.unique(pred_labels)[1:]
            
        matches = []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(true_indices), len(pred_indices)))
        for i, true_idx in enumerate(true_indices):
            true_mask = true_labels == true_idx
            for j, pred_idx in enumerate(pred_indices):
                pred_mask = pred_labels == pred_idx
                intersection = np.logical_and(true_mask, pred_mask).sum()
                union = np.logical_or(true_mask, pred_mask).sum()
                iou_matrix[i, j] = intersection / (union + 1e-6)
        
        # Match objects greedily
        while True:
            if iou_matrix.size == 0 or iou_matrix.max() < self.min_iou:
                break
                
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matches.append((true_indices[i], pred_indices[j]))
            
            # Remove matched objects
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
            
        return matches
    
    def evaluate(self,
                true_labels: np.ndarray,
                pred_labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            true_labels: Ground truth instance labels
            pred_labels: Predicted instance labels
            
        Returns:
            Dictionary of all metrics
        """
        # Basic metrics
        basic_metrics = self._compute_basic_metrics(true_labels, pred_labels)
        
        # Size-based metrics
        size_metrics = self.compute_size_based_metrics(true_labels, pred_labels)
        
        # Boundary metrics
        boundary_metrics = self.compute_boundary_metrics(true_labels, pred_labels)
        
        # Scale consistency metrics
        scale_metrics = self.compute_scale_consistency(true_labels, pred_labels)
        
        # Shape metrics
        shape_metrics = self.compute_shape_metrics(true_labels, pred_labels)
        
        return {
            'overall': basic_metrics,
            'size_stratified': size_metrics,
            'boundary': boundary_metrics,
            'scale': scale_metrics,
            'shape': shape_metrics
        }
    
    def evaluate_dataset(self,
                        true_dir: Path,
                        pred_dir: Path,
                        output_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Evaluate predictions on a dataset.
        
        Args:
            true_dir: Directory with ground truth labels
            pred_dir: Directory with predicted labels
            output_file: Optional path to save results
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        # Find matching files
        true_files = sorted(true_dir.glob('*.tiff'))
        for true_file in true_files:
            pred_file = pred_dir / true_file.name
            if not pred_file.exists():
                self.logger.warning(f"No prediction found for {true_file.name}")
                continue
                
            # Load labels
            true_labels = np.load(true_file)
            pred_labels = np.load(pred_file)
            
            # Evaluate
            metrics = self.evaluate(true_labels, pred_labels)
            metrics['file_name'] = true_file.name
            results.append(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save if requested
        if output_file is not None:
            if output_file.suffix == '.csv':
                df.to_csv(output_file, index=False)
            else:
                df.to_json(output_file, orient='records', indent=2)
        
        return df
