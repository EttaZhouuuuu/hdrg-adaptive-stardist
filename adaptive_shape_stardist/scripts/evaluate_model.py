"""
Model Evaluation Script: Adaptive Shape StarDist
==============================================

Comprehensive evaluation of cell segmentation performance using multiple metrics.

Metrics:
--------
1. **AP (Average Precision)**: Detection precision at different IoU thresholds
2. **mIoU (mean Intersection over Union)**: Average segmentation overlap
3. **Dice Coefficient**: Spatial overlap between prediction and ground truth
4. **Counting Accuracy**: Cell counting precision
5. **Panoptic Quality**: Combined metric for panoptic segmentation
6. **F1 Score**: Harmonic mean of precision and recall

Usage:
------
    python scripts/evaluate_model.py --model_path models/xenium_he_model --data_dir data
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.models.adaptive_shape_model import AdaptiveShapeConfig
from adaptive_shape_stardist.utils.xenium_preprocessing import XeniumDataLoader
from scipy import ndimage
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops
from skimage.draw import polygon
import matplotlib.pyplot as plt
from collections import defaultdict


class CellSegmentationEvaluator:
    """
    Comprehensive evaluator for cell segmentation models.
    """
    
    def __init__(self, iou_thresholds=[0.5, 0.75, 0.9]):
        """
        Initialize evaluator.
        
        Args:
            iou_thresholds: List of IoU thresholds for AP calculation
        """
        self.iou_thresholds = iou_thresholds
        self.results = {}
    
    def compute_iou(self, pred_mask, true_mask):
        """
        Compute Intersection over Union between prediction and ground truth.
        
        Args:
            pred_mask: Predicted binary mask [H, W]
            true_mask: Ground truth binary mask [H, W]
            
        Returns:
            float: IoU score
        """
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union == 0:
            return 1.0 if true_mask.sum() == 0 else 0.0
        
        return intersection / union
    
    def compute_dice(self, pred_mask, true_mask):
        """
        Compute Dice coefficient.
        
        Args:
            pred_mask: Predicted binary mask [H, W]
            true_mask: Ground truth binary mask [H, W]
            
        Returns:
            float: Dice score
        """
        intersection = np.logical_and(pred_mask, true_mask).sum()
        total = pred_mask.sum() + true_mask.sum()
        
        if total == 0:
            return 1.0
        
        return 2 * intersection / total
    
    def compute_panoptic_quality(self, pred_labels, true_labels):
        """
        Compute Panoptic Quality.
        
        PQ = Σ(TP) / (Σ(TP) + 0.5Σ(FP) + 0.5Σ(FN))
        
        Args:
            pred_labels: Predicted instance labels [H, W]
            true_labels: Ground truth instance labels [H, W]
            
        Returns:
            dict: Panoptic Quality metrics
        """
        true_props = regionprops(true_labels)
        pred_props = regionprops(pred_labels)
        
        true_centers = {i: (p.centroid[1], p.centroid[0]) for i, p in enumerate(true_props, 1)}
        pred_centers = {i: (p.centroid[1], p.centroid[0]) for i, p in enumerate(pred_props, 1)}
        
        # Match predictions to ground truth based on center distance
        matched = set()
        true_to_pred = {}
        pred_to_true = {}
        
        for t_idx, t_center in true_centers.items():
            min_dist = float('inf')
            best_match = None
            for p_idx, p_center in pred_centers.items():
                if p_idx in matched:
                    continue
                dist = np.sqrt((t_center[0] - p_center[0])**2 + (t_center[1] - p_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = p_idx
            if best_match is not None and min_dist < 50:  # 50 pixel distance threshold
                matched.add(best_match)
                true_to_pred[t_idx] = best_match
                pred_to_true[best_match] = t_idx
        
        # Calculate PQ components
        true_matched = set(true_to_pred.keys())
        pred_matched = set(pred_to_true.keys())
        
        tp = len(true_matched)
        fp = len(pred_centers) - len(pred_matched)
        fn = len(true_centers) - len(true_matched)
        
        # Calculate IoU for matched pairs
        ious = []
        for t_idx, p_idx in true_to_pred.items():
            t_mask = (true_labels == t_idx)
            p_mask = (pred_labels == p_idx)
            ious.append(self.compute_iou(p_mask, t_mask))
        
        pq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0
        sq = np.mean(ious) if ious else 0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0
        
        return {
            'pq': pq,
            'sq': sq,
            'rq': rq,
            'true_count': len(true_centers),
            'pred_count': len(pred_centers),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def compute_ap(self, pred_prob, true_labels, iou_thresholds=None):
        """
        Compute Average Precision at different IoU thresholds.
        
        Args:
            pred_prob: Predicted probability map [H, W]
            true_labels: Ground truth instance labels [H, W]
            iou_thresholds: IoU thresholds to evaluate
            
        Returns:
            dict: AP at each threshold
        """
        if iou_thresholds is None:
            iou_thresholds = self.iou_thresholds
        
        # Get ground truth instances
        true_props = regionprops(true_labels)
        true_masks = [(i, (true_labels == i)) for i in range(1, len(true_props) + 1)]
        
        # Generate detections from probability map
        thresholds = np.arange(0.1, 1.0, 0.1)
        ap_scores = {}
        
        for iou_thresh in iou_thresholds:
            precisions = []
            recalls = []
            
            for prob_thresh in thresholds:
                # Threshold probability map
                pred_binary = pred_prob > prob_thresh
                pred_labeled, num_pred = label(pred_binary, return_num=True)
                
                # Calculate TP, FP, FN at this IoU threshold
                tp, fp, fn = 0, 0, 0
                
                matched_pred = set()
                for i, (t_idx, t_mask) in enumerate(true_masks):
                    best_iou = 0
                    best_pred = None
                    for p_idx in range(1, num_pred + 1):
                        if p_idx in matched_pred:
                            continue
                        p_mask = (pred_labeled == p_idx)
                        iou = self.compute_iou(p_mask, t_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = p_idx
                    
                    if best_iou >= iou_thresh:
                        tp += 1
                        matched_pred.add(best_pred)
                    else:
                        fn += 1
                
                fp = num_pred - len(matched_pred)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate AP using precision-recall curve
            ap = self._calculate_ap(precisions, recalls)
            ap_scores[f'AP@{iou_thresh:.2f}'] = ap
        
        return ap_scores
    
    def _calculate_ap(self, precisions, recalls):
        """Calculate Average Precision from precision-recall curve."""
        # Add sentinel values
        recalls = [0] + recalls + [1]
        precisions = [0] + precisions + [0]
        
        # Compute the precision envelope
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate AP as area under curve
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
        
        return ap
    
    def compute_counting_accuracy(self, pred_labels, true_labels):
        """
        Compute cell counting accuracy metrics.
        
        Args:
            pred_labels: Predicted instance labels [H, W]
            true_labels: Ground truth instance labels [H, W]
            
        Returns:
            dict: Counting metrics
        """
        pred_count = len(regionprops(pred_labels))
        true_count = len(regionprops(true_labels))
        
        abs_error = abs(pred_count - true_count)
        rel_error = abs_error / true_count if true_count > 0 else 0
        
        return {
            'true_count': true_count,
            'pred_count': pred_count,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'accuracy': 1 - rel_error
        }
    
    def compute_boundary_f1(self, pred_mask, true_mask, distances=[1, 2, 5]):
        """
        Compute boundary F1 score at multiple distances.
        
        Args:
            pred_mask: Predicted binary mask [H, W]
            true_mask: Ground truth binary mask [H, W]
            distances: Distances to evaluate
            
        Returns:
            dict: Boundary F1 scores at each distance
        """
        from skimage.morphology import binary_dilation, disk
        
        boundary_f1 = {}
        
        for dist in distances:
            # Create boundaries
            true_boundary = binary_dilation(true_mask, disk(dist)) & ~true_mask
            pred_boundary = binary_dilation(pred_mask, disk(dist)) & ~pred_mask
            
            # Calculate precision and recall
            intersection = np.logical_and(pred_boundary, true_boundary).sum()
            
            precision = intersection / pred_boundary.sum() if pred_boundary.sum() > 0 else 0
            recall = intersection / true_boundary.sum() if true_boundary.sum() > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            boundary_f1[f'BF1@{dist}px'] = f1
        
        return boundary_f1
    
    def evaluate_image(self, pred_output, true_labels):
        """
        Evaluate a single image.
        
        Args:
            pred_output: Model prediction dictionary
            true_labels: Ground truth instance labels [H, W]
            
        Returns:
            dict: Metrics for this image
        """
        metrics = {}
        
        # Extract predictions
        pred_prob = pred_output['prob'][0, :, :, 0].numpy() if 'prob' in pred_output else None
        pred_dist = pred_output['dist'][0].numpy() if 'dist' in pred_output else None
        sampling_points = pred_output['sampling_points'][0].numpy() if 'sampling_points' in pred_output else None
        
        # Get predicted probability map
        if pred_prob is None and 'multiscale_predictions' in pred_output:
            # Use finest scale for single-scale evaluation
            pred_prob = pred_output['multiscale_predictions']['p2']['prob'][0, :, :, 0].numpy()
        
        # Get true labels shape
        true_height, true_width = true_labels.shape
        
        # Resize prediction to match ground truth size
        pred_prob_resized = tf.image.resize(
            pred_prob[np.newaxis, :, :, np.newaxis],
            [true_height, true_width],
            method='bilinear'
        )[0, :, 0].numpy()
        
        # Create binary mask from probability
        pred_mask = (pred_prob_resized > 0.5).astype(np.int32)
        
        # Compute instance-level metrics
        pred_labeled, num_pred = label(pred_mask, return_num=True)
        
        # IoU
        metrics['IoU'] = self.compute_iou(pred_mask.astype(bool), true_labels.astype(bool) > 0)
        
        # Dice
        metrics['Dice'] = self.compute_dice(pred_mask.astype(bool), true_labels.astype(bool) > 0)
        
        # Counting accuracy
        counting_metrics = self.compute_counting_accuracy(pred_labeled, true_labels)
        metrics.update({f'Counting/{k}': v for k, v in counting_metrics.items()})
        
        # Panoptic Quality
        pq_metrics = self.compute_panoptic_quality(pred_labeled, true_labels)
        metrics.update({f'PQ/{k}': v for k, v in pq_metrics.items()})
        
        # Boundary F1
        boundary_metrics = self.compute_boundary_f1(pred_mask.astype(bool), true_labels.astype(bool) > 0)
        metrics.update(boundary_metrics)
        
        # AP scores
        ap_scores = self.compute_ap(pred_prob_resized, true_labels)
        metrics.update(ap_scores)
        
        return metrics
    
    def evaluate_dataset(self, model, X_test, Y_test, batch_size=8):
        """
        Evaluate the model on a dataset.
        
        Args:
            model: Trained AdaptiveShapeStarDist model
            X_test: Test images [N, H, W, C]
            Y_test: Test labels [N, H, W] (instance labels)
            batch_size: Batch size for inference
            
        Returns:
            dict: Aggregated metrics
        """
        print("=" * 60)
        print("EVALUATING MODEL PERFORMANCE")
        print("=" * 60)
        
        all_metrics = []
        
        for i in range(len(X_test)):
            print(f"  Evaluating image {i + 1}/{len(X_test)}...")
            
            # Get single image
            X_batch = X_test[i:i + 1]
            true_labels = Y_test[i]
            
            # Run inference
            pred_output = model(X_batch, training=False)
            
            # Evaluate
            metrics = self.evaluate_image(pred_output, true_labels)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        
        aggregated = {}
        
        # Aggregate each metric
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}/mean'] = np.mean(values)
            aggregated[f'{key}/std'] = np.std(values)
            aggregated[f'{key}/min'] = np.min(values)
            aggregated[f'{key}/max'] = np.max(values)
        
        return aggregated
    
    def print_results(self, results):
        """Print evaluation results in a formatted table."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        # Group metrics by category
        categories = {
            'Overall': ['IoU', 'Dice'],
            'Counting': [k for k in results.keys() if k.startswith('Counting/')],
            'Panoptic Quality': [k for k in results.keys() if k.startswith('PQ/')],
            'Boundary': [k for k in results.keys() if k.startswith('BF1')],
            'Detection AP': [k for k in results.keys() if k.startswith('AP@')],
        }
        
        for category, metric_names in categories.items():
            if not metric_names:
                continue
            
            print(f"\n{category}:")
            print("-" * 40)
            
            for metric in metric_names:
                mean_key = f'{metric}/mean'
                std_key = f'{metric}/std'
                if mean_key in results:
                    print(f"  {metric:20s}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")
        
        # Print key metrics summary
        print("\n" + "=" * 60)
        print("KEY METRICS SUMMARY")
        print("=" * 60)
        
        key_metrics = [
            ('Dice', 'Segmentation Quality'),
            ('IoU', 'Segmentation Overlap'),
            ('Counting/accuracy', 'Counting Accuracy'),
            ('PQ/pq', 'Panoptic Quality'),
            ('AP@0.50', 'Detection AP@0.50'),
            ('AP@0.75', 'Detection AP@0.75'),
        ]
        
        for metric_key, description in key_metrics:
            if metric_key in results:
                print(f"  {description:25s}: {results[metric_key]:.4f}")
    
    def create_report(self, results, save_path='evaluation_report.txt'):
        """Save evaluation report to file."""
        with open(save_path, 'w') as f:
            f.write("Adaptive Shape StarDist Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
            
            print(f"\nReport saved to: {save_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Adaptive Shape StarDist model')
    parser.add_argument('--model_path', type=str, default='models/xenium_he_model',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size for evaluation')
    parser.add_argument('--num_patches', type=int, default=50,
                        help='Number of patches to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Load model
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    
    model = AdaptiveShapeStarDist.load_model(args.model_path)
    config = model.config
    
    print(f"  Model loaded from: {args.model_path}")
    print(f"  Backbone: {config.backbone}")
    print(f"  Use FPN: {config.use_fpn}")
    print(f"  Multi-scale: {config.multiscale_prediction}")
    
    # Load evaluation data
    print("\n" + "=" * 60)
    print("LOADING EVALUATION DATA")
    print("=" * 60)
    
    cells_zarr_path = os.path.join(args.data_dir, 'cells.zarr')
    image_path = os.path.join(args.data_dir, 'morphology_focus.ome.tif')
    
    # Create evaluation dataset
    print("\n" + "=" * 60)
    print("LOADING EVALUATION DATA")
    print("=" * 60)
    
    cells_zarr_path = os.path.join(args.data_dir, 'cells.zarr')
    image_path = os.path.join(args.data_dir, 'morphology_focus.ome.tif')
    
    # Load data using the same function as training
    loader = XeniumDataLoader(
        cells_zarr_path=cells_zarr_path,
        image_path=image_path,
        load_image=True
    )
    
    # Extract patches for evaluation
    images, labels = loader.extract_patches(
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        plane=0,
        seed=42,
        use_precomputed=True
    )
    
    # Use all patches for evaluation (no train/val split)
    X_eval = images
    Y_eval = labels
    
    print(f"  Evaluation samples: {len(X_eval)}")
    print(f"  Image shape: {X_eval.shape[1:]}")

    # Normalize images
    X_eval = X_eval.astype(np.float32)
    for i in range(len(X_eval)):
        img = X_eval[i]
        img_mean = img.mean()
        img_std = img.std()
        if img_std > 0:
            X_eval[i] = (img - img_mean) / (img_std + 1e-7)
    
    # Evaluate
    evaluator = CellSegmentationEvaluator()
    results = evaluator.evaluate_dataset(model, X_eval, Y_eval, batch_size=args.batch_size)
    
    # Print results
    evaluator.print_results(results)
    
    # Save report
    evaluator.create_report(results, save_path=f'{args.model_path}/evaluation_report.txt')
    
    # Save detailed metrics
    np.save(f'{args.model_path}/evaluation_metrics.npy', results)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()

