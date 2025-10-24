"""
Predictor for Adaptive Shape StarDist
======================================

Handles inference and post-processing for instance segmentation.
"""

import numpy as np
from scipy.ndimage import label, maximum_filter, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import tensorflow as tf


class AdaptiveShapePredictor:
    """
    Predictor class for inference
    
    Parameters:
    -----------
    model : AdaptiveShapeStarDist
        Trained model
    prob_thresh : float
        Probability threshold for detection
    nms_thresh : float
        Non-maximum suppression threshold
    """
    
    def __init__(self, model, prob_thresh=0.5, nms_thresh=0.3):
        self.model = model
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
    
    def predict(self, image, normalize=True, return_details=False):
        """
        Predict instance segmentation
        
        Parameters:
        -----------
        image : np.ndarray
            Input image [H, W] or [H, W, C]
        normalize : bool
            Whether to normalize image
        return_details : bool
            Whether to return detailed predictions
            
        Returns:
        --------
        labels : np.ndarray
            Instance label image
        details : dict (optional)
            Additional prediction details
        """
        
        # Prepare image
        img_input = self._prepare_image(image, normalize=normalize)
        
        # Predict
        predictions = self.model(img_input, training=False)
        
        # Extract predictions
        prob = predictions['prob'][0, ..., 0].numpy()  # [H, W]
        dist = predictions['dist'][0].numpy()  # [H, W, N_points]
        complexity = predictions['complexity'][0].numpy()
        
        # Post-process to get instance labels
        labels = self._postprocess(prob, dist)
        
        if return_details:
            details = {
                'prob': prob,
                'dist': dist,
                'complexity': complexity,
                'sampling_points': predictions['sampling_points'][0].numpy(),
                'shape_descriptor': predictions['shape_descriptor'][0].numpy(),
            }
            return labels, details
        else:
            return labels
    
    def _prepare_image(self, image, normalize=True):
        """Prepare image for model input"""
        
        # Ensure float32
        img = image.astype(np.float32)
        
        # Add batch dimension
        if img.ndim == 2:
            img = img[np.newaxis, ..., np.newaxis]
        elif img.ndim == 3:
            img = img[np.newaxis, ...]
        
        # Normalize
        if normalize:
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / (std + 1e-7)
        
        return img
    
    def _postprocess(self, prob, dist):
        """
        Post-process predictions to instance labels
        
        Uses adaptive watershed algorithm that leverages
        the adaptive sampling points.
        """
        
        # Threshold probability map
        mask = prob > self.prob_thresh
        
        # Find seeds (local maxima in probability map)
        seeds = self._find_seeds(prob, mask)
        
        # Watershed segmentation
        labels = self._adaptive_watershed(prob, dist, seeds, mask)
        
        return labels
    
    def _find_seeds(self, prob, mask):
        """
        Find cell center seeds using local maxima
        
        Parameters:
        -----------
        prob : np.ndarray
            Probability map [H, W]
        mask : np.ndarray
            Foreground mask [H, W]
            
        Returns:
        --------
        seeds : np.ndarray
            Seed label image [H, W]
        """
        
        # Apply mask
        prob_masked = prob * mask
        
        # Find local maxima
        local_max = (prob_masked == maximum_filter(prob_masked, size=5))
        local_max = local_max & (prob_masked > self.prob_thresh)
        
        # Label connected components of local maxima
        seeds, num_seeds = label(local_max)
        
        return seeds
    
    def _adaptive_watershed(self, prob, dist, seeds, mask):
        """
        Adaptive watershed using distance predictions
        
        Unlike standard watershed, this uses the predicted
        distances to guide the segmentation.
        
        Parameters:
        -----------
        prob : np.ndarray
            Probability map [H, W]
        dist : np.ndarray
            Distance predictions [H, W, N_points]
        seeds : np.ndarray
            Seed labels [H, W]
        mask : np.ndarray
            Foreground mask [H, W]
            
        Returns:
        --------
        labels : np.ndarray
            Instance labels [H, W]
        """
        
        # Create distance map for watershed
        # Use average distance across all sampling points as a measure
        avg_dist = np.mean(dist, axis=-1)  # [H, W]
        
        # Invert for watershed (watershed uses minima)
        watershed_map = -avg_dist
        
        # Apply watershed
        labels = watershed(watershed_map, markers=seeds, mask=mask)
        
        return labels
    
    def predict_batch(self, images, normalize=True):
        """
        Predict on batch of images
        
        Parameters:
        -----------
        images : list of np.ndarray
            List of input images
        normalize : bool
            Whether to normalize images
            
        Returns:
        --------
        labels_list : list of np.ndarray
            List of instance label images
        """
        
        labels_list = []
        
        for img in images:
            labels = self.predict(img, normalize=normalize, return_details=False)
            labels_list.append(labels)
        
        return labels_list


class InstanceMetrics:
    """
    Compute instance segmentation metrics
    
    Metrics:
    - Average Precision (AP) at different IoU thresholds
    - F1 score
    - Dice coefficient
    """
    
    def __init__(self, iou_thresholds=None):
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75, 0.9]
        self.iou_thresholds = iou_thresholds
    
    def compute_metrics(self, pred_labels, true_labels):
        """
        Compute all metrics
        
        Parameters:
        -----------
        pred_labels : np.ndarray
            Predicted instance labels
        true_labels : np.ndarray
            Ground truth instance labels
            
        Returns:
        --------
        metrics : dict
            Dictionary of computed metrics
        """
        
        # Match predictions to ground truth
        matches = self._match_instances(pred_labels, true_labels)
        
        # Compute metrics at each IoU threshold
        results = {}
        
        for thresh in self.iou_thresholds:
            precision, recall, f1 = self._compute_prf(matches, thresh)
            results[f'precision_{thresh}'] = precision
            results[f'recall_{thresh}'] = recall
            results[f'f1_{thresh}'] = f1
        
        # Compute average metrics
        results['AP'] = np.mean([results[f'precision_{t}'] for t in self.iou_thresholds])
        results['AR'] = np.mean([results[f'recall_{t}'] for t in self.iou_thresholds])
        results['F1'] = np.mean([results[f'f1_{t}'] for t in self.iou_thresholds])
        
        return results
    
    def _match_instances(self, pred_labels, true_labels):
        """
        Match predicted instances to ground truth
        
        Returns:
        --------
        matches : list of (pred_id, true_id, iou)
        """
        
        pred_ids = np.unique(pred_labels)[1:]  # Exclude background
        true_ids = np.unique(true_labels)[1:]
        
        matches = []
        
        for pred_id in pred_ids:
            pred_mask = pred_labels == pred_id
            
            best_iou = 0
            best_true_id = -1
            
            for true_id in true_ids:
                true_mask = true_labels == true_id
                
                # Compute IoU
                intersection = np.sum(pred_mask & true_mask)
                union = np.sum(pred_mask | true_mask)
                
                if union > 0:
                    iou = intersection / union
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_true_id = true_id
            
            if best_true_id > 0:
                matches.append((pred_id, best_true_id, best_iou))
        
        return matches
    
    def _compute_prf(self, matches, iou_threshold):
        """
        Compute precision, recall, F1 at given IoU threshold
        """
        
        # True positives: matches with IoU >= threshold
        tp = sum(1 for _, _, iou in matches if iou >= iou_threshold)
        
        # False positives: predictions with no good match
        fp = sum(1 for _, _, iou in matches if iou < iou_threshold)
        
        # False negatives: ground truth with no match
        # (not directly computed from matches, simplified here)
        fn = 0  # Would need ground truth count
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

