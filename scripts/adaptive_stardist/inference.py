"""
Inference pipeline with dynamic parameter adjustment for Multi-Scale StarDist.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from skimage import io, transform
from csbdeep.utils import normalize_mi_ma
from .utils import compute_local_statistics, estimate_object_sizes
from .param_optimizer import ParamOptimizer

class AdaptiveInferencePipeline:
    """
    Advanced inference pipeline with dynamic parameter adjustment.
    """
    
    def __init__(self,
                 model,
                 tile_size: int = 2048,
                 overlap: float = 0.2,
                 batch_size: int = 4,
                 normalize_percentiles: Tuple[float, float] = (1, 99.8),
                 prob_thresh: float = 0.5,
                 nms_thresh: float = 0.4,
                 min_object_size: int = 10,
                 scale_sampling: str = 'adaptive'):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained StarDist model
            tile_size: Size of tiles for processing large images
            overlap: Overlap between tiles (0-1)
            batch_size: Batch size for inference
            normalize_percentiles: Percentiles for image normalization
            prob_thresh: Probability threshold for object detection
            nms_thresh: Non-maximum suppression threshold
            min_object_size: Minimum object size in pixels
            scale_sampling: How to sample scales ('fixed', 'adaptive', or 'dynamic')
        """
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.normalize_percentiles = normalize_percentiles
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.min_object_size = min_object_size
        self.scale_sampling = scale_sampling
        
        # Initialize parameter optimizer
        self.param_optimizer = ParamOptimizer(
            min_n_rays=model.config.min_n_rays,
            max_n_rays=model.config.max_n_rays,
            min_grid=model.config.min_grid,
            max_grid=model.config.max_grid
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using percentile-based normalization.
        """
        p_low, p_high = np.percentile(img, self.normalize_percentiles)
        return normalize_mi_ma(img, p_low, p_high)
    
    def _compute_tile_positions(self,
                              image_shape: Tuple[int, int],
                              tile_size: int,
                              overlap: float) -> List[Tuple[slice, slice]]:
        """
        Compute optimal tile positions with overlap.
        """
        step = int(tile_size * (1 - overlap))
        
        tile_positions = []
        for y in range(0, image_shape[0], step):
            for x in range(0, image_shape[1], step):
                # Adjust tile boundaries
                y_end = min(y + tile_size, image_shape[0])
                x_end = min(x + tile_size, image_shape[1])
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                
                tile_positions.append((
                    slice(y_start, y_end),
                    slice(x_start, x_end)
                ))
        
        return tile_positions
    
    def _determine_scales(self,
                         image: np.ndarray,
                         tile_stats: Optional[Dict] = None) -> List[float]:
        """
        Determine appropriate scales for processing based on image characteristics.
        """
        if self.scale_sampling == 'fixed':
            return [1.0, 0.5, 0.25]
            
        # Compute image statistics if not provided
        if tile_stats is None:
            tile_stats = compute_local_statistics(image)
            
        if self.scale_sampling == 'adaptive':
            # Use scale distribution to determine appropriate scales
            scale_dist = tile_stats['gradient']
            scale_peaks = np.percentile(scale_dist[scale_dist > 0],
                                      [25, 50, 75])
            
            # Convert scale peaks to relative scales
            base_scale = np.median(scale_peaks)
            scales = [s/base_scale for s in scale_peaks]
            scales = [s for s in scales if 0.1 <= s <= 2.0]
            
        elif self.scale_sampling == 'dynamic':
            # Dynamically determine number and values of scales
            scale_var = np.var(tile_stats['gradient'])
            if scale_var < 0.1:  # Low variance - use fewer scales
                scales = [1.0, 0.5]
            elif scale_var < 0.3:  # Medium variance
                scales = [1.0, 0.5, 0.25]
            else:  # High variance - use more scales
                scales = [1.0, 0.75, 0.5, 0.25]
                
        return sorted(scales, reverse=True)
    
    def _process_tile(self,
                     tile: np.ndarray,
                     tile_stats: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Process a single tile with optimized parameters.
        """
        # Optimize parameters based on tile characteristics
        params = self.param_optimizer.optimize_parameters(
            image=tile,
            scale_map=tile_stats['gradient']
        )
        
        # Update model parameters
        self.model.config.n_rays = params['n_rays']
        self.model.config.grid = params['grid']
        
        # Determine scales for this tile
        scales = self._determine_scales(tile, tile_stats)
        
        # Process at multiple scales
        predictions = []
        for scale in scales:
            if scale != 1.0:
                scaled_tile = transform.rescale(
                    tile,
                    scale,
                    preserve_range=True,
                    anti_aliasing=True
                )
            else:
                scaled_tile = tile
                
            # Get predictions
            labels, details = self.model.predict_instances(scaled_tile)
            
            # Rescale back if needed
            if scale != 1.0:
                labels = transform.resize(
                    labels,
                    tile.shape,
                    order=0,
                    preserve_range=True
                ).astype(np.uint16)
                
            predictions.append((labels, details, scale))
            
        return self._merge_predictions(predictions, tile.shape)
    
    def _merge_predictions(self,
                         predictions: List[Tuple],
                         output_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
        """
        Merge predictions from multiple scales.
        """
        merged_labels = np.zeros(output_shape, dtype=np.uint16)
        merged_details = {
            'scales_used': [],
            'confidence_maps': [],
            'object_counts': []
        }
        
        current_id = 1
        
        for labels, details, scale in predictions:
            # Record scale information
            merged_details['scales_used'].append(scale)
            merged_details['confidence_maps'].append(details['prob'])
            merged_details['object_counts'].append(np.max(labels))
            
            # Add non-overlapping objects
            for label_id in range(1, np.max(labels) + 1):
                obj_mask = labels == label_id
                if obj_mask.sum() < self.min_object_size:
                    continue
                    
                # Check overlap with existing objects
                overlap = np.sum(merged_labels[obj_mask] > 0) / obj_mask.sum()
                if overlap < 0.3:  # Low overlap - add as new object
                    merged_labels[obj_mask] = current_id
                    current_id += 1
        
        return merged_labels, merged_details
    
    def _stitch_tiles(self,
                     tiles_data: List[Tuple],
                     image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
        """
        Stitch together processed tiles.
        """
        final_labels = np.zeros(image_shape, dtype=np.uint16)
        confidence_map = np.zeros(image_shape, dtype=float)
        
        current_id = 1
        global_details = {
            'tile_statistics': [],
            'object_counts': [],
            'processing_params': []
        }
        
        for (y_slice, x_slice), (labels, details) in tiles_data:
            # Record tile information
            global_details['tile_statistics'].append({
                'position': (y_slice.start, x_slice.start),
                'object_count': np.max(labels)
            })
            
            # Handle overlapping regions
            overlap_mask = final_labels[y_slice, x_slice] > 0
            if overlap_mask.any():
                # Resolve conflicts in overlap region
                conf_current = confidence_map[y_slice, x_slice]
                conf_new = details['prob']
                
                # Keep higher confidence predictions
                keep_new = conf_new > conf_current
                keep_existing = ~keep_new
                
                # Update labels
                new_labels = np.zeros_like(labels)
                new_labels[keep_new] = labels[keep_new]
                if new_labels.max() > 0:
                    # Remap labels to avoid conflicts
                    unique_labels = np.unique(new_labels[new_labels > 0])
                    for old_id in unique_labels:
                        new_labels[new_labels == old_id] = current_id
                        current_id += 1
                
                # Combine labels
                final_region = final_labels[y_slice, x_slice]
                final_region[keep_new] = new_labels[keep_new]
                
                # Update confidence map
                confidence_map[y_slice, x_slice][keep_new] = conf_new[keep_new]
            else:
                # No overlap - directly add with remapped IDs
                unique_labels = np.unique(labels[labels > 0])
                new_labels = np.zeros_like(labels)
                for old_id in unique_labels:
                    new_labels[labels == old_id] = current_id
                    current_id += 1
                
                final_labels[y_slice, x_slice] = new_labels
                confidence_map[y_slice, x_slice] = details['prob']
        
        return final_labels, {
            'confidence_map': confidence_map,
            'details': global_details
        }
    
    def process_image(self,
                     image: Union[np.ndarray, str, Path],
                     output_dir: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Process a single image with dynamic parameter adjustment.
        
        Args:
            image: Input image or path to image
            output_dir: Optional directory to save results
            
        Returns:
            Tuple of (instance labels, processing details)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = io.imread(image)
            
        # Normalize image
        image = self._normalize_image(image)
        
        # Compute tile positions
        tile_positions = self._compute_tile_positions(
            image.shape,
            self.tile_size,
            self.overlap
        )
        
        # Process tiles
        tiles_data = []
        for pos in tile_positions:
            # Extract tile
            tile = image[pos[0], pos[1]]
            
            # Compute tile statistics
            tile_stats = compute_local_statistics(tile)
            
            # Process tile
            labels, details = self._process_tile(tile, tile_stats)
            tiles_data.append((pos, (labels, details)))
        
        # Stitch tiles together
        final_labels, global_details = self._stitch_tiles(
            tiles_data,
            image.shape
        )
        
        # Save results if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save segmentation mask
            io.imsave(
                output_dir / 'segmentation.tiff',
                final_labels.astype(np.uint16)
            )
            
            # Save confidence map
            io.imsave(
                output_dir / 'confidence_map.tiff',
                global_details['confidence_map']
            )
            
            # Save processing details
            with open(output_dir / 'processing_details.json', 'w') as f:
                json.dump(global_details['details'], f, indent=2)
        
        return final_labels, global_details
