import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class Visualizer:
    """
    Visualization tools for Shape-aware StarDist model.
    
    Provides methods for visualizing:
    - Predictions and ground truth
    - Attention maps
    - Training progress
    - Shape analysis
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up custom colormaps
        self.attention_cmap = self._create_attention_colormap()
    
    @staticmethod
    def _create_attention_colormap() -> LinearSegmentedColormap:
        """Create custom colormap for attention visualization."""
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
                 '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        return LinearSegmentedColormap.from_list('attention', colors)
    
    def visualize_prediction(
        self,
        image: torch.Tensor,
        pred_mask: torch.Tensor,
        true_mask: Optional[torch.Tensor] = None,
        attention_maps: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize model prediction with optional ground truth and attention maps.
        
        Args:
            image: Input image tensor (C, H, W)
            pred_mask: Predicted mask tensor (H, W)
            true_mask: Optional ground truth mask tensor (H, W)
            attention_maps: Optional attention maps tensor (N, H, W)
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        # Convert tensors to numpy arrays
        image = image.cpu().numpy().transpose(1, 2, 0)
        pred_mask = pred_mask.cpu().numpy()
        if true_mask is not None:
            true_mask = true_mask.cpu().numpy()
        if attention_maps is not None:
            attention_maps = attention_maps.cpu().numpy()
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create figure
        n_cols = 2 + (true_mask is not None) + (attention_maps is not None)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        if n_cols == 1:
            axes = [axes]
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot prediction
        axes[1].imshow(image)
        axes[1].imshow(pred_mask, alpha=0.5, cmap='viridis')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        col_idx = 2
        
        # Plot ground truth if available
        if true_mask is not None:
            axes[col_idx].imshow(image)
            axes[col_idx].imshow(true_mask, alpha=0.5, cmap='viridis')
            axes[col_idx].set_title('Ground Truth')
            axes[col_idx].axis('off')
            col_idx += 1
        
        # Plot attention maps if available
        if attention_maps is not None:
            mean_attention = attention_maps.mean(axis=0)
            axes[col_idx].imshow(mean_attention, cmap=self.attention_cmap)
            axes[col_idx].set_title('Attention Map')
            axes[col_idx].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = self.save_dir / save_path if self.save_dir else save_path
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Convert figure to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_image
    
    def visualize_training_progress(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize training progress metrics.
        
        Args:
            metrics: Dictionary of metric names and their values over time
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values, label=metric_name)
            ax.set_title(f'{metric_name} over Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.grid(True)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = self.save_dir / save_path if self.save_dir else save_path
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Convert figure to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_image
    
    def visualize_attention_analysis(
        self,
        attention_maps: torch.Tensor,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detailed analysis of attention maps.
        
        Args:
            attention_maps: Attention maps tensor (N, H, W)
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        attention_maps = attention_maps.cpu().numpy()
        n_maps = attention_maps.shape[0]
        
        # Create figure
        n_cols = min(4, n_maps)
        n_rows = (n_maps + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
        
        # Plot individual attention maps
        for i in range(n_maps):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].imshow(attention_maps[i], cmap=self.attention_cmap)
            axes[row, col].set_title(f'Attention Head {i+1}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_maps, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = self.save_dir / save_path if self.save_dir else save_path
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Convert figure to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_image
    
    def visualize_shape_analysis(
        self,
        pred_distances: torch.Tensor,
        true_distances: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize shape analysis including ray distances.
        
        Args:
            pred_distances: Predicted distance tensor (n_rays, H, W)
            true_distances: Optional ground truth distance tensor (n_rays, H, W)
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        pred_distances = pred_distances.cpu().numpy()
        if true_distances is not None:
            true_distances = true_distances.cpu().numpy()
        
        n_rays = pred_distances.shape[0]
        angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot mean distances
        mean_pred = pred_distances.mean(axis=0)
        axes[0].imshow(mean_pred, cmap='viridis')
        axes[0].set_title('Mean Predicted Distances')
        axes[0].axis('off')
        
        # Plot polar representation
        center_pred = pred_distances[:, pred_distances.shape[1]//2, pred_distances.shape[2]//2]
        axes[1].plot(angles, center_pred, label='Prediction')
        
        if true_distances is not None:
            center_true = true_distances[:, true_distances.shape[1]//2, true_distances.shape[2]//2]
            axes[1].plot(angles, center_true, label='Ground Truth')
        
        axes[1].set_title('Ray Distances (Center Point)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = self.save_dir / save_path if self.save_dir else save_path
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Convert figure to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_image
