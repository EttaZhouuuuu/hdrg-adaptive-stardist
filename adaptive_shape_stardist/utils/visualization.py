"""
Visualization Utilities
========================

Functions for visualizing model predictions and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb


def visualize_predictions(image, labels, prob=None, sampling_points=None, 
                         title=None, figsize=(15, 5)):
    """
    Visualize image, predictions, and optional details
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    labels : np.ndarray
        Instance label image
    prob : np.ndarray, optional
        Probability map
    sampling_points : np.ndarray, optional
        Adaptive sampling points [N, 2]
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    """
    
    # Determine number of subplots
    n_plots = 2
    if prob is not None:
        n_plots += 1
    if sampling_points is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot original image
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    
    axes[plot_idx].imshow(image, cmap='gray')
    axes[plot_idx].set_title('Input Image')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Plot instance labels with boundaries
    label_overlay = label2rgb(labels, image=image, bg_label=0, alpha=0.3)
    boundaries = find_boundaries(labels, mode='thick')
    label_overlay[boundaries] = [1, 0, 0]  # Red boundaries
    
    axes[plot_idx].imshow(label_overlay)
    axes[plot_idx].set_title(f'Instances (n={len(np.unique(labels))-1})')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Plot probability map if provided
    if prob is not None:
        im = axes[plot_idx].imshow(prob, cmap='viridis')
        axes[plot_idx].set_title('Probability Map')
        axes[plot_idx].axis('off')
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
        plot_idx += 1
    
    # Plot sampling points if provided
    if sampling_points is not None:
        axes[plot_idx].imshow(image, cmap='gray')
        axes[plot_idx].scatter(
            sampling_points[:, 1], 
            sampling_points[:, 0],
            c='red', s=10, alpha=0.5
        )
        axes[plot_idx].set_title('Adaptive Sampling Points')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Parameters:
    -----------
    history : dict
        Training history with 'loss' and 'val_loss' keys
    save_path : str, optional
        Path to save figure
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['loss']) + 1)
    
    ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_shape_complexity(image, labels, complexity_scores, figsize=(12, 4)):
    """
    Visualize predicted shape complexity for each instance
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    labels : np.ndarray
        Instance labels
    complexity_scores : dict
        Dictionary mapping instance ID to complexity score
    figsize : tuple
        Figure size
    """
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Instance labels
    axes[1].imshow(label2rgb(labels, bg_label=0))
    axes[1].set_title('Instance Labels')
    axes[1].axis('off')
    
    # Complexity map
    complexity_map = np.zeros_like(labels, dtype=float)
    for instance_id, score in complexity_scores.items():
        complexity_map[labels == instance_id] = score
    
    im = axes[2].imshow(complexity_map, cmap='plasma')
    axes[2].set_title('Shape Complexity')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    return fig


def visualize_shape_prototypes(prototype_weights, prototype_embeddings=None, 
                               figsize=(12, 6)):
    """
    Visualize learned shape prototypes
    
    Parameters:
    -----------
    prototype_weights : np.ndarray
        Weights for each prototype [batch, num_prototypes]
    prototype_embeddings : np.ndarray, optional
        Prototype embeddings for visualization
    figsize : tuple
        Figure size
    """
    
    num_prototypes = prototype_weights.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Prototype usage histogram
    avg_weights = np.mean(prototype_weights, axis=0)
    axes[0].bar(range(num_prototypes), avg_weights)
    axes[0].set_xlabel('Prototype Index')
    axes[0].set_ylabel('Average Weight')
    axes[0].set_title('Prototype Usage')
    axes[0].grid(True, alpha=0.3)
    
    # Prototype weight heatmap
    im = axes[1].imshow(prototype_weights, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Prototype Index')
    axes[1].set_ylabel('Sample Index')
    axes[1].set_title('Prototype Weights per Sample')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    return fig


def compare_with_stardist(image, adaptive_labels, stardist_labels, 
                         ground_truth=None, figsize=(15, 5)):
    """
    Compare Adaptive Shape StarDist with original StarDist
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    adaptive_labels : np.ndarray
        Predictions from Adaptive Shape StarDist
    stardist_labels : np.ndarray
        Predictions from original StarDist
    ground_truth : np.ndarray, optional
        Ground truth labels
    figsize : tuple
        Figure size
    """
    
    n_plots = 3 if ground_truth is None else 4
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    
    plot_idx = 0
    
    # Original image
    axes[plot_idx].imshow(image, cmap='gray')
    axes[plot_idx].set_title('Input Image')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Ground truth (if provided)
    if ground_truth is not None:
        axes[plot_idx].imshow(label2rgb(ground_truth, bg_label=0))
        axes[plot_idx].set_title(f'Ground Truth\n(n={len(np.unique(ground_truth))-1})')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Adaptive Shape StarDist
    axes[plot_idx].imshow(label2rgb(adaptive_labels, bg_label=0))
    axes[plot_idx].set_title(f'Adaptive Shape StarDist\n(n={len(np.unique(adaptive_labels))-1})')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Original StarDist
    axes[plot_idx].imshow(label2rgb(stardist_labels, bg_label=0))
    axes[plot_idx].set_title(f'Original StarDist\n(n={len(np.unique(stardist_labels))-1})')
    axes[plot_idx].axis('off')
    
    plt.tight_layout()
    return fig

