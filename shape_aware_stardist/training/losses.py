import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

class StarDistLoss(nn.Module):
    """
    Combined loss function for StarDist model with shape-aware components.
    
    Args:
        dist_loss_weight (float): Weight for distance loss
        prob_loss_weight (float): Weight for probability loss
        shape_loss_weight (float): Weight for shape consistency loss
        reg_loss_weight (float): Weight for shape regularization loss
        focal_gamma (float): Focal loss gamma parameter
        adaptive_weight (bool): Whether to use adaptive loss weighting
    """
    def __init__(
        self,
        dist_loss_weight: float = 1.0,
        prob_loss_weight: float = 1.0,
        shape_loss_weight: float = 0.5,
        reg_loss_weight: float = 0.1,
        focal_gamma: float = 2.0,
        adaptive_weight: bool = True
    ):
        super().__init__()
        self.dist_loss_weight = dist_loss_weight
        self.prob_loss_weight = prob_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.focal_gamma = focal_gamma
        self.adaptive_weight = adaptive_weight
        
        # Initialize adaptive weights if enabled
        if adaptive_weight:
            self.register_buffer('running_dist_loss', torch.tensor(0.0))
            self.register_buffer('running_prob_loss', torch.tensor(0.0))
            self.register_buffer('running_shape_loss', torch.tensor(0.0))
            self.register_buffer('running_reg_loss', torch.tensor(0.0))
            self.momentum = 0.9
    
    def forward(
        self,
        pred_distances: torch.Tensor,
        pred_probabilities: torch.Tensor,
        true_distances: torch.Tensor,
        true_probabilities: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred_distances: Predicted distance map (B, n_rays, H, W)
            pred_probabilities: Predicted probability map (B, 1, H, W)
            true_distances: Ground truth distance map (B, n_rays, H, W)
            true_probabilities: Ground truth probability map (B, H, W)
            attention_maps: Optional attention maps from shape-aware modules
            
        Returns:
            tuple: (total_loss, loss_components)
        """
        # Compute individual losses
        dist_loss = self._distance_loss(pred_distances, true_distances, true_probabilities)
        prob_loss = self._probability_loss(pred_probabilities, true_probabilities)
        shape_loss = self._shape_consistency_loss(pred_distances, true_distances, true_probabilities)
        reg_loss = self._shape_regularization_loss(pred_distances, attention_maps)
        
        # Update running averages for adaptive weighting
        if self.adaptive_weight and self.training:
            self._update_running_losses(dist_loss, prob_loss, shape_loss, reg_loss)
            weights = self._compute_adaptive_weights()
            dist_weight = weights['dist']
            prob_weight = weights['prob']
            shape_weight = weights['shape']
            reg_weight = weights['reg']
        else:
            dist_weight = self.dist_loss_weight
            prob_weight = self.prob_loss_weight
            shape_weight = self.shape_loss_weight
            reg_weight = self.reg_loss_weight
        
        # Combine losses
        total_loss = (
            dist_weight * dist_loss +
            prob_weight * prob_loss +
            shape_weight * shape_loss +
            reg_weight * reg_loss
        )
        
        # Return total loss and components
        return total_loss, {
            'distance_loss': dist_loss,
            'probability_loss': prob_loss,
            'shape_loss': shape_loss,
            'regularization_loss': reg_loss,
            'dist_weight': dist_weight,
            'prob_weight': prob_weight,
            'shape_weight': shape_weight,
            'reg_weight': reg_weight
        }
    
    def _distance_loss(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance prediction loss.
        Uses masked MSE loss for distance predictions.
        """
        # Expand mask to match prediction dimensions
        mask = mask.unsqueeze(1).expand_as(pred)
        
        # Compute masked MSE loss
        loss = F.mse_loss(pred * mask, true * mask, reduction='sum')
        norm = mask.sum() + 1e-8
        return loss / norm
    
    def _probability_loss(
        self,
        pred: torch.Tensor,
        true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute probability prediction loss.
        Uses focal loss for better handling of class imbalance.
        """
        pred = pred.squeeze(1)  # Remove channel dimension
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, true, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        
        return focal_loss.mean()
    
    def _shape_consistency_loss(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute shape consistency loss.
        Ensures consistent predictions across different rays.
        """
        # Compute gradients along ray dimension
        pred_grad = torch.diff(pred, dim=1)
        true_grad = torch.diff(true, dim=1)
        
        # Expand mask for gradient computation
        mask = mask.unsqueeze(1).expand_as(pred_grad)
        
        # Compute masked gradient loss
        loss = F.mse_loss(pred_grad * mask, true_grad * mask, reduction='sum')
        norm = mask.sum() + 1e-8
        return loss / norm
    
    def _shape_regularization_loss(
        self,
        pred: torch.Tensor,
        attention_maps: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute shape regularization loss.
        Encourages smooth and consistent shape predictions.
        """
        if attention_maps is None:
            # Compute smoothness loss on predictions
            grad_y = torch.diff(pred, dim=2)
            grad_x = torch.diff(pred, dim=3)
            loss = (grad_y.abs().mean() + grad_x.abs().mean()) / 2
        else:
            # Use attention maps to guide regularization
            attention_weights = F.softmax(attention_maps, dim=1)
            weighted_pred = pred * attention_weights
            grad_y = torch.diff(weighted_pred, dim=2)
            grad_x = torch.diff(weighted_pred, dim=3)
            loss = (grad_y.abs().mean() + grad_x.abs().mean()) / 2
        
        return loss
    
    def _update_running_losses(
        self,
        dist_loss: torch.Tensor,
        prob_loss: torch.Tensor,
        shape_loss: torch.Tensor,
        reg_loss: torch.Tensor
    ):
        """Update running averages of losses for adaptive weighting."""
        self.running_dist_loss = (
            self.momentum * self.running_dist_loss +
            (1 - self.momentum) * dist_loss.detach()
        )
        self.running_prob_loss = (
            self.momentum * self.running_prob_loss +
            (1 - self.momentum) * prob_loss.detach()
        )
        self.running_shape_loss = (
            self.momentum * self.running_shape_loss +
            (1 - self.momentum) * shape_loss.detach()
        )
        self.running_reg_loss = (
            self.momentum * self.running_reg_loss +
            (1 - self.momentum) * reg_loss.detach()
        )
    
    def _compute_adaptive_weights(self) -> Dict[str, float]:
        """
        Compute adaptive weights based on running losses.
        Uses gradient normalization strategy.
        """
        # Get current loss values
        losses = torch.stack([
            self.running_dist_loss,
            self.running_prob_loss,
            self.running_shape_loss,
            self.running_reg_loss
        ])
        
        # Compute weights using gradient normalization
        weights = 1 / (losses + 1e-8)
        weights = weights / weights.sum()
        
        return {
            'dist': weights[0].item() * self.dist_loss_weight,
            'prob': weights[1].item() * self.prob_loss_weight,
            'shape': weights[2].item() * self.shape_loss_weight,
            'reg': weights[3].item() * self.reg_loss_weight
        }
