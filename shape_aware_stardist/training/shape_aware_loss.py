import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

class ShapeAwareHybridLoss(nn.Module):
    """
    Shape-aware hybrid loss function combining multiple components:
    1. Basic StarDist losses (distance + probability)
    2. Shape consistency loss
    3. Boundary smoothness loss
    4. Adversarial loss
    
    Args:
        n_rays: Number of rays in StarDist
        consistency_weight: Weight for shape consistency loss
        smoothness_weight: Weight for boundary smoothness loss
        adversarial_weight: Weight for adversarial loss
        use_focal_loss: Whether to use focal loss for probability
        focal_gamma: Gamma parameter for focal loss
        use_adaptive_weights: Whether to use adaptive loss weighting
    """
    def __init__(
        self,
        n_rays: int = 32,
        consistency_weight: float = 1.0,
        smoothness_weight: float = 0.5,
        adversarial_weight: float = 0.1,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        use_adaptive_weights: bool = True
    ):
        super().__init__()
        self.n_rays = n_rays
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_adaptive_weights = use_adaptive_weights
        
        # Initialize discriminator for adversarial loss
        self.discriminator = ShapeDiscriminator(n_rays)
        self.discriminator_criterion = nn.BCEWithLogitsLoss()
        
        # Initialize adaptive weights if enabled
        if use_adaptive_weights:
            self.register_buffer('running_losses', torch.zeros(5))  # 5 loss components
            self.momentum = 0.9
    
    def compute_distance_loss(
        self,
        pred_distances: torch.Tensor,
        true_distances: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted distance loss."""
        # Apply mask
        mask = mask.unsqueeze(1).expand_as(pred_distances)
        
        # Compute Huber loss for robustness
        loss = F.smooth_l1_loss(
            pred_distances * mask,
            true_distances * mask,
            reduction='none'
        )
        
        # Weight loss by distance to favor accurate boundary prediction
        distance_weights = torch.exp(-true_distances * mask)
        weighted_loss = loss * distance_weights
        
        return weighted_loss.sum() / (mask.sum() + 1e-8)
    
    def compute_probability_loss(
        self,
        pred_prob: torch.Tensor,
        true_prob: torch.Tensor
    ) -> torch.Tensor:
        """Compute probability loss with optional focal loss."""
        if self.use_focal_loss:
            # Focal loss implementation
            ce_loss = F.binary_cross_entropy_with_logits(
                pred_prob,
                true_prob,
                reduction='none'
            )
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = focal_weight * ce_loss
        else:
            loss = F.binary_cross_entropy_with_logits(
                pred_prob,
                true_prob,
                reduction='none'
            )
        
        return loss.mean()
    
    def compute_shape_consistency_loss(
        self,
        pred_distances: torch.Tensor,
        true_distances: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute shape consistency loss to ensure consistent predictions
        across different rays and spatial locations.
        """
        # Compute gradients along ray dimension
        pred_grad_rays = torch.diff(pred_distances, dim=1)
        true_grad_rays = torch.diff(true_distances, dim=1)
        
        # Compute gradients along spatial dimensions
        pred_grad_y = torch.diff(pred_distances, dim=2)
        pred_grad_x = torch.diff(pred_distances, dim=3)
        true_grad_y = torch.diff(true_distances, dim=2)
        true_grad_x = torch.diff(true_distances, dim=3)
        
        # Compute consistency losses
        ray_consistency = F.mse_loss(pred_grad_rays, true_grad_rays)
        spatial_consistency_y = F.mse_loss(pred_grad_y, true_grad_y)
        spatial_consistency_x = F.mse_loss(pred_grad_x, true_grad_x)
        
        return ray_consistency + spatial_consistency_y + spatial_consistency_x
    
    def compute_boundary_smoothness_loss(
        self,
        pred_distances: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary smoothness loss to encourage smooth shape boundaries.
        Uses total variation regularization with edge-awareness.
        """
        # Compute gradients
        grad_y = torch.diff(pred_distances, dim=2)
        grad_x = torch.diff(pred_distances, dim=3)
        
        # Compute edge weights
        edge_weights_y = torch.exp(-torch.abs(grad_y))
        edge_weights_x = torch.exp(-torch.abs(grad_x))
        
        # Compute weighted total variation
        smoothness_y = torch.abs(grad_y) * edge_weights_y
        smoothness_x = torch.abs(grad_x) * edge_weights_x
        
        return (smoothness_y.mean() + smoothness_x.mean()) / 2
    
    def compute_adversarial_loss(
        self,
        pred_distances: torch.Tensor,
        true_distances: torch.Tensor,
        is_training_discriminator: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute adversarial loss for improving shape quality.
        Returns generator loss and optional discriminator loss.
        """
        batch_size = pred_distances.size(0)
        real_label = torch.ones(batch_size, 1).to(pred_distances.device)
        fake_label = torch.zeros(batch_size, 1).to(pred_distances.device)
        
        # Discriminator forward pass
        real_output = self.discriminator(true_distances)
        fake_output = self.discriminator(pred_distances.detach())
        
        if is_training_discriminator:
            # Compute discriminator loss
            d_loss_real = self.discriminator_criterion(real_output, real_label)
            d_loss_fake = self.discriminator_criterion(fake_output, fake_label)
            d_loss = (d_loss_real + d_loss_fake) / 2
            return None, d_loss
        else:
            # Compute generator loss
            fake_output = self.discriminator(pred_distances)
            g_loss = self.discriminator_criterion(fake_output, real_label)
            return g_loss, None
    
    def update_adaptive_weights(self, losses: List[torch.Tensor]):
        """Update running averages for adaptive weight computation."""
        if not self.use_adaptive_weights:
            return
            
        with torch.no_grad():
            for i, loss in enumerate(losses):
                self.running_losses[i] = (
                    self.momentum * self.running_losses[i] +
                    (1 - self.momentum) * loss.detach()
                )
    
    def compute_adaptive_weights(self) -> torch.Tensor:
        """Compute adaptive weights based on running losses."""
        if not self.use_adaptive_weights:
            return torch.ones(5, device=self.running_losses.device)
            
        # Use inverse of running losses as weights
        weights = 1 / (self.running_losses + 1e-8)
        return weights / weights.sum()
    
    def forward(
        self,
        pred_distances: torch.Tensor,
        pred_probabilities: torch.Tensor,
        true_distances: torch.Tensor,
        true_probabilities: torch.Tensor,
        is_training_discriminator: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass computing all loss components.
        
        Args:
            pred_distances: Predicted distance map (B, n_rays, H, W)
            pred_probabilities: Predicted probability map (B, 1, H, W)
            true_distances: Ground truth distance map (B, n_rays, H, W)
            true_probabilities: Ground truth probability map (B, H, W)
            is_training_discriminator: Whether to compute discriminator loss
            
        Returns:
            tuple: (total_loss, loss_components)
        """
        # Compute individual losses
        distance_loss = self.compute_distance_loss(
            pred_distances, true_distances, true_probabilities
        )
        
        probability_loss = self.compute_probability_loss(
            pred_probabilities, true_probabilities
        )
        
        consistency_loss = self.compute_shape_consistency_loss(
            pred_distances, true_distances, true_probabilities
        )
        
        smoothness_loss = self.compute_boundary_smoothness_loss(
            pred_distances, true_probabilities
        )
        
        adv_loss, d_loss = self.compute_adversarial_loss(
            pred_distances,
            true_distances,
            is_training_discriminator
        )
        
        # Get loss weights
        if self.use_adaptive_weights:
            weights = self.compute_adaptive_weights()
            distance_weight, prob_weight, cons_weight, smooth_weight, adv_weight = weights
        else:
            distance_weight = 1.0
            prob_weight = 1.0
            cons_weight = self.consistency_weight
            smooth_weight = self.smoothness_weight
            adv_weight = self.adversarial_weight
        
        # Combine losses
        if is_training_discriminator:
            total_loss = d_loss
        else:
            losses = [
                distance_loss * distance_weight,
                probability_loss * prob_weight,
                consistency_loss * cons_weight,
                smoothness_loss * smooth_weight,
                adv_loss * adv_weight
            ]
            total_loss = sum(losses)
            
            # Update adaptive weights
            if self.use_adaptive_weights:
                self.update_adaptive_weights([
                    distance_loss, probability_loss,
                    consistency_loss, smoothness_loss, adv_loss
                ])
        
        # Return loss components
        loss_components = {
            'distance_loss': distance_loss,
            'probability_loss': probability_loss,
            'consistency_loss': consistency_loss,
            'smoothness_loss': smoothness_loss,
            'adversarial_loss': adv_loss if adv_loss is not None else d_loss,
            'distance_weight': distance_weight,
            'probability_weight': prob_weight,
            'consistency_weight': cons_weight,
            'smoothness_weight': smooth_weight,
            'adversarial_weight': adv_weight
        }
        
        return total_loss, loss_components


class ShapeDiscriminator(nn.Module):
    """
    Discriminator network for adversarial training.
    Takes distance predictions and determines if they are real or fake.
    """
    def __init__(self, n_rays: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Initial convolution
            nn.Conv2d(n_rays, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Intermediate layers
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Global average pooling and final classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output.view(x.size(0), -1)
