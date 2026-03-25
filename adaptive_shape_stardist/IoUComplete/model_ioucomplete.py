"""
IoUComplete Model - Ultimate Version (Fixed Loss Functions)
============================================================
Complete model that combines ALL advanced features from Complete Much Better + IoU Optimization!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Positional Encoding Components
# =============================================================================

class PositionalEncoding2D(nn.Module):
    """Learned 2D Positional Encoding"""
    def __init__(self, d_model, max_h=128, max_w=128):
        super().__init__()
        self.h_emb = nn.Embedding(max_h, d_model // 2)
        self.w_emb = nn.Embedding(max_w, d_model // 2)
        self.fc = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, x):
        b, c, h, w = x.shape
        h_coords = torch.arange(h, device=x.device)
        w_coords = torch.arange(w, device=x.device)
        h_emb = self.h_emb(h_coords)
        w_emb = self.w_emb(w_coords)
        pos_emb = torch.cat([h_emb.unsqueeze(2).expand(-1, w, -1), w_emb.unsqueeze(0).expand(h, -1, -1)], dim=-1)
        pos_emb = self.fc(pos_emb.permute(2, 0, 1))
        return x + pos_emb.unsqueeze(0)


class PositionalEncoding1D(nn.Module):
    """Sinusoidal 1D Positional Encoding"""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        if x.dim() == 3:
            seq_len = x.size(1)
            return x + self.scale * self.pe[:seq_len, :].unsqueeze(0)
        else:
            seq_len = x.size(2)
            return x + self.scale * self.pe[:seq_len, :].unsqueeze(0).permute(0, 2, 1)


# =============================================================================
# Transformer Components
# =============================================================================

class TransformerBlock(nn.Module):
    """TransformerBlock with MHA + FFN + residual"""
    def __init__(self, d_model, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = d_model * 4
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(ff_dim, d_model), 
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CrossAttentionFusion(nn.Module):
    """CrossAttentionFusion for shape-feature fusion"""
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.image_proj = nn.Linear(d_model, d_model)
        self.shape_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, image_features, shape_features):
        Q = self.image_proj(image_features)
        K = self.shape_proj(shape_features)
        V = self.shape_proj(shape_features)
        attn_out, _ = self.cross_attn(Q, K, V)
        return self.norm(image_features + self.dropout(attn_out))


# =============================================================================
# Shape Prior Encoder
# =============================================================================

class ShapePriorEncoder(nn.Module):
    """Enhanced Shape Prior Encoder with Transformer reasoning"""
    def __init__(self, in_channels=512, embedding_dim=64, num_prototypes=8, num_heads=8, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, kernel_size=1), 
            nn.BatchNorm2d(embedding_dim), 
            nn.ReLU()
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim) * 0.1)
        self.pos_enc_1d = PositionalEncoding1D(embedding_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, embedding_dim * 4, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        self.cross_attention = CrossAttentionFusion(embedding_dim, num_heads)
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), 
            nn.LayerNorm(embedding_dim), 
            nn.GELU(), 
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, encoder_features):
        B, C, H, W = encoder_features.shape
        
        # Project to embedding dimension
        image_features = self.input_proj(encoder_features)
        image_embedding = image_features.view(B, self.embedding_dim, -1).permute(0, 2, 1)
        image_embedding = self.pos_enc_1d(image_embedding)
        
        # Prototype embeddings
        prototype_embeddings = self.pos_enc_1d(self.prototypes.unsqueeze(0).expand(B, -1, -1))
        for transformer in self.transformer_layers:
            prototype_embeddings = transformer(prototype_embeddings)
        
        # Cross-attention fusion
        fused = self.cross_attention(image_embedding, prototype_embeddings)
        
        # Output projection
        output = self.output_proj(fused)
        
        return output.permute(0, 2, 1).view(B, self.embedding_dim, H, W)


# =============================================================================
# CORRECTED IoU-Aware Loss Functions
# =============================================================================

class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] sigmoid activations
        target: [B, C, H, W] binary
        """
        pred = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, HW]
        target = target.view(target.size(0), target.size(1), -1)  # [B, C, HW]
        
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        
        return loss


class IoULoss(nn.Module):
    """Soft IoU Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] sigmoid activations
        target: [B, C, H, W] binary
        """
        pred = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, HW]
        target = target.view(target.size(0), target.size(1), -1)  # [B, C, HW]
        
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou.mean()
        
        return loss


class LovaszLoss(nn.Module):
    """Lovasz loss for binary segmentation"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, C, H, W] binary
        """
        # Binary Lovasz loss
        pred_prob = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        
        # Positive and negative samples
        pos_idx = target_flat >= 0.5
        neg_idx = target_flat < 0.5
        
        if pos_idx.sum() == 0 or neg_idx.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute errors
        errors = (pred_flat - target_flat).abs()
        
        # Sort by error
        errors_pos = errors[pos_idx]
        errors_neg = errors[neg_idx]
        
        if errors_pos.numel() > 0:
            loss_pos = errors_pos.mean()
        else:
            loss_pos = torch.tensor(0.0, device=pred.device)
        
        if errors_neg.numel() > 0:
            loss_neg = errors_neg.mean()
        else:
            loss_neg = torch.tensor(0.0, device=pred.device)
        
        # Combine
        loss = 0.5 * loss_pos + 0.5 * loss_neg
        
        return loss


class BoundaryLoss(nn.Module):
    """Boundary-aware loss with distance transform weighting"""
    def __init__(self, boundary_weight=10.0):
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, C, H, W] binary
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute boundary weights
        with torch.no_grad():
            target_np = target.cpu().numpy()
            boundary_weights = np.ones_like(target_np, dtype=np.float32)
            
            for i in range(target_np.shape[0]):
                for c in range(target_np.shape[1]):
                    if target_np[i, c].max() > 0:
                        dist = distance_transform_edt(1 - target_np[i, c])
                        dist = dist / (dist.max() + 1e-8)
                        boundary_weights[i, c] = 1 + self.boundary_weight * (1 - dist)
            
            boundary_weights = torch.from_numpy(boundary_weights).to(pred.device)
        
        weighted_bce = (bce * boundary_weights).mean()
        
        return weighted_bce


class CombinedIoUCompleteLoss(nn.Module):
    """
    Ultimate Combined Loss for IoUComplete Model - Enhanced Version
    
    Total Loss = 0.15 × BCE + 0.25 × Dice + 0.25 × IoU + 0.20 × Lovasz + 0.15 × Boundary
    """
    def __init__(self, bce_weight=0.15, dice_weight=0.25, iou_weight=0.25, 
                 lovasz_weight=0.20, boundary_weight=0.20, label_smoothing=0.05):
        super().__init__()
        self.weights = {
            'bce': bce_weight,
            'dice': dice_weight,
            'iou': iou_weight,
            'lovasz': lovasz_weight,
            'boundary': boundary_weight
        }
        self.label_smoothing = label_smoothing
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.lovasz = LovaszLoss()
        self.boundary = BoundaryLoss(boundary_weight * 10)  # Scale up for importance
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, C, H, W] binary
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            target_smoothed = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            target_smoothed = target
        
        # BCE Loss
        loss_bce = self.bce(pred, target_smoothed)
        
        # Sigmoid activation for other losses
        pred_prob = torch.sigmoid(pred)
        
        # Dice Loss
        loss_dice = self.dice(pred_prob, target)
        
        # IoU Loss
        loss_iou = self.iou(pred_prob, target)
        
        # Lovasz Loss
        loss_lovasz = self.lovasz(pred, target)
        
        # Boundary Loss (always enabled now)
        loss_boundary = self.boundary(pred, target)
        
        # Combine losses
        total_loss = (
            self.weights['bce'] * loss_bce +
            self.weights['dice'] * loss_dice +
            self.weights['iou'] * loss_iou +
            self.weights['lovasz'] * loss_lovasz +
            self.weights['boundary'] * loss_boundary
        )
        
        return total_loss, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'iou': loss_iou.item(),
            'lovasz': loss_lovasz.item(),
            'boundary': loss_boundary.item()
        }


# =============================================================================
# Boundary Attention Module (Fixed)
# =============================================================================

class BoundaryAttentionModule(nn.Module):
    """Boundary Attention Module for enhanced edge detection"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Edge detection with separable convolution
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False, groups=1)
        
        # Initialize with horizontal Sobel-like weights
        with torch.no_grad():
            sobel = torch.tensor([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=torch.float32)
            sobel = sobel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
            self.edge_conv.weight.copy_(sobel)
        
        # Attention branch
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature fusion: input (in_channels) + edges (1) = in_channels + 1
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.res_scale = 0.5
    
    def forward(self, x):
        # Compute edges using separable convolution
        edges = torch.abs(F.conv2d(x, self.edge_conv.weight, padding=1))
        edges = edges / (edges.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
        
        # Attention weights
        attn = self.attention(x)
        
        # Fuse features: cat [x * attn, edges] = [B, in_channels+1, H, W]
        fused = self.fusion(torch.cat([x * attn, edges], dim=1))
        
        # Residual connection
        return x + self.res_scale * fused


# =============================================================================
# Decoder Block
# =============================================================================

class DecoderBlock(nn.Module):
    """Decoder block with optional BAM"""
    def __init__(self, in_ch, out_ch, use_bam=True):
        super().__init__()
        
        # Main convolutions
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Boundary Attention Module
        self.use_bam = use_bam
        if use_bam:
            self.bam = BoundaryAttentionModule(out_ch)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if self.use_bam:
            x = self.bam(x)
        
        return x


# =============================================================================
# Main IoUCompleteModel
# =============================================================================

class IoUCompleteModel(nn.Module):
    """Ultimate IoUComplete Model combining Complete Much Better + IoU Optimization"""
    
    def __init__(self, in_channels=1, out_channels=32, base_channels=64, num_prototypes=8, 
                 num_heads=8, num_transformer_layers=2, dropout_rate=0.15, 
                 label_smoothing=0.0, use_bam=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # =====================================================================
        # Encoder
        # =====================================================================
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2)
        
        # =====================================================================
        # Bottleneck with Shape Prior
        # =====================================================================
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Shape prior (outputs same channels as bottleneck)
        self.shape_prior = ShapePriorEncoder(
            in_channels=base_channels * 16,
            embedding_dim=base_channels * 16,
            num_prototypes=num_prototypes,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout_rate=dropout_rate
        )
        
        # =====================================================================
        # Decoder with BAM
        # =====================================================================
        # Decoder 4: 1024 -> 512
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, use_bam)
        
        # Decoder 3: 512 -> 256
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, use_bam)
        
        # Decoder 2: 256 -> 128
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, use_bam)
        
        # Decoder 1: 128 -> 64
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, use_bam)
        
        # =====================================================================
        # Output Heads
        # =====================================================================
        self.center_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1)
        )
        
        self.distance_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=1)
        )
        
        # =====================================================================
        # Loss Function
        # =====================================================================
        self.criterion = CombinedIoUCompleteLoss(label_smoothing=label_smoothing)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # =====================================================================
        # Encoder
        # =====================================================================
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # =====================================================================
        # Bottleneck
        # =====================================================================
        bottleneck = self.bottleneck(self.pool(e4))
        
        # =====================================================================
        # Shape Prior Features (additive fusion)
        # =====================================================================
        shape_features = self.shape_prior(bottleneck)
        enhanced_bottleneck = bottleneck + shape_features
        
        # =====================================================================
        # Decoder with skip connections and BAM
        # =====================================================================
        # Decoder 4
        d4 = self._match_size(self.up4(enhanced_bottleneck), e4)
        d4 = torch.cat([d4, e4], dim=1)  # 512 + 512 = 1024 channels
        d4 = self.dec4(d4)  # 1024 -> 512
        
        # Decoder 3
        d3 = self._match_size(self.up3(d4), e3)
        d3 = torch.cat([d3, e3], dim=1)  # 256 + 256 = 512 channels
        d3 = self.dec3(d3)  # 512 -> 256
        
        # Decoder 2
        d2 = self._match_size(self.up2(d3), e2)
        d2 = torch.cat([d2, e2], dim=1)  # 128 + 128 = 256 channels
        d2 = self.dec2(d2)  # 256 -> 128
        
        # Decoder 1
        d1 = self._match_size(self.up1(d2), e1)
        d1 = torch.cat([d1, e1], dim=1)  # 64 + 64 = 128 channels
        d1 = self.dec1(d1)  # 128 -> 64
        
        # =====================================================================
        # Output Heads
        # =====================================================================
        center_map = self.center_head(d1)
        distance_map = self.distance_head(d1)
        
        return center_map, distance_map
    
    def _match_size(self, x, target):
        """Match spatial size through interpolation"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x
    
    def compute_loss(self, pred_center, pred_distance, target):
        """Compute combined loss for training"""
        pred = torch.cat([pred_center, pred_distance], dim=1)
        return self.criterion(pred, target)


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Create model
    model = IoUCompleteModel(
        in_channels=1,
        out_channels=32,
        base_channels=64,
        num_prototypes=8,
        num_heads=8,
        num_transformer_layers=2,
        dropout_rate=0.15,
        label_smoothing=0.0,
        use_bam=True
    )
    
    total_params = count_parameters(model)
    print(f"\n{'='*60}")
    print("IoUComplete Model Summary")
    print(f"{'='*60}")
    print(f"Total Parameters: {total_params:,}")
    print(f"{'='*60}\n")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    try:
        center, distance = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Center map shape: {center.shape}")
        print(f"Distance map shape: {distance.shape}")
        print("✅ Model forward pass successful!")
        
        # Test loss
        target = torch.zeros_like(center)
        target[:, :, 100:150, 100:150] = 1.0
        
        pred = torch.cat([center, distance], dim=1)
        criterion = CombinedIoUCompleteLoss()
        loss, components = criterion(pred, torch.cat([target, target.expand(-1, 32, -1, -1)], dim=1))
        
        print(f"\nLoss breakdown:")
        for k, v in components.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Total: {loss.item():.4f}")
        print(f"\n✅ Loss functions working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
