from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class ShapeAwareBackboneConfig:
    """Configuration for Shape-aware Backbone"""
    in_channels: int = 3
    base_channels: int = 64
    num_levels: int = 4
    num_transformer_blocks: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    
@dataclass
class AttentionConfig:
    """Configuration for Attention modules"""
    reduction_ratio: int = 8
    use_channel_attention: bool = True
    use_spatial_attention: bool = True
    use_shape_context: bool = True

@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # Learning rate scheduling
    lr_schedule_patience: int = 10
    lr_schedule_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 20
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 180.0
    flip_probability: float = 0.5
    scale_range: Tuple[float, float] = (0.8, 1.2)
    
    # Validation
    validation_interval: int = 1
    save_best_only: bool = True

@dataclass
class ModelConfig:
    """Complete model configuration"""
    backbone: ShapeAwareBackboneConfig = ShapeAwareBackboneConfig()
    attention: AttentionConfig = AttentionConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Input/Output settings
    input_size: Tuple[int, int] = (512, 512)
    n_rays: int = 32  # Number of rays for StarDist
    
    # Model saving
    model_dir: str = "trained_models"
    
    def __post_init__(self):
        if not isinstance(self.backbone, ShapeAwareBackboneConfig):
            self.backbone = ShapeAwareBackboneConfig(**self.backbone)
        if not isinstance(self.attention, AttentionConfig):
            self.attention = AttentionConfig(**self.attention)
        if not isinstance(self.training, TrainingConfig):
            self.training = TrainingConfig(**self.training)

# Default configuration
default_config = ModelConfig()
