"""
Configuration Templates
=======================

Pre-defined configurations for different use cases.
"""

from ..models.adaptive_shape_model import AdaptiveShapeConfig


def get_default_config():
    """
    Default configuration for general cell segmentation
    """
    return AdaptiveShapeConfig(
        n_channel_in=1,
        backbone='unet',
        n_depth=3,
        n_filter_base=32,
        min_sampling_points=32,
        max_sampling_points=128,
        use_shape_prior=True,
        num_shape_prototypes=16,
        deformable_groups=4,
        grid=(1, 1),
        train_learning_rate=3e-4,
        train_epochs=100,
        train_steps_per_epoch=100,
        train_batch_size=4,
    )


def get_high_complexity_config():
    """
    Configuration for highly irregular cell shapes
    
    Uses more sampling points and stronger shape prior encoding
    """
    return AdaptiveShapeConfig(
        n_channel_in=1,
        backbone='unet',
        n_depth=4,
        n_filter_base=64,
        min_sampling_points=64,
        max_sampling_points=256,
        use_shape_prior=True,
        num_shape_prototypes=32,
        deformable_groups=8,
        grid=(1, 1),
        train_learning_rate=1e-4,
        train_epochs=150,
        train_steps_per_epoch=200,
        train_batch_size=2,
    )


def get_fast_config():
    """
    Lightweight configuration for fast inference
    
    Uses fewer sampling points and smaller backbone
    """
    return AdaptiveShapeConfig(
        n_channel_in=1,
        backbone='resnet',
        n_depth=2,
        n_filter_base=16,
        min_sampling_points=16,
        max_sampling_points=64,
        use_shape_prior=False,
        num_shape_prototypes=8,
        deformable_groups=2,
        grid=(2, 2),
        train_learning_rate=5e-4,
        train_epochs=50,
        train_steps_per_epoch=50,
        train_batch_size=8,
    )


def get_multi_channel_config(n_channels=3):
    """
    Configuration for multi-channel images (e.g., RGB, fluorescence)
    """
    return AdaptiveShapeConfig(
        n_channel_in=n_channels,
        backbone='unet',
        n_depth=3,
        n_filter_base=32,
        min_sampling_points=32,
        max_sampling_points=128,
        use_shape_prior=True,
        num_shape_prototypes=16,
        deformable_groups=4,
        grid=(1, 1),
        train_learning_rate=3e-4,
        train_epochs=100,
        train_steps_per_epoch=100,
        train_batch_size=4,
    )


def get_fpn_config():
    """
    FPN-based configuration for multi-scale feature extraction
    
    Uses Feature Pyramid Network backbone for robust multi-scale segmentation.
    Best for images with cells of varying sizes.
    """
    return AdaptiveShapeConfig(
        n_channel_in=1,
        backbone='resnet34',  # Ignored when use_fpn=True
        n_depth=3,
        n_filter_base=32,
        min_sampling_points=32,
        max_sampling_points=128,
        use_shape_prior=True,
        num_shape_prototypes=16,
        deformable_groups=4,
        grid=(1, 1),
        # FPN-specific parameters
        use_fpn=True,
        fpn_channels=256,
        fpn_levels=['p2', 'p3', 'p4', 'p5'],
        multiscale_prediction=False,  # Use single scale (p2) by default
        train_learning_rate=3e-4,
        train_epochs=100,
        train_steps_per_epoch=100,
        train_batch_size=4,
    )


def get_fpn_multiscale_config():
    """
    FPN with multi-scale prediction
    
    Predicts at all FPN levels (p2, p3, p4, p5) for maximum robustness.
    Suitable for challenging datasets with extreme size variation.
    """
    return AdaptiveShapeConfig(
        n_channel_in=1,
        backbone='resnet34',
        n_depth=3,
        n_filter_base=32,
        min_sampling_points=32,
        max_sampling_points=128,
        use_shape_prior=True,
        num_shape_prototypes=16,
        deformable_groups=4,
        grid=(1, 1),
        # FPN with multi-scale prediction
        use_fpn=True,
        fpn_channels=256,
        fpn_levels=['p2', 'p3', 'p4', 'p5'],
        multiscale_prediction=True,  # Predict at all scales
        train_learning_rate=3e-4,
        train_epochs=120,  # Slightly longer training
        train_steps_per_epoch=150,
        train_batch_size=2,  # Smaller batch due to memory
    )


def get_fpn_fast_config():
    """
    Lightweight FPN configuration
    
    Uses fewer FPN levels for faster inference while maintaining multi-scale benefits.
    """
    return AdaptiveShapeConfig(
        n_channel_in=1,
        backbone='resnet34',
        n_depth=3,
        n_filter_base=32,
        min_sampling_points=32,
        max_sampling_points=64,
        use_shape_prior=False,  # Disable for speed
        num_shape_prototypes=8,
        deformable_groups=2,
        grid=(1, 1),
        # Simplified FPN
        use_fpn=True,
        fpn_channels=128,  # Fewer channels
        fpn_levels=['p2', 'p3'],  # Only 2 scales
        multiscale_prediction=False,
        train_learning_rate=5e-4,
        train_epochs=80,
        train_steps_per_epoch=100,
        train_batch_size=8,
    )

