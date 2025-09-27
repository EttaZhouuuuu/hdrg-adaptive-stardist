"""
Configuration for comparative experiments.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a model in the experiment."""
    name: str
    model_type: str  # 'baseline', 'adaptive', 'multi_scale'
    params: Dict
    
@dataclass
class DatasetConfig:
    """Configuration for a dataset in the experiment."""
    name: str
    path: Path
    slides: List[str]  # List of slide names to use
    split_ratio: float = 0.8  # train/test split ratio
    
@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    models: List[ModelConfig]
    dataset: DatasetConfig
    metrics: List[str]
    n_runs: int = 3  # number of runs for statistical significance
    
# Define baseline StarDist configuration
BASELINE_CONFIG = ModelConfig(
    name="baseline_stardist",
    model_type="baseline",
    params={
        "n_rays": 32,
        "grid": (2, 2),
        "train_patch_size": (256, 256),
        "n_channel_in": 1
    }
)

# Define adaptive StarDist configuration
ADAPTIVE_CONFIG = ModelConfig(
    name="adaptive_stardist",
    model_type="adaptive",
    params={
        "min_n_rays": 32,
        "max_n_rays": 128,
        "min_grid": 1,
        "max_grid": 4,
        "train_patch_size": (256, 256),
        "n_channel_in": 1
    }
)

# Define multi-scale StarDist configuration
MULTI_SCALE_CONFIG = ModelConfig(
    name="multi_scale_stardist",
    model_type="multi_scale",
    params={
        "scale_factors": [1.0, 0.5, 0.25],
        "fusion_mode": "attention",
        "attention_channels": 64,
        "min_n_rays": 32,
        "max_n_rays": 128,
        "min_grid": 1,
        "max_grid": 4,
        "train_patch_size": (256, 256),
        "n_channel_in": 1
    }
)

# Define experiment configurations
EXPERIMENTS = [
    # Experiment 1: Basic Comparison
    ExperimentConfig(
        name="basic_comparison",
        description="Compare baseline, adaptive, and multi-scale approaches on standard data",
        models=[BASELINE_CONFIG, ADAPTIVE_CONFIG, MULTI_SCALE_CONFIG],
        dataset=DatasetConfig(
            name="standard_dataset", 
            path=Path("data"),
            slides=["240819_Ji_N1_H_EScan", "test_slide_1"]
        ),
        metrics=["f1", "boundary_f1", "shape_preservation"]
    ),
    
    # Experiment 2: Scale Variation
    ExperimentConfig(
        name="scale_variation",
        description="Compare performance on images with high scale variation",
        models=[BASELINE_CONFIG, ADAPTIVE_CONFIG, MULTI_SCALE_CONFIG],
        dataset=DatasetConfig(
            name="scale_variation_dataset",
            path=Path("data"),
            slides=["scale_variation_slide"]
        ),
        metrics=["size_stratified_f1", "scale_consistency"]
    ),
    
    # Experiment 3: Boundary Complexity
    ExperimentConfig(
        name="boundary_complexity",
        description="Compare performance on objects with complex boundaries",
        models=[BASELINE_CONFIG, ADAPTIVE_CONFIG, MULTI_SCALE_CONFIG],
        dataset=DatasetConfig(
            name="complex_boundary_dataset",
            path=Path("data"),
            slides=["test_slide_2"]
        ),
        metrics=["boundary_precision", "boundary_recall", "shape_preservation"]
    ),
    
    # Experiment 4: Density Variation
    ExperimentConfig(
        name="density_variation",
        description="Compare performance on images with varying object density",
        models=[BASELINE_CONFIG, ADAPTIVE_CONFIG, MULTI_SCALE_CONFIG],
        dataset=DatasetConfig(
            name="density_variation_dataset",
            path=Path("data"),
            slides=["240819_Ji_N1_H_EScan", "scale_variation_slide"]
        ),
        metrics=["f1", "boundary_f1", "processing_time"]
    )
]
