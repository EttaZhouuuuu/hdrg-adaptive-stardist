import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
import json
import yaml
from datetime import datetime
import git
import sys
import platform
import logging
import numpy as np

class ExperimentTracker:
    """
    Experiment tracking and logging utility.
    
    Features:
    - TensorBoard integration
    - Experiment configuration management
    - Git version control tracking
    - System information logging
    - Metric tracking and visualization
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments
            config: Optional configuration dictionary
            tags: Optional list of tags for the experiment
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config = config or {}
        self.tags = tags or []
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_dir()
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.experiment_dir / "tensorboard"))
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Save experiment metadata
        self._save_experiment_metadata()
        
        self.logger.info(f"Initialized experiment tracker: {experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _create_experiment_dir(self) -> Path:
        """Create and return experiment directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.base_dir / f"{self.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.experiment_dir / "experiment.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                'commit_hash': repo.head.object.hexsha,
                'branch': repo.active_branch.name,
                'is_dirty': repo.is_dirty()
            }
        except:
            return {
                'commit_hash': 'N/A',
                'branch': 'N/A',
                'is_dirty': 'N/A'
            }
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
    
    def _save_experiment_metadata(self):
        """Save experiment metadata."""
        metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'tags': self.tags,
            'config': self.config,
            'git_info': self._get_git_info(),
            'system_info': self._get_system_info()
        }
        
        # Save as JSON
        with open(self.experiment_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save config as YAML for better readability
        with open(self.experiment_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            metric_name = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(metric_name, value, step)
    
    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log images to TensorBoard.
        
        Args:
            images: Dictionary of image names and tensors
            step: Optional step number
            prefix: Optional prefix for image names
        """
        for name, image in images.items():
            image_name = f"{prefix}/{name}" if prefix else name
            self.writer.add_image(image_name, image, step)
    
    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_shape: tuple
    ):
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
        """
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        self.writer.add_graph(model, dummy_input)
    
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log hyperparameters and optional metrics.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional dictionary of metric values
        """
        if metrics:
            self.writer.add_hparams(hparams, metrics)
        
        # Also save to config file
        self.config.update({'hyperparameters': hparams})
        with open(self.experiment_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_name: Optional[str] = None
    ):
        """
        Save an artifact file to the experiment directory.
        
        Args:
            artifact_path: Path to the artifact file
            artifact_name: Optional name for the artifact
        """
        artifact_path = Path(artifact_path)
        if not artifact_name:
            artifact_name = artifact_path.name
        
        dest_path = self.experiment_dir / "artifacts" / artifact_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        import shutil
        shutil.copy2(artifact_path, dest_path)
        
        self.logger.info(f"Saved artifact: {artifact_name}")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        filename: str = "checkpoint.pth"
    ):
        """
        Save a checkpoint.
        
        Args:
            state: State dictionary to save
            filename: Name of the checkpoint file
        """
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        torch.save(state, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {filename}")
    
    def finish(self):
        """Clean up and finish experiment tracking."""
        self.writer.close()
        self.logger.info("Finished experiment tracking")
