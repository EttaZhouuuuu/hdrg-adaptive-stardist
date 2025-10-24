import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
from datetime import datetime

from ..models.shape_aware_backbone import ShapeAwareBackbone
from .losses import StarDistLoss

class ShapeAwareTrainer:
    """
    Trainer class for Shape-aware StarDist model.
    
    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to use
        config: Training configuration
        scheduler: Optional learning rate scheduler
        experiment_dir: Directory for saving experiments
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: StarDistLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: Dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        experiment_dir: Optional[str] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        
        # Setup experiment directory
        self.experiment_dir = Path(experiment_dir or self._create_experiment_dir())
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Early stopping
        self.patience = config.get('early_stopping_patience', 10)
        self.patience_counter = 0
        
        self.logger.info(f"Initialized trainer with experiment dir: {self.experiment_dir}")
        self._log_config()
    
    def train(self, num_epochs: int) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            dict: Training history
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validation phase
            val_metrics = self._validate_epoch()
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                self.logger.info(f"Current learning rate: {current_lr:.6f}")
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(val_metrics['loss'])
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['loss']):
                self.logger.info("Early stopping triggered")
                break
            
            # Save training state
            self._save_training_state()
        
        # Load best model
        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path))
        
        return self._get_training_history()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_components = {}
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device)
                true_distances = batch['distances'].to(self.device)
                true_probabilities = batch['probabilities'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                pred_distances, attention_maps = self.model(images)
                pred_probabilities = pred_distances[:, :1]  # Use first channel as probability
                
                # Compute loss
                loss, components = self.loss_fn(
                    pred_distances,
                    pred_probabilities,
                    true_distances,
                    true_probabilities,
                    attention_maps
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                for name, value in components.items():
                    if name not in total_components:
                        total_components[name] = 0
                    total_components[name] += value.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
                })
        
        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
            **{f"{k}_avg": v / num_batches for k, v in total_components.items()}
        }
        
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_components = {}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Move data to device
                    images = batch['image'].to(self.device)
                    true_distances = batch['distances'].to(self.device)
                    true_probabilities = batch['probabilities'].to(self.device)
                    
                    # Forward pass
                    pred_distances, attention_maps = self.model(images)
                    pred_probabilities = pred_distances[:, :1]
                    
                    # Compute loss
                    loss, components = self.loss_fn(
                        pred_distances,
                        pred_probabilities,
                        true_distances,
                        true_probabilities,
                        attention_maps
                    )
                    
                    # Update metrics
                    total_loss += loss.item()
                    for name, value in components.items():
                        if name not in total_components:
                            total_components[name] = 0
                        total_components[name] += value.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'val_loss': f"{loss.item():.4f}",
                        'avg_val_loss': f"{total_loss/(batch_idx+1):.4f}"
                    })
        
        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
            **{f"{k}_avg": v / num_batches for k, v in total_components.items()}
        }
        
        return metrics
    
    def _save_checkpoint(self, val_loss: float):
        """Save model checkpoint if validation loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save model
            checkpoint_path = self.experiment_dir / f"model_best.pth"
            torch.save(self.model.state_dict(), checkpoint_path)
            self.best_model_path = checkpoint_path
            
            self.logger.info(f"Saved best model checkpoint with val_loss: {val_loss:.4f}")
        else:
            self.patience_counter += 1
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if self.patience_counter >= self.patience:
            return True
        return False
    
    def _save_training_state(self):
        """Save current training state."""
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        
        state_path = self.experiment_dir / "training_state.pth"
        torch.save(state, state_path)
    
    def _get_training_history(self) -> Dict:
        """Get training history."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_model_path': str(self.best_model_path) if self.best_model_path else None
        }
    
    def _create_experiment_dir(self) -> Path:
        """Create unique experiment directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"experiments/shape_aware_stardist_{timestamp}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('ShapeAwareTrainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.experiment_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _log_config(self):
        """Log configuration parameters."""
        self.logger.info("Training configuration:")
        for key, value in self.config.items():
            self.logger.info(f"{key}: {value}")
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training and validation metrics."""
        self.logger.info(
            f"Epoch {self.current_epoch+1} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )
        
        # Log detailed metrics
        for name, value in train_metrics.items():
            if name != 'loss':
                self.logger.info(f"Train {name}: {value:.4f}")
        
        for name, value in val_metrics.items():
            if name != 'loss':
                self.logger.info(f"Val {name}: {value:.4f}")
