import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from ..configs.config import ModelConfig, default_config
from ..models.shape_aware_backbone import ShapeAwareBackbone

def setup_model(config: ModelConfig = default_config) -> ShapeAwareBackbone:
    """
    Set up the Shape-aware StarDist model.
    
    Args:
        config: Model configuration
        
    Returns:
        Configured ShapeAwareBackbone model
    """
    model = ShapeAwareBackbone(
        in_channels=config.backbone.in_channels,
        base_channels=config.backbone.base_channels,
        num_levels=config.backbone.num_levels,
        num_transformer_blocks=config.backbone.num_transformer_blocks,
        num_heads=config.backbone.num_heads,
        dropout=config.backbone.dropout
    )
    return model

def train_step(model: nn.Module,
               data: torch.Tensor,
               target: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module) -> float:
    """
    Single training step.
    
    Args:
        model: The neural network model
        data: Input data batch
        target: Target batch
        optimizer: The optimizer
        criterion: Loss function
        
    Returns:
        float: Loss value
    """
    optimizer.zero_grad()
    output, _ = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module) -> float:
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        criterion: Loss function
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            output, _ = model(data)
            val_loss += criterion(output, target).item()
    return val_loss / len(val_loader)

def main():
    """Main training function"""
    # Load configuration
    config = default_config
    
    # Set up model
    model = setup_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    criterion = nn.MSELoss()  # Example loss function
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training.lr_schedule_factor,
        patience=config.training.lr_schedule_patience,
        min_lr=config.training.min_lr
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(config.training.num_epochs):
        model.train()
        
        # Training loop would go here
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(device), target.to(device)
        #     loss = train_step(model, data, target, optimizer, criterion)
        
        # Validation
        # val_loss = validate(model, val_loader, criterion)
        
        # For example purposes
        val_loss = 0.5  # Placeholder
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model
            save_path = Path(config.model_dir) / "best_model.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")

if __name__ == "__main__":
    main()
