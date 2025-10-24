import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path

from ..training.trainer import ShapeAwareTrainer
from ..training.losses import StarDistLoss
from ..models.shape_aware_backbone import ShapeAwareBackbone

class TestShapeAwareTrainer(unittest.TestCase):
    """Test cases for Shape-aware Trainer"""
    
    def setUp(self):
        """Set up test cases"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy model
        self.model = ShapeAwareBackbone(
            in_channels=3,
            base_channels=16,
            num_levels=3
        ).to(self.device)
        
        # Create dummy data
        self.batch_size = 2
        self.n_rays = 32
        self.height = 64
        self.width = 64
        
        # Training data
        train_images = torch.randn(10, 3, self.height, self.width)
        train_distances = torch.abs(torch.randn(10, self.n_rays, self.height, self.width))
        train_probabilities = torch.randint(0, 2, (10, self.height, self.width)).float()
        
        # Validation data
        val_images = torch.randn(5, 3, self.height, self.width)
        val_distances = torch.abs(torch.randn(5, self.n_rays, self.height, self.width))
        val_probabilities = torch.randint(0, 2, (5, self.height, self.width)).float()
        
        # Create data loaders
        self.train_loader = DataLoader(
            TensorDataset(train_images, train_distances, train_probabilities),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            TensorDataset(val_images, val_distances, val_probabilities),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Create loss function and optimizer
        self.loss_fn = StarDistLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create temporary directory for experiments
        self.temp_dir = tempfile.mkdtemp()
        
        # Training config
        self.config = {
            'num_epochs': 2,
            'early_stopping_patience': 5,
            'learning_rate': 0.001,
            'batch_size': self.batch_size
        }
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = ShapeAwareTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            config=self.config,
            experiment_dir=self.temp_dir
        )
        
        self.assertIsNotNone(trainer.logger)
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.best_val_loss, float('inf'))
        self.assertEqual(trainer.patience_counter, 0)
    
    def test_training_loop(self):
        """Test basic training loop"""
        trainer = ShapeAwareTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            config=self.config,
            experiment_dir=self.temp_dir
        )
        
        # Run training
        history = trainer.train(num_epochs=2)
        
        # Check training history
        self.assertEqual(len(history['train_losses']), 2)
        self.assertEqual(len(history['val_losses']), 2)
        self.assertIsNotNone(history['best_model_path'])
        
        # Check model checkpoints
        checkpoint_path = Path(history['best_model_path'])
        self.assertTrue(checkpoint_path.exists())
    
    def test_early_stopping(self):
        """Test early stopping mechanism"""
        # Create trainer with small patience
        config = self.config.copy()
        config['early_stopping_patience'] = 1
        
        trainer = ShapeAwareTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            config=config,
            experiment_dir=self.temp_dir
        )
        
        # Run training
        history = trainer.train(num_epochs=10)
        
        # Should stop early
        self.assertLess(len(history['train_losses']), 10)
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduling"""
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=2
        )
        
        trainer = ShapeAwareTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            config=self.config,
            scheduler=scheduler,
            experiment_dir=self.temp_dir
        )
        
        # Run training
        history = trainer.train(num_epochs=2)
        
        # Check learning rates were recorded
        self.assertEqual(len(history['learning_rates']), 2)
    
    def test_model_saving(self):
        """Test model checkpoint saving"""
        trainer = ShapeAwareTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            config=self.config,
            experiment_dir=self.temp_dir
        )
        
        # Run training
        trainer.train(num_epochs=2)
        
        # Check training state was saved
        state_path = Path(self.temp_dir) / "training_state.pth"
        self.assertTrue(state_path.exists())
        
        # Load and check state
        state = torch.load(state_path)
        self.assertIn('model_state_dict', state)
        self.assertIn('optimizer_state_dict', state)
        self.assertIn('train_losses', state)
        self.assertIn('val_losses', state)

if __name__ == '__main__':
    unittest.main()
