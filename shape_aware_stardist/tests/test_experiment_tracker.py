import unittest
import torch
import tempfile
import shutil
from pathlib import Path
import json
import yaml
import numpy as np

from ..tracking.experiment_tracker import ExperimentTracker
from ..models.shape_aware_backbone import ShapeAwareBackbone

class TestExperimentTracker(unittest.TestCase):
    """Test cases for experiment tracking"""
    
    def setUp(self):
        """Set up test cases"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create tracker
        self.config = {
            'model': {
                'in_channels': 3,
                'base_channels': 64,
                'num_levels': 4
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 0.001,
                'num_epochs': 100
            }
        }
        
        self.tracker = ExperimentTracker(
            experiment_name="test_experiment",
            base_dir=self.temp_dir,
            config=self.config,
            tags=['test', 'unit_test']
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.tracker.finish()
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test tracker initialization"""
        # Check directory creation
        self.assertTrue(self.tracker.experiment_dir.exists())
        self.assertTrue((self.tracker.experiment_dir / "tensorboard").exists())
        
        # Check metadata file
        metadata_path = self.tracker.experiment_dir / "metadata.json"
        self.assertTrue(metadata_path.exists())
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['experiment_name'], "test_experiment")
        self.assertEqual(metadata['tags'], ['test', 'unit_test'])
        self.assertEqual(metadata['config'], self.config)
        
        # Check config file
        config_path = self.tracker.experiment_dir / "config.yaml"
        self.assertTrue(config_path.exists())
        
        with open(config_path) as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config, self.config)
    
    def test_metric_logging(self):
        """Test metric logging"""
        metrics = {
            'loss': 0.5,
            'accuracy': 0.95
        }
        
        # Log metrics
        self.tracker.log_metrics(metrics, step=0)
        self.tracker.log_metrics(metrics, step=1, prefix='train')
        
        # Check that TensorBoard files were created
        event_files = list(self.tracker.experiment_dir.glob("tensorboard/events*"))
        self.assertTrue(len(event_files) > 0)
    
    def test_image_logging(self):
        """Test image logging"""
        # Create test image
        image = torch.randn(3, 64, 64)
        images = {'test_image': image}
        
        # Log image
        self.tracker.log_images(images, step=0)
        
        # Check that TensorBoard files were created
        event_files = list(self.tracker.experiment_dir.glob("tensorboard/events*"))
        self.assertTrue(len(event_files) > 0)
    
    def test_model_graph_logging(self):
        """Test model graph logging"""
        # Create model
        model = ShapeAwareBackbone(
            in_channels=3,
            base_channels=16,
            num_levels=3
        )
        
        # Log model graph
        self.tracker.log_model_graph(model, (1, 3, 64, 64))
        
        # Check that TensorBoard files were created
        event_files = list(self.tracker.experiment_dir.glob("tensorboard/events*"))
        self.assertTrue(len(event_files) > 0)
    
    def test_hyperparameter_logging(self):
        """Test hyperparameter logging"""
        hparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam'
        }
        
        metrics = {
            'final_accuracy': 0.95,
            'final_loss': 0.1
        }
        
        # Log hyperparameters
        self.tracker.log_hyperparameters(hparams, metrics)
        
        # Check config file was updated
        config_path = self.tracker.experiment_dir / "config.yaml"
        with open(config_path) as f:
            saved_config = yaml.safe_load(f)
        
        self.assertIn('hyperparameters', saved_config)
        self.assertEqual(saved_config['hyperparameters'], hparams)
    
    def test_artifact_saving(self):
        """Test artifact saving"""
        # Create test artifact
        artifact_content = "Test artifact content"
        artifact_path = Path(self.temp_dir) / "test_artifact.txt"
        
        with open(artifact_path, 'w') as f:
            f.write(artifact_content)
        
        # Save artifact
        self.tracker.log_artifact(artifact_path, "test_artifact.txt")
        
        # Check artifact was saved
        saved_path = self.tracker.experiment_dir / "artifacts" / "test_artifact.txt"
        self.assertTrue(saved_path.exists())
        
        with open(saved_path) as f:
            saved_content = f.read()
        
        self.assertEqual(saved_content, artifact_content)
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving"""
        # Create test state
        state = {
            'epoch': 10,
            'model_state': {'weight': torch.randn(5, 5)},
            'optimizer_state': {'step': 100}
        }
        
        # Save checkpoint
        self.tracker.save_checkpoint(state, "test_checkpoint.pth")
        
        # Check checkpoint was saved
        checkpoint_path = self.tracker.experiment_dir / "checkpoints" / "test_checkpoint.pth"
        self.assertTrue(checkpoint_path.exists())
        
        # Load and verify checkpoint
        loaded_state = torch.load(checkpoint_path)
        self.assertEqual(loaded_state['epoch'], state['epoch'])
        self.assertTrue(torch.equal(
            loaded_state['model_state']['weight'],
            state['model_state']['weight']
        ))

if __name__ == '__main__':
    unittest.main()
