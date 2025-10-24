import unittest
import torch
import torch.nn as nn
from ..optimization.model_optimizer import ModelOptimizer
from ..models.shape_aware_backbone import ShapeAwareBackbone

class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TestModelOptimizer(unittest.TestCase):
    """Test cases for model optimization"""
    
    def setUp(self):
        """Set up test cases"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = ShapeAwareBackbone(
            in_channels=3,
            base_channels=16,
            num_levels=3
        ).to(self.device)
        
        self.optimizer = ModelOptimizer(self.model)
        
        # Create simple model for basic tests
        self.simple_model = SimpleModel().to(self.device)
        self.simple_optimizer = ModelOptimizer(self.simple_model)
    
    def test_quantization(self):
        """Test model quantization"""
        if self.device.type == 'cuda':
            # Quantization currently only supported on CPU
            return
        
        # Quantize model
        quantized_model = self.simple_optimizer.quantize_model()
        
        self.assertIsInstance(quantized_model, nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 3, 8, 8).to(self.device)
        output = quantized_model(x)
        
        self.assertIsInstance(output, torch.Tensor)
    
    def test_pruning(self):
        """Test model pruning"""
        # Prune model
        pruned_model = self.simple_optimizer.prune_model(
            method='l1_unstructured',
            amount=0.2
        )
        
        self.assertIsInstance(pruned_model, nn.Module)
        
        # Check if parameters are actually pruned
        pruned_params = 0
        total_params = 0
        for module in pruned_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = module.weight_mask
                pruned_params += torch.sum(mask == 0).item()
                total_params += mask.numel()
        
        self.assertGreater(pruned_params, 0)
    
    def test_knowledge_distillation(self):
        """Test knowledge distillation setup"""
        # Create teacher model
        teacher_model = SimpleModel().to(self.device)
        
        # Setup distillation
        distill_params = self.simple_optimizer.setup_knowledge_distillation(
            teacher_model,
            temperature=2.0,
            alpha=0.5
        )
        
        self.assertIn('teacher_model', distill_params)
        self.assertIn('temperature', distill_params)
        self.assertIn('alpha', distill_params)
        
        # Test distillation loss
        batch_size = 2
        x = torch.randn(batch_size, 3, 8, 8).to(self.device)
        
        student_outputs = self.simple_model(x)
        teacher_outputs = teacher_model(x)
        targets = torch.randint(0, 10, (batch_size,)).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        
        loss = self.simple_optimizer.knowledge_distillation_loss(
            student_outputs,
            teacher_outputs,
            targets,
            criterion,
            distill_params
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
    
    def test_model_ensemble(self):
        """Test model ensemble"""
        # Create multiple models
        models = [
            SimpleModel().to(self.device)
            for _ in range(3)
        ]
        
        # Create ensemble
        ensemble = ModelOptimizer.create_ensemble(models)
        
        self.assertIsInstance(ensemble, nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 3, 8, 8).to(self.device)
        output = ensemble(x)
        
        self.assertIsInstance(output, torch.Tensor)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_mixed_precision(self):
        """Test mixed precision setup"""
        # Setup mixed precision
        mp_params = self.optimizer.setup_mixed_precision()
        
        self.assertIn('scaler', mp_params)
        self.assertIn('enabled', mp_params)
        
        # Test mixed precision step
        x = torch.randn(2, 3, 64, 64).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        loss, output = ModelOptimizer.mixed_precision_step(
            self.model,
            x,
            criterion,
            optimizer,
            mp_params['scaler']
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
    
    def test_model_profiling(self):
        """Test model profiling"""
        # Profile model
        profile_results = self.simple_optimizer.profile_model(
            input_shape=(3, 8, 8),
            batch_size=2,
            n_iterations=10
        )
        
        expected_metrics = {
            'inference_time_ms',
            'memory_allocated_mb',
            'memory_cached_mb',
            'throughput'
        }
        
        self.assertEqual(set(profile_results.keys()), expected_metrics)
        
        for value in profile_results.values():
            self.assertIsInstance(value, float)
            self.assertGreater(value, 0)
    
    def test_model_restore(self):
        """Test model restoration"""
        # Modify model
        self.simple_optimizer.prune_model(amount=0.5)
        
        # Restore model
        restored_model = self.simple_optimizer.restore_original_model()
        
        # Check if parameters are restored
        for orig_param, restored_param in zip(
            self.simple_model.parameters(),
            restored_model.parameters()
        ):
            self.assertTrue(torch.equal(orig_param, restored_param))

if __name__ == '__main__':
    unittest.main()
