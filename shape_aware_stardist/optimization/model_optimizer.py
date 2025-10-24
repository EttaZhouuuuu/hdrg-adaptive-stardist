import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.quantization import quantize_dynamic
from torch.nn.utils import prune
import copy

class ModelOptimizer:
    """
    Model optimization and performance enhancement utility.
    
    Features:
    - Model quantization
    - Pruning
    - Knowledge distillation
    - Model ensemble
    - FP16 mixed precision training
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
        """
        self.model = model
        self.original_model = copy.deepcopy(model)
    
    def quantize_model(
        self,
        dtype: Optional[torch.dtype] = torch.qint8
    ) -> nn.Module:
        """
        Quantize model to reduce memory usage and improve inference speed.
        
        Args:
            dtype: Quantization data type
            
        Returns:
            Quantized model
        """
        # Configure quantization
        quantized_model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )
        
        return quantized_model
    
    def prune_model(
        self,
        method: str = 'l1_unstructured',
        amount: float = 0.2,
        parameters: Optional[List[Tuple[nn.Module, str]]] = None
    ) -> nn.Module:
        """
        Prune model parameters to reduce model size.
        
        Args:
            method: Pruning method ('l1_unstructured', 'random_unstructured')
            amount: Amount of parameters to prune (0-1)
            parameters: Optional list of (module, name) tuples to prune
            
        Returns:
            Pruned model
        """
        if parameters is None:
            parameters = [
                (module, 'weight')
                for module in self.model.modules()
                if isinstance(module, (nn.Conv2d, nn.Linear))
            ]
        
        # Apply pruning
        for module, name in parameters:
            if method == 'l1_unstructured':
                prune.l1_unstructured(module, name=name, amount=amount)
            elif method == 'random_unstructured':
                prune.random_unstructured(module, name=name, amount=amount)
            else:
                raise ValueError(f"Unknown pruning method: {method}")
        
        return self.model
    
    def setup_knowledge_distillation(
        self,
        teacher_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5
    ) -> Dict:
        """
        Setup knowledge distillation training.
        
        Args:
            teacher_model: Teacher model for distillation
            temperature: Temperature for softening probability distributions
            alpha: Weight for balancing student and distillation loss
            
        Returns:
            Dictionary with distillation parameters
        """
        return {
            'teacher_model': teacher_model,
            'temperature': temperature,
            'alpha': alpha
        }
    
    def knowledge_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        distill_params: Dict
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            targets: Ground truth targets
            criterion: Loss criterion
            distill_params: Distillation parameters
            
        Returns:
            Combined loss
        """
        temperature = distill_params['temperature']
        alpha = distill_params['alpha']
        
        # Student loss
        student_loss = criterion(student_outputs, targets)
        
        # Distillation loss
        soft_targets = (teacher_outputs / temperature).softmax(dim=1)
        soft_prob = (student_outputs / temperature).softmax(dim=1)
        distillation_loss = torch.sum(-soft_targets * torch.log(soft_prob)) * (temperature ** 2)
        
        # Combined loss
        loss = alpha * student_loss + (1 - alpha) * distillation_loss
        
        return loss
    
    @staticmethod
    def create_ensemble(models: List[nn.Module]) -> nn.Module:
        """
        Create model ensemble.
        
        Args:
            models: List of models for ensemble
            
        Returns:
            Ensemble model
        """
        class ModelEnsemble(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models)
            
            def forward(self, x):
                outputs = [model(x) for model in self.models]
                if isinstance(outputs[0], tuple):
                    # Handle multiple outputs
                    num_outputs = len(outputs[0])
                    ensemble_outputs = []
                    for i in range(num_outputs):
                        output_i = torch.stack([out[i] for out in outputs])
                        ensemble_outputs.append(torch.mean(output_i, dim=0))
                    return tuple(ensemble_outputs)
                else:
                    # Single output
                    outputs = torch.stack(outputs)
                    return torch.mean(outputs, dim=0)
        
        return ModelEnsemble(models)
    
    def setup_mixed_precision(self) -> Dict:
        """
        Setup mixed precision training.
        
        Returns:
            Dictionary with mixed precision settings
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for mixed precision training")
        
        from torch.cuda.amp import GradScaler
        
        return {
            'scaler': GradScaler(),
            'enabled': True
        }
    
    @staticmethod
    def mixed_precision_step(
        model: nn.Module,
        data: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one mixed precision training step.
        
        Args:
            model: PyTorch model
            data: Input data
            criterion: Loss criterion
            optimizer: Optimizer
            scaler: Gradient scaler
            
        Returns:
            tuple: (loss, output)
        """
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        return loss, output
    
    def profile_model(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        n_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Profile model performance.
        
        Args:
            input_shape: Input tensor shape
            batch_size: Batch size for profiling
            n_iterations: Number of iterations
            
        Returns:
            Dictionary with profiling results
        """
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        
        # Warm up
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # Profile
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(n_iterations):
            _ = self.model(dummy_input)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / n_iterations
        
        # Calculate memory usage
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_cached = torch.cuda.memory_reserved(device)
        
        return {
            'inference_time_ms': elapsed_time,
            'memory_allocated_mb': memory_allocated / 1024 / 1024,
            'memory_cached_mb': memory_cached / 1024 / 1024,
            'throughput': batch_size / (elapsed_time / 1000)
        }
    
    def restore_original_model(self) -> nn.Module:
        """Restore model to its original state."""
        self.model = copy.deepcopy(self.original_model)
        return self.model
