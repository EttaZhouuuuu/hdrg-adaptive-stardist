# Shape-Aware StarDist

An enhanced version of StarDist with shape-aware backbone network design, incorporating Transformer architecture and specialized shape attention mechanisms.

## Features

### 1. Shape-aware Backbone Network
- Transformer-based architecture for global context modeling
- Specialized shape attention module
- Multi-level feature aggregation mechanism
- Enhanced long-range dependency modeling

### 2. Innovative Shape-aware Hybrid Loss
- Distance and Probability Loss with adaptive weighting
- Shape Consistency Loss
- Boundary Smoothness Loss
- Adversarial Loss

### 3. Advanced Training Pipeline
- Adaptive loss weighting mechanism
- Mixed precision training support
- Knowledge distillation capabilities
- Model ensemble support

## Installation

### Local Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/shape-aware-stardist.git
cd shape-aware-stardist

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup
We provide a Colab notebook for easy training with GPU support. See [train_on_colab.ipynb](colab/train_on_colab.ipynb).

## Project Structure
```
shape_aware_stardist/
├── models/                    # Model implementations
│   ├── adaptive_shape_encoder.py
│   ├── transformer_block.py
│   ├── attention_module.py
│   └── feature_aggregation.py
├── training/                  # Training components
│   ├── trainer.py
│   └── shape_aware_loss.py
├── data/                      # Data handling
│   ├── dataset.py
│   └── prepare_data.py
├── evaluation/                # Evaluation metrics
│   └── metrics.py
├── visualization/             # Visualization tools
│   └── visualizer.py
├── configs/                   # Configuration files
│   ├── config.py
│   └── data_config.py
└── colab/                    # Colab training scripts
    └── train_on_colab.ipynb
```

## Usage

### Data Preparation
```python
from shape_aware_stardist.data.prepare_data import prepare_dataset

# Prepare training data
prepare_dataset(source_dir='path/to/data', split='train')

# Verify dataset
prepare_dataset(split='train', verify_only=True)
```

### Training
```python
from shape_aware_stardist.models import AdaptiveShapeEncoder
from shape_aware_stardist.training import ShapeAwareTrainer

# Create model
model = AdaptiveShapeEncoder(
    in_channels=3,
    n_rays=32,
    base_channels=64
)

# Train model
trainer = ShapeAwareTrainer(model=model, ...)
trainer.train(num_epochs=100)
```

### Inference
```python
# Load trained model
model.load_state_dict(torch.load('path/to/weights.pth'))

# Run inference
outputs = model(images)
```

## Training on Google Colab

1. Upload data to Google Drive
2. Open [train_on_colab.ipynb](colab/train_on_colab.ipynb) in Colab
3. Follow the notebook instructions
4. Results will be saved to your Drive

## Model Architecture

### Shape-aware Backbone
- Transformer-based feature extraction
- Deformable convolutions
- Multi-scale feature processing
- Shape attention mechanisms

### Loss Function Components
1. Distance and Probability Loss
   - Robust Huber loss
   - Focal loss for class imbalance
2. Shape Consistency Loss
   - Ray-direction consistency
   - Spatial gradient consistency
3. Boundary Smoothness Loss
   - Edge-aware regularization
   - Adaptive smoothness constraints
4. Adversarial Loss
   - Shape-specific discriminator
   - Multi-scale feature matching

## Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU support)
- Other dependencies in requirements.txt

## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{shape_aware_stardist,
  title={Shape-Aware StarDist: Enhanced Instance Segmentation with Shape-Guided Features},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Original StarDist implementation
- Transformer architecture designs
- Shape analysis techniques