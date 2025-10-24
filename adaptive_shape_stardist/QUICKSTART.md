# Quick Start Guide

## Installation

### 1. Clone or Download the Project

```bash
cd adaptive_shape_stardist
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### 3. Verify Installation

```python
import adaptive_shape_stardist as ass
print(f"Version: {ass.__version__}")
```

## Basic Usage

### Training a Model

```python
import numpy as np
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.configs import get_default_config

# 1. Prepare your data
# X_train: [N, H, W] or [N, H, W, C] - training images
# Y_train: [N, H, W] - instance labels (0=background, 1,2,3...=instances)

# 2. Configure the model
config = get_default_config()
config.train_epochs = 100
config.train_batch_size = 4

# 3. Create model
model = AdaptiveShapeStarDist(config, name='my_model', basedir='./models')

# 4. Train
history = model.train_model(
    X_train, Y_train,
    validation_data=(X_val, Y_val)
)

# 5. Save
model.save_model()
```

### Running Inference

```python
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.inference import AdaptiveShapePredictor

# 1. Load trained model
model = AdaptiveShapeStarDist.load_model('./models/my_model')

# 2. Create predictor
predictor = AdaptiveShapePredictor(model, prob_thresh=0.5)

# 3. Predict
labels, details = predictor.predict(image, return_details=True)

# 4. Get results
print(f"Detected {len(np.unique(labels))-1} instances")
print(f"Shape complexity: {details['complexity']}")
```

### Visualization

```python
from adaptive_shape_stardist.utils import visualize_predictions

# Visualize results
fig = visualize_predictions(
    image=image,
    labels=labels,
    prob=details['prob'],
    sampling_points=details['sampling_points']
)

plt.savefig('results.png')
```

## Pre-configured Settings

### For Highly Irregular Shapes

```python
from adaptive_shape_stardist.configs import get_high_complexity_config

config = get_high_complexity_config()
# Uses 64-256 sampling points
# More prototypes (32)
# Deeper network
```

### For Fast Inference

```python
from adaptive_shape_stardist.configs import get_fast_config

config = get_fast_config()
# Uses 16-64 sampling points
# Lightweight backbone
# No shape prior
```

### For Multi-Channel Images

```python
from adaptive_shape_stardist.configs import get_multi_channel_config

config = get_multi_channel_config(n_channels=3)
# For RGB or multi-channel fluorescence
```

## Examples

Run the example scripts:

```bash
# Training example
cd examples
python train_example.py

# Inference example
python inference_example.py
```

## Tips for Best Results

### 1. Data Preparation

- **Normalize images**: Zero mean, unit variance
- **Instance labels**: Each cell has unique ID, background = 0
- **Sufficient data**: 50+ images recommended, 200+ for best results

### 2. Training

- **Start with default config**: Then adjust based on results
- **Use validation set**: Monitor for overfitting
- **Data augmentation**: Improves generalization
  ```python
  def augmenter(img, label):
      # Add rotation, flipping, etc.
      return img_aug, label_aug
  
  model.train_model(X, Y, augmenter=augmenter)
  ```

### 3. Hyperparameter Tuning

- **More irregular shapes** → Increase `max_sampling_points`
- **Densely packed cells** → Increase `num_shape_prototypes`
- **Large images** → Use grid subsampling: `grid=(2, 2)`
- **Limited GPU memory** → Reduce `train_batch_size`

### 4. Inference

- **Adjust thresholds**:
  - `prob_thresh`: Higher = fewer but more confident detections
  - `nms_thresh`: Lower = less overlap between instances

## Troubleshooting

### Out of Memory

1. Reduce `train_batch_size`
2. Use grid subsampling: `config.grid = (2, 2)`
3. Reduce `max_sampling_points`

### Poor Segmentation

1. Check data quality and labels
2. Increase training epochs
3. Use data augmentation
4. Adjust `max_sampling_points` based on shape complexity

### Training Too Slow

1. Use `get_fast_config()`
2. Reduce `train_steps_per_epoch`
3. Use GPU acceleration
4. Disable shape prior: `config.use_shape_prior = False`

## Next Steps

- Read [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for architecture details
- See [examples/](examples/) for more comprehensive examples
- Check [README.md](README.md) for full documentation

## Support

- **Issues**: https://github.com/yourusername/adaptive-shape-stardist/issues
- **Documentation**: See README.md and TECHNICAL_REPORT.md
- **Examples**: Check the examples/ directory

Happy segmenting! 🔬🧬

