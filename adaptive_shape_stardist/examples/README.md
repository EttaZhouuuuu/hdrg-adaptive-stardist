# Adaptive Shape StarDist - Examples

This directory contains example scripts demonstrating how to use Adaptive Shape StarDist.

## Examples

### 1. Training Example (`train_example.py`)

Shows how to train the model on your custom dataset.

```bash
python train_example.py
```

Key steps:
- Load training data
- Configure the model
- Train with validation
- Save the trained model

### 2. Inference Example (`inference_example.py`)

Shows how to use a trained model for prediction.

```bash
python inference_example.py
```

Key steps:
- Load a trained model
- Predict on new images
- Visualize results

### 3. Comparison Example (`compare_with_stardist.py`)

Compare Adaptive Shape StarDist with original StarDist.

```bash
python compare_with_stardist.py
```

## Quick Start

### Basic Training

```python
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.configs import get_default_config

# Configure
config = get_default_config()

# Create model
model = AdaptiveShapeStarDist(config, name='my_model', basedir='./models')

# Train
history = model.train_model(X_train, Y_train, validation_data=(X_val, Y_val))

# Save
model.save_model()
```

### Basic Inference

```python
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.inference import AdaptiveShapePredictor

# Load model
model = AdaptiveShapeStarDist.load_model('./models/my_model')

# Create predictor
predictor = AdaptiveShapePredictor(model, prob_thresh=0.5)

# Predict
labels, details = predictor.predict(image, return_details=True)
```

## Custom Configuration

Create a custom configuration for your specific use case:

```python
from adaptive_shape_stardist.models import AdaptiveShapeConfig

config = AdaptiveShapeConfig(
    n_channel_in=1,
    backbone='unet',
    min_sampling_points=64,
    max_sampling_points=256,
    use_shape_prior=True,
    num_shape_prototypes=32,
    train_epochs=100,
    train_batch_size=4,
)
```

## Tips

1. **For irregular shapes**: Use `get_high_complexity_config()` with more sampling points
2. **For fast inference**: Use `get_fast_config()` with a lightweight backbone
3. **For multi-channel images**: Adjust `n_channel_in` in the configuration
4. **Data augmentation**: Implement custom augmenter function for better generalization

