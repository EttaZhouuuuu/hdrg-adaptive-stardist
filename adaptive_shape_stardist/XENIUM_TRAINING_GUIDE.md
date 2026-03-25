# Xenium Mouse Brain Data: Training Guide

This guide explains how to use the Xenium mouse brain data with Adaptive Shape StarDist for training.

## 📊 Data Overview

### Xenium Output Files

| File | Description | Size |
|------|-------------|------|
| `cells.zarr` | Cell polygon segmentation | 162,033 cells |
| `morphology.ome.tif` | H&E morphology image | ~3.2 GB |

### Xenium Data Structure

The `cells.zarr` file contains polygon-based cell segmentation:

```
cells.zarr/
├── cell_id               (162033,) uint32     - Unique cell IDs (1-162033)
├── polygon_vertices      (2, 162033, 26) f32  - Polygon vertices [z, cell, vertex]
├── polygon_num_vertices  (2, 162033) i32      - Vertices per polygon
└── cell_summary          (162033, 7) f64      - [x, y, area, eccentricity, major_axis, minor_axis, orientation]

Polygon format: interleaved [y0, x0, y1, x1, y2, x2, ...]
```

## 🚀 Quick Start

### 1. Verify Environment

```bash
# Activate conda environment
conda activate /data/common/conda_envs_shared/yitong.zhou/adaptive_shape_stardist

# Verify packages
python -c "import adaptive_shape_stardist; print('✓ Package:', adaptive_shape_stardist.__file__)"
```

### 2. Explore Data

```bash
python scripts/quick_explore.py
```

### 3. Create Training Dataset

```bash
python -c "
from pathlib import Path
from utils.xenium_preprocessing import create_training_dataset

DATA_DIR = Path('data')

# Create training dataset (500 patches)
images, labels = create_training_dataset(
    cells_zarr_path=str(DATA_DIR / 'cells.zarr'),
    image_path=str(DATA_DIR / 'morphology.ome.tif'),
    output_path=str(DATA_DIR / 'training_data.npz'),
    patch_size=256,
    num_patches=500
)

print(f'✓ Training data: {images.shape}, {labels.shape}')
"
```

### 4. Train Model

```bash
python scripts/train_xenium_data.py \
    --patches 500 \
    --epochs 50 \
    --batch-size 2 \
    --backbone resnet50
```

## 📁 File Structure

```
adaptive_shape_stardist/
├── data/
│   ├── cells.zarr/              # Xenium cell segmentation
│   ├── morphology.ome.tif       # H&E image (corrupted)
│   └── training_data.npz        # Generated training patches
├── scripts/
│   ├── quick_explore.py         # Explore Xenium data structure
│   ├── train_xenium_data.py     # Main training script
│   └── explore_xenium_data.py   # Detailed analysis
├── utils/
│   ├── xenium_preprocessing.py   # Data preprocessing module
│   ├── visualization.py
│   └── __init__.py
└── models/                      # Saved models
```

## 🔧 Data Preprocessing

### Loading Xenium Data

```python
from utils.xenium_preprocessing import XeniumDataLoader

# Initialize loader
loader = XeniumDataLoader(
    cells_zarr_path='data/cells.zarr',
    image_path='data/morphology.ome.tif',
    load_image=False  # Set True if image loads correctly
)

# Access data
print(f"Cells: {len(loader.cell_id)}")
print(f"Image size: {loader.image_height} x {loader.image_width}")
```

### Converting Polygons to Masks

```python
# Single cell polygon to binary mask
mask = loader.polygon_to_mask(cell_idx=0, plane=0)
print(f"Mask shape: {mask.shape}, Cell area: {mask.sum()} px")

# Get cell bounding box
bbox = loader.get_cell_bbox(cell_idx=0, plane=0, padding=10)
print(f"Bounding box: {bbox}")
```

### Creating Training Patches

```python
from utils.xenium_preprocessing import create_training_dataset

images, labels = create_training_dataset(
    cells_zarr_path='data/cells.zarr',
    image_path='data/morphology.ome.tif',
    output_path='data/training_data.npz',
    patch_size=256,    # Patch dimensions
    num_patches=1000,  # Number of patches
    seed=42            # Random seed
)
```

## 🏋️ Training

### Basic Training

```python
from scripts.train_xenium_data import load_xenium_data, train_xenium_model

# Load data
X_train, Y_train, X_val, Y_val = load_xenium_data(
    cells_zarr_path='data/cells.zarr',
    image_path='data/morphology.ome.tif',
    patch_size=256,
    num_train_patches=500,
    num_val_patches=50
)

# Train
model, history = train_xenium_model(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    backbone='resnet50',
    epochs=50,
    batch_size=2
)
```

### Command Line Training

```bash
# Quick test (100 patches, 10 epochs)
python scripts/train_xenium_data.py --patches 100 --epochs 10

# Full training (1000 patches, 100 epochs)
python scripts/train_xenium_data.py \
    --patches 1000 \
    --epochs 100 \
    --batch-size 2 \
    --backbone resnet50 \
    --lr 1e-4
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--patches` | 500 | Number of training patches |
| `--val-patches` | 50 | Number of validation patches |
| `--patch-size` | 256 | Patch dimensions (square) |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--backbone` | resnet50 | Backbone network |

## ⚠️ Known Issues

### morphology.ome.tif Loading Error

The H&E morphology image (`morphology.ome.tif`) has JPEG2000 compression issues:

```
TiffFileError: opj_decode or opj_end_decompress failed
```

**Solutions:**

1. **Re-export from Xenium**: Ask for a re-exported version with standard compression
2. **Use alternative viewer**: Open in napari or Bio-Formats
3. **Synthetic data**: The training script falls back to synthetic data

### Memory Issues

For large-scale training:

```python
# Reduce patch size for limited GPU memory
patch_size = 128  # Instead of 256

# Process in batches
num_patches = 500  # Instead of 1000
```

## 📈 Expected Results

- **Training samples**: 162,033 cells from Xenium
- **Image size**: 7065 x 9985 pixels
- **Average vertices**: 13 per cell polygon
- **Typical training**: 50 epochs for convergence

## 🔮 Next Steps

1. **Fix morphology image**: Get properly compressed version
2. **Add data augmentation**: H&E-specific augmentation (color jitter, elastic deformation)
3. **Hyperparameter tuning**: Adjust sampling points, learning rate schedule
4. **Inference**: Use trained model on new H&E images

## 📞 Support

For issues with:
- **Xenium data**: Check Xenium documentation
- **StarDist training**: See `examples/train_example.py`
- **Model configuration**: See `configs/config.py`

