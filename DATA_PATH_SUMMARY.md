# Data Path Configuration Summary

## 📂 Data Structure Found in Original Files

From analyzing `pipeline.sh`, `train_seg_neuron.py`, and `infer_main_2d.py`:

### Original Pipeline Configuration
- **DATA_PATH**: `/hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/`
- **SLIDE**: `240819_Ji_N1_H_EScan`
- **Structure**:
  ```
  /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/
  └── 240819_Ji_N1_H_EScan/
      ├── patches_2048/           # Input images (.png)
      │   ├── patch_col_00_row_00.png
      │   ├── patch_col_00_row_01.png
      │   └── ...
      └── pseudo_gt_masks_2048/   # Ground truth masks (.png)
          ├── patch_col_00_row_00.png
          ├── patch_col_00_row_01.png
          └── ...
  ```

### Key Characteristics
- **Format**: PNG images (not TIFF)
- **Size**: 2048x2048 pixels per patch
- **Naming**: `patch_col_XX_row_YY.png`
- **Masks**: Binary values 0 (background) and 255 (object)
- **Loading**: Uses `mask // 255` to convert to 0/1 binary

## 🔧 Updated Files for Correct Data Paths

### 1. Training Script (`scripts/train_adaptive_stardist.py`)
**Updated**: `load_training_data()` function
- ✅ Supports multiple slides: `--slides slide1,slide2`
- ✅ Looks for `patches_2048/` and `pseudo_gt_masks_2048/` directories
- ✅ Processes `.png` files with correct naming pattern
- ✅ Uses `normalize_mi_ma(img, 0, 255)` following original
- ✅ Uses `mask // 255` for binary conversion

### 2. Inference Script (`scripts/run_adaptive_inference.py`)
**Updated**: File glob patterns and help text
- ✅ Supports both `.png` and `.tiff` files
- ✅ Updated help text to mention correct path structure

### 3. Test Setup (`scripts/test_setup.py`)
**Updated**: Complete rewrite to match original structure
- ✅ Creates slides: `240819_Ji_N1_H_EScan`, `test_slide_1`, `test_slide_2`, `scale_variation_slide`
- ✅ Uses correct directory structure: `DATA_PATH/SLIDE/patches_2048/`
- ✅ Generates PNG files with correct naming: `patch_col_XX_row_YY.png`
- ✅ Creates strictly binary masks: 0 and 255 only

### 4. Test Pipeline (`scripts/test_pipeline.py`)
**Updated**: Data loading paths
- ✅ Uses `data/240819_Ji_N1_H_EScan/patches_2048/*.png`
- ✅ Uses `data/240819_Ji_N1_H_EScan/pseudo_gt_masks_2048/*.png`

### 5. Experiment Configuration (`scripts/experiments/experiment_config.py`)
**Updated**: DatasetConfig and experiment datasets
- ✅ Added `slides` field to `DatasetConfig`
- ✅ Updated all experiments to use proper slide names
- ✅ Base path is `./data` with specific slides

### 6. Experiment Runner (`scripts/experiments/experiment_runner.py`)
**Updated**: Dataset preparation and evaluation
- ✅ `prepare_dataset()` loads from multiple slides
- ✅ `evaluate_model()` uses `pseudo_gt_masks_2048/` for ground truth
- ✅ Handles original directory structure

## 🚀 Correct Usage Examples

### Training
```bash
python scripts/train_adaptive_stardist.py \
    --data_path ./data \
    --slides 240819_Ji_N1_H_EScan,test_slide_1 \
    --ckpt_path ./models/adaptive \
    --n_epochs 50
```

### Inference
```bash
python scripts/run_adaptive_inference.py \
    --model_path ./models/adaptive \
    --input_path ./data/240819_Ji_N1_H_EScan/patches_2048 \
    --output_dir ./results
```

### Evaluation
```bash
python scripts/evaluate_results.py \
    --pred_dir ./results \
    --true_dir ./data/240819_Ji_N1_H_EScan/pseudo_gt_masks_2048 \
    --output_dir ./evaluation
```

### Experiments
```bash
python -m scripts.experiments.experiment_runner \
    --experiment_name basic_comparison \
    --data_path ./data
```

## 🔍 Verification Tools

### 1. Demo Script
```bash
python scripts/demo_usage.py
```
- Shows data structure verification
- Displays example commands
- Validates mask format

### 2. Real Data Helper
```bash
python scripts/run_with_real_data.py --action train --data_path /your/data/path
```
- Helps with real HPC data paths
- Provides command templates
- Validates data structure

### 3. Test Data Generation
```bash
python scripts/test_setup.py
```
- Creates synthetic data with correct structure
- Generates proper PNG files
- Uses exact naming convention

## ✅ Key Changes Made

1. **Data Structure**: Changed from generic `data/experiment_type/` to specific `data/SLIDE_NAME/patches_2048/`
2. **File Format**: Ensured PNG support instead of TIFF-only
3. **Naming Convention**: Uses `patch_col_XX_row_YY.png` pattern
4. **Mask Format**: Strictly binary 0/255 values
5. **Multi-slide Support**: Can handle multiple slides in single training run
6. **Path Consistency**: All scripts now use the same path structure

## 🎯 Benefits

- **Drop-in Compatibility**: Works with original hDRG data structure
- **No Data Migration**: Can use existing data directly
- **Consistent Interface**: All new scripts follow same path convention
- **Easy Testing**: Synthetic data matches real data structure exactly
- **Flexible**: Supports both test data and real HPC data paths
