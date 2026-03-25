# Baseline Methods Evaluation Report

## Executive Summary

This report presents a comprehensive evaluation of baseline instance segmentation methods on the Xenium spatial transcriptomics dataset. We compared three approaches: **U-Net with Watershed post-processing**, **Standalone Watershed**, and **Cellpose/Cellpose-SAM**. Our experiments demonstrate that trained U-Net significantly outperforms traditional image processing methods, achieving **97.6% pixel-level IoU** while highlighting the challenges of instance-level segmentation.

---

## 1. Experimental Setup

### 1.1 Dataset Description

| Property | Value |
|----------|-------|
| **Source** | Xenium spatial transcriptomics |
| **Full Image Size** | 33,131 × 48,358 pixels |
| **Total Cells** | 162,033 |
| **Mean Vertices per Cell** | 13.0 |
| **Image Data Type** | uint16 |
| **Output Directory** | `baselines/cellpose_results/` |

### 1.2 Evaluation Protocol

- **Patch Size**: 256 × 256 pixels
- **Number of Test Samples**: 21 patches
- **Evaluation Metrics**:
  - **Pixel IoU** (Intersection over Union at pixel level)
  - **Instance IoU** (Average best-match IoU between predicted and ground truth instances)
  - **AJI** (Aggregate Jaccard Index - standard metric for instance segmentation)

### 1.3 Hardware Configuration

- **GPU**: Available (CUDA)
- **Processing**: CPU fallback available for all methods

---

## 2. Baseline Methods

### 2.1 Method 1: U-Net with Watershed Post-processing

#### Architecture
- **Encoder**: ResNet34 backbone (pretrained on ImageNet)
- **Decoder**: U-Net style with skip connections
- **Output**: Binary segmentation mask (cell vs. background)

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 100 |
| **Batch Size** | 8 |
| **Learning Rate** | 1e-4 |
| **Loss Function** | BCE + Dice Loss |
| **Optimizer** | Adam |
| **Input Size** | 256 × 256 |

#### Post-processing Pipeline
```python
# Watershed post-processing for instance extraction
1. Binary thresholding (Otsu's method)
2. Morphological opening (remove small noise)
3. Distance transform (EDT)
4. Local maxima detection (cell centers)
5. Watershed segmentation with compactness
```

#### Results

| Metric | Mean | Std Dev |
|--------|------|---------|
| **Pixel IoU** | 0.9761 | - |
| **Pixel Dice** | 0.9879 | - |
| **Instance IoU** | 0.0869 | - |
| **Instance Dice** | 0.1439 | - |
| **Training Time** | ~8 minutes | (57 epochs) |

**Interpretation**:
- Pixel-level performance is excellent (97.6% IoU)
- Instance-level performance is limited (8.7% IoU)
- The gap suggests the Watershed post-processing is the bottleneck

---

### 2.2 Method 2: Standalone Watershed (Traditional Image Processing)

#### Algorithm
A classical image segmentation approach using gradient information and marker-controlled watershed:

1. **Gaussian smoothing** (σ=1.0) for denoising
2. **Intensity thresholding** (Otsu's method or fixed threshold)
3. **Morphological opening** for noise removal
4. **Euclidean Distance Transform (EDT)**
5. **Peak local maxima** for cell center detection
6. **Watershed segmentation** for instance separation

#### Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `diameter` | 30.0 | Expected cell diameter in pixels |
| `threshold` | 0.5 | Intensity threshold for binarization |
| `min_distance` | 3 | Minimum distance between cell centers |
| `compactness` | 0.001 | Watershed compactness parameter |

#### Results

| Metric | Mean | Std Dev |
|--------|------|---------|
| **Pixel IoU** | 0.0136 | 0.0105 |
| **Instance IoU** | 0.0421 | 0.0380 |
| **AJI** | 0.0366 | 0.0227 |
| **Avg Pred/Cell** | 2.8 | - |
| **Avg GT/Cell** | 10.9 | - |
| **Processing Time** | ~0.4 seconds | (21 samples) |

**Interpretation**:
- Very low pixel-level performance (1.4%)
- Slightly better than random at instance level (4.2%)
- Significantly under-segments (predicts 2.8 cells vs 10.9 ground truth)
- No learning capability; purely data-agnostic

---

### 2.3 Method 3: Cellpose / Cellpose-SAM

#### Description
Cellpose is a generalist segmentation model that can be used with or without the SAM (Segment Anything Model) backbone.

| Variant | Description | Pros | Cons |
|---------|------------|------|------|
| **Cellpose (v3.x)** | Traditional Cellpose model | No external downloads | Less generalizable |
| **Cellpose-SAM (v4.x)** | Cellpose with SAM backbone | State-of-the-art generalization | Requires model download (~1GB) |

#### Attempted Execution

```python
# Standard Cellpose usage
from cellpose import models
model = models.CellposeModel(gpu=True, model_type="cyto2")
masks, flows, styles = model.eval(image)
```

#### Status: ⚠️ Blocked by Network Restrictions

| Issue | Details |
|-------|---------|
| **Error Type** | DNS resolution failure (`socket.gaierror`) |
| **Cause** | Network connectivity issues in server environment |
| **Impact** | Unable to download Cellpose-SAM model weights |
| **Fallback** | Using standard Cellpose (v3.x) not viable due to v4.x auto-download |

#### Recommendation for Future Work

1. **Local Model Deployment**: Transfer pre-downloaded model weights to the server
2. **Alternative Installation**: Use Docker container with pre-installed Cellpose
3. **Manual Weight Download**: Download weights from official repository on a connected machine

---

## 3. Comparative Analysis

### 3.1 Performance Comparison Table

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Method              │  Pixel IoU  │  Instance IoU  │    AJI   │  Relative  │
├─────────────────────────────────────────────────────────────────────────────┤
│  U-Net + Watershed  │   0.9761    │     0.0869     │     -    │   BASELINE │
│  Watershed Only     │   0.0136    │     0.0421     │  0.0366  │    72x✗    │
│  Cellpose-SAM       │     N/A     │       N/A      │    N/A   │   Unknown  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Findings

#### Finding 1: Learning is Essential for This Dataset

The Xenium morphological data presents unique challenges:
- **Complex cell morphology** with irregular shapes
- **Dense cell populations** with touching instances
- **Variable cell sizes** across the tissue

Traditional image processing (Watershed) fails because:
- Fixed parameters cannot adapt to morphological variations
- No learned features to distinguish cells from background
- Sensitive to intensity variations and noise

#### Finding 2: Pixel-Level vs Instance-Level Discrepancy

| Method | Pixel IoU | Instance IoU | Gap |
|--------|-----------|---------------|-----|
| U-Net + Watershed | 97.6% | 8.7% | 88.9% |

**Analysis**:
1. The U-Net encoder-decoder effectively learns pixel-level features
2. However, the Watershed post-processing cannot reliably separate instances
3. **Bottleneck**: The post-processing step limits instance segmentation performance

#### Finding 3: Instance Under-segmentation

| Method | Predicted/GT Ratio |
|--------|-------------------|
| U-Net + Watershed | ~0.5-1.0 (variable) |
| Watershed Only | 0.26 (2.8 / 10.9) |

Both methods tend to under-segment, likely due to:
- Touching cells merged into single instances
- Overly strict instance separation criteria
- Incomplete distance transform information

---

## 4. Recommendations

### 4.1 Short-term Improvements

1. **Optimize Watershed Parameters**
   ```python
   # Try adaptive parameters based on cell density
   diameter = estimate_cell_size(image_patch)
   min_distance = diameter // 4
   ```

2. **Use Alternative Post-processing**
   - **Connected Components** on high-confidence regions
   - **Shape priors** from training data
   - **Conditional Random Fields (CRF)** for boundary refinement

3. **Multi-scale Processing**
   - Detect cells at multiple scales
   - Use hierarchical segmentation for overlapping instances

### 4.2 Medium-term Improvements

1. **Advanced Instance Segmentation**
   - **Mask R-CNN**: Joint detection and segmentation
   - **YOLACT**: Real-time instance segmentation
   - **SOLOv2**: Anchor-free instance segmentation

2. **Better Training Strategies**
   - Hard negative mining
   - Instance-aware loss functions
   - Curriculum learning (easy → hard examples)

3. **Ensemble Methods**
   - Combine U-Net with traditional methods
   - Voting-based instance merging

### 4.3 Long-term Improvements

1. **Integration with SAM**
   - Use SAM (Segment Anything Model) for mask refinement
   - Trainable SAM adapters for domain adaptation

2. **Transformer-based Methods**
   - **SegFormer**: Efficient semantic segmentation
   - **Mask2Former**: Universal image segmentation
   - **Open-vocabulary** segmentation for novel cell types

3. **3D Segmentation**
   - If z-stack data available
   - Leverage continuity across z-planes

---

## 5. Reproducibility

### 5.1 Running the Experiments

```bash
# U-Net + Watershed Baseline
python baselines/unet_watershed_baseline.py --mode evaluate --max_samples 21

# Watershed Only (No training required)
python baselines/watershed_baseline.py --max_samples 21

# Cellpose-SAM (Requires network access)
python baselines/run_cellpose_sam.py --max_samples 21
```

### 5.2 Dependencies

```txt
# Core dependencies
torch>=2.0
torchvision>=0.15
numpy>=1.24
scipy>=1.10
scikit-image>=0.21
tqdm>=4.65

# Optional dependencies
tifffile>=2023.3
matplotlib>=3.7
cellpose>=4.0  # For Cellpose-SAM baseline
```

### 5.3 Output Files

| File | Description |
|------|-------------|
| `baselines/cellpose_results/metrics.json` | Watershed metrics |
| `baselines/unet_watershed/metrics.json` | U-Net metrics |
| `baselines/cellpose_results/masks/*.tif` | Predicted instance masks |

---

## 6. Conclusion

This evaluation demonstrates that **learning-based methods significantly outperform traditional image processing** for instance segmentation on Xenium morphological data. The U-Net achieves **97.6% pixel-level accuracy** but struggles with **8.7% instance-level IoU**, indicating that post-processing is the current bottleneck.

The primary recommendations are:
1. **Improve post-processing** to better separate touching instances
2. **Explore advanced architectures** (Mask R-CNN, transformers)
3. **Integrate SAM** for mask refinement when network access is available

Future work should focus on:
- Training domain-specific instance segmentation models
- Leveraging spatial transcriptomics information
- Developing multi-modal segmentation approaches

---

## 7. References

1. **U-Net**: Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

2. **Cellpose**: Stringer, C., et al. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods.

3. **SAM**: Kirillov, A., et al. (2023). Segment Anything. ICCV.

4. **Watershed**: Beucher, S., & Meyer, F. (1992). The morphological approach to segmentation: the watershed transformation. Mathematical Morphology in Image Processing.

---

*Report generated on: February 2026*  
*Author: Adaptive StarDist Project Team*
