# Adaptive Shape StarDist - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
│                    [Batch, Height, Width, Channels]              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKBONE NETWORK                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │   U-Net    │  │  ResNet    │  │   Custom   │                │
│  │  Encoder   │  │   Blocks   │  │  Backbone  │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│         │                │                │                      │
│         └────────────────┴────────────────┘                      │
│                         │                                        │
│              Multi-Scale Features                                │
│         [B, H/s, W/s, C1], [B, H/2s, W/2s, C2]                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              SHAPE PRIOR ENCODER (Optional)                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │  1. 2D Positional Encoding                       │           │
│  │     • Sinusoidal position embeddings             │           │
│  │                                                   │           │
│  │  2. Learnable Shape Prototypes                   │           │
│  │     • [num_prototypes, embedding_dim]            │           │
│  │     • Domain-specific shape knowledge            │           │
│  │                                                   │           │
│  │  3. Transformer Encoder Layers (×N)              │           │
│  │     • Multi-Head Self-Attention                  │           │
│  │     • Feed-Forward Networks                      │           │
│  │     • Layer Normalization                        │           │
│  │                                                   │           │
│  │  4. Cross-Attention Fusion                       │           │
│  │     • Prototypes as Keys/Values                  │           │
│  │     • Features as Queries                        │           │
│  └──────────────────────────────────────────────────┘           │
│                         │                                        │
│              Prior-Enhanced Features                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ADAPTIVE SHAPE ENCODER                              │
│  ┌──────────────────────────────────────────────────┐           │
│  │  1. Deformable Convolution Blocks (×3)           │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Offset Prediction Network   │              │           │
│  │     │  • Conv2D → offsets         │              │           │
│  │     └─────────────────────────────┘              │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Modulation Prediction       │              │           │
│  │     │  • Conv2D → weights         │              │           │
│  │     └─────────────────────────────┘              │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Deformable Sampling         │              │           │
│  │     │  • Sample at offset locs    │              │           │
│  │     └─────────────────────────────┘              │           │
│  │                                                   │           │
│  │  2. Adaptive Sampler                             │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Complexity Predictor        │              │           │
│  │     │  • GlobalPool → Dense       │              │           │
│  │     │  • Output: complexity [0-1] │              │           │
│  │     └─────────────────────────────┘              │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Num Points Calculator       │              │           │
│  │     │  • Map complexity to N      │              │           │
│  │     │  • N ∈ [min_pts, max_pts]   │              │           │
│  │     └─────────────────────────────┘              │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Position Predictor          │              │           │
│  │     │  • Conv2D → point coords    │              │           │
│  │     └─────────────────────────────┘              │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Point Feature Extractor     │              │           │
│  │     │  • Sample features at pts   │              │           │
│  │     └─────────────────────────────┘              │           │
│  │     ┌─────────────────────────────┐              │           │
│  │     │ Self-Attention Refinement   │              │           │
│  │     │  • Q, K, V projections      │              │           │
│  │     │  • Attention over points    │              │           │
│  │     └─────────────────────────────┘              │           │
│  │                                                   │           │
│  │  3. Shape Descriptor                             │           │
│  │     • Global shape embedding                     │           │
│  │     • [batch, descriptor_dim]                    │           │
│  └──────────────────────────────────────────────────┘           │
│                         │                                        │
│        Shape Features + Sampling Points + Complexity             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION HEADS                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Probability  │  │   Distance   │  │  Complexity  │          │
│  │     Head     │  │     Head     │  │     Head     │          │
│  │              │  │              │  │              │          │
│  │ Conv → 1ch   │  │ Conv → N ch  │  │ Pool → 1val  │          │
│  │  Sigmoid     │  │    ReLU      │  │   Sigmoid    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         ▼                 ▼                  ▼                  │
│   Prob Map          Distance Map      Complexity Score          │
│  [B,H,W,1]          [B,H,W,N_pts]          [B,1]               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING                                │
│  ┌──────────────────────────────────────────────────┐           │
│  │  1. Threshold Probability Map                    │           │
│  │     • prob > prob_thresh                         │           │
│  │                                                   │           │
│  │  2. Find Seeds (Local Maxima)                    │           │
│  │     • Maximum filter                             │           │
│  │     • Connected components                       │           │
│  │                                                   │           │
│  │  3. Adaptive Watershed                           │           │
│  │     • Use predicted distances                    │           │
│  │     • Grow from seeds                            │           │
│  │                                                   │           │
│  │  4. Instance Reconstruction                      │           │
│  │     • Convert distances to polygons              │           │
│  │     • Fill polygons for masks                    │           │
│  └──────────────────────────────────────────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT INSTANCES                              │
│                Instance Label Map [H, W]                         │
│         (Each cell has unique ID, background = 0)                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Mode

```
Input Batch
    ↓
Backbone → Features
    ↓
Shape Prior Encoder → Enhanced Features
    ↓
Adaptive Shape Encoder → Shape Features + Points
    ↓
Prediction Heads → Prob + Dist + Complexity
    ↓
Loss Computation:
    • Focal Loss (probability)
    • Smooth L1 (distance)
    • L2 (complexity)
    • Smoothness (boundary)
    • Entropy (prototypes)
    ↓
Backpropagation → Update Weights
```

### Inference Mode

```
Input Image
    ↓
Normalize
    ↓
Forward Pass → Predictions
    ↓
Post-Processing:
    • Threshold
    • Find seeds
    • Watershed
    • Reconstruct instances
    ↓
Instance Labels + Details
```

## Key Components Detail

### 1. Deformable Convolution

**Purpose**: Learn adaptive sampling locations

**Mechanism**:
```
Input Features [B, H, W, C]
    ↓
Offset Conv → Offsets [B, H, W, 2×K×K]
Modulation Conv → Weights [B, H, W, K×K]
    ↓
Sample at (regular_pos + offset) × weight
    ↓
Regular Convolution on sampled features
    ↓
Output [B, H, W, C']
```

**Innovation**: 
- Adapts to shape irregularities
- Learns where to look for boundary information
- Multiple deformable groups for efficiency

### 2. Adaptive Sampler

**Purpose**: Generate variable number of boundary points

**Mechanism**:
```
Features → Complexity Score (0-1)
    ↓
N = min_pts + complexity × (max_pts - min_pts)
    ↓
Generate N point positions (angles, radii)
    ↓
Extract features at point locations
    ↓
Self-attention refinement
    ↓
Refined points + features
```

**Innovation**:
- Simple shapes use fewer points (efficient)
- Complex shapes use more points (accurate)
- Attention focuses on important points

### 3. Shape Prior Encoder

**Purpose**: Encode domain-specific shape knowledge

**Mechanism**:
```
Features + Positional Encoding
    ↓
Flatten to sequence [B, H×W, C]
    ↓
Concat with prototypes [B, N_proto+H×W, C]
    ↓
Transformer layers (self-attention + FFN)
    ↓
Split: prototypes [B, N_proto, C], features [B, H×W, C]
    ↓
Cross-attention: features ← prototypes
    ↓
Reshape to [B, H, W, C]
```

**Innovation**:
- Learns common shape patterns
- Transfers knowledge across instances
- Interpretable (prototype weights)

## Loss Function Architecture

```
Ground Truth:
  • Probability Map
  • Distance Map  
  • Valid Mask

Predictions:
  • Probability Map
  • Distance Map
  • Complexity Score
  • Prototype Weights

    ↓
    
L_total = Σ (weight_i × loss_i)

where:
  loss_1: Focal Loss (prob)
  loss_2: Smooth L1 (dist)
  loss_3: MSE (complexity)
  loss_4: L2 (smoothness)
  loss_5: Entropy (prototypes)
```

## Memory Layout

### Feature Maps

```
Layer            Shape                Memory (batch=4, H=256, W=256)
─────────────────────────────────────────────────────────────────
Input            [4, 256, 256, 1]    256 KB
Backbone L1      [4, 256, 256, 32]   8 MB
Backbone L2      [4, 128, 128, 64]   4 MB
Backbone L3      [4, 64, 64, 128]    2 MB
Prior Features   [4, 64, 64, 256]    4 MB
Shape Features   [4, 64, 64, 256]    4 MB
Prob Output      [4, 256, 256, 1]    256 KB
Dist Output      [4, 256, 256, 128]  33 MB
─────────────────────────────────────────────────────────────────
Total (approx)                       ~60 MB per batch
```

### Model Parameters

```
Component               Parameters
─────────────────────────────────────
Backbone (U-Net)        ~5M
Shape Prior Encoder     ~3M
Adaptive Shape Encoder  ~2M
Prediction Heads        ~1M
─────────────────────────────────────
Total                   ~11M
```

## Computational Complexity

### FLOPs Analysis (single image, 256×256)

```
Operation                    FLOPs
────────────────────────────────────
Backbone Forward             ~5 GFLOPs
Shape Prior (Transformer)    ~2 GFLOPs
Deformable Convolutions      ~3 GFLOPs
Adaptive Sampling            ~0.5 GFLOPs
Prediction Heads             ~1 GFLOPs
────────────────────────────────────
Total Forward Pass           ~12 GFLOPs
```

### Inference Time (typical)

```
Hardware              Time per Image (256×256)
──────────────────────────────────────────────
CPU (Intel i7)        ~2-5 seconds
GPU (RTX 3090)        ~50-100 ms
GPU (A100)            ~30-50 ms
```

## Design Patterns

### 1. Modular Design
- Each component is self-contained
- Clear interfaces between modules
- Easy to swap implementations

### 2. Configuration-Driven
- Single config object controls all aspects
- Pre-defined configs for common scenarios
- Easy to reproduce experiments

### 3. TensorFlow Best Practices
- @tf.function for performance
- Proper use of training flags
- Memory-efficient operations

## Extensibility Points

### Easy Extensions

1. **New Backbone**: Implement in `backbone.py`
2. **Custom Loss**: Add to `loss.py`
3. **New Sampler**: Extend `AdaptiveSampler`
4. **Post-Processing**: Modify `predictor.py`

### Future Enhancements

1. **3D Extension**: Add Z-dimension handling
2. **Multi-Task**: Add classification head
3. **Tracking**: Add temporal consistency
4. **Active Learning**: Add uncertainty estimation

---

This architecture balances:
- **Flexibility**: Handles diverse shapes
- **Efficiency**: Adaptive complexity
- **Accuracy**: Multi-scale + prior knowledge
- **Interpretability**: Visualizable components

