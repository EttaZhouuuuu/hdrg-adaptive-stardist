# Multi-Scale StarDist Training Comparison Report

Generated on: 2025-10-01 11:40:35

## Overview

Compared 2 different multi-scale configurations:

1. **multiscale_long**: 2 scales
2. **multiscale_5scales**: 5 scales

## Detailed Results

### multiscale_long

- **Total Training Time**: 292.45 seconds
- **Successful Scales**: 2/2

#### Scale-wise Results:

| Scale | Training Time (s) | Final Loss | Val Loss | Objects | Efficiency |
|-------|------------------|------------|----------|---------|------------|
| 1.0 | 145.87 | 1.0415 | 0.8911 | 3 | 0.0206 |
| 0.5 | 144.41 | 0.3365 | 0.4668 | 1 | 0.0069 |

### multiscale_5scales

- **Total Training Time**: 980.08 seconds
- **Successful Scales**: 3/5

#### Scale-wise Results:

| Scale | Training Time (s) | Final Loss | Val Loss | Objects | Efficiency |
|-------|------------------|------------|----------|---------|------------|
| 1.0 | 326.89 | 0.9353 | 0.8388 | 3 | 0.0092 |
| 0.8 | 325.00 | 0.5534 | 0.6692 | 2 | 0.0062 |
| 0.6 | 325.06 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 0.4 | 0.00 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 0.2 | 0.00 | 0.0000 | 0.0000 | 0 | 0.0000 |

## Analysis and Recommendations

### Key Findings:

- **Best Loss Performance**: Scale 0.40 (Loss: 0.0000)
- **Most Objects Detected**: Scale 1.00 (3 objects)
- **Highest Efficiency**: Scale 1.00

### Recommendations:

1. **For accuracy**: Use scale 0.40 (lowest loss)
2. **For detection**: Use scale 1.00 (most objects)
3. **For efficiency**: Use scale 1.00 (best time/performance ratio)

### Scale Trends:

- **Larger scales** tend to detect more objects but may have higher loss
- **Smaller scales** often train faster and achieve lower loss
- **Multi-scale fusion** combines the benefits of different scales
