# Shape-Aware StarDist++ with Instance Separation Head

This repository extends an existing **Adaptive Shape-Aware StarDist** framework with an explicit **Instance Separation Geometry Head**, aiming to improve robustness and generalization of neuron soma segmentation across slides, stains, and acquisition conditions.

The objective of this work is **not** to optimize performance on a fixed dataset, but to address a structural failure mode—**merging of touching instances**—that becomes increasingly severe under domain shift.

---

## 1. Motivation

### Problem Setting
Dorsal root ganglion (DRG) neuron soma segmentation is challenging due to:
- highly irregular and concave soma geometry,
- dense clustering and frequent instance contact,
- substantial variation in staining and contrast across slides.

While the base Adaptive StarDist model achieves strong in-domain performance, systematic error analysis reveals that a dominant failure mode is:

> **Merging of adjacent neuron somata into a single instance**, particularly in low-contrast or densely packed regions.

This error is not dataset-specific and tends to worsen under cross-domain conditions.

### Key Insight
StarDist effectively models **instance shape** via radial distances, but does not explicitly encode **instance separability** when boundaries are ambiguous. Under domain shift, boundary cues degrade first, leading to merge errors.

**Hypothesis:**  
Injecting an explicit instance-level geometric signal—independent of texture or staining—can improve separation robustness and generalization.

---

## 2. Method Overview

We introduce an **Instance Separation Head** that predicts a per-pixel offset vector pointing toward the centroid of the instance to which the pixel belongs.

This head complements, rather than replaces, StarDist’s shape reconstruction mechanism.

### Architecture Summary
The extended model consists of:

- **Backbone (unchanged)**
  - Multi-scale FPN
  - Transformer-based shape prior encoder
  - Deformable convolution refinement

- **Existing heads**
  - Foreground probability head
  - Radial distance head (StarDist-style)
  - Shape complexity head

- **New head**
  - **Instance offset head**: predicts a 2D offset field \((dx, dy)\)

All heads share the same high-resolution feature map.

---

## 3. Instance Offset Representation

### Definition
For a pixel \(p=(x,y)\) belonging to instance \(k\) with centroid \(c_k=(x_k,y_k)\), the ground-truth offset is defined as:

\[
O_{gt}(p) = \left(\frac{x_k - x}{s_k}, \frac{y_k - y}{s_k}\right)
\]

where \(s_k\) is a scale normalization factor (e.g., equivalent radius or bounding-box diagonal).

Offsets are defined only on foreground pixels.

### Rationale
- Pixels of the same instance share a common geometric attractor.
- Touching instances produce diverging offset fields even when boundaries are weak.
- Offset geometry depends on spatial structure rather than appearance, making it more stable under domain shift.

---

## 4. Loss Design

The total loss augments the base StarDist objective with an instance separation term.

### Offset Regression Loss
A Smooth L1 loss is applied on foreground pixels:

\[
L_{off} = \text{SmoothL1}(O_{pred}, O_{gt})
\]

### Hard-Region Reweighting
To emphasize touching instances:
1. Instance masks are slightly dilated.
2. Overlapping dilation regions define *touching zones*.
3. Offset loss is upweighted in these zones.

### Total Loss
\[
L = \alpha_p L_{prob}
  + \alpha_d L_{dist}
  + \alpha_c L_{complex}
  + \alpha_{off} L_{off}^{weighted}
  + L_{regularization}
\]

---

## 5. Inference Pipeline

The offset head is used for **instance separation**, while StarDist distances are used for **boundary refinement**.

1. **Foreground masking** using probability head.
2. **Center voting**:  
   \(c(p) = p + s(p)\cdot O_{pred}(p)\)
3. **Seed detection** via peak finding in center space.
4. **Instance separation** using seeded watershed or clustering.
5. **Boundary refinement** using StarDist radial distances.

This decouples separability from boundary appearance.

---

## 6. Training Protocol

- Uses existing instance annotations; no additional labeling required.
- Offset labels are generated directly from instance masks.
- Training schedule follows the base model.
- Only generic augmentations are used (rotation, scaling, noise).

---

## 7. Evaluation Strategy

### Standard Metrics
- IoU, Dice, Precision, Recall.

### Separation-Specific Metrics
- **Merge rate**
- **Split rate**
- Reported globally and on touching-instance subsets.

### Domain Robustness
Synthetic domain shifts (contrast, blur, noise) are applied at inference time.  
Reduced performance degradation is taken as evidence of improved generalization.

---

## 8. Expected Outcomes

- Reduced merge errors in touching regions.
- Improved robustness under domain shift.
- Comparable or improved IoU without dataset-specific tuning.
- Clear separation of shape modeling and instance separability roles.

---

## 9. Limitations

- Focuses on instance separation, not semantic misclassification.
- Extremely ambiguous or heavily overlapping regions remain challenging.

---

## 10. Summary

This work introduces explicit instance-level geometry into a shape-aware StarDist framework. By decoupling boundary modeling from instance separation, the model achieves improved robustness and generalization while remaining compatible with existing StarDist pipelines.
