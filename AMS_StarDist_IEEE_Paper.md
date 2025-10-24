# Adaptive Multi-Scale StarDist: A Dynamic Parameter Selection Framework for Robust Neuronal Image Segmentation

*IEEE Conference Format*

Author Name(s)  
Department/Institution  
Email Address(es)

## Abstract

Accurate segmentation of neuronal structures in microscopy images remains a challenging task due to varying object scales, densities, and boundary complexities. While the StarDist algorithm has shown promising results in instance segmentation tasks, its fixed parameters often limit performance across diverse image characteristics. This paper presents Adaptive Multi-Scale StarDist (AMS-StarDist), a novel framework that dynamically adjusts segmentation parameters based on local image properties and processes images at multiple scales simultaneously. Our key contributions include: 1) an adaptive parameter selection mechanism that optimizes the number of radial rays and grid size based on object complexity; 2) a multi-scale architecture that processes images at different resolutions (1.0, 0.75, 0.5) with scale-specific parameter sets; and 3) a scale-aware training strategy with custom loss functions for boundary accuracy and cross-scale consistency. Experimental results on human dorsal root ganglion (hDRG) microscopy images demonstrate that AMS-StarDist achieves superior segmentation performance compared to the baseline StarDist model, with particular improvements in handling varying object sizes and complex boundaries. Our approach shows stable convergence across all scales with no overfitting, processes each scale in approximately 24 seconds per 10 epochs, and scales linearly with the number of resolution levels.

**Keywords**: neuronal segmentation, instance segmentation, deep learning, adaptive algorithms, microscopy image analysis

[Rest of paper content...]

## I. Introduction

[Previous introduction section content...]

## II. Related Work

[Previous related work section content...]

## III. Methodology

[Previous methodology section content...]

## IV. Experiments

[Previous experiments section content...]

## V. Discussion

[Previous discussion section content...]

## VI. Conclusion and Future Work

[Previous conclusion section content...]

## Acknowledgment

The authors would like to thank the StarDist team for the excellent base framework, Duke Computing Cluster (DCC) for computational resources, and the hDRG dataset providers for high-quality training data. This work was supported in part by [funding sources].

## References

[1] U. Schmidt et al., "Cell Detection with Star-Convex Polygons," in MICCAI, 2018.

[2] D. Svoboda et al., "A Survey of Image Processing Algorithms in Medical Image Analysis," Med. Image Anal., 2017.

[3] J. Liu et al., "DeepNuc: Deep Learning for Nucleus Segmentation," in ISBI, 2019.

[4] K. He et al., "Mask R-CNN," in ICCV, 2017.

[5] O. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," in MICCAI, 2015.

[6] M. Weigert et al., "Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy," in WACV, 2020.

[7] L. Wang et al., "Attention-guided Instance Segmentation," in CVPR, 2019.

[8] T. Lindeberg, "Scale-space theory: A basic tool for analyzing structures at different scales," J. Appl. Stat., 1994.

[9] P. Perona et al., "Scale-space and edge detection using anisotropic diffusion," IEEE PAMI, 1990.

[10] X. Jia et al., "Dynamic Filter Networks," in NeurIPS, 2016.

[11] J. Hu et al., "Squeeze-and-Excitation Networks," in CVPR, 2018.

[12] T.-Y. Lin et al., "Feature Pyramid Networks for Object Detection," in CVPR, 2017.

[13] S. Liu et al., "Path Aggregation Network for Instance Segmentation," in CVPR, 2018.

[14] K. Sun et al., "Deep High-Resolution Representation Learning for Human Pose Estimation," in CVPR, 2019.

