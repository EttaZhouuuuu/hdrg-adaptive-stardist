#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation.py

Perform quantitative evaluation of a single image's segmentation result (predicted labels vs. ground-truth labels),
including:
  - Object-level IoU (Jaccard) and F1 score (derived from StarDist.matching TP/FP/FN)
  - Pixel-level Dice coefficient
  - Hausdorff Distance (on global masks, measuring worst-case boundary distance)

Optionally generates a visualization overlay (TP=green, FP=red, FN=gray) either on the original image
(if provided) or on a blank background.

This version avoids reliance on render_label_pred (which may have a mismatched signature),
and instead builds the overlay “by hand.”
"""

import os
import argparse
import json

import numpy as np
from skimage import io, img_as_ubyte, img_as_float
from skimage.transform import resize
from skimage.metrics import hausdorff_distance
from stardist.matching import matching       # for object‐level TP/FP/FN


def load_label(path):
    """
    Load a label image from disk (uint8 or uint16). Return a 2D numpy array of type int.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    arr = io.imread(path)
    # If RGB or RGBA, take the first channel
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Ensure integer type
    return arr.astype(np.int32)


def compute_pixel_dice(y_true, y_pred):
    """
    Compute pixel‐level Dice coefficient between two integer label images.
    Convert to binary masks (any label > 0).
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    bin_true = (y_true > 0).astype(np.bool_)
    bin_pred = (y_pred > 0).astype(np.bool_)
    intersection = np.logical_and(bin_true, bin_pred).sum()
    denom = bin_true.sum() + bin_pred.sum()
    if denom == 0:
        # If both GT and pred have no foreground, define Dice = 1.0
        return 1.0
    return float(2 * intersection / denom)


def make_overlay(y_true, y_pred, src_image=None, alpha_tp=0.5, alpha_fp=0.5, alpha_fn=0.5):
    """
    Build an RGB overlay image highlighting:
      - True Positives (TP) in green
      - False Positives (FP) in red
      - False Negatives (FN) in gray

    If src_image is provided (H×W or H×W×3/4), it is used as background. Otherwise,
    a blank white background is used.

    All input arrays must be aligned in shape.
    """
    # Determine output height, width
    H, W = y_true.shape

    # 1. Load or create background
    if src_image is not None:
        if not os.path.isfile(src_image):
            print(f"[WARNING] src_image not found: {src_image}. Using blank white background.")
            bg_rgb = np.ones((H, W, 3), dtype=np.float32)
        else:
            bg = io.imread(src_image)
            # If grayscale, convert to RGB
            if bg.ndim == 2:
                bg_rgb = np.stack([bg] * 3, axis=-1)
            elif bg.ndim == 3 and bg.shape[2] in (3, 4):
                bg_rgb = bg[..., :3]
            else:
                bg_rgb = bg
            # Ensure float in [0,1]
            bg_rgb = img_as_float(bg_rgb)
            # If bg image size != (H,W), resize it (bilinear) to (H,W)
            if bg_rgb.shape[0] != H or bg_rgb.shape[1] != W:
                bg_rgb = resize(bg_rgb, (H, W), order=1, preserve_range=True, anti_aliasing=True)
                bg_rgb = np.clip(bg_rgb, 0.0, 1.0).astype(np.float32)
    else:
        # White background
        bg_rgb = np.ones((H, W, 3), dtype=np.float32)

    # 2. Build binary maps
    bin_true = (y_true > 0)
    bin_pred = (y_pred > 0)

    tp_mask = np.logical_and(bin_true, bin_pred)
    fp_mask = np.logical_and(~bin_true, bin_pred)
    fn_mask = np.logical_and(bin_true, ~bin_pred)

    # 3. Prepare color arrays
    color_tp = np.array([0.0, 1.0, 0.0], dtype=np.float32)    # green
    color_fp = np.array([1.0, 0.0, 0.0], dtype=np.float32)    # red
    color_fn = np.array([0.5, 0.5, 0.5], dtype=np.float32)    # gray

    overlay = bg_rgb.copy()

    # 4. Overlay TP
    if tp_mask.any():
        overlay[tp_mask] = (1 - alpha_tp) * overlay[tp_mask] + alpha_tp * color_tp

    # 5. Overlay FP
    if fp_mask.any():
        overlay[fp_mask] = (1 - alpha_fp) * overlay[fp_mask] + alpha_fp * color_fp

    # 6. Overlay FN
    if fn_mask.any():
        overlay[fn_mask] = (1 - alpha_fn) * overlay[fn_mask] + alpha_fn * color_fn

    # 7. Clip and return uint8
    overlay_uint8 = img_as_ubyte(np.clip(overlay, 0.0, 1.0))
    return overlay_uint8


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single‐image segmentation: object‐level metrics (IoU/F1), "
                    "pixel‐level Dice, Hausdorff distance, and optional TP/FP/FN overlay."
    )
    parser.add_argument(
        "--src_image", type=str, default=None,
        help="(Optional) Path to the original grayscale/RGB image for overlay. "
             "If omitted, overlay will be on a blank background."
    )
    parser.add_argument(
        "--gt_label", type=str, required=True,
        help="Path to the ground‐truth label image (integer mask)."
    )
    parser.add_argument(
        "--pred_label", type=str, required=True,
        help="Path to the predicted label image (integer mask)."
    )
    parser.add_argument(
        "--iou_thresh", type=float, default=0.50,
        help="IoU threshold for object‐level matching (default: 0.50)."
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Path to output JSON file where metrics will be saved."
    )
    parser.add_argument(
        "--vis_out", type=str, default=None,
        help="(Optional) Path to save TP/FP/FN overlay image (PNG)."
    )
    args = parser.parse_args()

    # 1. Load GT and prediction labels
    y_true = load_label(args.gt_label)
    y_pred = load_label(args.pred_label)

    # 2. If dimensions differ, resize prediction to match ground‐truth
    if y_true.shape != y_pred.shape:
        print(f"[WARNING] GT shape {y_true.shape} != pred shape {y_pred.shape}. "
              "Resizing pred to match GT (nearest‐neighbor).")
        y_pred_resized = resize(
            y_pred,
            output_shape=y_true.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(y_pred.dtype)
        y_pred = y_pred_resized

    # 3. Object‐level matching: TP, FP, FN, and mean IoU over matched pairs
    match_res = matching(y_true, y_pred, thresh=args.iou_thresh, report_matches=True)
    tp = int(match_res.tp)
    fp = int(match_res.fp)
    fn = int(match_res.fn)

    precision_obj = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_obj = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_obj = (2 * precision_obj * recall_obj / (precision_obj + recall_obj)
              if (precision_obj + recall_obj) > 0 else 0.0)

    ious_matched = match_res.matched_scores  # list or array of IoU for matched objects
    mean_iou_obj = float(np.mean(ious_matched)) if len(ious_matched) > 0 else 0.0

    # 4. Pixel‐level Dice coefficient
    dice_pixel = compute_pixel_dice(y_true, y_pred)

    # 5. Hausdorff distance on binary masks
    bin_true = (y_true > 0).astype(np.bool_)
    bin_pred = (y_pred > 0).astype(np.bool_)
    if bin_true.sum() == 0 or bin_pred.sum() == 0:
        hd_global = float("nan")
    else:
        hd_global = float(hausdorff_distance(bin_true, bin_pred))

    # 6. Aggregate metrics into a dictionary
    metrics = {
        # Object‐level
        "iou_thresh":       float(args.iou_thresh),
        "tp":               tp,
        "fp":               fp,
        "fn":               fn,
        "precision_obj":    precision_obj,
        "recall_obj":       recall_obj,
        "f1_obj":           f1_obj,
        "mean_iou_obj":     mean_iou_obj,
        # Pixel‐level
        "dice_pixel":       dice_pixel,
        # Boundary‐level
        "hausdorff_dist":   hd_global
    }

    # 7. Save metrics to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    # 8. Print a summary to console
    print("======== Evaluation Metrics ========")
    print(f"IoU Threshold (object‐level): {args.iou_thresh:.2f}")
    print(f"  TP (object‐level):        {metrics['tp']}")
    print(f"  FP (object‐level):        {metrics['fp']}")
    print(f"  FN (object‐level):        {metrics['fn']}")
    print(f"  Precision (object‐level): {metrics['precision_obj']:.3f}")
    print(f"  Recall (object‐level):    {metrics['recall_obj']:.3f}")
    print(f"  F1 (object‐level):        {metrics['f1_obj']:.3f}")
    print(f"  Mean IoU (matched):       {metrics['mean_iou_obj']:.3f}")
    print(f"  Dice (pixel‐level):       {metrics['dice_pixel']:.3f}")
    if np.isnan(metrics["hausdorff_dist"]):
        print("  Hausdorff Dist:           NaN (one mask is empty)")
    else:
        print(f"  Hausdorff Dist:           {metrics['hausdorff_dist']:.3f}")

    # 9. Optionally generate a TP/FP/FN overlay
    if args.vis_out:
        os.makedirs(os.path.dirname(args.vis_out), exist_ok=True)
        overlay_img = make_overlay(
            y_true, y_pred,
            src_image=args.src_image,
            alpha_tp=0.5,
            alpha_fp=0.5,
            alpha_fn=0.5
        )
        io.imsave(args.vis_out, overlay_img)
        print(f"[INFO] Visualization saved to: {args.vis_out}")


if __name__ == "__main__":
    main()
