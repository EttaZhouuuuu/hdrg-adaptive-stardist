import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
from skimage import io
from skimage.transform import resize
from csbdeep.utils import normalize_mi_ma
import openslide
from openslide.deepzoom import DeepZoomGenerator
from stardist import random_label_cmap
from stardist.models import Config2D, StarDist2D
from time import time

np.random.seed(42)

parser = argparse.ArgumentParser(description="Inference with StarDist")
parser.add_argument("--model_ckpt_path", type=str, required=True)
parser.add_argument("--slide_fp", type=str, required=True)
parser.add_argument("--process_by_patches", action="store_true")
parser.add_argument("--pixel_intensity_threshold", type=float)
parser.add_argument("--patch_len", type=int, default=2048)
parser.add_argument("--target_len", type=int, default=256,
                    help="Target image/patch size for the model")
parser.add_argument("--prob_threshold", type=float, default=0.5,
                    help="Probability threshold for the neuron cell prediction")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, required=True)

args = parser.parse_args()
model_ckpt_path     = args.model_ckpt_path
slide_fp            = args.slide_fp
process_by_patches  = args.process_by_patches
threshold           = args.pixel_intensity_threshold
patch_len           = args.patch_len
target_len          = args.target_len
prob_threshold      = args.prob_threshold
device              = args.device
output_dir          = args.output_dir
scale               = target_len / patch_len

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

start = time()

# Load pretrained model
model = StarDist2D(None, name="stardist", basedir=model_ckpt_path)
model.thresholds = dict(prob=prob_threshold, nms=0.4)
print("Now the thresholds are:", model.thresholds)

# Predict on the whole slide image (wsi) or a test image
if process_by_patches:
    print("Processing a whole slide image (may take a while)...")
    slide = openslide.open_slide(slide_fp)
    patches = DeepZoomGenerator(slide, tile_size=patch_len, overlap=0)
    last_level = patches.level_count - 1 # last level has the highest resolution
    cols, rows = patches.level_tiles[last_level]
    whole_mask = Image.new('1', (cols * patch_len, rows * patch_len))

    for i in range(cols):
        for j in range(rows):
            patch = patches.get_tile(last_level, (i, j))
            patch = np.array(patch)
            if threshold and patch.mean() >= threshold:
                # skip patch not on the tisssue
                continue
            print(f"Processing the patch at col {i} and row {j}")
            patch = normalize_mi_ma(patch, mi=0, ma=255)
            labels, details = model.predict_instances(
                patch, prob_thresh=prob_threshold, scale=scale
            )
            coords, probs = details["coord"], details["prob"]
            pred_coords_dir = os.path.join(output_dir, "patch_pred_coords/")
            os.makedirs(pred_coords_dir, exist_ok=True)
            np.save(os.path.join(pred_coords_dir, f"patch_col_{i}_row_{j}.npy"), coords)
            pred_mask = (labels > 0)
            pred_mask = Image.fromarray(pred_mask)
            whole_mask.paste(pred_mask, (i * patch_len, j * patch_len))

    # whole_mask.save(os.path.join(output_dir, f"pred_wsi_seg_mask.png"))
    whole_mask.thumbnail((1000, 1000))
    whole_mask.save(os.path.join(output_dir, f"pred_wsi_seg_mask_thumbnail.png"))

else:
    print("Processing a normal-size image...")
    normal_image = io.imread(slide_fp)
    normal_image = normalize_mi_ma(normal_image, mi=0, ma=255)
    labels, details = model.predict_instances(
        normal_image, prob_thresh=prob_threshold, scale=scale
    )
    coords, probs = details["coord"], details["prob"]
    np.save(os.path.join(output_dir, "pred_labels.npy"), labels)
    np.save(os.path.join(output_dir, "pred_coords.npy"), coords)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    lbl_cmap = random_label_cmap()
    ax.imshow(normal_image, alpha=0.7)
    ax.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, f"pred_seg_mask.png"),
                bbox_inches="tight", dpi=300, transparent=True)

end = time()
print("Done!")
print(f"Time elapsed: {(end-start)/60:.2f} minutes.")