# #!/usr/bin/env python
# """
# Stitch all files named   patch_col_<col>_row_<row>.png
# inside ORIGINAL_DIR into one big PNG called mask_full.png.

# Usage (no arguments needed):
#     conda install pillow      # one-time setup
#     python stitch_mask.py
# """
# import re, sys
# from pathlib import Path
# from PIL import Image

# # ----------- paths -----------
# ORIGINAL_DIR = Path(
#     "/hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/"
#     "240819_Ji_N1_H_EScan/original_masks_2048"
# )
# OUT_PATH = "/hpc/home/yz814/yizhanglab_yz814/hDRG-autoseg/images/mask_full.png"

# # ----------- collect patches -----------
# pat   = re.compile(r"patch_col_(\d+)_row_(\d+)\.png$")
# tiles = [(int(r), int(c), f) for f in ORIGINAL_DIR.glob("patch_col_*_row_*.png")
#          if (m := pat.match(f.name)) for c, r in [m.groups()]]

# if not tiles:
#     sys.exit(f"No patch_col_*_row_*.png files found in {ORIGINAL_DIR}")

# rows = sorted({r for r, _, _ in tiles})
# cols = sorted({c for _, c, _ in tiles})
# row0, col0 = min(rows), min(cols)

# # size/mode from first tile
# with Image.open(tiles[0][2]) as sample:
#     w, h   = sample.size
#     mode   = sample.mode

# canvas = Image.new(mode, (w * len(cols), h * len(rows)))

# for r, c, f in tiles:
#     with Image.open(f) as img:
#         x = (c - col0) * w
#         y = (r - row0) * h
#         canvas.paste(img, (x, y))

# canvas.save(OUT_PATH)
# print(f"✓ stitched {len(tiles)} patches → {OUT_PATH}")








#!/usr/bin/env python
"""
Stitch patch_col_<col>_row_<row>.png files and save a
down-scaled mosaic (if needed) as mask_full.png.

Output path: /hpc/.../original_masks_2048/mask_full.png
"""
import re, sys, math
from pathlib import Path
from PIL import Image

# ---- paths ----
ORIG_DIR = Path(
    "/hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/"
    "240819_Ji_N1_H_EScan/original_masks_2048"
)
OUT_PATH = "/hpc/home/yz814/yizhanglab_yz814/hDRG-autoseg/images/mask_full_resized.png"

# ---- collect tiles ----
pat = re.compile(r"patch_col_(\d+)_row_(\d+)\.png$")
tiles = [(int(r), int(c), f)
         for f in ORIG_DIR.glob("patch_col_*_row_*.png")
         if (m := pat.match(f.name)) for c, r in [m.groups()]]

if not tiles:
    sys.exit(f"No patch_col_*_row_*.png files found in {ORIG_DIR}")

rows = sorted({r for r, _, _ in tiles})
cols = sorted({c for _, c, _ in tiles})
row0, col0 = min(rows), min(cols)

with Image.open(tiles[0][2]) as sample:
    w, h = sample.size
    mode = sample.mode

W_full, H_full = w * len(cols), h * len(rows)
MAX_PIX = Image.MAX_IMAGE_PIXELS or 178_956_970  # default limit

# ---- build full-size canvas (still in memory) ----
canvas = Image.new(mode, (W_full, H_full))
for r, c, f in tiles:
    with Image.open(f) as img:
        canvas.paste(img, ((c - col0) * w, (r - row0) * h))

# ---- auto-down-scale if the image is too large ----
pixels = W_full * H_full
if pixels > MAX_PIX:
    scale = math.sqrt(MAX_PIX / pixels)
    new_size = (int(W_full * scale), int(H_full * scale))
    print(f"Image is {pixels:,} px → resizing to {new_size} "
          f"({new_size[0]*new_size[1]:,} px)")
    canvas = canvas.resize(new_size, resample=Image.NEAREST)
else:
    print(f"Image size {pixels:,} px within limit; no resize needed.")

canvas.save(OUT_PATH)
print(f"✓ stitched {len(tiles)} patches → {OUT_PATH}")
