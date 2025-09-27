#!/bin/bash
set -e
set -o pipefail

# Suppress TensorFlow/XLA spam
export TF_CPP_MIN_LOG_LEVEL=2        # hide INFO/WARN
export XLA_FLAGS="--xla_gpu_autotune_level=0"


# ========================
# CONFIGURATION
# ========================
VERSION="v1.0.4"

DATA_PATH="/hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/"
SLIDE="240819_Ji_N1_H_EScan"
CKPT_BASE="../ckpts"
OUTPUT_BASE="../output"

# Hyperparameter search grid
LRS=("7e-4" "1e-3")
BATCH_SIZES=(16)
N_RAYS=(24 96)
GRID_SIZES=("2 2" "4 4" "8 8")

# ========================
# FUNCTIONS
# ========================

run_train () {
    LR=$1
    BS=$2
    RAYS=$3
    GRID=$4
    RUN_NAME="run_${VERSION}_lr${LR}_bs${BS}_rays${RAYS}_grid${GRID// /x}"

    echo ">>> Training: $RUN_NAME"

    python3 ../scripts/train_seg_neuron.py \
        --exp_id "$RUN_NAME" \
        --wandb_run_name "stardist-h&e2d-$RUN_NAME-$VERSION" \
        --data_path "$DATA_PATH" \
        --slides "$SLIDE" \
        --ckpt_path "$CKPT_BASE/$RUN_NAME" \
        --use_gpu \
        --train_batch_size $BS \
        --n_epochs 100 \
        --n_rays $RAYS \
        --train_lr $LR \
        --grid_size $GRID
}

run_infer () {
    RUN_NAME=$1
    PATCH_DIR="$DATA_PATH/$SLIDE/patches_2048"
    MODEL_CKPT="$CKPT_BASE/$RUN_NAME/"
    OUTPUT_DIR="$OUTPUT_BASE/${RUN_NAME}_infer"

    echo ">>> Inference with model: $RUN_NAME"

    mkdir -p "$OUTPUT_DIR"

    for patch_path in "$PATCH_DIR"/patch_col_*_row_*.png; do
        image_part_name=$(basename "$patch_path")
        patch_out_dir="${OUTPUT_DIR}/${image_part_name%.*}"
        mkdir -p "$patch_out_dir"

        python3 ../scripts/infer_main_2d.py \
            --model_ckpt_path "$MODEL_CKPT" \
            --slide_fp "$patch_path" \
            --patch_len 2048 \
            --target_len 256 \
            --prob_threshold 0.2 \
            --device "cpu" \
            --output_dir "$patch_out_dir"
    done
}

run_eval () {
    RUN_NAME=$1
    PRED_BASE_DIR="$OUTPUT_BASE/${RUN_NAME}_infer"
    RESULT_FILE="$OUTPUT_BASE/${RUN_NAME}_iou.txt"

    echo ">>> Evaluating IoU for $RUN_NAME"
    echo "Results will be saved to $RESULT_FILE"

    python3 <<EOF > "$RESULT_FILE" 2>&1
import os, numpy as np
from skimage import io
from glob import glob

gt_dir = "$DATA_PATH/$SLIDE/pseudo_gt_masks_2048"
pred_base_dir = "$PRED_BASE_DIR"

gt_paths = sorted(glob(os.path.join(gt_dir, "patch_col_*_row_*.png")))
ious, empty_cases = [], 0

for gt_path in gt_paths:
    filename = os.path.basename(gt_path)
    pred_dir = os.path.join(pred_base_dir, filename.replace(".png", ""))
    pred_path = os.path.join(pred_dir, "pred_labels.npy")

    if not os.path.exists(pred_path):
        print(f"[SKIP] Missing prediction for {filename}")
        continue

    gt_mask = io.imread(gt_path)
    gt_binary = (gt_mask == 255).astype(np.uint8)
    pred_mask = np.load(pred_path)
    pred_binary = (pred_mask > 0).astype(np.uint8)

    if gt_binary.shape != pred_binary.shape:
        print(f"[SKIP] Shape mismatch for {filename}")
        continue

    if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
        iou, empty_cases = 1.0, empty_cases + 1
    else:
        inter = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        iou = inter / union if union > 0 else 0.0

    ious.append(iou)
    print(f"{filename}: IoU = {iou:.4f}")

if ious:
    print("\\n=== Summary for $RUN_NAME ===")
    print(f"Evaluated {len(ious)} patches")
    print(f"Average IoU: {np.mean(ious):.4f}")
    print(f"Empty-background matches (IoU=1): {empty_cases}")
else:
    print("No valid IoUs were computed.")
EOF
}


# ========================
# MAIN LOOP
# ========================

for LR in "${LRS[@]}"; do
for BS in "${BATCH_SIZES[@]}"; do
for RAYS in "${N_RAYS[@]}"; do
for GRID in "${GRID_SIZES[@]}"; do

    RUN_NAME="run_${VERSION}_lr${LR}_bs${BS}_rays${RAYS}_grid${GRID// /x}"

    run_train $LR $BS $RAYS "$GRID"
    run_infer $RUN_NAME
    run_eval $RUN_NAME

done
done
done
done
