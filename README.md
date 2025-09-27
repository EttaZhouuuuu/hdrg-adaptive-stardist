# hDRG-autoseg — Reproducible Segmentation Pipeline

This document describes a reproducible, “one-click” pipeline for training, inference, and evaluation of a StarDist-style 2D neuronal segmentation model on histological slide tiles. It formalizes the expected data layout, software environment, configurable hyperparameters, and end-to-end execution, with emphasis on clarity and reproducibility.

---

## Contents

**Overview**
**Repository structure**
**Data requirements**
**Software requirements**
**Configuration**
**End-to-end execution**
**Stage-wise execution**
**Hyperparameter sweep customization**
**Outputs**
**Troubleshooting**
**HPC/Slurm option (optional)**
**Reproducibility**
**Appendix: optional `.env` example**

---

## Overview

For a selected slide, the pipeline performs a grid search over a user-specified hyperparameter set (learning rate, batch size, number of rays, and grid size). For each hyperparameter configuration, the pipeline executes:

1. **Train** a StarDist-style 2D model and save the checkpoint
2. **Infer** per-patch predictions on that slide
3. **Evaluate** with IoU (per patch + average), writing a plain-text report

Each run has a unique `run_name`. Model artifacts and evaluation outputs are written under `ckpts/` and `output/`, respectively.

---

## Repository structure

```
hDRG-autoseg/
├── ckpts/                          # Model checkpoints per run_name
├── commands/
│   └── pipeline.sh                 # End-to-end grid pipeline (train → infer → eval)
├── data/results/                   # (optional) local scratch for data/results
├── envs/
│   └── hdrg.yml                    # Conda env (exported with --no-builds)
├── output/
│   ├── <run_name>_infer/           # Per-patch predictions (e.g., pred_labels.npy)
│   └── <run_name>_iou.txt          # IoU report (per patch + average)
├── results/
│   ├── evaluation/                 # Long-term eval archive (optional)
│   └── inference/                  # Long-term inference archive (optional)
├── scripts/
│   ├── compute_iou....py           # IoU utility (matches pipeline’s evaluator)
│   ├── constants.py                # Global constants (WANDB, labels, etc.)
│   ├── evaluation.py               # Eval/visualization helpers
│   ├── infer_main_2d.py            # Inference entry (per-patch prediction)
│   ├── stitch.py                   # (optional) Stitch patches / post-processing
│   ├── train_seg_neuron.py         # Training entry (LR, batch, n_rays, grid, etc.)
│   └── wandb_helper.py             # Weights & Biases helpers
├── stardist/                       # Vendored/modified StarDist code (if any)
└── stardist.egg-info/              # Package metadata created by install
```

### What the key scripts do

**`scripts/train_seg_neuron.py`** — trains a 2D model. CLI accepts `--data_path`, `--slides`, `--ckpt_path`, `--train_batch_size`, `--n_epochs`, `--n_rays`, `--train_lr`, `--grid_size`, etc.
**`scripts/infer_main_2d.py`** — runs inference on tiles under `patches_2048/`, writing `pred_labels.npy` under `output/<run_name>_infer/<patch_id>/`.
**`scripts/evaluation.py` / `scripts/compute_iou*.py`** — evaluation helpers; the pipeline embeds a small IoU evaluator that mirrors these.
`pipeline.sh` orchestrates the full grid (training → inference → evaluation) and writes a plain-text IoU report.

---

## Data requirements

For a selected `SLIDE`, the pipeline expects:

```
${DATA_PATH}/${SLIDE}/
  ├── patches_2048/                 # Input tiles: patch_col_<C>_row_<R>.png
  └── pseudo_gt_masks_2048/         # Matched GT masks (PNG; foreground = 255)
```

Patch file names must be paired between `patches_2048/` and `pseudo_gt_masks_2048/` using the pattern `patch_col_*_row_*.png`.

---

## Software requirements

A Conda environment is recommended.

```bash
conda env create -f envs/hdrg.yml
conda activate hdrg
# Install your project requirements here (csbdeep, stardist, skimage, numpy, matplotlib, wandb, openslide, etc.)
```

GPU is optional. Training supports CPU/GPU. Inference defaults to CPU in the pipeline but can be switched to `cuda:0` where appropriate.

---

## Configuration

Open `pipeline.sh` and review the variables near the top.

**Versioning and naming**
  `VERSION` — included in `run_name` to group runs

**Paths and slide**
  `DATA_PATH` — root directory containing slides
  `SLIDE` — slide identifier (directory name under `DATA_PATH`)
  `CKPT_BASE` — checkpoint root (e.g., `./ckpts`)
  `OUTPUT_BASE` — output root for inference and evaluation (e.g., `./output`)

**Hyperparameter grid**
  `LRS` — e.g., `("7e-4" "1e-3")`
  `BATCH_SIZES` — e.g., `(16)`
  `N_RAYS` — e.g., `(24 96)`
  `GRID_SIZES` — strings of the form `"r c"`, e.g., `("2 2" "4 4" "8 8")`

**Stage defaults**
  Training: `EPOCHS`, `--train_batch_size`, `--n_rays`, `--train_lr`, `--grid_size`
  Inference: `PATCH_LEN=2048`, `TARGET_LEN=256`, `PROB_THRESHOLD=0.2`, `DEVICE="cpu"`
  Evaluation: embedded Python (IoU computation)

To prefer environment variables, an optional `.env` can be sourced at the top of `pipeline.sh` (see the Appendix).

---

## End-to-end execution

From the repository root:

```bash
bash pipeline.sh
```

For each `(LR × BATCH_SIZE × N_RAYS × GRID)`, the script executes:

### 1) Training

```bash
python3 train_seg_neuron.py \
  --exp_id         "<run_name>" \
  --wandb_run_name "stardist-h&e2d-<run_name>-<VERSION>" \
  --data_path      "${DATA_PATH}" \
  --slides         "${SLIDE}" \
  --ckpt_path      "${CKPT_BASE}/<run_name>/" \
  --use_gpu \
  --train_batch_size <BATCH_SIZE> \
  --n_epochs       <EPOCHS> \
  --n_rays         <N_RAYS> \
  --train_lr       <LR> \
  --grid_size      "<GRID_ROW> <GRID_COL>"
```

Outputs: `${CKPT_BASE}/<run_name>/` (model files)

### 2) Inference (per patch)

```bash
python3 infer_main_2d.py \
  --model_ckpt_path "${CKPT_BASE}/<run_name>/" \
  --slide_fp        "${DATA_PATH}/${SLIDE}/patches_2048/patch_col_<C>_row_<R>.png" \
  --patch_len       2048 \
  --target_len      256 \
  --prob_threshold  0.2 \
  --device          "cpu"            # change to "cuda:0" for GPU \
  --output_dir      "${OUTPUT_BASE}/<run_name>_infer/patch_col_<C>_row_<R>"
```

Outputs: `${OUTPUT_BASE}/<run_name>_infer/patch_col_*_row_*/pred_labels.npy`

### 3) Evaluation (embedded)

Compares predictions to GT masks under `${DATA_PATH}/${SLIDE}/pseudo_gt_masks_2048/`
Produces `${OUTPUT_BASE}/<run_name>_iou.txt` with per-patch IoU and an average summary

---

## Stage-wise execution

**Train only**

```bash
python3 scripts/train_seg_neuron.py  # same args as above
```

**Infer only (all patches)**

```bash
PATCH_DIR="${DATA_PATH}/${SLIDE}/patches_2048"
OUT_DIR="output/<run_name>_infer"
mkdir -p "$OUT_DIR"
for p in "$PATCH_DIR"/patch_col_*_row_*.png; do
  id=$(basename "${p%.*}")
  python3 scripts/infer_main_2d.py \
    --model_ckpt_path "ckpts/<run_name>/" \
    --slide_fp "$p" \
    --patch_len 2048 --target_len 256 --prob_threshold 0.2 \
    --device "cuda:0" \
    --output_dir "$OUT_DIR/$id"
done
```

**Evaluate only**

Use `scripts/compute_iou*.py`, or copy the small embedded evaluator from the pipeline into a standalone script.
Inputs: `${OUTPUT_BASE}/<run_name>_infer` and GT folder.
Output: `${OUTPUT_BASE}/<run_name>_iou.txt`.

---

## Hyperparameter sweep customization

Adjust the grid near the top of `pipeline.sh`:

```bash
LRS=("7e-4" "1e-3")
GRID_SIZES=("2 2" "4 4" "8 8")
```

Switch the inference device to GPU:

```bash
DEVICE="cuda:0"
```

Modify the probability threshold:

```bash
PROB_THRESHOLD=0.25
```

---

## Outputs

**Checkpoints**: `ckpts/<run_name>/`
**Predictions**: `output/<run_name>_infer/patch_col_*_row_*/pred_labels.npy`
**IoU report**: `output/<run_name>_iou.txt`
(Optional) Long-term archival under `results/inference/` and `results/evaluation/`

---

## Troubleshooting

**Missing tiles or GT masks**: Ensure both directories exist and contain paired files.

```
${DATA_PATH}/${SLIDE}/patches_2048/
${DATA_PATH}/${SLIDE}/pseudo_gt_masks_2048/
```

**“Missing prediction …” during evaluation**: Some patch predictions are absent. Re-run inference for the missing patches and verify `pred_labels.npy` paths.

**Shape mismatch**: Ensure `--patch_len` and `--target_len` are consistent with preprocessing. Confirm GT masks align with tile dimensions.

**Slow inference**: Switch to GPU (`--device cuda:0`) and verify `torch.cuda.is_available()`.

**Weights & Biases authentication**: If using W&B, set `WANDB_API_KEY` via environment or `.env` (do not commit secrets).

---

## HPC/Slurm option (optional)

On HPC clusters, it is typical to run one job per hyperparameter combination and chain stages with job dependencies, e.g., `--dependency=afterok:<JOBID>`. A thin wrapper can generate `train.sh`, `infer.sh`, and `eval.sh` for each run and submit them via a helper (e.g., `submit_all.sh`).

---

## Reproducibility

Pin the environment and avoid embedding paths:

```bash
conda env export --no-builds | awk '!/^prefix: /' > envs/hdrg.yml
python -m pip freeze > envs/requirements.txt
```

Use a consistent run-name scheme:

```
run_<VERSION>_lr<LR>_bs<BATCH>_rays<N_RAYS>_grid<R>x<C>
```

Optionally record per-run metadata (e.g., slide, seed, grid) in a small `run_meta.json` colocated with checkpoints.

---

## Appendix: optional `.env` example

Create `.env` (never commit secrets):

```dotenv
DATA_PATH=/hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data
SLIDE=240819_Ji_N1_H_EScan
CKPT_BASE=./ckpts
OUTPUT_BASE=./output
VERSION=v1.0.4
PROB_THRESHOLD=0.20
DEVICE=cpu
```

Source it at the top of `pipeline.sh`:

```bash
[ -f ".env" ] && set -a && source .env && set +a
```

The pipeline can then be executed end-to-end, and outputs will be available under `ckpts/` and `output/` as described above.
