#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=multiscale_work
#SBATCH --output=logs/multiscale_%j.out
#SBATCH --error=logs/multiscale_%j.err

source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

mkdir -p models results logs

python scripts/train_multiscale_working.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/multiscale \
    --n_epochs 3 \
    --max_samples 12 \
    --scale_factors "1.0,0.5"
