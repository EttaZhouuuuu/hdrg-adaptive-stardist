#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=multiscale_5scales
#SBATCH --output=logs/multiscale_5scales_%j.out
#SBATCH --error=logs/multiscale_5scales_%j.err

source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

mkdir -p models/multiscale_5scales logs

python scripts/train_multiscale_enhanced.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/multiscale_5scales \
    --n_epochs 120 \
    --batch_size 4 \
    --max_samples 40 \
    --scale_factors "1.0,0.8,0.6,0.4,0.2"
