#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=simple_test
#SBATCH --output=logs/simple_%j.out
#SBATCH --error=logs/simple_%j.err

source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

mkdir -p models results logs

python scripts/train_simple_test.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/simple_test \
    --n_epochs 3 \
    --max_samples 10
