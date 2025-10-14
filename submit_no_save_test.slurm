#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --job-name=no_save_test
#SBATCH --output=logs/no_save_%j.out
#SBATCH --error=logs/no_save_%j.err

source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

mkdir -p logs

python scripts/train_no_save_test.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --n_epochs 2 \
    --max_samples 4
