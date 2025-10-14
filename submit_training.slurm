#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=adaptive_stardist
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# 激活环境
source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

# 创建必要的目录
mkdir -p models results logs

# 运行训练（使用正确的数据路径）
python scripts/train_adaptive_stardist.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/adaptive \
    --n_epochs 50 \
    --use_gpu
