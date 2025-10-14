#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=multiscale_fixed
#SBATCH --output=logs/multiscale_fixed_%j.out
#SBATCH --error=logs/multiscale_fixed_%j.err

# 激活环境
source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

# 创建必要的目录
mkdir -p models/multiscale_fixed logs

# 运行修复版本的多尺度训练
echo "=== 开始多尺度训练 (修复版本) ==="
python scripts/train_multiscale_fixed.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/multiscale_fixed \
    --n_epochs 50 \
    --batch_size 2 \
    --max_samples 12 \
    --scale_factors "1.0,0.5"

echo "=== 多尺度训练完成 ==="
