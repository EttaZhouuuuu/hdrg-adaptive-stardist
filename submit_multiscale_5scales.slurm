#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --job-name=multiscale_3scales
#SBATCH --output=logs/multiscale_3scales_%j.out
#SBATCH --error=logs/multiscale_3scales_%j.err

# 激活环境
source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

# 创建必要的目录
mkdir -p models/multiscale_5scales logs

# 运行3尺度多尺度训练 (稳定版本)
echo "=== 开始5尺度多尺度训练 (稳定版本) ==="
echo "开始时间: $(date)"
echo "尺度因子: 1.0, 0.75, 0.5, 0.25, 0.125"

python scripts/train_multiscale_enhanced.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/multiscale_3scales \
    --n_epochs 120 \
    --batch_size 3 \
    --max_samples 50 \
    --scale_factors "1.0,0.75,0.5, 0.25, 0.125" \
    --resume

echo "结束时间: $(date)"
echo "=== 5尺度多尺度训练完成 ==="
