#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --job-name=multiscale_4scales
#SBATCH --output=logs/multiscale_4scales_%j.out
#SBATCH --error=logs/multiscale_4scales_%j.err

# 激活环境
source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

# 创建必要的目录
mkdir -p models/multiscale_4scales logs

# 运行4尺度多尺度训练
echo "=== 开始4尺度多尺度训练 ==="
echo "开始时间: $(date)"
echo "尺度因子: 1.0, 0.75, 0.5, 0.25"

python scripts/train_multiscale_enhanced.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/multiscale_4scales \
    --n_epochs 40 \
    --batch_size 3 \
    --max_samples 60 \
    --scale_factors "1.0,0.75,0.5,0.25" \
    --resume

echo "结束时间: $(date)"
echo "=== 4尺度多尺度训练完成 ==="
