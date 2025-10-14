#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=multiscale_long
#SBATCH --output=logs/multiscale_long_%j.out
#SBATCH --error=logs/multiscale_long_%j.err

# 激活环境
source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

# 创建必要的目录
mkdir -p models/multiscale_long logs

# 运行长时间多尺度训练
echo "=== 开始长时间多尺度训练 ==="
echo "开始时间: $(date)"

python scripts/train_multiscale_enhanced.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/multiscale_long \
    --n_epochs 50 \
    --batch_size 4 \
    --max_samples 50 \
    --scale_factors "1.0,0.5" \
    --resume

echo "结束时间: $(date)"
echo "=== 长时间多尺度训练完成 ==="
