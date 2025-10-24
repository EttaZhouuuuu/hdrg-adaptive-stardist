#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=multiscale_4scales_test
#SBATCH --output=logs/multiscale_4scales_test_%j.out
#SBATCH --error=logs/multiscale_4scales_test_%j.err

# 激活环境
source /hpc/group/yizhanglab/yz814/anaconda3/etc/profile.d/conda.sh
conda activate /hpc/group/yizhanglab/yz814/envs/hdrg_adaptive

# 创建必要的目录
mkdir -p models/multiscale_4scales_test logs

# 运行4尺度多尺度训练 (测试版本 - 较少epochs)
echo "=== 开始4尺度多尺度训练 (测试版本) ==="
echo "开始时间: $(date)"
echo "尺度因子: 1.0, 0.75, 0.5, 0.25"

python scripts/train_multiscale_enhanced.py \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data \
    --slides 240819_Ji_N1_H_EScan \
    --ckpt_path ./models/multiscale_4scales_test \
    --n_epochs 10 \
    --batch_size 2 \
    --max_samples 30 \
    --scale_factors "1.0,0.75,0.5,0.25"

echo "结束时间: $(date)"
echo "=== 4尺度多尺度训练测试完成 ==="
