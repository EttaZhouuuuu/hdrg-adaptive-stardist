#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=adaptive_stardist
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@duke.edu

echo "🚀 Starting Multi-Scale Adaptive StarDist Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Load modules
module load Anaconda3
module load CUDA/11.8

# Activate environment
source activate hdrg_adaptive

# Set up environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS="--xla_gpu_autotune_level=0"

# Print system info
echo "GPU Info:"
nvidia-smi

echo "Python Environment:"
which python
python --version
pip list | grep -E "(tensorflow|numpy|scikit-image)"

# Run training with real data (modify paths as needed)
echo "🎯 Starting training..."

# Option 1: Use real HPC data (if available)
# python scripts/train_adaptive_stardist.py \
#     --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/ \
#     --slides 240819_Ji_N1_H_EScan \
#     --ckpt_path ./models/adaptive_real_$(date +%Y%m%d_%H%M) \
#     --n_epochs 100 \
#     --use_gpu \
#     --scale_factors 1.0,0.5,0.25 \
#     --fusion_mode attention

# Option 2: Use test data (for demonstration)
python scripts/train_adaptive_stardist.py \
    --data_path ./data \
    --slides 240819_Ji_N1_H_EScan,test_slide_1 \
    --ckpt_path ./models/adaptive_test_$(date +%Y%m%d_%H%M) \
    --n_epochs 20 \
    --use_gpu \
    --scale_factors 1.0,0.5,0.25 \
    --fusion_mode attention

echo "✅ Training completed at $(date)"

# Run inference test
echo "🔮 Testing inference..."
python scripts/run_adaptive_inference.py \
    --model_path ./models/adaptive_test_$(date +%Y%m%d_%H%M) \
    --input_path ./data/240819_Ji_N1_H_EScan/patches_2048 \
    --output_dir ./results/test_$(date +%Y%m%d_%H%M) \
    --save_confidence \
    --save_parameters

echo "🎉 Job completed successfully at $(date)"
