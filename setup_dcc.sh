#!/bin/bash
# Setup script for Duke DCC server

echo "🚀 Setting up Multi-Scale Adaptive StarDist on Duke DCC"

# Check if we're on DCC
if [[ ! "$HOSTNAME" == *"dcc"* ]]; then
    echo "❌ This script should be run on Duke DCC server"
    echo "💡 Transfer this project to DCC first using:"
    echo "   scp -r /path/to/hDRG-autoseg-etta_dev your_netid@dcc-login.oit.duke.edu:~/projects/"
    exit 1
fi

# Load required modules
echo "📦 Loading modules..."
module load Anaconda3
module load CUDA/11.8 2>/dev/null || module load CUDA/11.0 2>/dev/null || echo "⚠️  CUDA module not found"

# Create conda environment
echo "🐍 Setting up conda environment..."
if conda env list | grep -q "hdrg_adaptive"; then
    echo "✓ Environment already exists"
    conda activate hdrg_adaptive
else
    # Try to use the provided environment file, fallback to manual setup
    if [ -f "envs/hdrg_mac.yml" ]; then
        conda env create -f envs/hdrg_mac.yml -n hdrg_adaptive
    else
        echo "📋 Creating environment manually..."
        conda create -n hdrg_adaptive python=3.9 -y
        conda activate hdrg_adaptive
        pip install tensorflow==2.10.0 scikit-image matplotlib seaborn pandas opencv-python tqdm
        pip install numpy==1.24.3  # Compatibility fix
    fi
    conda activate hdrg_adaptive
fi

# Install the package
echo "⚙️  Installing package..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs results evaluation experiments/results

# Test data setup
echo "🧪 Setting up test data..."
python scripts/test_setup.py

# Verify installation
echo "✅ Verifying installation..."
python scripts/demo_usage.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "💡 Next steps:"
echo "   1. For interactive testing:"
echo "      srun --partition=gpu-common --gres=gpu:1 --mem=16G --time=2:00:00 --pty bash"
echo "      conda activate hdrg_adaptive"
echo ""
echo "   2. For batch training:"
echo "      sbatch submit_training.slurm"
echo ""
echo "   3. Check available data:"
echo "      ls -la /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/"
echo ""
echo "   4. Run with real data:"
echo "      python scripts/run_with_real_data.py --action train --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/"
