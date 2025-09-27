# 🚀 Duke DCC Server Transfer Guide

## 📋 **Quick Transfer Checklist**

- [ ] Choose transfer method (Git recommended)
- [ ] Upload code to DCC server
- [ ] Set up conda environment
- [ ] Install dependencies
- [ ] Verify data access
- [ ] Test pipeline
- [ ] Submit training job

---

## 🔄 **Transfer Methods**

### **Method 1: Git Repository (Recommended) ⭐**

**Step 1: Create Repository Locally**
```bash
# In your local project directory
cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev
git init
git add .
git commit -m "Multi-Scale Adaptive StarDist project"

# Push to GitHub (create repo first on github.com)
git remote add origin https://github.com/YOUR_USERNAME/hdrg-adaptive-stardist.git
git branch -M main
git push -u origin main
```

**Step 2: Clone on DCC**
```bash
# SSH to DCC
ssh YOUR_NETID@dcc-login.oit.duke.edu

# Clone repository
cd ~/projects
git clone https://github.com/YOUR_USERNAME/hdrg-adaptive-stardist.git
cd hdrg-adaptive-stardist
```

### **Method 2: Direct SCP Transfer**

```bash
# From your Mac terminal
scp -r /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev \
    YOUR_NETID@dcc-login.oit.duke.edu:~/projects/hdrg-adaptive-stardist
```

### **Method 3: rsync (Efficient for updates)**

```bash
# Initial transfer
rsync -avz --progress /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/ \
    YOUR_NETID@dcc-login.oit.duke.edu:~/projects/hdrg-adaptive-stardist/

# For updates (only transfers changed files)
rsync -avz --progress --delete /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/ \
    YOUR_NETID@dcc-login.oit.duke.edu:~/projects/hdrg-adaptive-stardist/
```

---

## ⚙️ **DCC Setup Process**

### **Step 1: SSH to DCC**
```bash
ssh YOUR_NETID@dcc-login.oit.duke.edu
cd ~/projects/hdrg-adaptive-stardist
```

### **Step 2: Run Automated Setup**
```bash
# Use the provided setup script
./setup_dcc.sh
```

**Or Manual Setup:**

### **Step 3: Load Modules**
```bash
module load Anaconda3
module load CUDA/11.8  # Check: module avail CUDA
```

### **Step 4: Create Environment**
```bash
# Option A: Use provided environment file
conda env create -f envs/hdrg_mac.yml -n hdrg_adaptive

# Option B: Manual setup
conda create -n hdrg_adaptive python=3.9 -y
conda activate hdrg_adaptive
pip install tensorflow==2.10.0 scikit-image matplotlib seaborn pandas opencv-python tqdm
pip install numpy==1.24.3
```

### **Step 5: Install Package**
```bash
conda activate hdrg_adaptive
pip install -e .
```

### **Step 6: Create Directories**
```bash
mkdir -p models logs results evaluation experiments/results
```

---

## 🧪 **Testing the Setup**

### **1. Generate Test Data**
```bash
python scripts/test_setup.py
```

### **2. Verify Installation**
```bash
python scripts/demo_usage.py
```

### **3. Check Real Data Access**
```bash
ls -la /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/
```

---

## 🎯 **Running Jobs on DCC**

### **Interactive Testing**
```bash
# Request GPU node for testing
srun --partition=gpu-common --gres=gpu:1 --mem=16G --time=2:00:00 --pty bash

# Activate environment
conda activate hdrg_adaptive

# Quick test
python scripts/demo_usage.py
```

### **Batch Training**

**1. Edit SLURM script:**
```bash
nano submit_training.slurm
# Update your email address and adjust paths if needed
```

**2. Submit job:**
```bash
sbatch submit_training.slurm
```

**3. Monitor job:**
```bash
squeue -u YOUR_NETID
tail -f logs/training_JOBID.out
```

---

## 📊 **Data Configuration**

### **Option A: Use Original HPC Data**
```bash
# Check data availability
ls -la /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/240819_Ji_N1_H_EScan/

# Use real data helper
python scripts/run_with_real_data.py --action train \
    --data_path /hpc/group/yizhanglab/shared/hDRG_autoseg/processed_data/
```

### **Option B: Use Test Data**
```bash
# Generate synthetic test data
python scripts/test_setup.py

# Verify structure
python scripts/demo_usage.py
```

---

## 🚨 **Common Issues & Solutions**

### **1. Module Not Found**
```bash
# Check available modules
module avail
module avail CUDA
module avail Anaconda
```

### **2. GPU Not Available**
```bash
# Check GPU partitions
sinfo -p gpu-common
sinfo -p gpu-scavenger

# Alternative GPU request
srun --partition=gpu-scavenger --gres=gpu:1 --mem=16G --time=2:00:00 --pty bash
```

### **3. Memory Issues**
```bash
# Request more memory
srun --partition=gpu-common --gres=gpu:1 --mem=32G --time=4:00:00 --pty bash
```

### **4. Conda Environment Issues**
```bash
# Reset environment
conda env remove -n hdrg_adaptive
conda create -n hdrg_adaptive python=3.9 -y
```

---

## 📝 **Quick Command Reference**

```bash
# SSH to DCC
ssh YOUR_NETID@dcc-login.oit.duke.edu

# Transfer files
scp -r LOCAL_PATH YOUR_NETID@dcc-login.oit.duke.edu:~/projects/

# Setup environment
./setup_dcc.sh

# Interactive GPU session
srun --partition=gpu-common --gres=gpu:1 --mem=16G --time=2:00:00 --pty bash

# Submit training job
sbatch submit_training.slurm

# Check job status
squeue -u YOUR_NETID

# Monitor logs
tail -f logs/training_*.out
```

---

## 🎉 **Success Verification**

Your setup is successful when:
- ✅ `python scripts/demo_usage.py` runs without errors
- ✅ GPU is detected: `nvidia-smi` shows GPU info
- ✅ Data structure is correct: Binary masks with values [0, 255]
- ✅ Training starts without module import errors

**You're ready to run the Multi-Scale Adaptive StarDist pipeline on Duke DCC! 🚀**
