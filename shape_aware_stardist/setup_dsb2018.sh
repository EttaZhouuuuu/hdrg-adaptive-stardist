#!/bin/bash

# DSB2018 Dataset Setup Script
# This script automates the setup process for DSB2018 dataset

set -e  # Exit on error

echo "================================================"
echo "DSB2018 Dataset Setup for Shape-aware StarDist"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if kaggle.json exists
echo -e "\n${YELLOW}Step 1: Checking Kaggle API credentials...${NC}"
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}Error: kaggle.json not found!${NC}"
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Click 'Create New API Token'"
    echo "3. Move kaggle.json to ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
else
    echo -e "${GREEN}✓ Kaggle API credentials found${NC}"
fi

# Install dependencies
echo -e "\n${YELLOW}Step 2: Installing dependencies...${NC}"
pip3 install kaggle scikit-image tqdm albumentations --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Download and prepare dataset
echo -e "\n${YELLOW}Step 3: Downloading DSB2018 dataset...${NC}"
echo "This may take several minutes depending on your internet speed..."
python3 -m shape_aware_stardist.data.download_dsb2018 --download

# Verify dataset
echo -e "\n${YELLOW}Step 4: Verifying dataset...${NC}"
python3 -m shape_aware_stardist.data.dsb2018_dataset

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "1. Review the dataset at: data/dsb2018/"
echo "2. Start training: python examples/train_dsb2018.py"
echo "3. Or use Colab: colab/train_on_colab.ipynb"
echo ""
echo "For more information, see: DSB2018_SETUP.md"
