#!/bin/bash

# Upload DSB2018 dataset to Google Drive
# This script helps prepare and upload data for Colab training

set -e

echo "================================================"
echo "Upload DSB2018 Dataset to Google Drive"
echo "================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if data exists
DATA_DIR="../data/dsb2018"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: DSB2018 dataset not found at $DATA_DIR${NC}"
    echo "Please run ./setup_dsb2018.sh first"
    exit 1
fi

echo -e "${GREEN}✓ Found DSB2018 dataset${NC}"

# Create zip file
echo -e "\n${YELLOW}Step 1: Compressing dataset...${NC}"
cd ../data
if [ -f "dsb2018.zip" ]; then
    echo "Removing old zip file..."
    rm dsb2018.zip
fi

zip -r dsb2018.zip dsb2018/ -q
ZIP_SIZE=$(du -h dsb2018.zip | cut -f1)
echo -e "${GREEN}✓ Created dsb2018.zip (${ZIP_SIZE})${NC}"

# Instructions for upload
echo -e "\n${YELLOW}Step 2: Upload to Google Drive${NC}"
echo ""
echo "Please follow these steps:"
echo ""
echo "1. Open Google Drive in your browser:"
echo "   https://drive.google.com"
echo ""
echo "2. Create folder structure:"
echo "   My Drive → New Folder → 'shape_data'"
echo ""
echo "3. Upload the file:"
echo "   File location: $(pwd)/dsb2018.zip"
echo "   Upload to: My Drive/shape_data/"
echo ""
echo "4. Wait for upload to complete"
echo ""
echo -e "${YELLOW}Alternative: Use Google Drive Desktop Client${NC}"
echo "If you have Google Drive installed:"
echo "  cp dsb2018.zip ~/Google\\ Drive/shape_data/"
echo ""

# Verify
read -p "Press Enter when upload is complete..."

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Data preparation complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "1. Push code to GitHub:"
echo "   cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev"
echo "   git push origin feature/shape-aware-backbone"
echo ""
echo "2. Open Colab notebook:"
echo "   https://colab.research.google.com"
echo "   → GitHub → EttaZhouuuuu/hdrg-adaptive-stardist"
echo "   → feature/shape-aware-backbone"
echo "   → shape_aware_stardist/colab/train_on_colab.ipynb"
echo ""
echo "3. Follow COLAB_TRAINING_GUIDE.md for detailed instructions"
echo ""
