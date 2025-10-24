"""
Download and prepare DSB2018 dataset for Shape-aware StarDist
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import numpy as np
from skimage import io
from tqdm import tqdm
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('DSB2018_Downloader')

def download_dsb2018(data_dir: Path):
    """
    Download DSB2018 dataset using Kaggle API.
    
    Args:
        data_dir: Directory to save the dataset
    """
    logger = setup_logging()
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle API not found. Installing...")
        os.system("pip install kaggle")
        import kaggle
    
    # Check if kaggle.json exists
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        logger.error(
            "Kaggle API token not found!\n"
            "Please follow these steps:\n"
            "1. Go to https://www.kaggle.com/settings\n"
            "2. Click 'Create New API Token'\n"
            "3. Move the downloaded kaggle.json to ~/.kaggle/\n"
            "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        )
        return False
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    logger.info("Downloading DSB2018 dataset from Kaggle...")
    os.system(f"kaggle competitions download -c data-science-bowl-2018 -p {data_dir}")
    
    # Unzip files
    logger.info("Extracting files...")
    zip_files = list(data_dir.glob("*.zip"))
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        zip_file.unlink()  # Remove zip file after extraction
    
    logger.info("Download completed!")
    return True

def prepare_dsb2018(data_dir: Path, output_dir: Path):
    """
    Prepare DSB2018 dataset for training.
    
    Args:
        data_dir: Directory containing raw DSB2018 data
        output_dir: Directory to save processed data
    """
    logger = setup_logging()
    logger.info("Preparing DSB2018 dataset...")
    
    # Create output directories
    train_images_dir = output_dir / 'train' / 'images'
    train_masks_dir = output_dir / 'train' / 'masks'
    test_images_dir = output_dir / 'test' / 'images'
    test_masks_dir = output_dir / 'test' / 'masks'
    
    for dir_path in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    train_dir = data_dir / 'stage1_train'
    if train_dir.exists():
        logger.info("Processing training data...")
        process_split(train_dir, train_images_dir, train_masks_dir, logger)
    
    # Process test data
    test_dir = data_dir / 'stage1_test'
    if test_dir.exists():
        logger.info("Processing test data...")
        process_split(test_dir, test_images_dir, test_masks_dir, logger)
    
    # Create validation split from training data
    logger.info("Creating validation split...")
    create_validation_split(train_images_dir, train_masks_dir, output_dir, split_ratio=0.2)
    
    logger.info("Dataset preparation completed!")
    
    # Print statistics
    print_dataset_statistics(output_dir, logger)

def process_split(input_dir: Path, images_dir: Path, masks_dir: Path, logger):
    """
    Process a data split (train or test).
    
    Args:
        input_dir: Input directory containing image folders
        images_dir: Output directory for images
        masks_dir: Output directory for masks
        logger: Logger instance
    """
    image_folders = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    for idx, image_folder in enumerate(tqdm(image_folders, desc="Processing images")):
        image_id = image_folder.name
        
        # Load image
        image_path = image_folder / 'images' / f'{image_id}.png'
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        image = io.imread(str(image_path))
        
        # Load and combine masks
        masks_folder = image_folder / 'masks'
        if masks_folder.exists():
            mask_files = sorted(list(masks_folder.glob('*.png')))
            
            if len(mask_files) == 0:
                logger.warning(f"No masks found for {image_id}")
                continue
            
            # Create instance mask
            instance_mask = np.zeros(image.shape[:2], dtype=np.uint16)
            
            for instance_id, mask_file in enumerate(mask_files, start=1):
                mask = io.imread(str(mask_file))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                instance_mask[mask > 0] = instance_id
        else:
            # For test set without masks, create empty mask
            instance_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        
        # Save processed data
        output_image_path = images_dir / f'{image_id}.png'
        output_mask_path = masks_dir / f'{image_id}.png'
        
        io.imsave(str(output_image_path), image, check_contrast=False)
        io.imsave(str(output_mask_path), instance_mask, check_contrast=False)

def create_validation_split(train_images_dir: Path, train_masks_dir: Path, 
                           output_dir: Path, split_ratio: float = 0.2):
    """
    Create validation split from training data.
    
    Args:
        train_images_dir: Training images directory
        train_masks_dir: Training masks directory
        output_dir: Output directory
        split_ratio: Ratio of validation data
    """
    # Get all image files
    image_files = sorted(list(train_images_dir.glob('*.png')))
    
    # Calculate split
    n_val = int(len(image_files) * split_ratio)
    
    # Randomly select validation images
    np.random.seed(42)
    val_indices = np.random.choice(len(image_files), n_val, replace=False)
    
    # Create validation directories
    val_images_dir = output_dir / 'val' / 'images'
    val_masks_dir = output_dir / 'val' / 'masks'
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Move validation files
    for idx in val_indices:
        image_file = image_files[idx]
        mask_file = train_masks_dir / image_file.name
        
        shutil.move(str(image_file), str(val_images_dir / image_file.name))
        shutil.move(str(mask_file), str(val_masks_dir / mask_file.name))

def print_dataset_statistics(output_dir: Path, logger):
    """Print dataset statistics."""
    splits = ['train', 'val', 'test']
    
    logger.info("\n" + "="*50)
    logger.info("Dataset Statistics:")
    logger.info("="*50)
    
    for split in splits:
        images_dir = output_dir / split / 'images'
        masks_dir = output_dir / split / 'masks'
        
        if images_dir.exists():
            n_images = len(list(images_dir.glob('*.png')))
            n_masks = len(list(masks_dir.glob('*.png')))
            
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Images: {n_images}")
            logger.info(f"  Masks: {n_masks}")
            
            # Calculate average number of instances
            if n_masks > 0:
                total_instances = 0
                for mask_file in masks_dir.glob('*.png'):
                    mask = io.imread(str(mask_file))
                    n_instances = len(np.unique(mask)) - 1  # Exclude background
                    total_instances += n_instances
                
                avg_instances = total_instances / n_masks
                logger.info(f"  Average instances per image: {avg_instances:.2f}")
    
    logger.info("="*50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare DSB2018 dataset")
    parser.add_argument('--download', action='store_true',
                      help='Download dataset from Kaggle')
    parser.add_argument('--data_dir', type=str, 
                      default='/Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/data/dsb2018_raw',
                      help='Directory to save raw data')
    parser.add_argument('--output_dir', type=str,
                      default='/Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/data/dsb2018',
                      help='Directory to save processed data')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Download dataset
    if args.download:
        success = download_dsb2018(data_dir)
        if not success:
            sys.exit(1)
    
    # Prepare dataset
    prepare_dsb2018(data_dir, output_dir)
