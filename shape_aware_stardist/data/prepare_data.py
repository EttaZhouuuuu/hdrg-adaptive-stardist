"""Data preparation utilities for Shape-aware StarDist"""

import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from tqdm import tqdm
import cv2
from skimage import io, transform

from ..configs.data_config import (
    DATASET_CONFIG,
    PROCESSING_CONFIG,
    INSTANCE_CONFIG,
    verify_data_paths
)

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('data_preparation')
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    
    return logger

def prepare_dataset(
    source_dir: Path,
    split: str = 'train',
    force_rebuild: bool = False
) -> None:
    """
    Prepare dataset from source directory.
    
    Args:
        source_dir: Source directory containing images and masks
        split: Dataset split ('train', 'val', 'test')
        force_rebuild: Whether to force rebuild the dataset
    """
    logger = setup_logging()
    logger.info(f"Preparing {split} dataset from {source_dir}")
    
    # Get target directories
    target_images = DATASET_CONFIG[split]['images']
    target_masks = DATASET_CONFIG[split]['masks']
    
    # Create directories
    target_images.mkdir(parents=True, exist_ok=True)
    target_masks.mkdir(parents=True, exist_ok=True)
    
    # Clear existing data if force_rebuild
    if force_rebuild:
        logger.info("Forcing rebuild - clearing existing data")
        shutil.rmtree(target_images, ignore_errors=True)
        shutil.rmtree(target_masks, ignore_errors=True)
        target_images.mkdir(parents=True)
        target_masks.mkdir(parents=True)
    
    # Process images
    image_files = sorted(list(source_dir.glob("*.tiff")) + list(source_dir.glob("*.png")))
    mask_dir = source_dir / "ground_truth"
    
    for img_file in tqdm(image_files, desc=f"Processing {split} data"):
        # Load and process image
        image = io.imread(str(img_file))
        
        # Load corresponding mask
        mask_file = mask_dir / f"{img_file.stem}_mask{img_file.suffix}"
        if not mask_file.exists():
            mask_file = mask_dir / f"{img_file.stem}.tiff"
        
        if not mask_file.exists():
            logger.warning(f"No mask found for {img_file.name}")
            continue
        
        mask = io.imread(str(mask_file))
        
        # Process data
        processed_image, processed_mask = process_data_pair(image, mask)
        
        # Save processed data
        io.imsave(
            str(target_images / img_file.name),
            processed_image,
            check_contrast=False
        )
        io.imsave(
            str(target_masks / img_file.name),
            processed_mask,
            check_contrast=False
        )
    
    logger.info(f"Dataset preparation completed. Processed {len(image_files)} images.")

def process_data_pair(
    image: np.ndarray,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process image and mask pair.
    
    Args:
        image: Input image
        mask: Input mask
        
    Returns:
        tuple: (processed_image, processed_mask)
    """
    # Ensure correct dimensions
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Resize to target size
    target_size = PROCESSING_CONFIG['input_size']
    if image.shape[:2] != target_size:
        image = transform.resize(
            image,
            target_size + (3,),
            preserve_range=True
        ).astype(np.uint8)
        mask = transform.resize(
            mask,
            target_size,
            preserve_range=True,
            order=0
        ).astype(np.uint16)
    
    # Filter instances by size
    if INSTANCE_CONFIG['min_object_size'] or INSTANCE_CONFIG['max_object_size']:
        mask = filter_instances_by_size(
            mask,
            min_size=INSTANCE_CONFIG['min_object_size'],
            max_size=INSTANCE_CONFIG['max_object_size']
        )
    
    return image, mask

def filter_instances_by_size(
    mask: np.ndarray,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None
) -> np.ndarray:
    """
    Filter instances by size.
    
    Args:
        mask: Instance segmentation mask
        min_size: Minimum instance size
        max_size: Maximum instance size
        
    Returns:
        Filtered mask
    """
    filtered_mask = np.zeros_like(mask)
    instance_ids = np.unique(mask)[1:]  # Exclude background
    
    for instance_id in instance_ids:
        instance_mask = mask == instance_id
        instance_size = np.sum(instance_mask)
        
        if ((min_size is None or instance_size >= min_size) and
            (max_size is None or instance_size <= max_size)):
            filtered_mask[instance_mask] = instance_id
    
    return filtered_mask

def verify_dataset(split: str) -> bool:
    """
    Verify dataset integrity.
    
    Args:
        split: Dataset split to verify
        
    Returns:
        bool: Whether verification passed
    """
    logger = setup_logging()
    logger.info(f"Verifying {split} dataset")
    
    image_dir = DATASET_CONFIG[split]['images']
    mask_dir = DATASET_CONFIG[split]['masks']
    
    # Check directories exist
    if not image_dir.exists() or not mask_dir.exists():
        logger.error(f"Dataset directories not found: \n{image_dir}\n{mask_dir}")
        return False
    
    logger.info(f"Found directories:\nImages: {image_dir}\nMasks: {mask_dir}")
    
    # Get file lists
    image_files = sorted(list(image_dir.glob("*.tiff")) + list(image_dir.glob("*.png")))
    mask_files = sorted(list(mask_dir.glob("*.tiff")) + list(mask_dir.glob("*.png")))
    
    logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Check number of files
    if len(image_files) == 0:
        logger.error("No image files found!")
        return False
        
    if len(mask_files) == 0:
        logger.error("No mask files found!")
        return False
    
    if len(image_files) != len(mask_files):
        logger.error(
            f"Mismatch in number of files: "
            f"{len(image_files)} images vs {len(mask_files)} masks"
        )
        return False
    
    # Check each pair
    total_instances = 0
    instance_sizes = []
    
    for img_file, mask_file in tqdm(zip(image_files, mask_files), 
                                   desc="Verifying files",
                                   total=len(image_files)):
        # Check names match
        if img_file.stem != mask_file.stem:
            logger.error(f"Filename mismatch: {img_file.name} vs {mask_file.name}")
            return False
        
        # Load files
        try:
            image = io.imread(str(img_file))
            mask = io.imread(str(mask_file))
        except Exception as e:
            logger.error(f"Error loading {img_file.name}: {str(e)}")
            return False
        
        # Check dimensions
        if image.shape[:2] != mask.shape[:2]:
            logger.error(
                f"Shape mismatch in {img_file.name}: "
                f"{image.shape} vs {mask.shape}"
            )
            return False
            
        # Verify image properties
        if image.shape[:2] != (2048, 2048):
            logger.error(
                f"Unexpected image size in {img_file.name}: "
                f"{image.shape[:2]} (expected (2048, 2048))"
            )
            return False
        
        # Check image type and range
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            logger.error(f"Unexpected image type in {img_file.name}: {image.dtype}")
            return False
        
        # Analyze instances in mask
        unique_instances = np.unique(mask)
        num_instances = len(unique_instances) - 1  # Exclude background
        total_instances += num_instances
        
        # Collect instance sizes
        for instance_id in unique_instances[1:]:  # Skip background
            instance_size = np.sum(mask == instance_id)
            instance_sizes.append(instance_size)
    
    # Report statistics
    logger.info(f"\nDataset Statistics for {split}:")
    logger.info(f"Total number of images: {len(image_files)}")
    logger.info(f"Total number of instances: {total_instances}")
    logger.info(f"Average instances per image: {total_instances/len(image_files):.2f}")
    
    if instance_sizes:
        logger.info(f"Instance size statistics:")
        logger.info(f"  Min: {min(instance_sizes)}")
        logger.info(f"  Max: {max(instance_sizes)}")
        logger.info(f"  Mean: {np.mean(instance_sizes):.2f}")
        logger.info(f"  Median: {np.median(instance_sizes):.2f}")
    
    logger.info(f"\nDataset verification passed for {split}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for Shape-aware StarDist")
    parser.add_argument('--source_dir', type=str, required=True,
                      help='Source directory containing images and masks')
    parser.add_argument('--split', type=str, default='train',
                      choices=['train', 'val', 'test'],
                      help='Dataset split')
    parser.add_argument('--force_rebuild', action='store_true',
                      help='Force rebuild dataset')
    parser.add_argument('--verify_only', action='store_true',
                      help='Only verify dataset without preparation')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.split)
    else:
        prepare_dataset(
            Path(args.source_dir),
            split=args.split,
            force_rebuild=args.force_rebuild
        )
