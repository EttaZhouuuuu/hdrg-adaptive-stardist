"""
Xenium Data Preprocessing Module
================================

Convert Xenium polygon-based cell segmentation to bitmap instance masks
for training StarDist models.

Key functionality:
- Load polygon data from cells.zarr
- Convert polygons to rasterized instance masks
- Extract image patches and corresponding labels
- Generate training data in StarDist format

Key findings:
- Zarr contains 'masks/1' array with pre-rasterized instance masks at full resolution
- polygon_vertices coordinates need to be scaled by ~4.89x (X) and ~3.35x (Y) to match image
- morphology_focus.ome.tif is the best 2D image for training (2D best-focus projection)
"""

import zarr
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from scipy import ndimage
from skimage import draw, measure, transform
import warnings

warnings.filterwarnings('ignore')


class XeniumDataLoader:
    """
    Load and preprocess Xenium data for StarDist training.
    
    This class handles:
    - Loading polygon-based cell segmentation from Xenium zarr
    - Converting polygons to rasterized instance masks
    - Extracting image patches with corresponding labels
    
    Attributes:
        cells_zarr_path: Path to cells.zarr directory
        image_path: Path to morphology image (H&E)
        cell_id: Array of cell IDs
        polygon_vertices: Polygon vertices for each cell
        polygon_num_vertices: Number of vertices per polygon
        cell_summary: Summary statistics per cell
        image_height, image_width: Actual image dimensions
        scale_x, scale_y: Scaling factors for polygon coordinates
    """
    
    def __init__(
        self,
        cells_zarr_path: str,
        image_path: Optional[str] = None,
        load_image: bool = False
    ):
        """
        Initialize Xenium data loader.
        
        Args:
            cells_zarr_path: Path to cells.zarr directory
            image_path: Optional path to morphology image (morphology_focus.ome.tif recommended)
            load_image: Whether to load the image immediately
        """
        self.cells_zarr_path = Path(cells_zarr_path)
        self.image_path = Path(image_path) if image_path else None
        
        # Pre-computed masks from zarr (if available)
        self.precomputed_masks: Optional[np.ndarray] = None
        
        # Load zarr data
        self._load_zarr_data()
        
        # Load image if requested
        self.image = None
        if load_image and self.image_path:
            self._load_image()
    
    def _load_zarr_data(self):
        """Load polygon data from zarr archive."""
        root = zarr.open(str(self.cells_zarr_path), mode='r')
        
        # Key arrays for polygon-based segmentation
        self.cell_id = root['cell_id'][:]  # (N,) cell IDs
        self.polygon_vertices = root['polygon_vertices'][:]  # (2, N, max_vertices)
        self.polygon_num_vertices = root['polygon_num_vertices'][:]  # (2, N)
        self.cell_summary = root['cell_summary'][:]  # (N, 7)
        
        # Check for pre-computed masks (Xenium provides these at full resolution)
        if 'masks/1' in root:
            self.precomputed_masks = root['masks/1'][:]
            print(f"✓ Loaded pre-computed masks from zarr: shape={self.precomputed_masks.shape}")
        
        # Calculate scaling factors for polygon coordinates
        # polygon_vertices coordinates are in a normalized space
        # They need to be scaled to match the actual morphology image
        coord_x_max = self.polygon_vertices[0].max()
        coord_y_max = self.polygon_vertices[1].max()
        
        # These will be updated when image is loaded
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.image_height: int = int(coord_y_max) + 100
        self.image_width: int = int(coord_x_max) + 100
        
        print(f"✓ Loaded Xenium data:")
        print(f"  - Total cells: {len(self.cell_id):,}")
        print(f"  - Polygon coord range: X=[0, {coord_x_max:.1f}], Y=[0, {coord_y_max:.1f}]")
        print(f"  - Mean vertices per cell: {self.polygon_num_vertices[0, :].mean():.1f}")
    
    def _load_image(self):
        """Load morphology image (morphology_focus.ome.tif recommended)."""
        if self.image_path is None:
            print("⚠️  No image path provided")
            self.image = None
            return
            
        try:
            import tifffile
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with tifffile.TiffFile(str(self.image_path)) as tif:
                    if len(tif.pages) > 0:
                        self.image = tif.pages[0].asarray()
                    else:
                        self.image = tif.asarray()
            
            # Update image dimensions
            self.image_height, self.image_width = self.image.shape[:2]
            
            # Calculate scaling factors
            coord_x_max = self.polygon_vertices[0].max()
            coord_y_max = self.polygon_vertices[1].max()
            
            self.scale_x = self.image_width / coord_x_max
            self.scale_y = self.image_height / coord_y_max
            
            print(f"✓ Loaded image: shape={self.image.shape}, dtype={self.image.dtype}")
            print(f"  - Scaling factors: X={self.scale_x:.3f}, Y={self.scale_y:.3f}")
            
        except Exception as e:
            print(f"⚠️  Failed to load image: {e}")
            self.image = None
    
    def polygon_to_mask(
        self,
        cell_idx: int,
        plane: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        use_precomputed: bool = True
    ) -> np.ndarray:
        """
        Convert a single cell polygon to a binary mask.
        
        Args:
            cell_idx: Index into the cell arrays (0-based)
            plane: Z-plane (0=cell, 1=nucleus)
            image_size: Output mask size (height, width)
            use_precomputed: If True and masks exist, extract from precomputed masks
            
        Returns:
            Binary mask of the cell polygon
        """
        # Use precomputed masks if available and cell is within bounds
        if use_precomputed and self.precomputed_masks is not None:
            cell_id = self.cell_id[cell_idx]
            if 0 <= cell_id < self.precomputed_masks.max():
                if image_size is None:
                    image_size = (self.image_height, self.image_width)
                mask = (self.precomputed_masks == cell_id).astype(np.uint8)
                if image_size != mask.shape:
                    mask = transform.resize(
                        mask, image_size, order=0, preserve_range=True
                    ).astype(np.uint8)
                return mask
        
        # Fallback: rasterize from polygon
        if image_size is None:
            image_size = (self.image_height, self.image_width)
        
        num_verts = self.polygon_num_vertices[plane, cell_idx]
        vertices = self.polygon_vertices[plane, cell_idx, :num_verts]
        
        # Format: interleaved [y0, x0, y1, x1, ...]
        y_coords = vertices[0::2] * self.scale_y
        x_coords = vertices[1::2] * self.scale_x
        
        rr, cc = draw.polygon(y_coords, x_coords, shape=image_size)
        mask = np.zeros(image_size, dtype=np.uint8)
        mask[rr, cc] = 1
        
        return mask
    
    def get_cell_bbox(
        self,
        cell_idx: int,
        plane: int = 0,
        padding: int = 10
    ) -> Tuple[int, int, int, int]:
        """
        Get bounding box for a cell.
        
        Args:
            cell_idx: Index into the cell arrays
            plane: Z-plane (0=cell, 1=nucleus)
            padding: Padding around the cell
            
        Returns:
            (y_min, y_max, x_min, x_max)
        """
        # Use precomputed masks for accurate bbox if available
        if self.precomputed_masks is not None:
            cell_id = self.cell_id[cell_idx]
            coords = np.where(self.precomputed_masks == cell_id)
            if len(coords[0]) > 0:
                y_min = max(0, coords[0].min() - padding)
                y_max = min(self.image_height, coords[0].max() + padding + 1)
                x_min = max(0, coords[1].min() - padding)
                x_max = min(self.image_width, coords[1].max() + padding + 1)
                return y_min, y_max, x_min, x_max
        
        # Fallback: calculate from polygon
        num_verts = self.polygon_num_vertices[plane, cell_idx]
        vertices = self.polygon_vertices[plane, cell_idx, :num_verts]
        
        y_coords = vertices[0::2] * self.scale_y
        x_coords = vertices[1::2] * self.scale_x
        
        y_min = max(0, int(y_coords.min()) - padding)
        y_max = min(self.image_height, int(y_coords.max()) + padding)
        x_min = max(0, int(x_coords.min()) - padding)
        x_max = min(self.image_width, int(x_coords.max()) + padding)
        
        return y_min, y_max, x_min, x_max
    
    def create_instance_mask(
        self,
        image_size: Optional[Tuple[int, int]] = None,
        plane: int = 0,
        cell_ids: Optional[np.ndarray] = None,
        use_precomputed: bool = True
    ) -> np.ndarray:
        """
        Create a full instance segmentation mask.
        
        Args:
            image_size: Output mask size (height, width)
            plane: Z-plane (0=cell, 1=nucleus)
            cell_ids: Optional subset of cell IDs to include
            use_precomputed: If True and available, return precomputed masks
            
        Returns:
            Instance mask where each cell has unique integer ID (0=background)
        """
        # Return precomputed masks if available
        if use_precomputed and self.precomputed_masks is not None:
            if image_size is None or image_size == self.precomputed_masks.shape:
                print(f"✓ Using precomputed masks: {self.precomputed_masks.shape}")
                return self.precomputed_masks.copy()
            else:
                # Resize precomputed masks to desired size
                resized = transform.resize(
                    self.precomputed_masks.astype(np.float32),
                    image_size,
                    order=0,  # Nearest neighbor
                    preserve_range=True
                ).astype(np.uint32)
                print(f"✓ Resized precomputed masks to {image_size}")
                return resized
        
        # Fallback: rasterize from polygons
        if image_size is None:
            image_size = (self.image_height, self.image_width)
        
        instance_mask = np.zeros(image_size, dtype=np.uint32)
        
        if cell_ids is None:
            cell_indices = range(len(self.cell_id))
        else:
            cell_indices = [np.where(self.cell_id == cid)[0][0] for cid in cell_ids]
        
        print(f"Rasterizing {len(cell_indices)} cells from polygons...")
        
        for i, cell_idx in enumerate(cell_indices):
            if (i + 1) % 20000 == 0:
                print(f"  Processed {i + 1}/{len(cell_indices)} cells")
            
            num_verts = self.polygon_num_vertices[plane, cell_idx]
            vertices = self.polygon_vertices[plane, cell_idx, :num_verts]
            
            # Apply scaling
            y_coords = vertices[0::2] * self.scale_y
            x_coords = vertices[1::2] * self.scale_x
            
            rr, cc = draw.polygon(y_coords, x_coords, shape=image_size)
            
            cell_label = self.cell_id[cell_idx]
            instance_mask[rr, cc] = cell_label
        
        print(f"✓ Created instance mask with {len(cell_indices)} cells")
        
        return instance_mask
    
    def extract_patches(
        self,
        patch_size: int = 256,
        num_patches: int = 1000,
        plane: int = 0,
        seed: int = 42,
        use_precomputed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract random patches for training.
        
        Args:
            patch_size: Size of patches (square)
            num_patches: Number of patches to extract
            plane: Z-plane to use
            seed: Random seed
            use_precomputed: Use precomputed masks if available
            
        Returns:
            Tuple of (images, labels):
            - images: (N, H, W) array of image patches (grayscale)
            - labels: (N, H, W) array of instance labels
        """
        np.random.seed(seed)
        
        if self.image is None:
            raise ValueError("Image not loaded. Call with load_image=True or load manually.")
        
        # Create instance mask
        instance_mask = self.create_instance_mask(
            image_size=self.image.shape[:2],
            plane=plane,
            use_precomputed=use_precomputed
        )
        
        # Get cell centers - always use cell_summary for efficiency
        # The precomputed_masks are only used for creating instance_mask
        cell_centers = self.cell_summary[:, :2]  # (x, y) columns
        cell_coords = np.column_stack([cell_centers, self.cell_id])
        
        print(f"Sampling from {len(cell_coords)} cells...")
        
        images = []
        labels = []
        half_size = patch_size // 2
        
        for _ in range(num_patches * 3):  # Try more times to get valid patches
            if len(images) >= num_patches:
                break
                
            # Randomly select a cell
            idx = np.random.randint(0, len(cell_coords))
            cx, cy, cell_id = cell_coords[idx]
            
            # Add jitter
            cx += np.random.randint(-50, 50)
            cy += np.random.randint(-50, 50)
            
            # Calculate patch boundaries
            x1 = max(0, int(cx) - half_size)
            x2 = min(self.image_width, x1 + patch_size)
            y1 = max(0, int(cy) - half_size)
            y2 = min(self.image_height, y1 + patch_size)
            
            # Skip if patch is too small or mostly background
            if (x2 - x1 < patch_size * 0.8) or (y2 - y1 < patch_size * 0.8):
                continue
            
            # Extract image patch
            img_patch = self.image[y1:y2, x1:x2]
            label_patch = instance_mask[y1:y2, x1:x2]
            
            # Skip if mostly background
            if label_patch.sum() < 100:  # At least 100 pixels of cells
                continue
            
            # Resize to exact size
            if img_patch.shape[:2] != (patch_size, patch_size):
                img_patch = transform.resize(
                    img_patch, (patch_size, patch_size), 
                    preserve_range=True, mode='constant'
                )
                label_patch = transform.resize(
                    label_patch.astype(np.float32), 
                    (patch_size, patch_size),
                    order=0, preserve_range=True
                ).astype(np.uint32)
            
            images.append(img_patch)
            labels.append(label_patch)
        
        images = np.stack(images).astype(np.float32)
        labels = np.stack(labels)
        
        print(f"✓ Extracted {len(images)} patches of size {patch_size}x{patch_size}")
        
        return images, labels
    
    def export_for_stardist(
        self,
        output_path: str,
        patch_size: int = 256,
        num_patches: int = 1000,
        plane: int = 0,
        seed: int = 42
    ):
        """
        Export training data in StarDist-compatible format.
        
        Args:
            output_path: Path to save npz file
            patch_size: Size of image patches
            num_patches: Number of patches to extract
            plane: Z-plane to use
            seed: Random seed
        """
        # Extract patches
        images, labels = self.extract_patches(
            patch_size=patch_size,
            num_patches=num_patches,
            plane=plane,
            seed=seed
        )
        
        # Normalize images (for DAPI/ fluorescence images)
        if images.ndim == 3:
            # Grayscale - normalize to [0, 1]
            img_min, img_max = images.min(), images.max()
            if img_max > img_min:
                images = (images - img_min) / (img_max - img_min)
            else:
                images = np.zeros_like(images)
        
        # Save as compressed npz
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_file, X=images, Y=labels)
        
        print(f"✓ Saved training data to {output_path}")
        print(f"  - X (images): {images.shape}, dtype={images.dtype}")
        print(f"  - Y (labels): {labels.shape}, dtype={labels.dtype}")


def create_training_dataset(
    cells_zarr_path: str,
    image_path: str,
    output_path: str,
    patch_size: int = 256,
    num_patches: int = 1000,
    plane: int = 0,
    seed: int = 42,
    use_precomputed_masks: bool = True
):
    """
    Create a training dataset from Xenium data.
    
    This function creates patches centered on cells for more efficient training.
    Uses precomputed masks from zarr if available (recommended).
    
    Args:
        cells_zarr_path: Path to cells.zarr
        image_path: Path to morphology_focus.ome.tif (recommended)
        output_path: Output npz file path
        patch_size: Patch size
        num_patches: Number of patches to extract
        plane: Z-plane (0=cell, 1=nucleus)
        seed: Random seed
        use_precomputed_masks: Use precomputed masks from zarr if available
        
    Returns:
        Tuple of (images, labels) arrays
    """
    np.random.seed(seed)
    
    print("=" * 60)
    print("CREATING TRAINING DATASET FROM XENIUM DATA")
    print("=" * 60)
    
    # Load data
    loader = XeniumDataLoader(
        cells_zarr_path=cells_zarr_path,
        image_path=image_path,
        load_image=True
    )
    
    print(f"\nDataset info:")
    print(f"  - Total cells: {len(loader.cell_id):,}")
    print(f"  - Image size: {loader.image_height} x {loader.image_width}")
    print(f"  - Scaling: X={loader.scale_x:.3f}, Y={loader.scale_y:.3f}")
    print(f"  - Precomputed masks: {'Yes' if loader.precomputed_masks is not None else 'No'}")
    
    # Extract patches
    images, labels = loader.extract_patches(
        patch_size=patch_size,
        num_patches=num_patches,
        plane=plane,
        seed=seed,
        use_precomputed=use_precomputed_masks
    )
    
    # Normalize images
    print(f"\nNormalizing images...")
    if images.ndim == 3:
        # Per-image normalization for fluorescence images
        for i in range(images.shape[0]):
            img = images[i]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                images[i] = (img - img_min) / (img_max - img_min)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=images, Y=labels)
    
    print(f"\n" + "=" * 60)
    print("TRAINING DATASET CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"  - Images: {images.shape}, dtype={images.dtype}")
    print(f"  - Labels: {labels.shape}, dtype={labels.dtype}")
    print(f"  - Unique cell labels: {len(np.unique(labels))}")
    print(f"  - Patch size: {patch_size}x{patch_size}")
    
    return images, labels


if __name__ == "__main__":
    # Example usage
    DATA_DIR = Path("/home/yitongzhou/workspace/hdrg-adaptive-stardist/adaptive_shape_stardist/data")
    
    print("=" * 70)
    print("XENIUM DATA PREPROCESSING - MORPHOLOGY_FOCUS")
    print("=" * 70)
    print("\nThis script uses morphology_focus.ome.tif (2D best-focus projection)")
    print("which is the recommended 2D image from Xenium Onboard Analysis.")
    
    loader = XeniumDataLoader(
        cells_zarr_path=str(DATA_DIR / "cells.zarr"),
        image_path=str(DATA_DIR / "morphology_focus.ome.tif"),
        load_image=True
    )
    
    print("\n" + "=" * 70)
    print("XENIUM DATA LOADED SUCCESSFULLY")
    print("=" * 70)
    
    print("\nTo create training dataset, run:")
    print("-" * 70)
    print("python -c \"")
    print("from utils.xenium_preprocessing import create_training_dataset")
    print("from pathlib import Path")
    print()
    print("create_training_dataset(")
    print(f"    cells_zarr_path='{DATA_DIR / 'cells.zarr'}',")
    print(f"    image_path='{DATA_DIR / 'morphology_focus.ome.tif'}',")
    print(f"    output_path='training_data.npz',")
    print(f"    patch_size=256,")
    print(f"    num_patches=1000")
    print(")")
    print("\"")
    print("-" * 70)
