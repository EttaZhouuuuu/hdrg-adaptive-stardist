import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime

from models.adaptive_shape_encoder import AdaptiveShapeEncoder
from training.shape_aware_loss import ShapeAwareHybridLoss
from data.dataset import ShapeAwareDataset
from visualization.visualizer import Visualizer
from evaluation.metrics import SegmentationMetrics
from tracking.experiment_tracker import ExperimentTracker
from configs.test_config import TEST_CONFIG

def setup_logging(save_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(save_dir / 'test.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def test_model(args):
    """Main testing function."""
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"test_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir)
    logger.info("Starting model testing...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = AdaptiveShapeEncoder(
        **TEST_CONFIG['model']
    ).to(device)
    
    # Load model weights
    logger.info(f"Loading model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Create loss function
    loss_fn = ShapeAwareHybridLoss(
        **TEST_CONFIG['loss']
    ).to(device)
    
    # Create dataset
    test_dataset = ShapeAwareDataset(
        image_dir=args.test_data,
        mask_dir=args.test_masks,
        input_size=TEST_CONFIG['data']['input_size']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_CONFIG['testing']['test_batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create metrics
    metrics = SegmentationMetrics(n_rays=TEST_CONFIG['model']['n_rays'])
    
    # Create visualizer
    visualizer = Visualizer(save_dir=save_dir / "visualizations")
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_name="shape_aware_test",
        base_dir=save_dir,
        config=TEST_CONFIG
    )
    
    # Testing loop
    logger.info("Starting testing...")
    total_metrics = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get data
            images = batch['image'].to(device)
            true_distances = batch['distances'].to(device)
            true_probabilities = batch['probabilities'].to(device)
            
            # Forward pass
            features, points, intermediates = model(images)
            pred_distances = points.squeeze(2)  # Remove channel dimension
            pred_probabilities = torch.sigmoid(features[:, :1])  # Use first channel as probability
            
            # Compute loss
            loss, loss_components = loss_fn(
                pred_distances,
                pred_probabilities,
                true_distances,
                true_probabilities
            )
            
            # Compute metrics
            batch_metrics = metrics.compute_all_metrics(
                pred_distances,
                pred_probabilities,
                true_distances,
                true_probabilities
            )
            
            # Update total metrics
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)
            
            # Visualize results
            if TEST_CONFIG['testing']['visualization']['save_predictions']:
                vis_image = visualizer.visualize_prediction(
                    images[0],
                    pred_probabilities[0],
                    true_probabilities[0],
                    attention_maps=intermediates[-1][0] if intermediates else None,
                    save_path=f"prediction_{batch_idx}.png"
                )
                
                if TEST_CONFIG['testing']['visualization']['save_attention_maps']:
                    visualizer.visualize_attention_analysis(
                        intermediates[-1][0],
                        save_path=f"attention_{batch_idx}.png"
                    )
                
                if TEST_CONFIG['testing']['visualization']['save_shape_analysis']:
                    visualizer.visualize_shape_analysis(
                        pred_distances[0],
                        true_distances[0],
                        save_path=f"shape_analysis_{batch_idx}.png"
                    )
            
            # Log progress
            logger.info(f"Processed batch {batch_idx+1}/{len(test_loader)}")
            
            # Log metrics
            tracker.log_metrics(batch_metrics, batch_idx)
    
    # Compute and log final metrics
    final_metrics = {
        key: np.mean(values) for key, values in total_metrics.items()
    }
    
    logger.info("Testing completed. Final metrics:")
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Save final metrics
    tracker.log_metrics(final_metrics, prefix='final')
    
    return final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Shape-aware StarDist model")
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model weights')
    parser.add_argument('--test_data', type=str, required=True,
                      help='Path to test images')
    parser.add_argument('--test_masks', type=str, required=True,
                      help='Path to test masks')
    parser.add_argument('--save_dir', type=str, default='test_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    test_model(args)
