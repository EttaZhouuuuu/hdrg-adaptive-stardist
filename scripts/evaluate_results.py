"""
Script to evaluate multi-scale StarDist results.
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adaptive_stardist.evaluation import MultiScaleEvaluator

def setup_logging(log_file: Path = None):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate StarDist results")
    
    # Input/output arguments
    parser.add_argument("--true_dir", type=str, required=True,
                       help="Directory with ground truth labels")
    parser.add_argument("--pred_dir", type=str, required=True,
                       help="Directory with predicted labels")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save evaluation results")
    
    # Evaluation parameters
    parser.add_argument("--min_iou", type=float, default=0.5,
                       help="Minimum IoU for matching")
    parser.add_argument("--size_bins", type=str, default="10,50,100,200,500",
                       help="Comma-separated size bin thresholds")
    parser.add_argument("--boundary_tolerance", type=int, default=2,
                       help="Pixel tolerance for boundary evaluation")
    
    # Visualization options
    parser.add_argument("--create_plots", action="store_true",
                       help="Create visualization plots")
    parser.add_argument("--plot_format", type=str, default="png",
                       choices=["png", "pdf", "svg"],
                       help="Format for plots")
    
    return parser.parse_args()

def create_evaluation_plots(results_df, output_dir: Path, plot_format: str):
    """
    Create visualization plots for evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
        plot_format: Plot file format
    """
    # Set up plotting style
    plt.style.use('seaborn')
    
    # 1. Size-stratified performance
    plt.figure(figsize=(12, 6))
    size_metrics = results_df['size_stratified'].apply(pd.Series)
    sns.boxplot(data=size_metrics.melt(), x='variable', y='value')
    plt.xticks(rotation=45)
    plt.title('Performance Across Object Sizes')
    plt.tight_layout()
    plt.savefig(output_dir / f'size_performance.{plot_format}')
    plt.close()
    
    # 2. Boundary metrics distribution
    plt.figure(figsize=(8, 6))
    boundary_metrics = results_df['boundary'].apply(pd.Series)
    sns.boxplot(data=boundary_metrics)
    plt.title('Boundary Detection Performance')
    plt.tight_layout()
    plt.savefig(output_dir / f'boundary_metrics.{plot_format}')
    plt.close()
    
    # 3. Scale consistency metrics
    plt.figure(figsize=(8, 6))
    scale_metrics = results_df['scale'].apply(pd.Series)
    sns.boxplot(data=scale_metrics)
    plt.title('Scale Consistency Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / f'scale_metrics.{plot_format}')
    plt.close()
    
    # 4. Shape preservation metrics
    plt.figure(figsize=(8, 6))
    shape_metrics = results_df['shape'].apply(pd.Series)
    sns.boxplot(data=shape_metrics)
    plt.title('Shape Preservation Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / f'shape_metrics.{plot_format}')
    plt.close()
    
    # 5. Overall performance summary
    plt.figure(figsize=(10, 6))
    overall_metrics = results_df['overall'].apply(pd.Series)
    metrics_to_plot = ['precision', 'recall', 'f1']
    sns.boxplot(data=overall_metrics[metrics_to_plot])
    plt.title('Overall Performance Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / f'overall_performance.{plot_format}')
    plt.close()

def main():
    args = parse_args()
    
    # Set up paths
    true_dir = Path(args.true_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir / 'evaluation.log')
    logger = logging.getLogger(__name__)
    
    # Parse size bins
    size_bins = [int(x) for x in args.size_bins.split(',')]
    
    # Create evaluator
    evaluator = MultiScaleEvaluator(
        min_iou=args.min_iou,
        size_bins=size_bins,
        boundary_tolerance=args.boundary_tolerance
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results_df = evaluator.evaluate_dataset(
        true_dir=true_dir,
        pred_dir=pred_dir,
        output_file=output_dir / 'evaluation_results.json'
    )
    
    # Create summary statistics
    summary = {
        'overall_performance': results_df['overall'].apply(pd.Series).mean().to_dict(),
        'size_performance': {
            size: results_df['size_stratified'].apply(
                lambda x: x.get(size, {}).get('f1', 0)
            ).mean()
            for size in results_df['size_stratified'].iloc[0].keys()
        },
        'boundary_performance': results_df['boundary'].apply(pd.Series).mean().to_dict(),
        'scale_consistency': results_df['scale'].apply(pd.Series).mean().to_dict(),
        'shape_preservation': results_df['shape'].apply(pd.Series).mean().to_dict()
    }
    
    # Save summary
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create plots if requested
    if args.create_plots:
        logger.info("Creating visualization plots...")
        create_evaluation_plots(results_df, output_dir, args.plot_format)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
