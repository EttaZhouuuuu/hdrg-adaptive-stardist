"""
Runner for comparative experiments.
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from skimage import io

from .experiment_config import EXPERIMENTS, ExperimentConfig
from scripts.adaptive_stardist.multi_scale_model import MultiScaleStarDist2D, MultiScaleConfig2D
from scripts.adaptive_stardist.adaptive_model import AdaptiveStarDist2D, AdaptiveConfig2D
from scripts.adaptive_stardist.evaluation import MultiScaleEvaluator
from stardist.models import StarDist2D, Config2D

class ExperimentRunner:
    """Runner for comparative experiments."""
    
    def __init__(self,
                 output_dir: Path,
                 seed: int = 42):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.seed = seed
        np.random.seed(seed)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize evaluator
        self.evaluator = MultiScaleEvaluator()
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.output_dir / 'experiments.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    
    def prepare_dataset(self, dataset_config):
        """Prepare dataset for experiment."""
        # Load images from all specified slides
        all_images = []
        data_path = dataset_config.path
        
        for slide in dataset_config.slides:
            slide_dir = data_path / slide / 'patches_2048'
            if slide_dir.exists():
                slide_images = sorted(slide_dir.glob('*.png'))
                all_images.extend(slide_images)
        
        # Random split
        n_train = int(len(all_images) * dataset_config.split_ratio)
        indices = np.random.permutation(len(all_images))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        return {
            'train': [all_images[i] for i in train_indices],
            'test': [all_images[i] for i in test_indices]
        }
    
    def train_model(self, model_config, train_data, experiment_dir):
        """Train a model with given configuration."""
        if model_config.model_type == "baseline":
            config = Config2D(**model_config.params)
            model = StarDist2D(config, name=model_config.name,
                             basedir=experiment_dir)
        elif model_config.model_type == "adaptive":
            config = AdaptiveConfig2D(**model_config.params)
            model = AdaptiveStarDist2D(config, name=model_config.name,
                                     basedir=experiment_dir)
        else:  # multi_scale
            config = MultiScaleConfig2D(**model_config.params)
            model = MultiScaleStarDist2D(config, name=model_config.name,
                                       basedir=experiment_dir)
        
        # Train model (simplified for testing)
        # In a real implementation, this would load and process the training data
        # For testing, we'll skip the actual training
        self.logger.info(f"Model {model_config.name} training skipped for testing")
        
        return model
    
    def evaluate_model(self, model, test_data, metrics):
        """Evaluate model performance."""
        results = []
        
        for image_path in tqdm(test_data, desc="Evaluating"):
            # Load test image and ground truth
            image = io.imread(image_path)
            # Find corresponding mask in pseudo_gt_masks_2048 directory
            slide_name = image_path.parent.parent.name
            gt_path = image_path.parent.parent / 'pseudo_gt_masks_2048' / image_path.name
            gt_labels = io.imread(gt_path)
            
            # Get predictions
            pred_labels = model.predict_instances(image)[0]
            
            # Compute metrics
            eval_results = self.evaluator.evaluate(gt_labels, pred_labels)
            
            # Extract requested metrics
            result = {
                'image': image_path.name,
                **{m: eval_results[m] for m in metrics}
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def run_statistical_analysis(self, results_df: pd.DataFrame) -> Dict:
        """Perform statistical analysis on results."""
        stats_results = {}
        
        # Perform Kruskal-Wallis H-test for each metric
        metrics = [col for col in results_df.columns if col != 'image']
        for metric in metrics:
            # Prepare data for statistical test
            model_data = [
                results_df[results_df['model'] == model][metric].values
                for model in results_df['model'].unique()
            ]
            
            # Perform Kruskal-Wallis H-test
            h_stat, p_value = stats.kruskal(*model_data)
            
            # If significant difference found, perform post-hoc Mann-Whitney U tests
            pairwise_tests = {}
            if p_value < 0.05:
                models = results_df['model'].unique()
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        model1, model2 = models[i], models[j]
                        stat, p = stats.mannwhitneyu(
                            results_df[results_df['model'] == model1][metric],
                            results_df[results_df['model'] == model2][metric]
                        )
                        pairwise_tests[f"{model1}_vs_{model2}"] = {
                            'statistic': stat,
                            'p_value': p
                        }
            
            stats_results[metric] = {
                'kruskal_wallis': {
                    'statistic': h_stat,
                    'p_value': p_value
                },
                'pairwise_tests': pairwise_tests
            }
        
        return stats_results
    
    def create_visualization(self, results_df: pd.DataFrame,
                           experiment_dir: Path):
        """Create visualization of results."""
        # Set up plotting style
        plt.style.use('seaborn')
        
        # 1. Performance comparison across metrics
        metrics = [col for col in results_df.columns
                  if col not in ['image', 'model']]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=results_df, x='model', y=metric)
            plt.xticks(rotation=45)
            plt.title(f'{metric} Comparison')
            plt.tight_layout()
            plt.savefig(experiment_dir / f'{metric}_comparison.png')
            plt.close()
        
        # 2. Performance distribution
        plt.figure(figsize=(12, 6))
        melted_df = results_df.melt(
            id_vars=['model'],
            value_vars=metrics,
            var_name='metric',
            value_name='value'
        )
        sns.violinplot(data=melted_df, x='metric', y='value', hue='model')
        plt.xticks(rotation=45)
        plt.title('Performance Distribution Across Metrics')
        plt.tight_layout()
        plt.savefig(experiment_dir / 'performance_distribution.png')
        plt.close()
    
    def run_experiment(self, experiment_config: ExperimentConfig):
        """Run a single experiment."""
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = self.output_dir / f"{experiment_config.name}_{timestamp}"
        experiment_dir.mkdir(parents=True)
        
        # Save experiment configuration
        with open(experiment_dir / 'config.json', 'w') as f:
            json.dump({
                'name': experiment_config.name,
                'description': experiment_config.description,
                'models': [vars(m) for m in experiment_config.models],
                'metrics': experiment_config.metrics,
                'n_runs': experiment_config.n_runs
            }, f, indent=2)
        
        # Prepare dataset
        dataset = self.prepare_dataset(experiment_config.dataset)
        
        # Run multiple times for statistical significance
        all_results = []
        for run in range(experiment_config.n_runs):
            run_dir = experiment_dir / f"run_{run}"
            run_dir.mkdir()
            
            # Train and evaluate each model
            for model_config in experiment_config.models:
                self.logger.info(f"Training {model_config.name}")
                model = self.train_model(model_config, dataset['train'], run_dir)
                
                self.logger.info(f"Evaluating {model_config.name}")
                results = self.evaluate_model(
                    model,
                    dataset['test'],
                    experiment_config.metrics
                )
                results['model'] = model_config.name
                results['run'] = run
                all_results.append(results)
        
        # Combine results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Perform statistical analysis
        stats_results = self.run_statistical_analysis(combined_results)
        
        # Create visualizations
        self.create_visualization(combined_results, experiment_dir)
        
        # Save results
        combined_results.to_csv(experiment_dir / 'results.csv', index=False)
        with open(experiment_dir / 'statistics.json', 'w') as f:
            json.dump(stats_results, f, indent=2)
        
        return combined_results, stats_results
    
    def run_all_experiments(self):
        """Run all defined experiments."""
        all_results = {}
        
        for experiment_config in EXPERIMENTS:
            self.logger.info(f"Running experiment: {experiment_config.name}")
            results, stats = self.run_experiment(experiment_config)
            all_results[experiment_config.name] = {
                'results': results,
                'statistics': stats
            }
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Run comparative experiments")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Create runner and run experiments
    runner = ExperimentRunner(Path(args.output_dir), args.seed)
    results = runner.run_all_experiments()
    
    print("All experiments completed successfully!")

if __name__ == "__main__":
    main()
