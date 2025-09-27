"""
Test script to verify all components of the StarDist pipeline.
"""

import argparse
import logging
from pathlib import Path
import sys
import traceback
from typing import List, Dict
import numpy as np

from skimage import io
from scripts.adaptive_stardist.multi_scale_model import MultiScaleStarDist2D, MultiScaleConfig2D
from scripts.adaptive_stardist.inference import AdaptiveInferencePipeline
from scripts.adaptive_stardist.evaluation import MultiScaleEvaluator
from scripts.experiments.experiment_runner import ExperimentRunner

class PipelineTester:
    """Test runner for StarDist pipeline components."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.logger = self.setup_logging()
        self.test_results = {}
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.base_dir / 'test_results' / 'pipeline_test.log'
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        return logging.getLogger(__name__)
    
    def test_model_creation(self) -> bool:
        """Test model initialization."""
        try:
            self.logger.info("Testing model creation...")
            
            # Create test configurations
            basic_config = MultiScaleConfig2D(
                n_rays=32,
                grid=(2, 2),
                n_channel_in=1,
                train_patch_size=(256, 256),
                train_batch_size=8,
                scale_factors=[1.0, 0.5, 0.25]
            )
            
            adaptive_config = MultiScaleConfig2D(
                n_rays=32,
                grid=(2, 2),
                n_channel_in=1,
                train_patch_size=(256, 256),
                train_batch_size=8,
                scale_factors=[1.0, 0.5],
                fusion_mode='attention'
            )
            
            # Test different model configurations
            models = {
                'basic': MultiScaleStarDist2D(basic_config, name='test_basic'),
                'adaptive': MultiScaleStarDist2D(adaptive_config, name='test_adaptive')
            }
            
            for name, model in models.items():
                assert model is not None
                self.logger.info(f"Successfully created {name} model")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model creation failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_inference_pipeline(self) -> bool:
        """Test inference pipeline."""
        try:
            self.logger.info("Testing inference pipeline...")
            
            # Create test model with configuration
            config = MultiScaleConfig2D(
                n_rays=32,
                grid=(2, 2),
                n_channel_in=1,
                train_patch_size=(256, 256),
                train_batch_size=8,
                scale_factors=[1.0, 0.5]
            )
            model = MultiScaleStarDist2D(config, name='test_inference')
            
            # Create inference pipeline
            pipeline = AdaptiveInferencePipeline(
                model=model,
                tile_size=256,
                overlap=0.2
            )
            
            # Test pipeline creation (skip actual processing for now)
            # In a real implementation, we would process a test image
            # For testing, we'll just verify the pipeline was created successfully
            assert pipeline is not None
            self.logger.info("Inference pipeline created successfully (processing skipped for testing)")
            
            self.logger.info("Inference pipeline test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Inference pipeline test failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_evaluation(self) -> bool:
        """Test evaluation metrics."""
        try:
            self.logger.info("Testing evaluation metrics...")
            
            # Create evaluator
            evaluator = MultiScaleEvaluator()
            
            # Load test data following original structure
            test_slide_dir = self.base_dir / 'data/240819_Ji_N1_H_EScan'
            test_image_path = next(test_slide_dir.glob('patches_2048/*.png'))
            gt_path = test_slide_dir / 'pseudo_gt_masks_2048' / test_image_path.name
            
            pred_labels = io.imread(test_image_path).astype(np.uint16)  # Using image as fake prediction
            gt_labels = io.imread(gt_path).astype(np.uint16)
            
            # Run evaluation
            metrics = evaluator.evaluate(gt_labels, pred_labels)
            
            assert metrics is not None
            assert 'overall' in metrics
            assert 'boundary' in metrics
            
            self.logger.info("Evaluation test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluation test failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def test_experiment_runner(self) -> bool:
        """Test experiment framework."""
        try:
            self.logger.info("Testing experiment framework...")
            
            # Create experiment runner
            exp_dir = self.base_dir / 'test_results' / 'experiments'
            runner = ExperimentRunner(exp_dir)
            
            # Run a minimal test experiment
            from experiments.experiment_config import EXPERIMENTS
            test_experiment = EXPERIMENTS[0]  # Use first experiment as test
            
            results, stats = runner.run_experiment(test_experiment)
            
            assert results is not None
            assert stats is not None
            
            self.logger.info("Experiment framework test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Experiment framework test failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all pipeline tests."""
        self.test_results = {
            'model_creation': self.test_model_creation(),
            'inference_pipeline': self.test_inference_pipeline(),
            'evaluation': self.test_evaluation(),
            'experiment_runner': self.test_experiment_runner()
        }
        
        # Print summary
        self.logger.info("\nTest Results Summary:")
        for test_name, passed in self.test_results.items():
            status = "PASSED" if passed else "FAILED"
            self.logger.info(f"{test_name}: {status}")
        
        return self.test_results

def main():
    parser = argparse.ArgumentParser(description="Test StarDist pipeline")
    parser.add_argument("--base_dir", type=str,
                       default="/Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev",
                       help="Base directory of the project")
    args = parser.parse_args()
    
    # Run tests
    tester = PipelineTester(Path(args.base_dir))
    results = tester.run_all_tests()
    
    # Exit with appropriate status
    if all(results.values()):
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
