#!/usr/bin/env python3
"""
AutoML Text Classification Pipeline
----------------------------------

This script runs the complete AutoML pipeline with:
1. Meta-feature extraction from Phase 1 datasets
2. Leave-one-out cross-validation on training datasets
3. RL agent training with BOHB optimization
4. Final model selection and training on exam dataset
5. Test prediction generation and evaluation

Usage:
  python run.py [--time HOURS] [--output DIR] [--data-path PATH]

Example:
  python run.py --time 24.0 --output final_submission --data-path data
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add the project directory to the Python path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import AutoMLPipeline
from automl.constants import FEATURE_ORDER


def setup_environment():
    """Setup environment and verify dependencies."""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from automl.meta_features import MetaFeatureExtractor
        from automl.rl_agent import RLModelSelector
        from automl.datasets import load_dataset
        from automl.logging_utils import AutoMLLogger

        print("All required dependencies are available.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False


def run_pipeline(args):
    """Run the AutoML pipeline with provided arguments."""
    print("\n======================================================================")
    print("STARTING AUTOML PIPELINE WITH LEAVE-ONE-OUT CROSS-VALIDATION")
    print("======================================================================")
    print(f"Maximum runtime: {args.time:.1f} hours")
    print(f"Output directory: {args.output}")
    print("----------------------------------------------------------------------")

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Always use experiments folder as the parent directory
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)

    # Create output directory inside experiments folder
    output_dir = experiments_dir / args.output / f"run_{timestamp}"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Initialize and run pipeline
    pipeline = AutoMLPipeline(
        max_runtime_hours=args.time,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
    )

    start_time = time.time()
    results = pipeline.run_pipeline()

    total_time = time.time() - start_time

    # display final summary
    print("\n======================================================================")
    print("AUTOML PIPELINE COMPLETED")
    print("======================================================================")
    print(f"Actual runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    print()

    # Display cross-validation results
    print("CROSS-VALIDATION RESULTS:")
    print("----------------------------------------------------------------------")
    for dataset, score in sorted(
        pipeline.results["cv_results"]["performance"].items(), key=lambda x: x[1], reverse=True
    ):
        cv_model = next(
            (
                f["selected_model"]
                for f in pipeline.results["cv_results"]["folds"]
                if f["held_out"] == dataset
            ),
            None,
        )
        print(f"  {dataset}: {score:.4f} (model: {cv_model})")

    print(f"  MEAN: {results['cv_mean_performance']:.4f}")
    print()

    # Display final model selections
    print("FINAL MODEL SELECTIONS:")
    final_selections = results.get("final_selections", {})
    if final_selections:
        for agent_name, selection in final_selections.items():
            print(f"📊 Final Model Selection from {agent_name.upper()} Agent:")
            print(f"   • Selected Model: {selection['model_type'].title()}")
            print(f"   • BOHB Score: {selection['bohb_score']:.4f}")
            print(f"   • Confidence: {selection.get('confidence', 0):.4f}")
            print(f"   • Best Config: {len(selection['best_config'])} parameters")

            # Show key config parameters
            config = selection["best_config"]
            if "algorithm" in config:
                print(f"     - Algorithm: {config['algorithm']}")
            if "max_features" in config:
                print(f"     - Max Features: {config['max_features']}")
            if "C" in config:
                print(f"     - C Parameter: {config['C']:.4f}")
            if "learning_rate" in config:
                print(f"     - Learning Rate: {config['learning_rate']:.4f}")

    # Display prediction summary
    print(f"\n🎯 Prediction Results:")
    print(f"   • Test dataset: {pipeline.exam_dataset}")
    print(f"   • Predictions saved to: data/exam_dataset/predictions.npy")
    print(f"   • Total runtime: {(time.time() - pipeline.start_time)/60:.1f} minutes")

    # Display training summary
    total_bohb_evals = len(
        [eval for eval in results["bohb_evaluations"] if eval["cv_fold"] == "final"]
    )
    print(f"\n📈 Training Summary:")
    print(f"   • RL Training Iterations: {len(results.get('rl_training_iterations', []))}")
    print(f"   • Total BOHB Evaluations: {total_bohb_evals}")

    return results


def main():
    """Main function parsing arguments and running pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoML Pipeline with Leave-One-Out CV")
    parser.add_argument("--time", type=float, default=1.0, help="Maximum runtime in hours")
    parser.add_argument(
        "--output",
        type=str,
        default="test_run",
        help="Name for this experiment (saved in experiments folder)",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=10, help="Number of trials for optimization"
    )

    args = parser.parse_args()

    # Validate environment
    if not setup_environment():
        return 1

    try:
        run_pipeline(args)
        return 0
    except KeyboardInterrupt:
        print("\n Pipeline interrupted by user.")
        return 130
    except Exception as e:
        import traceback

        print(f"\n Pipeline failed with error: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
