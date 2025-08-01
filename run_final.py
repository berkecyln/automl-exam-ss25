#!/usr/bin/env python3
"""
AutoML Pipeline Runner with Leave-One-Out Cross-Validation
---------------------------------------------------------

This script runs the complete AutoML pipeline with:
1. Meta-feature extraction
2. Leave-one-out cross-validation
3. Final training on all datasets
4. Evaluation and visualization

Usage:
  python run_final.py [--time HOURS] [--output DIR] [--datasets LIST]
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add the project directory to the Python path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_final import AutoMLPipeline
from automl.visualizer import save_all_figures
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
        print("✅ All required dependencies are available.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False


def run_pipeline(args):
    """Run the AutoML pipeline with provided arguments."""
    print("\n======================================================================")
    print("STARTING AUTOML PIPELINE WITH LEAVE-ONE-OUT CROSS-VALIDATION")
    print("======================================================================")
    print(f"⏱️ Maximum runtime: {args.time:.1f} hours")
    print(f"💾 Output directory: {args.output}")
    print(f"📊 Datasets: {', '.join(args.datasets)}")
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
    vis_dir = output_dir / "visualizations"                       # <<< NEW >>>
    vis_dir.mkdir(parents=True, exist_ok=True)                    # <<< NEW >>>
    try:                                                          # <<< NEW >>>
        save_all_figures(pipeline.results, vis_dir, feature_order=FEATURE_ORDER)              # <<< NEW >>>
    except Exception as e:                                        # <<< NEW >>>
        print(f"⚠️  Could not create visuals: {e}")               # <<< NEW >>>

    # Print final summary
    print("\n======================================================================")
    print("AUTOML PIPELINE COMPLETED")
    print("======================================================================")
    print(f"🕐 Actual runtime: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📈 Visualisations stored   : {vis_dir}")              # <<< NEW >>>
    print()
    
    # Print cross-validation results
    print("📊 CROSS-VALIDATION RESULTS:")
    print("----------------------------------------------------------------------")
    for dataset, score in sorted(pipeline.results['cv_results']['performance'].items(), 
                               key=lambda x: x[1], reverse=True):
        cv_model = next((f['selected_model'] for f in pipeline.results['cv_results']['folds'] 
                         if f['held_out'] == dataset), None)
        print(f"  {dataset}: {score:.4f} (model: {cv_model})")
    
    print(f"  MEAN: {results['cv_mean_performance']:.4f}")
    print()
    
    # Print final model selections
    print("🎯 FINAL MODEL SELECTIONS:")
    print("----------------------------------------------------------------------")
    for dataset, selection in sorted(results.get('final_selections', {}).items()):
        print(f"  {dataset}: {selection.get('model_type')} (confidence: {selection.get('confidence', 0):.3f})")
    
    print()
    print(f"🎯 CV vs FINAL MODEL AGREEMENT: {results['cv_vs_final_agreement']:.1%}")
    print()
    print(f"📈 VISUALIZATION FILES: {output_dir / 'visualizations'}")
    
    return results


def main():
    """Main function parsing arguments and running pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML Pipeline with Leave-One-Out CV")
    parser.add_argument('--time', type=float, default=1.0, help="Maximum runtime in hours")
    parser.add_argument('--output', type=str, default="test_run", 
                        help="Name for this experiment (saved in experiments folder)")
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['amazon', 'ag_news', 'dbpedia', 'imdb'],
                        help="Datasets to use")
    parser.add_argument('--max_iterations', type=int, default=10,
                        help="Number of trials for optimization")
    
    args = parser.parse_args()
    
    # Validate environment
    if not setup_environment():
        return 1
    
    try:
        # Run pipeline
        run_pipeline(args)
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user.")
        return 130
    except Exception as e:
        import traceback
        print(f"\n❌ Pipeline failed with error: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
