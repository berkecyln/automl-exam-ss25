#!/usr/bin/env python
"""
Test the medium model (TruncatedSVD + LogisticRegression) in isolation.
"""
import sys
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path to import automl package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from automl.datasets import load_dataset
from automl.bohb_optimization import BOHBOptimizer, BOHBConfig

def test_medium_model(dataset_name="imdb", sample_size=80, n_trials=2):
    """Test the medium model with TruncatedSVD + LogisticRegression."""
    print("\n" + "="*80)
    print(f"Testing MEDIUM model on {dataset_name} dataset")
    print("="*80)
    
    # Load a small dataset
    print(f"Loading {dataset_name} dataset with {sample_size} samples...")
    
    X_train, y_train = load_dataset(dataset_name, split='train', data_path="../data")
    X_test, y_test = load_dataset(dataset_name, split='test', data_path="../data")
    
    # Limit samples for faster testing
    X_train = X_train[:sample_size]
    y_train = y_train[:sample_size]
    X_test = X_test[:sample_size//4]  # Smaller test set
    y_test = y_test[:sample_size//4]
    
    # Create train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Dataset prepared: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    
    # Force garbage collection before starting
    import gc
    gc.collect()
    
    # Create config for medium model
    config = BOHBConfig(
        max_budget=3.0,
        min_budget=1.0,
        n_trials=n_trials,
        wall_clock_limit=120.0,  # 2 minutes
        num_initial_design=1
    )
    
    # Create optimizer
    optimizer = BOHBOptimizer(
        model_type="medium",
        config=config,
        random_state=42
    )
    
    # Run optimization
    start_time = time.time()
    print(f"Starting medium model optimization...")
    
    try:
        best_config, best_score, stats = optimizer.optimize(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fidelity_mode="low"
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Medium model optimization completed in {elapsed_time:.2f}s")
        print(f"Best score: {best_score:.4f}")
        print(f"Best config:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
            
        # Print optimization statistics
        print("\nOptimization statistics:")
        eval_count = stats.get("num_evaluations", "unknown")
        print(f"  Evaluations: {eval_count}")
        print(f"  Total time: {stats.get('total_time', elapsed_time):.2f}s")
        
        return True, best_score, best_config
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Medium model optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0, {}

if __name__ == "__main__":
    # Run medium model test with default parameters
    success, score, config = test_medium_model()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
