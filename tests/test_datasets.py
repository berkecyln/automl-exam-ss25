#!/usr/bin/env python
"""
Test dataset loading and statistics.
"""
import sys
import os
import time
import numpy as np
from collections import Counter

# Add parent directory to path to import automl package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from automl.datasets import load_dataset

def test_dataset(dataset_name, max_samples=100):
    """Test dataset loading and show basic statistics."""
    print(f"\n{'-'*20} Testing {dataset_name} dataset {'-'*20}")
    
    try:
        # Load datasets
        start_time = time.time()
        X_train, y_train = load_dataset(dataset_name, split='train', data_path="../data")
        X_test, y_test = load_dataset(dataset_name, split='test', data_path="../data")
        load_time = time.time() - start_time
        
        # Basic statistics
        train_size = len(X_train)
        test_size = len(X_test)
        train_class_dist = dict(sorted(Counter(y_train).items()))
        test_class_dist = dict(sorted(Counter(y_test).items()))
        num_classes = len(train_class_dist)
        
        # Sample lengths
        sample_lengths = [len(str(x).split()) for x in X_train[:max_samples]]
        avg_length = np.mean(sample_lengths)
        max_length = np.max(sample_lengths)
        min_length = np.min(sample_lengths)
        
        # Print results
        print(f"Dataset loaded in {load_time:.2f} seconds")
        print(f"Training set: {train_size} samples, Test set: {test_size} samples")
        print(f"Number of classes: {num_classes}")
        print(f"Class distribution (training):")
        for cls, count in train_class_dist.items():
            print(f"  Class {cls}: {count} samples ({count/train_size*100:.1f}%)")
        print(f"Text statistics (first {max_samples} samples):")
        print(f"  Average length: {avg_length:.1f} words")
        print(f"  Min length: {min_length} words")
        print(f"  Max length: {max_length} words")
        
        print(f"✅ {dataset_name} dataset test successful")
        return True
        
    except Exception as e:
        print(f"❌ {dataset_name} dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Test all available datasets
    datasets = ["imdb", "amazon", "dbpedia", "ag_news"]
    results = {}
    
    for dataset in datasets:
        results[dataset] = test_dataset(dataset)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Testing Summary")
    print("="*60)
    
    all_success = True
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"{dataset:<10}: {status}")
        if not success:
            all_success = False
    
    print("\n" + "="*60)
    if all_success:
        print("✅ All dataset tests passed!")
    else:
        print("❌ Some dataset tests failed.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
