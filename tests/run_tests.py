#!/usr/bin/env python
"""
Master test runner that can run all models or specific models.
"""
import sys
import os
import time
import argparse

# Add parent directory to path to import automl package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from test_simple_model import test_simple_model
from test_medium_model import test_medium_model
from test_complex_model import test_complex_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run model tests')
    parser.add_argument('--models', type=str, default='simple,medium,complex',
                        help='Comma-separated list of models to test (simple,medium,complex)')
    parser.add_argument('--dataset', type=str, default='imdb',
                        help='Dataset to use for testing (imdb,amazon,dbpedia,ag_news)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Override number of samples for all models')
    parser.add_argument('--trials', type=int, default=2,
                        help='Number of trials for each model')
    args = parser.parse_args()
    
    # Parse models to test
    models_to_test = args.models.split(',')
    
    # Store test results
    results = []
    all_success = True
    start_time = time.time()
    
    # Test each requested model
    if 'simple' in models_to_test:
        print("\nRunning simple model test...")
        samples = args.samples or 100
        success, score, config = test_simple_model(args.dataset, samples, args.trials)
        results.append({
            'model_type': 'simple',
            'success': success,
            'score': score,
            'samples': samples
        })
        if not success:
            all_success = False
    
    if 'medium' in models_to_test:
        print("\nRunning medium model test...")
        samples = args.samples or 80
        success, score, config = test_medium_model(args.dataset, samples, args.trials)
        results.append({
            'model_type': 'medium',
            'success': success,
            'score': score,
            'samples': samples
        })
        if not success:
            all_success = False
    
    if 'complex' in models_to_test:
        print("\nRunning complex model test...")
        samples = args.samples or 50
        success, score, config = test_complex_model(args.dataset, samples, args.trials)
        results.append({
            'model_type': 'complex',
            'success': success,
            'score': score,
            'samples': samples
        })
        if not success:
            all_success = False
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Print summary table
    print("\n\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"{'Model Type':<10} | {'Success':<8} | {'Score':<8} | {'Samples':<8}")
    print("-"*60)
    
    for result in results:
        success_str = "✓" if result["success"] else "✗"
        print(f"{result['model_type']:<10} | {success_str:<8} | {result['score']:<8.4f} | {result['samples']:<8}")
    
    print("\n" + "="*60)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if all_success:
        print("✅ All model tests passed!")
    else:
        print("❌ Some model tests failed.")
    
    # Return appropriate exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
