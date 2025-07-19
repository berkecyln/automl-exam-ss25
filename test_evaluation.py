#!/usr/bin/env python3
"""
Test Set Evaluation for AutoML Pipeline

This module provides true test set evaluation functionality for the RL+BOHB pipeline,
including:
- Train/validation/test splits
- Final model retraining on train+val
- Unbiased test performance evaluation
- Overfitting detection

Author: AutoML Pipeline Team
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from automl.datasets import DatasetLoader
from automl.models import ModelFactory, ModelConfig


class TestSetEvaluator:
    """Handles true test set evaluation for unbiased performance assessment."""
    
    def __init__(self, test_split_ratio: float = 0.2, random_state: int = 42):
        """
        Initialize test set evaluator.
        
        Args:
            test_split_ratio: Proportion of data to hold out for final testing
            random_state: Random seed for reproducible splits
        """
        self.test_split_ratio = test_split_ratio
        self.random_state = random_state
        self.dataset_loader = DatasetLoader()
        self.model_factory = ModelFactory()
        
    def create_train_val_test_splits(self, dataset_name: str) -> Dict[str, Any]:
        """
        Create proper train/validation/test splits for a dataset.
        
        Args:
            dataset_name: Name of the dataset to split
            
        Returns:
            Dictionary containing train, val, test splits and metadata
        """
        print(f"🔀 Creating train/val/test splits for {dataset_name}")
        
        # Load full dataset
        dataset = self.dataset_loader.load_dataset(dataset_name)
        
        # First split: separate test set
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            dataset['texts'], dataset['labels'],
            test_size=self.test_split_ratio,
            random_state=self.random_state,
            stratify=dataset['labels']
        )
        
        # Second split: train and validation from remaining data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=0.25,  # 25% of train_val = 20% of total for validation
            random_state=self.random_state,
            stratify=train_val_labels
        )
        
        splits = {
            'train': {
                'texts': train_texts,
                'labels': train_labels,
                'size': len(train_texts)
            },
            'validation': {
                'texts': val_texts,
                'labels': val_labels,
                'size': len(val_texts)
            },
            'test': {
                'texts': test_texts,
                'labels': test_labels,
                'size': len(test_labels)
            },
            'metadata': {
                'dataset_name': dataset_name,
                'total_size': len(dataset['texts']),
                'num_classes': len(set(dataset['labels'])),
                'test_ratio': self.test_split_ratio,
                'random_state': self.random_state
            }
        }
        
        print(f"✅ Split {dataset_name}: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
        
        return splits
    
    def retrain_best_model(self, dataset_splits: Dict[str, Any], 
                          best_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrain the best model found by BOHB on train+validation data.
        
        Args:
            dataset_splits: Train/val/test splits
            best_config: Best hyperparameter configuration from BOHB
            
        Returns:
            Trained model and training metrics
        """
        dataset_name = dataset_splits['metadata']['dataset_name']
        print(f"🔄 Retraining best model for {dataset_name} on train+val data")
        
        # Combine train and validation sets for final training
        combined_texts = dataset_splits['train']['texts'] + dataset_splits['validation']['texts']
        combined_labels = dataset_splits['train']['labels'] + dataset_splits['validation']['labels']
        
        print(f"📊 Training on {len(combined_texts)} samples (train+val combined)")
        
        # Create model configuration
        model_config = ModelConfig(
            model_type=best_config.get('model_type', 'complex'),
            max_features=best_config.get('max_features', 10000),
            ngram_range=best_config.get('ngram_range', (1, 2)),
            classifier_params=best_config.get('classifier_params', {})
        )
        
        # Train the model
        model = self.model_factory.create_model(model_config)
        training_start_time = time.time()
        
        model.fit(combined_texts, combined_labels)
        
        training_time = time.time() - training_start_time
        
        # Validate on the combined training data (for sanity check)
        train_predictions = model.predict(combined_texts)
        train_accuracy = accuracy_score(combined_labels, train_predictions)
        
        result = {
            'model': model,
            'config': model_config,
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'training_samples': len(combined_texts),
            'model_type': model_config.model_type
        }
        
        print(f"✅ Model retrained: {train_accuracy:.4f} accuracy on {len(combined_texts)} samples")
        
        return result
    
    def evaluate_on_test_set(self, model_result: Dict[str, Any], 
                           test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the retrained model on the held-out test set.
        
        Args:
            model_result: Result from retrain_best_model
            test_data: Test set data
            
        Returns:
            Comprehensive test evaluation results
        """
        model = model_result['model']
        test_texts = test_data['texts']
        test_labels = test_data['labels']
        
        print(f"🧪 Evaluating on test set ({len(test_texts)} samples)")
        
        # Make predictions
        test_predictions = model.predict(test_texts)
        
        # Calculate metrics
        test_accuracy = accuracy_score(test_labels, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average='weighted'
        )
        
        # Calculate per-class metrics
        unique_labels = sorted(set(test_labels))
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average=None
        )
        
        per_class_metrics = {}
        for i, label in enumerate(unique_labels):
            per_class_metrics[f'class_{label}'] = {
                'precision': per_class_precision[i],
                'recall': per_class_recall[i],
                'f1': per_class_f1[i]
            }
        
        # Detect potential overfitting
        train_accuracy = model_result['train_accuracy']
        overfitting_gap = train_accuracy - test_accuracy
        is_overfitting = overfitting_gap > 0.05  # 5% threshold
        
        evaluation_result = {
            'test_accuracy': test_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'per_class_metrics': per_class_metrics,
            'train_accuracy': train_accuracy,
            'overfitting_gap': overfitting_gap,
            'is_overfitting': is_overfitting,
            'test_samples': len(test_texts),
            'model_type': model_result['model_type'],
            'training_time': model_result['training_time']
        }
        
        # Print summary
        print(f"📊 Test Results:")
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Overfitting Gap: {overfitting_gap:+.4f}")
        if is_overfitting:
            print(f"   ⚠️ Potential overfitting detected!")
        else:
            print(f"   ✅ No significant overfitting")
        
        return evaluation_result
    
    def run_full_test_evaluation(self, pipeline_results: Dict[str, Any], 
                                output_dir: str = "pipeline_demo_results") -> Dict[str, Any]:
        """
        Run complete test set evaluation for all datasets in pipeline results.
        
        Args:
            pipeline_results: Results from the main RL+BOHB pipeline
            output_dir: Directory to save test evaluation results
            
        Returns:
            Comprehensive test evaluation results for all datasets
        """
        print(f"\n🧪 Starting True Test Set Evaluation")
        print(f"=" * 50)
        
        test_results = {
            'evaluation_metadata': {
                'test_split_ratio': self.test_split_ratio,
                'random_state': self.random_state,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'dataset_results': {}
        }
        
        # Get final selections from pipeline
        final_selections = pipeline_results.get('final_selections', {})
        
        for dataset_name, selection_result in final_selections.items():
            print(f"\n📊 Processing {dataset_name}")
            
            try:
                # Create train/val/test splits
                splits = self.create_train_val_test_splits(dataset_name)
                
                # Get best configuration (mock for now - in real implementation, 
                # extract from BOHB results)
                best_config = {
                    'model_type': selection_result.get('selected_model_type', 'complex'),
                    'max_features': 10000,
                    'ngram_range': (1, 2),
                    'classifier_params': {'random_state': 42}
                }
                
                # Retrain model on train+val
                model_result = self.retrain_best_model(splits, best_config)
                
                # Evaluate on test set
                test_evaluation = self.evaluate_on_test_set(
                    model_result, splits['test']
                )
                
                # Store results
                test_results['dataset_results'][dataset_name] = {
                    'splits_info': splits['metadata'],
                    'best_config': best_config,
                    'test_evaluation': test_evaluation,
                    'pipeline_score': selection_result.get('best_score', 0),
                    'rl_confidence': selection_result.get('confidence', 0)
                }
                
            except Exception as e:
                print(f"❌ Error evaluating {dataset_name}: {e}")
                test_results['dataset_results'][dataset_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Calculate summary statistics
        successful_evaluations = {
            name: result for name, result in test_results['dataset_results'].items()
            if 'test_evaluation' in result
        }
        
        if successful_evaluations:
            test_accuracies = [
                result['test_evaluation']['test_accuracy'] 
                for result in successful_evaluations.values()
            ]
            
            pipeline_scores = [
                result['pipeline_score']
                for result in successful_evaluations.values()
            ]
            
            test_results['summary'] = {
                'num_datasets_evaluated': len(successful_evaluations),
                'average_test_accuracy': np.mean(test_accuracies),
                'std_test_accuracy': np.std(test_accuracies),
                'average_pipeline_score': np.mean(pipeline_scores),
                'pipeline_vs_test_correlation': np.corrcoef(pipeline_scores, test_accuracies)[0, 1],
                'overfitting_datasets': [
                    name for name, result in successful_evaluations.items()
                    if result['test_evaluation']['is_overfitting']
                ]
            }
        
        # Save results
        output_path = Path(output_dir) / "test_evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n✅ Test evaluation completed!")
        print(f"📂 Results saved to: {output_path}")
        
        if 'summary' in test_results:
            summary = test_results['summary']
            print(f"\n📊 SUMMARY:")
            print(f"   Datasets evaluated: {summary['num_datasets_evaluated']}")
            print(f"   Average test accuracy: {summary['average_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")
            print(f"   Pipeline vs Test correlation: {summary['pipeline_vs_test_correlation']:.3f}")
            if summary['overfitting_datasets']:
                print(f"   ⚠️ Overfitting detected in: {', '.join(summary['overfitting_datasets'])}")
            else:
                print(f"   ✅ No overfitting detected")
        
        return test_results


def main():
    """Main function for running test set evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run test set evaluation')
    parser.add_argument('--results-file', default='pipeline_demo_results/full_pipeline_results.json',
                       help='Path to pipeline results JSON file')
    parser.add_argument('--output-dir', default='pipeline_demo_results',
                       help='Output directory for test results')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test set ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    try:
        # Load pipeline results
        with open(args.results_file, 'r') as f:
            pipeline_results = json.load(f)
        
        # Run test evaluation
        evaluator = TestSetEvaluator(test_split_ratio=args.test_ratio)
        test_results = evaluator.run_full_test_evaluation(
            pipeline_results, args.output_dir
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import time
    exit(main())
