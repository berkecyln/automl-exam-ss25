#!/usr/bin/env python3
"""
Full RL+BOHB Pipeline Test - As Intended

This script tests the COMPLETE intended pipeline:
1. Meta-feature extraction 
2. RL agent selection
3. BOHB evaluation with real optimization
4. Feedback to RL agent
5. Iterative improvement over 1 hour

This is the REAL test of our RL+BOHB integration.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl.meta_features import MetaFeatureExtractor
from automl.rl_agent import RLModelSelector, BOHBRewardEnv
from automl.datasets import load_dataset
from automl.logging_utils import AutoMLLogger
from automl.bohb_optimization import BOHBOptimizer, BOHBConfig


class FullPipelineTest:
    """Test the complete RL+BOHB pipeline as intended."""
    
    def __init__(self, max_runtime_hours: float = 1.0, output_dir: str = "pipeline_demo_results"):
        """Initialize full pipeline test.
        
        Args:
            max_runtime_hours: Maximum runtime in hours
            output_dir: Directory for results
        """
        self.max_runtime_hours = max_runtime_hours
        self.max_runtime_seconds = max_runtime_hours * 3600
        self.start_time = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = AutoMLLogger(
            experiment_name=f"pipeline_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            log_dir=self.output_dir / "logs"
        )
        
        # Results storage
        self.results = {
            'datasets': {},
            'meta_features': {},
            'rl_training_iterations': [],
            'bohb_evaluations': [],
            'model_selections': {},
            'performance_history': [],
            'timeline': []
        }
        
        # Available datasets - use ALL 4 datasets for full demo
        self.datasets = ['amazon', 'ag_news', 'dbpedia', 'imdb']
        
        # Define search spaces for each model type
        self.search_spaces = {
            'simple': {
                'model_type': 'simple',
                'learning_rate': (1e-4, 1e-2),
                'max_features': (1000, 5000),
                'regularization': (0.01, 1.0)
            },
            'medium': {
                'model_type': 'medium',
                'learning_rate': (1e-5, 1e-3),
                'hidden_size': (64, 256),
                'dropout': (0.1, 0.5),
                'batch_size': (16, 64)
            },
            'complex': {
                'model_type': 'complex',
                'learning_rate': (1e-6, 1e-4),
                'num_layers': (2, 8),
                'hidden_size': (128, 512),
                'attention_heads': (4, 16),
                'dropout': (0.1, 0.3)
            }
        }
        
    def _check_time_remaining(self) -> float:
        """Check remaining time."""
        if self.start_time is None:
            return self.max_runtime_seconds
            
        elapsed = time.time() - self.start_time
        remaining = self.max_runtime_seconds - elapsed
        
        return max(0, remaining)
    
    def _log_progress(self, stage: str, message: str, data: Dict[str, Any] = None):
        """Log progress with timestamp."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = self._check_time_remaining()
        
        progress_msg = f"[{elapsed/60:.1f}m elapsed, {remaining/60:.1f}m remaining] {stage}: {message}"
        print(progress_msg)
        
        self.logger.log_debug(progress_msg, data)
        
        # Add to timeline
        self.results['timeline'].append({
            'timestamp': time.time(),
            'elapsed_minutes': elapsed / 60,
            'stage': stage,
            'message': message,
            'data': data or {}
        })
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete RL+BOHB pipeline for 1 hour."""
        self.start_time = time.time()
        
        try:
            self._log_progress("STARTUP", "Initializing FULL RL+BOHB pipeline test")
            
            # Step 1: Extract meta-features from datasets
            self._log_progress("STEP_1", "Extracting meta-features")
            self._extract_meta_features()
            
            # Step 2: Prepare training datasets for BOHB-enhanced RL training
            self._log_progress("STEP_2", "Preparing RL training with BOHB feedback")
            training_datasets = self._prepare_training_datasets()
            
            # Step 3: Run iterative RL training with BOHB feedback
            self._log_progress("STEP_3", "Starting iterative RL+BOHB training")
            self._run_iterative_rl_bohb_training(training_datasets)
            
            # Step 4: Final evaluation with trained RL agent
            self._log_progress("STEP_4", "Final evaluation with trained agent")
            self._final_evaluation()
            
            # Step 5: Generate results
            self._log_progress("STEP_5", "Generating final results")
            final_results = self._generate_final_results()
            
            self._log_progress("COMPLETION", "Full pipeline completed successfully")
            
            return final_results
            
        except Exception as e:
            self._log_progress("ERROR", f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return self._handle_error()
            
        finally:
            self._save_results()
    
    def _extract_meta_features(self):
        """Extract meta-features from all datasets."""
        extractor = MetaFeatureExtractor()
        
        for dataset_name in self.datasets:
            try:
                self._log_progress("META_FEATURES", f"Processing {dataset_name}")
                
                # Load dataset
                texts, labels = load_dataset(dataset_name, split='train')
                
                # Sample for efficiency but use larger samples for full demo
                if len(texts) > 5000:  # Increased from 2000
                    indices = np.random.choice(len(texts), 5000, replace=False)
                    texts = [texts[i] for i in indices]
                    labels = [labels[i] for i in indices]
                
                # Convert to DataFrame
                train_df = pd.DataFrame({
                    'text': texts,
                    'label': labels
                })
                
                # Extract meta-features
                start_time = time.time()
                meta_features = extractor.extract_features(train_df)
                extraction_time = time.time() - start_time
                
                # Store results
                self.results['datasets'][dataset_name] = {
                    'num_samples': len(train_df),
                    'num_classes': len(set(train_df['label'])),
                    'extraction_time': extraction_time
                }
                
                self.results['meta_features'][dataset_name] = meta_features
                
                self._log_progress(
                    "META_FEATURES", 
                    f"Extracted {len(meta_features)} features from {dataset_name}",
                    {
                        'dataset': dataset_name,
                        'num_features': len(meta_features),
                        'extraction_time': extraction_time
                    }
                )
                
            except Exception as e:
                self._log_progress("ERROR", f"Failed to extract features from {dataset_name}: {e}")
                continue
    
    def _prepare_training_datasets(self) -> List[Tuple]:
        """Prepare training datasets for BOHB-enhanced RL training."""
        training_datasets = []
        
        for dataset_name in self.datasets:
            if dataset_name in self.results['meta_features']:
                texts, labels = load_dataset(dataset_name, split='train')
                
                # Sample for training (larger for full demo)
                if len(texts) > 1500:  # Increased from 800
                    indices = np.random.choice(len(texts), 1500, replace=False)
                    texts = [texts[i] for i in indices]
                    labels = [labels[i] for i in indices]
                
                meta_features = self.results['meta_features'][dataset_name]
                training_datasets.append((texts, labels, meta_features))
        
        self._log_progress(
            "PREPARE", 
            f"Prepared {len(training_datasets)} datasets for RL+BOHB training"
        )
        
        return training_datasets
    
    def _run_iterative_rl_bohb_training(self, training_datasets: List[Tuple]):
        """Run iterative RL training with BOHB feedback - FIXED CRITICAL ARCHITECTURE."""
        
        # Initialize RL selector
        rl_selector = RLModelSelector(
            meta_features_dim=35,
            model_save_path=self.output_dir / "models" / "rl_agent",
            logger=self.logger,
            random_state=42
        )
        
        iteration = 0
        last_performance = 0.0
        
        while self._check_time_remaining() > 600:  # Stop with 10 minutes remaining
            iteration += 1
            iteration_start = time.time()
            
            self._log_progress(
                "RL_BOHB_ITERATION", 
                f"Starting iteration {iteration}",
                {'iteration': iteration, 'remaining_time': self._check_time_remaining()/60}
            )
            
            # Limit iterations for proper exploration time
            if iteration > 10:
                self._log_progress("MAX_ITERATIONS", f"Reached maximum {iteration-1} iterations")
                break
            
            # Step 1: Train RL agent
            base_timesteps = int(self._check_time_remaining() * 20)
            timesteps_budget = min(50000, max(10000, base_timesteps))
            
            if iteration == 1:
                self._log_progress("RL_TRAINING", f"Initial RL training ({timesteps_budget} timesteps)")
                rl_selector.train(
                    total_timesteps=timesteps_budget,
                    learning_rate=5e-4,
                    exploration_fraction=0.7
                )
            else:
                continue_timesteps = max(5000, timesteps_budget // 2)
                self._log_progress("RL_TRAINING", f"Continuing RL training ({continue_timesteps} timesteps)")
                rl_selector.train(
                    total_timesteps=continue_timesteps,
                    learning_rate=2e-4,
                    exploration_fraction=0.3
                )
            
            # Step 2: RL+BOHB evaluation loop (FIXED ARCHITECTURE)
            current_performance = 0.0
            evaluation_results = []
            
            for dataset_name in self.datasets:
                if dataset_name not in self.results['meta_features']:
                    continue
                
                try:
                    meta_features = self.results['meta_features'][dataset_name]
                    texts, labels = load_dataset(dataset_name, split='train')
                    
                    # Sample for BOHB evaluation
                    if len(texts) > 1000:
                        indices = np.random.choice(len(texts), 1000, replace=False)
                        texts = [texts[i] for i in indices]
                        labels = [labels[i] for i in indices]
                    
                    self._log_progress(
                        "RL_SELECTION", 
                        f"Iteration {iteration}: RL selecting model for {dataset_name}"
                    )
                    
                    # CRITICAL FIX: RL selects ONLY the model type
                    chosen_model_type, action, debug_info = rl_selector.select_model(
                        meta_features, deterministic=True
                    )
                    
                    self._log_progress(
                        "RL_CHOICE", 
                        f"{dataset_name}: RL chose {chosen_model_type} (action={action})"
                    )
                    
                    # CRITICAL FIX: BOHB optimizes ONLY the chosen model type's hyperparameters
                    self._log_progress(
                        "BOHB_OPTIMIZE", 
                        f"BOHB optimizing {chosen_model_type} hyperparameters for {dataset_name}"
                    )
                    
                    # Create BOHB config for this specific optimization
                    bohb_config = BOHBConfig(
                        max_budget=50.0,
                        min_budget=10.0,
                        n_trials=15,
                        wall_clock_limit=120.0  # 2 minutes per dataset
                    )
                    
                    # Create BOHB optimizer for the chosen model type
                    bohb_optimizer = BOHBOptimizer(
                        model_type=chosen_model_type,
                        logger=self.logger,
                        config=bohb_config
                    )
                    
                    # BOHB optimizes only the chosen model type
                    best_config, best_score, bohb_info = bohb_optimizer.optimize(
                        X_train=texts,
                        y_train=labels,
                        fidelity_mode="low"
                    )
                    
                    self._log_progress(
                        "BOHB_RESULT", 
                        f"{dataset_name}: {chosen_model_type} achieved score {best_score:.4f}"
                    )
                    
                    # CRITICAL FIX: Feed back the score to RL for learning
                    try:
                        baseline_score = meta_features.get('baseline_accuracy', 0.5)
                        rl_selector.update_reward(
                            meta_features=meta_features,
                            action=action,
                            bohb_score=best_score,
                            baseline_score=baseline_score
                        )
                        self._log_progress("RL_UPDATE", f"Updated RL reward for action {action}: {best_score:.4f} vs baseline {baseline_score:.4f}")
                    except Exception as e:
                        self._log_progress("WARNING", f"Could not update RL reward: {e}")
                    
                    current_performance += best_score
                    
                    evaluation_results.append({
                        'iteration': iteration,
                        'dataset': dataset_name,
                        'selected_model': chosen_model_type,
                        'action': action,
                        'bohb_score': best_score,
                        'best_config': best_config,
                        'method': 'rl_then_bohb'
                    })
                    
                    # Store BOHB evaluation details
                    self.results['bohb_evaluations'].append({
                        'iteration': iteration,
                        'dataset': dataset_name,
                        'model_type': chosen_model_type,
                        'bohb_score': best_score,
                        'best_config': best_config,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    self._log_progress("ERROR", f"RL+BOHB evaluation failed for {dataset_name}: {e}")
                    continue
            
            # Step 3: Calculate iteration performance and improvement
            avg_performance = current_performance / len(evaluation_results) if evaluation_results else 0.0
            improvement = avg_performance - last_performance
            
            iteration_time = time.time() - iteration_start
            
            # Store iteration results
            iteration_summary = {
                'iteration': iteration,
                'avg_performance': avg_performance,
                'improvement': improvement,
                'evaluation_results': evaluation_results,
                'iteration_time': iteration_time,
                'remaining_time': self._check_time_remaining()
            }
            
            self.results['rl_training_iterations'].append(iteration_summary)
            self.results['performance_history'].append({
                'iteration': iteration,
                'performance': avg_performance,
                'improvement': improvement,
                'timestamp': time.time()
            })
            
            self._log_progress(
                "ITERATION_COMPLETE", 
                f"Iteration {iteration}: avg_performance={avg_performance:.4f}, "
                f"improvement={improvement:+.4f}, time={iteration_time:.1f}s",
                iteration_summary
            )
            
            # Check for convergence
            if improvement < 0.01 and iteration > 3:
                self._log_progress("CONVERGENCE", f"Performance converged after {iteration} iterations")
                break
            
            last_performance = avg_performance
            
            # Save intermediate results
            self._save_intermediate_results(iteration)
        
        # Store final trained selector
        self.final_rl_selector = rl_selector
        
        self._log_progress(
            "RL_BOHB_COMPLETE", 
            f"Completed {iteration} iterations of FIXED RL+BOHB training"
        )
    
    def _final_evaluation(self):
        """Final evaluation with the trained RL agent."""
        if not hasattr(self, 'final_rl_selector'):
            self._log_progress("ERROR", "No trained RL selector available for final evaluation")
            return
        
        final_selections = {}
        
        for dataset_name in self.datasets:
            if dataset_name not in self.results['meta_features']:
                continue
            
            try:
                meta_features = self.results['meta_features'][dataset_name]
                
                # Final selection with trained agent
                model_type, action, decision_info = self.final_rl_selector.select_model(
                    meta_features, deterministic=True
                )
                
                final_selections[dataset_name] = {
                    'model_type': model_type,
                    'action': action,
                    'confidence': decision_info.get('confidence', 0),
                    'method': 'final_trained_rl'
                }
                
                self._log_progress(
                    "FINAL_EVAL", 
                    f"{dataset_name} final selection: {model_type} (confidence: {decision_info.get('confidence', 0):.3f})"
                )
                
            except Exception as e:
                self._log_progress("ERROR", f"Final evaluation failed for {dataset_name}: {e}")
        
        self.results['final_selections'] = final_selections
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate comprehensive final results with visualizations."""
        total_time = time.time() - self.start_time
        
        # Calculate performance improvement over iterations
        performance_history = self.results.get('performance_history', [])
        initial_performance = performance_history[0]['performance'] if performance_history else 0.0
        final_performance = performance_history[-1]['performance'] if performance_history else 0.0
        total_improvement = final_performance - initial_performance
        
        # Calculate BOHB statistics
        bohb_evaluations = self.results.get('bohb_evaluations', [])
        avg_bohb_score = np.mean([e['bohb_score'] for e in bohb_evaluations]) if bohb_evaluations else 0.0
        
        # Generate visualizations
        self._generate_comprehensive_visualizations()
        
        final_results = {
            'experiment_summary': {
                'total_runtime_minutes': total_time / 60,
                'max_runtime_hours': self.max_runtime_hours,
                'datasets_processed': len(self.results['datasets']),
                'total_iterations': len(self.results['rl_training_iterations']),
                'total_bohb_evaluations': len(bohb_evaluations)
            },
            'performance_improvement': {
                'initial_performance': initial_performance,
                'final_performance': final_performance,
                'total_improvement': total_improvement,
                'avg_bohb_score': avg_bohb_score
            },
            'convergence_analysis': {
                'converged': total_improvement > 0.01,
                'iterations_to_convergence': len(performance_history),
                'performance_trend': [p['performance'] for p in performance_history]
            },
            'final_model_selections': self.results.get('final_selections', {}),
            'bohb_integration_success': len(bohb_evaluations) > 0
        }
        
        return final_results
    
    def _generate_comprehensive_visualizations(self):
        """Generate comprehensive visualizations of the RL+BOHB pipeline."""
        try:
            # Create visualization directory
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Set style for better plots
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            self._log_progress("VISUALIZATION", "Generating comprehensive visualizations")
            
            # 1. Meta-features comparison across datasets
            self._plot_meta_features_comparison(viz_dir)
            
            # 2. RL+BOHB iteration progress
            self._plot_iteration_progress(viz_dir)
            
            # 3. Model selection evolution
            self._plot_model_selection_evolution(viz_dir)
            
            # 4. BOHB performance analysis
            self._plot_bohb_performance_analysis(viz_dir)
            
            # 5. Timeline and resource usage
            self._plot_timeline_analysis(viz_dir)
            
            # 6. Final pipeline summary
            self._plot_pipeline_summary(viz_dir)
            
            self._log_progress("VISUALIZATION", f"Generated 6 comprehensive visualizations in {viz_dir}")
            
        except Exception as e:
            self._log_progress("ERROR", f"Visualization generation failed: {e}")
    
    def _plot_meta_features_comparison(self, viz_dir: Path):
        """Plot meta-features comparison across datasets."""
        if not self.results['meta_features']:
            return
        
        # Prepare data
        datasets = list(self.results['meta_features'].keys())
        features_data = []
        
        for dataset in datasets:
            features = self.results['meta_features'][dataset]
            for feature_name, value in features.items():
                features_data.append({
                    'dataset': dataset,
                    'feature': feature_name,
                    'value': value
                })
        
        df = pd.DataFrame(features_data)
        
        # Select top 10 most varying features
        feature_variance = df.groupby('feature')['value'].var().sort_values(ascending=False)
        top_features = feature_variance.head(10).index.tolist()
        
        df_top = df[df['feature'].isin(top_features)]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df = df_top.pivot(index='feature', columns='dataset', values='value')
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', ax=ax)
        ax.set_title('Meta-Features Comparison Across Datasets', fontsize=16)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Meta-Feature', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'meta_features_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_iteration_progress(self, viz_dir: Path):
        """Plot RL+BOHB iteration progress over time."""
        performance_history = self.results.get('performance_history', [])
        if not performance_history:
            return
        
        iterations = [p['iteration'] for p in performance_history]
        performances = [p['performance'] for p in performance_history]
        improvements = [p['improvement'] for p in performance_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Performance over iterations
        ax1.plot(iterations, performances, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('RL+BOHB Iteration', fontsize=12)
        ax1.set_ylabel('Average Performance', fontsize=12)
        ax1.set_title('RL+BOHB Performance Evolution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Performance improvement per iteration
        ax2.bar(iterations, improvements, alpha=0.7, color='skyblue')
        ax2.set_xlabel('RL+BOHB Iteration', fontsize=12)
        ax2.set_ylabel('Performance Improvement', fontsize=12)
        ax2.set_title('Performance Improvement per Iteration', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'iteration_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_selection_evolution(self, viz_dir: Path):
        """Plot how model selections evolved during training."""
        rl_iterations = self.results.get('rl_training_iterations', [])
        if not rl_iterations:
            return
        
        # Track model selections per iteration
        iteration_data = []
        for iteration_result in rl_iterations:
            for eval_result in iteration_result.get('evaluation_results', []):
                iteration_data.append({
                    'iteration': eval_result['iteration'],
                    'dataset': eval_result['dataset'],
                    'model': eval_result['selected_model'],
                    'score': eval_result['bohb_score']
                })
        
        if not iteration_data:
            return
        
        df = pd.DataFrame(iteration_data)
        
        # Model selection distribution over iterations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model selection counts per iteration
        model_counts = df.groupby(['iteration', 'model']).size().unstack(fill_value=0)
        model_counts.plot(kind='bar', stacked=True, ax=ax1, alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Number of Selections', fontsize=12)
        ax1.set_title('Model Selection Evolution', fontsize=14)
        ax1.legend(title='Model Type')
        ax1.tick_params(axis='x', rotation=0)
        
        # Average scores per model type
        avg_scores = df.groupby('model')['score'].mean()
        bars = ax2.bar(avg_scores.index, avg_scores.values, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_xlabel('Model Type', fontsize=12)
        ax2.set_ylabel('Average BOHB Score', fontsize=12)
        ax2.set_title('Average Performance by Model Type', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, avg_scores.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_selection_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bohb_performance_analysis(self, viz_dir: Path):
        """Plot detailed BOHB performance analysis."""
        bohb_evaluations = self.results.get('bohb_evaluations', [])
        if not bohb_evaluations:
            return
        
        # Prepare BOHB data
        datasets = []
        scores = []
        iterations = []
        
        for evaluation in bohb_evaluations:
            datasets.append(evaluation['dataset'])
            scores.append(evaluation['bohb_score'])
            iterations.append(evaluation.get('iteration', 1))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # BOHB scores by dataset
        df_bohb = pd.DataFrame({'dataset': datasets, 'score': scores, 'iteration': iterations})
        avg_scores = df_bohb.groupby('dataset')['score'].mean()
        
        bars = ax1.bar(avg_scores.index, avg_scores.values, alpha=0.7, color='lightblue')
        ax1.set_xlabel('Dataset', fontsize=12)
        ax1.set_ylabel('Average BOHB Score', fontsize=12)
        ax1.set_title('BOHB Performance by Dataset', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, avg_scores.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # BOHB score distribution
        ax2.hist(scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('BOHB Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('BOHB Score Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'bohb_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_timeline_analysis(self, viz_dir: Path):
        """Plot timeline analysis of pipeline execution."""
        timeline = self.results.get('timeline', [])
        if not timeline:
            return
        
        # Prepare timeline data
        timeline_df = pd.DataFrame(timeline)
        
        # Stage duration analysis
        stages = timeline_df['stage'].unique()
        stage_durations = {}
        stage_order = []
        
        for stage in ['STARTUP', 'STEP_1', 'STEP_2', 'STEP_3', 'STEP_4', 'STEP_5']:
            if stage in stages:
                stage_data = timeline_df[timeline_df['stage'] == stage]
                if len(stage_data) > 0:
                    start_time = stage_data['elapsed_minutes'].min()
                    end_time = stage_data['elapsed_minutes'].max()
                    duration = max(end_time - start_time, 0.1)
                    stage_durations[stage] = duration
                    stage_order.append(stage)
        
        # Create timeline plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Timeline progress
        ax1.plot(timeline_df['elapsed_minutes'], range(len(timeline_df)), 
                marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Elapsed Time (minutes)', fontsize=12)
        ax1.set_ylabel('Progress Steps', fontsize=12)
        ax1.set_title('Pipeline Execution Timeline', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Stage duration pie chart
        if stage_durations:
            ordered_durations = [stage_durations[stage] for stage in stage_order]
            ax2.pie(ordered_durations, labels=stage_order, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Time Distribution by Pipeline Stage', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'timeline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pipeline_summary(self, viz_dir: Path):
        """Plot comprehensive pipeline summary."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Datasets overview
        if self.results['datasets']:
            dataset_names = list(self.results['datasets'].keys())
            dataset_sizes = [self.results['datasets'][ds]['num_samples'] for ds in dataset_names]
            dataset_classes = [self.results['datasets'][ds]['num_classes'] for ds in dataset_names]
            
            x = np.arange(len(dataset_names))
            width = 0.35
            
            ax1_twin = ax1.twinx()
            bars1 = ax1.bar(x - width/2, dataset_sizes, width, label='Samples', alpha=0.8, color='skyblue')
            bars2 = ax1_twin.bar(x + width/2, dataset_classes, width, label='Classes', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('Dataset', fontsize=12)
            ax1.set_ylabel('Number of Samples', fontsize=12, color='blue')
            ax1_twin.set_ylabel('Number of Classes', fontsize=12, color='red')
            ax1.set_title('Dataset Overview', fontsize=14)
            ax1.set_xticks(x)
            ax1.set_xticklabels(dataset_names, rotation=45)
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # 2. RL+BOHB integration success
        total_evaluations = len(self.results.get('bohb_evaluations', []))
        successful_bohb = len([e for e in self.results.get('bohb_evaluations', []) 
                              if e.get('bohb_score', 0) > 0])
        
        if total_evaluations > 0:
            success_rate = successful_bohb / total_evaluations
            labels = ['BOHB Success', 'Fallback']
            sizes = [successful_bohb, total_evaluations - successful_bohb]
            colors = ['lightgreen', 'lightcoral']
            
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title(f'BOHB Integration Success\n({success_rate:.1%} success rate)', fontsize=14)
        
        # 3. Performance improvement
        performance_history = self.results.get('performance_history', [])
        if performance_history:
            iterations = [p['iteration'] for p in performance_history]
            performances = [p['performance'] for p in performance_history]
            
            ax3.plot(iterations, performances, marker='o', linewidth=3, markersize=8, color='green')
            ax3.fill_between(iterations, performances, alpha=0.3, color='green')
            ax3.set_xlabel('Iteration', fontsize=12)
            ax3.set_ylabel('Performance', fontsize=12)
            ax3.set_title('Performance Improvement Over Time', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
        
        # 4. Final model distribution
        final_selections = self.results.get('final_selections', {})
        if final_selections:
            model_types = [selection['model_type'] for selection in final_selections.values()]
            model_counts = pd.Series(model_types).value_counts()
            
            ax4.bar(model_counts.index, model_counts.values, alpha=0.8, color='purple')
            ax4.set_xlabel('Model Type', fontsize=12)
            ax4.set_ylabel('Number of Selections', fontsize=12)
            ax4.set_title('Final Model Selection Distribution', fontsize=14)
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, (model, count) in enumerate(model_counts.items()):
                ax4.text(i, count + 0.05, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'pipeline_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_intermediate_results(self, iteration: int):
        """Save intermediate results after each iteration."""
        import json
        
        # Save iteration summary
        iteration_file = self.output_dir / f"iteration_{iteration}_results.json"
        
        with open(iteration_file, 'w') as f:
            json.dump({
                'iteration': iteration,
                'performance_history': self.results['performance_history'],
                'bohb_evaluations': self.results['bohb_evaluations'][-len(self.datasets):],  # Last iteration
                'timestamp': time.time()
            }, f, indent=2)
    
    def _save_results(self):
        """Save all results to files."""
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        cleaned_results = clean_for_json(self.results)
        
        with open(self.output_dir / 'full_pipeline_results.json', 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        
        self._log_progress("SAVE", f"Results saved to {self.output_dir}")
    
    def _handle_error(self) -> Dict[str, Any]:
        """Handle pipeline errors gracefully."""
        return {
            'status': 'error',
            'partial_results': self.results,
            'message': 'Pipeline encountered errors'
        }


def main():
    """Main function to run the full RL+BOHB pipeline test."""
    print("=" * 70)
    print("AUTOML PIPELINE DEMO - FULL RL+BOHB INTEGRATION")
    print("Meta-features → RL Agent → BOHB Evaluation → Feedback → Iterate")
    print("Running with FULL datasets for 1 hour")
    print("=" * 70)
    
    # Initialize pipeline demo - SHORTER TIME FOR TESTING FIXES
    pipeline_test = FullPipelineTest(
        max_runtime_hours=0.2,  # 12 minutes for testing fixes
        output_dir="pipeline_demo_results"
    )
    
    try:
        # Run complete pipeline
        final_results = pipeline_test.run_full_pipeline()
        
        # Print summary
        print("\n" + "=" * 70)
        print("AUTOML PIPELINE DEMO COMPLETED")
        print("=" * 70)
        
        if 'experiment_summary' in final_results:
            summary = final_results['experiment_summary']
            print(f"🕐 Runtime: {summary['total_runtime_minutes']:.1f} minutes")
            print(f"📁 Datasets processed: {summary['datasets_processed']}")
            print(f"🔄 RL+BOHB iterations: {summary['total_iterations']}")
            print(f"🔧 BOHB evaluations: {summary['total_bohb_evaluations']}")
        
        if 'performance_improvement' in final_results:
            perf = final_results['performance_improvement']
            print(f"\n📈 PERFORMANCE IMPROVEMENT:")
            print(f"  Initial: {perf['initial_performance']:.4f}")
            print(f"  Final: {perf['final_performance']:.4f}")
            print(f"  Total improvement: {perf['total_improvement']:+.4f}")
            print(f"  Avg BOHB score: {perf['avg_bohb_score']:.4f}")
        
        if 'final_model_selections' in final_results:
            print(f"\n🎯 FINAL MODEL SELECTIONS:")
            for dataset, selection in final_results['final_model_selections'].items():
                print(f"  {dataset}: {selection['model_type']} (confidence: {selection['confidence']:.1f})")
        
        bohb_success = final_results.get('bohb_integration_success', False)
        print(f"\n✅ BOHB Integration: {'SUCCESS' if bohb_success else 'FAILED'}")
        
        print(f"\n📂 Results saved to: {pipeline_test.output_dir}")
        
        return 0 if bohb_success else 1
        
    except Exception as e:
        print(f"\n❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
