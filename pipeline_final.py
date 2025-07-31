"""
AutoML Pipeline
--------------------------------------------------

This module implements an AutoML pipeline. The pipeline:

1. Extracts meta-features from all datasets
2. Performs leave-one-out cross-validation where:
   - One dataset is held out
   - RL+BOHB training occurs on remaining datasets
   - The trained RL agent selects a model for the held-out dataset
   - Performance is evaluated on the held-out dataset
3. After CV, trains a final RL+BOHB model on all  4 datasets
4. Runs BOHB with result of RL agent model and hyperparameter configuration to tune more
5. Predicts labels for the exam dataset and saves results
"""

import time
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import torch
import argparse

from automl.meta_features import MetaFeatureExtractor
from automl.rl_agent import RLModelSelector
from automl.datasets import load_dataset
from automl.logging_utils import AutoMLLogger
from automl.bohb_optimization import BOHBOptimizer, BOHBConfig
from automl.models import create_model
from automl.constants import META_FEATURE_DIM, FEATURE_ORDER

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class AutoMLPipeline:
    """AutoML Pipeline"""

    def __init__(
        self, 
        max_runtime_hours: float = 1.0, 
        output_dir: str = "automl_pipeline_results",
        max_iterations: int = 10
    ):
        """Initialize AutoML pipeline.
        
        Args:
            max_runtime_hours: Maximum runtime in hours
            output_dir: Directory for results
            datasets: List of datasets to use (default: ['amazon', 'ag_news', 'dbpedia', 'imdb'])
            max_iterations: Maximum iterations for RL training
        """
        # Time Limit Management
        self.max_runtime_hours = max_runtime_hours
        self.max_runtime_seconds = max_runtime_hours * 3600
        self.start_time = None
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # base BOHB configuration
        self.bohb_base = BOHBConfig(max_budget=30, min_budget=10, n_trials=10, wall_clock_limit=300)
        # Maximum iterations for RL training
        self.max_iterations = max_iterations

        # Available datasets
        # (Phase 1) Training datasets
        #self.datasets = ['amazon', 'ag_news', 'dbpedia', 'imdb']
        self.datasets = ['amazon', 'ag_news']
        # (Phase 2) Exam dataset 
        self.exam_dataset = "yelp"
        # cv agents dictionary
        self.cv_agent_paths: Dict[str, Path] = {}
        # Create models directory
        models_dir = self.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = AutoMLLogger(
            experiment_name=f"automl_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
            'timeline': [],
            'cv_results': {},
            'detailed_logs': [],
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
        self.logger.logger.info(progress_msg)
        if data:
            self.logger.log_debug("extra", data)
        
        # Add to timeline
        self.results['timeline'].append({
            'timestamp': time.time(),
            'elapsed_minutes': elapsed / 60,
            'stage': stage,
            'message': message,
            'data': data or {}
        })
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete AutoML pipeline."""
        self.start_time = time.time()
        
        try:
            self._log_progress("STARTUP", "Initializing AutoML Pipeline")

            # Step 1: Extract meta-features from all datasets
            self._log_progress("STEP_1", "Extracting meta-features")
            self._extract_meta_features(is_test=False)
            
            # Step 2: Run leave-one-out cross-validation
            self._log_progress("STEP_2", "Starting leave-one-out cross-validation")
            self.run_leave_one_out_cv()

            # Step 3: Run training on all datasets
            self._log_progress("STEP_3", "Starting training on all datasets")
            self._run_final_training(is_test=True)
            
            # Step 4: Predict Test Labels and save results
            predicted_labels = self._predict_on_test_split()
            output_path = Path("data/exam_dataset")
            output_path.mkdir(parents=True, exist_ok=True)

            prediction_file = output_path / "predictions.npy"
            np.save(prediction_file, predicted_labels)
            print(f"Saved predictions to {prediction_file}")

            self._log_progress("COMPLETION", "Full pipeline completed successfully")


        except Exception as e:
            self._log_progress("CRITICAL_ERROR", f"Pipeline failed: {e}")
            import traceback
            self.logger.log_error(f"Pipeline error: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _extract_meta_features(self, is_test: bool = False):
        """Extract meta-features from datasets."""
        extractor = MetaFeatureExtractor()
        # tune datasets array based on the test mode
        if is_test:
            datasets = [self.exam_dataset]
        else:
            datasets = self.datasets
        # go over each dataset
        for dataset_name in datasets:
            try:
                self._log_progress("META_FEATURES", f"Processing {dataset_name}")
                
                # Load full dataset
                texts, labels = load_dataset(dataset_name, split='train')
                
                # Convert to DataFrame format expected by MetaFeatureExtractor
                train_df = pd.DataFrame({
                    'text': texts,
                    'label': labels
                })
                
                # Extract meta-features
                start_time = time.time()
                raw_meta_features = extractor.extract_features(train_df)
                
                # Check if extraction was successful
                if raw_meta_features is None:
                    self._log_progress("ERROR", f"Failed to extract features for {dataset_name}: raw_meta_features is None")
                    break

                # Normalize and order meta-features for the RL agent
                meta_features = extractor.get_ordered_features(raw_meta_features)
                extraction_time = time.time() - start_time
                
                # Add dataset name for reference
                meta_features['dataset_name'] = dataset_name
                
                # Store dataset info
                self.results['datasets'][dataset_name] = {
                    'num_samples': len(texts),
                    'num_classes': len(np.unique(labels)),
                    'extraction_time': extraction_time
                }
                
                self.results['meta_features'][dataset_name] = meta_features
                # If in test mode, return the meta features for the exam dataset
                if is_test:
                    return meta_features
                
                # Log the progress except for test mode
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

    def _prepare_training_datasets(self, exclude_dataset: Optional[str] = None) -> List[Tuple]:
        """Prepare training datasets for BOHB+RL training.

        Args:
            exclude_dataset: Optional dataset to exclude (for cross-validation)
            
        Returns:
            List of (texts, labels, meta_features) tuples
        """
        training_datasets = []
        
        for dataset_name in self.datasets:
            # Skip excluded dataset
            if exclude_dataset and dataset_name == exclude_dataset:
                continue
                
            if dataset_name in self.results['meta_features']:
                texts, labels = load_dataset(dataset_name, split='train')
                
                # Sample for training
                if len(texts) > 1500:
                    indices = np.random.choice(len(texts), 1500, replace=False)
                    texts = [texts[i] for i in indices]
                    labels = [labels[i] for i in indices]
                
                meta_features = self.results['meta_features'][dataset_name]
                training_datasets.append((texts, labels, meta_features))
        
        self._log_progress(
            "PREPARE", 
            f"Prepared {len(training_datasets)} datasets for training" +
            (f" (excluding {exclude_dataset})" if exclude_dataset else "")
        )
        
        return training_datasets

    def make_bohb_profile(self, iteration: int, max_iter: int, base_cfg: BOHBConfig, model_type: str = None, is_test: bool = False) -> BOHBConfig:
        """
        Returns a BOHBConfig tuned for iteration.
        
        - "explore" for early iterations (cheap, few trials)
        - "exploit" for later iterations (more trials, longer walltime)
        """
        cfg = deepcopy(base_cfg)

        base_wall = base_cfg.wall_clock_limit

        if is_test and model_type == "TEST":
            cfg.n_trials = min(30, cfg.n_trials)
            cfg.wall_clock_limit = min(600, base_wall * 3.0)
            return cfg

        if model_type == "simple":
            base_wall = base_cfg.wall_clock_limit * 1.0
        elif model_type == "medium":
            base_wall = base_cfg.wall_clock_limit * 1.5
        elif model_type == "complex":
            base_wall = base_cfg.wall_clock_limit * 2.0
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # threshold between explore/exploit
        explore_cutoff = max(3, int(0.3 * max_iter))

        if iteration <= explore_cutoff:
            # cheap exploration: few trials, short timeout
            cfg.n_trials = min(3, max(5, cfg.n_trials // 4))
            cfg.wall_clock_limit = max(60, base_wall * 0.3)
        else:
            # heavier exploitation
            cfg.n_trials = min(15, cfg.n_trials)
            cfg.wall_clock_limit = min(300, base_wall * 1.0)

        
        return cfg

    def _run_iterative_rl_bohb_training(
            self,
            training_datasets: List[Tuple],
            prefix: str = ""
        ) -> RLModelSelector:
        """
        RL + BOHB training that learns only from real-data experience.
        """
        log_prefix = f"{prefix}_" if prefix else ""

        # Initialise RL model selector
        rl_selector = RLModelSelector(
            meta_features_dim=META_FEATURE_DIM,
            model_save_path=self.output_dir / "models" / f"{log_prefix}rl_agent",
            logger=self.logger,
            random_state=42,
        )
        rl_selector.train(                
            total_timesteps=1,            # minimal “bootstrap” call
            learning_rate=5e-4,
            exploration_fraction=0.0,     # act deterministically
            target_update_interval=500,
        )

        # Max iterations for RL training
        max_iterations = self.max_iterations

        for iteration in range(1, max_iterations + 1):
            if self._check_time_remaining() < 600:
                self._log_progress(f"{log_prefix}TIME_LIMIT",
                                "Stopping - <10 min left")
                break

            #bohb_cfg = self.make_bohb_profile(iteration, max_iterations, self.bohb_base, model_type=None, is_test=False)

            iteration_perf, evals = 0.0, []
            self._log_progress(f"{log_prefix}ITER", f"-> iteration {iteration}")

            # ---- iterate over real datasets -----------------------------------
            for idx, (texts, labels, mfeat) in enumerate(training_datasets):
                dname = mfeat.get("dataset_name", "unknown")

                # ---- 1. env.reset with real meta‑features ---------------------
                obs, _ = rl_selector.env.reset(
                    options={"meta_features": rl_selector._prepare_observation(mfeat)}
                )

                # ---- 2. agent picks a model (greedy) --------------------------
                if iteration == 1 and idx == 0:
                    # We pick simple action for first iteration
                    # or we can pick a random action
                    # action = np.random.randint(rl_selector.env.action_space.n)
                    action = 0
                else:
                    # predict the next action after the first iteration (normal case)
                    action, _ = rl_selector.agent.predict(obs, deterministic=True)
                
                action = int(action)
                model_type = rl_selector.env.action_to_model[int(action)]
                self._log_progress(f"{log_prefix}CHOICE",
                                f"{dname}: picked {model_type}")

                # ---- 3. BOHB hyper‑parameter search ---------------------------
                bohb_cfg = self.make_bohb_profile(iteration, max_iterations, self.bohb_base, model_type=model_type, is_test=False)
                bo = BOHBOptimizer(model_type=model_type,
                                random_state=42,
                                config=bohb_cfg)
                
                best_cfg, acc, bohb_info = bo.optimize(
                    X_train=texts,
                    y_train=labels,
                    fidelity_mode="high" if iteration > 1 else "low",
                )
                self._log_progress(f"{log_prefix}BOHB",
                                f"{dname}: score {acc:.4f}")

                # ---- 4. reward: -----------------------
                # Send accuracy to RL environment and calculate reward
                # See calculate_enhanced_reward in rl_agent.py for reward calculation
                rl_selector.env.bohb_accuracy = float(acc)
                next_obs, reward, done, truncated, info = rl_selector.env.step(action)
                rl_selector.env.bohb_accuracy = None

                # ---- 5. store transition & learn ------------------------------
                rl_selector.agent.replay_buffer.add(
                    obs,                      # s
                    obs,                      # s' (single‑step episode)
                    np.array([action]),
                    np.array([reward]),
                    np.array([True]),         # done
                    [{}],
                )
                # short online update
                rl_selector.agent.learn(
                    total_timesteps=256,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )

                # --- DETAILED LOG ENTRY ----------------------------------------
                obs_tensor = torch.from_numpy(obs.reshape(1, -1).astype(np.float32))
                q_vals = rl_selector.agent.q_net(obs_tensor).detach().numpy()[0]

                self.results['detailed_logs'].append({
                    "fold": prefix or "final",
                    "iteration": iteration,
                    "dataset": dname,
                    "action": action,
                    "model_type": model_type,
                    "q_values": q_vals.tolist(),
                    "reward": reward,
                    "reward_components": info["reward_components"],
                    "bohb": {
                        "best_score": float(acc),
                        "best_config": best_cfg,
                        "incumbent_history": bohb_info
                                                .get("optimization_stats", {})
                                                .get("convergence_history", [])[-10:],
                        "n_trials": bohb_cfg.n_trials,
                        "runtime_s": bohb_info.get("total_time", None),
                        }
                })

                # --- Bookkeeping ----------------------------------------------------
                iteration_perf += acc
                evals.append({
                    "dataset": dname,
                    "model": model_type,
                    "acc": acc,
                    "reward": reward,
                    "best_cfg": best_cfg,
                })
                self.results['bohb_evaluations'].append({
                    "iteration": iteration,
                    "dataset": dname,
                    "model_type": model_type,
                    "bohb_score": acc,
                    "best_config": best_cfg,
                    "timestamp": time.time(),
                    "cv_fold": prefix or 'final',
                })


            # ---- Iteration summary --------------------------------------------
            mean_acc = iteration_perf / len(evals)
            self.results['rl_training_iterations'].append({
                "iteration": iteration,
                "avg_performance": mean_acc,
                "evaluation_results": evals,
                "cv_fold": prefix or 'final',
            })
            self._log_progress(f"{log_prefix}ITER_DONE",
                            f"iter  {iteration}: meanacc {mean_acc:.4f}")

            self.results['performance_history'].append({
                    'iteration': iteration,
                    'performance': mean_acc,
                    'timestamp': time.time(),
                    'cv_fold': prefix or 'final'
                })
            
            # Simple convergence check
            # TODO: TUNE THIS THRESHOLD (0.005)
            if iteration > 1 and \
            abs(self.results['rl_training_iterations'][-1]['avg_performance']
                - self.results['rl_training_iterations'][-2]['avg_performance']) < 0.005:
                break

        self._log_progress(f"{log_prefix}TRAINING", "RL-BOHB finished")
        rl_selector.save_model()

        return rl_selector

    def run_leave_one_out_cv(self):
        """Run leave-one-out cross-validation.
        
        For each dataset:
        1. Hold out one dataset
        2. Train RL+BOHB on remaining datasets
        3. Evaluate on held-out dataset
        """
        self.results['cv_results'] = {
            'folds': [],
            'performance': {},
            'summary': {}
        }
        
        # For each dataset as held-out
        for held_out in self.datasets:
            fold_start_time = time.time()
            
            self._log_progress("CV_FOLD", f"Starting fold with {held_out} as held-out dataset")
            
            # Prepare training datasets (excluding held-out)
            training_datasets = self._prepare_training_datasets(exclude_dataset=held_out)
            
            # Check if we have meta-features for the held-out dataset
            if held_out not in self.results['meta_features']:
                self._log_progress("CV_ERROR", f"Missing meta-features for {held_out}, skipping fold")
                continue
                
            # Run RL+BOHB training on remaining datasets
            rl_selector = self._run_iterative_rl_bohb_training(
                training_datasets, 
                prefix=f"cv_{held_out}"
            )
            
            # Save the agent path for this fold
            self.cv_agent_paths[held_out] = rl_selector.model_save_path
            
            # Evaluate on held-out dataset
            held_out_meta_features = self.results['meta_features'][held_out]
            texts, labels = load_dataset(held_out, split='train')
            
            # Select model for held-out dataset
            model_type, action, debug_info = rl_selector.select_model(
                held_out_meta_features, deterministic=True
            )
            
            self._log_progress(
                "CV_EVALUATION", 
                f"Selected {model_type} for held-out dataset {held_out}"
            )
            # Run with max budget for final evaluation for leaved out dataset
            final_cfg = self.make_bohb_profile(self.max_iterations + 1, self.max_iterations, self.bohb_base, model_type=model_type, is_test=True)
            
            bohb_optimizer = BOHBOptimizer(
                model_type=model_type,
                random_state=42,
                config=final_cfg
            )
            
            # Sample for BOHB evaluation
            if len(texts) > 1000:
                indices = np.random.choice(len(texts), 1000, replace=False)
                sampled_texts = [texts[i] for i in indices]
                sampled_labels = [labels[i] for i in indices]
            else:
                sampled_texts, sampled_labels = texts, labels
            
            # Run BOHB optimization with higher fidelity for better evaluation
            best_config, best_score, bohb_info = bohb_optimizer.optimize(
                X_train=sampled_texts,
                y_train=sampled_labels,
                fidelity_mode="high"  # Use high fidelity for better results
            )
            
            fold_time = time.time() - fold_start_time
            
            # Record CV results
            fold_result = {
                'held_out': held_out,
                'selected_model': model_type,
                'action': action,
                'bohb_score': best_score,
                'best_config': best_config,
                'fold_time': fold_time,
                'confidence': debug_info.get('confidence', 0)
            }
            
            self.results['cv_results']['folds'].append(fold_result)
            self.results['cv_results']['performance'][held_out] = best_score
            
            self._log_progress(
                "CV_RESULT", 
                f"Held-out {held_out}: {model_type} model achieved {best_score:.4f}"
            )
            
            # Save CV fold results
            self._save_intermediate_cv_results(held_out)
        
        # Calculate CV summary statistics
        if self.results['cv_results']['performance']:
            scores = list(self.results['cv_results']['performance'].values())
            self.results['cv_results']['summary'] = {
                'mean_score': np.mean(scores),
                'median_score': np.median(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'std_score': np.std(scores)
            }
            
            self._log_progress(
                "CV_SUMMARY", 
                f"Mean CV performance: {self.results['cv_results']['summary']['mean_score']:.4f} "
                f"(±{self.results['cv_results']['summary']['std_score']:.4f})"
            )

    def _run_final_training(self, is_test: bool = True) -> Dict[str, Any]:
        """Run final training on all datasets."""
        test_dataset = self.exam_dataset
        
        final_selection: Dict[str, Any] = {}
        # load meta‐features and data of exam data
        meta_feats = self._extract_meta_features(is_test=is_test)
        texts, labels = load_dataset(test_dataset, split='train')

        # Force exploit mode for final selection
        bohb_final_cfg = self.make_bohb_profile(
            iteration=self.max_iterations + 1,
            max_iter=self.max_iterations,
            base_cfg=self.bohb_base,
            model_type= "TEST",
            is_test=is_test
        )
        
        if not self.cv_agent_paths:
            raise RuntimeError("No CV agents found; cannot do final selection")

        # Loop over each CV agent
        for held_out, agent_path in self.cv_agent_paths.items():
            # Instantiate a new RLModelSelector pointing at agent_path
            rl_selector = RLModelSelector(
                meta_features_dim=META_FEATURE_DIM,
                model_save_path=agent_path,
                logger=self.logger,
                random_state=42,
            )
            
            # Load the trained model
            rl_selector.load_model()
            
            # call select_model_with_bohb to get both tier and its best config
            model_type, action, info = rl_selector.select_model_with_bohb(
                meta_features=meta_feats,
                training_data=(texts, labels),
                fidelity_mode="high",
                deterministic=True,
                bohb_config=bohb_final_cfg
            )

            # Extract information from the returned info
            best_cfg = info.get('bohb_info', {}).get('best_config', {})
            best_score = info.get('bohb_info', {}).get('best_score', None)
            confidence = info.get("confidence")
            q_values = info.get("q_values")
            
            # For plotting purposes for the 4th visualization function
            inc_history = info.get("bohb_info", {}).get("incumbent_history")
            if inc_history:
                self.results["detailed_logs"].append({
                    "fold": held_out,
                    "iteration": 0,
                    "dataset": test_dataset,                 # "yelp"
                    "action": action,
                    "model_type": model_type,
                    "q_values": q_values or [],
                    "reward": best_score,
                    "reward_components": {},
                    "bohb": {
                        "best_score": best_score,
                        "best_config": best_cfg,
                        "incumbent_history": inc_history,
                        "n_trials": len(inc_history),
                        "runtime_s": info.get("bohb_info", {}).get("total_time"),
                    },
                })

            # log & store
            self._log_progress(
                "FINAL_SELECTION",
                f"{held_out} agent on {test_dataset}: {model_type} @ acc≈{best_score:.4f}",
                {"best_config": best_cfg}
            )
            
            final_selection[held_out] = {
                "model_type":   model_type,
                "best_config":  best_cfg,
                "bohb_score":   best_score,
                "confidence":   confidence,
                "q_values":     q_values,
            }

        # save into results
        self.results['final_selections'] = final_selection
        
        # Calculate final mean performance as the mean of all BOHB scores
        if final_selection:
            all_scores = [sel["bohb_score"] for sel in final_selection.values() if sel["bohb_score"] is not None]
            self.results["final_mean_performance"] = np.mean(all_scores) if all_scores else 0.0
        else:
            self.results["final_mean_performance"] = 0.0
            
        return final_selection

    def _predict_on_test_split(self, dataset: str = "yelp", data_path: str = "data") -> np.ndarray:
        """predict test split."""
        final_selections = self.results.get('final_selections', {})
        if not final_selections:
            raise RuntimeError(f"No final selections found. Did you run _run_final_training()?")

        # Choose the best performing agent for prediction
        best_agent = None
        best_score = -1
        for agent_name, model_info in final_selections.items():
            if model_info['bohb_score'] is not None and model_info['bohb_score'] > best_score:
                best_score = model_info['bohb_score']
                best_agent = agent_name

        if best_agent is None:
            raise RuntimeError(f"No valid agent found for prediction")

        # Extract best model info
        model_info = final_selections[best_agent]
        model_type = model_info['model_type']
        best_config = model_info['best_config']
        self._log_progress("PREDICT:", f"Predicting on {dataset} using {model_type} from agent {best_agent} with config: {best_config}")
        print(best_config)
        # Load train + test splits
        df_train = pd.read_csv(Path(data_path) / dataset /"train.csv")
        df_test = pd.read_csv(Path(data_path) / dataset /"test.csv")
        texts_train = df_train["text"].tolist()
        labels_train = df_train["label"].tolist()
        texts_test = df_test["text"].tolist()

        # Test using best config - use unified model approach
        try:
            # Create model using the factory function from models.py
            model = create_model(
                model_type=model_type,
                config=best_config,
                random_state=42
            )
            
            # Train on full training data
            model.fit(texts_train, labels_train)
            
            # Predict on test data
            preds = model.predict(texts_test)
            
        except Exception as e:
            self._log_progress("PREDICT_ERROR", f"Failed to predict with {model_type}: {e}")
            raise
        return np.asarray(preds, dtype=int)

    def _save_intermediate_results(self, stage: str):
        """Save intermediate results."""
        try:
            results_file = self.output_dir / f"intermediate_{stage}_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(self.results, f)
                
            # Also save a JSON summary for easy viewing
            summary = {
                'timestamp': time.time(),
                'stage': stage,
                'datasets_processed': len(self.results.get('datasets', {})),
                'iterations_completed': len(self.results.get('rl_training_iterations', [])),
                'performance_history': self.results.get('performance_history', []),
                'elapsed_minutes': (time.time() - self.start_time) / 60 if self.start_time else 0
            }
            
            with open(self.output_dir / f"intermediate_{stage}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            self._log_progress("SAVE_ERROR", f"Error saving intermediate results: {e}")
    
    def _save_intermediate_cv_results(self, held_out: str):
        """Save intermediate CV results."""
        self._save_intermediate_results(f"cv_{held_out}")

def main():
    """Run AutoML pipeline main function."""
    
    parser = argparse.ArgumentParser(description="AutoML Pipeline")
    parser.add_argument('--time', type=float, default=1.0, help="Maximum runtime in hours")
    parser.add_argument('--experiment', type=str, default="default_experiment", 
                        help="Experiment name")
    parser.add_argument('--max_iterations', type=int, default=10,
                        help="Maximum iterations for RL training")
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Always use experiments folder as the parent directory
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    # Create output directory inside experiments folder
    output_dir = experiments_dir / args.experiment / f"run_{timestamp}"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    pipeline = AutoMLPipeline(
        max_runtime_hours=args.time,
        output_dir=output_dir,
        max_iterations=args.max_iterations
    )

    pipeline.run_pipeline()
    results = pipeline.results

    print("\n======================================================================")
    print("LEAVE-ONE-OUT CV COMPLETED")
    print("======================================================================")
    print(f"Folds: {len(results['cv_results']['folds'])}")
    print(f"Mean CV accuracy: {results['cv_results']['summary'].get('mean_score', 0):.4f}")
    print("Per-dataset CV scores:")
    for fold in results['cv_results']['folds']:
       held = fold['held_out']
       model = fold['selected_model']
       score = fold['bohb_score']
       print(f"  Held-out {held}: selected model = {model}, score = {score:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("\n======================================================================")
    print("FINAL TRAIN AND PREDICT COMPLETED")
    print("======================================================================")
    
    # Display final model selection results
    final_selections = results.get('final_selections', {})
    if final_selections:
        for agent_name, selection in final_selections.items():
            print(f"📊 Final Model Selection from {agent_name.upper()} Agent:")
            print(f"   • Selected Model: {selection['model_type'].title()}")
            print(f"   • BOHB Score: {selection['bohb_score']:.4f}")
            print(f"   • Confidence: {selection.get('confidence', 0):.4f}")
            print(f"   • Best Config: {len(selection['best_config'])} parameters")
            
            # Show key config parameters
            config = selection['best_config']
            if 'algorithm' in config:
                print(f"     - Algorithm: {config['algorithm']}")
            if 'max_features' in config:
                print(f"     - Max Features: {config['max_features']}")
            if 'C' in config:
                print(f"     - C Parameter: {config['C']:.4f}")
            if 'learning_rate' in config:
                print(f"     - Learning Rate: {config['learning_rate']:.4f}")
    
    # Display prediction summary
    print(f"\n🎯 Prediction Results:")
    print(f"   • Test dataset: {pipeline.exam_dataset}")
    print(f"   • Predictions saved to: data/exam_dataset/predictions.npy")
    print(f"   • Total runtime: {(time.time() - pipeline.start_time)/60:.1f} minutes")
    
    # Display training summary
    total_iterations = len([log for log in results['detailed_logs'] if log['fold'] == 'final'])
    total_bohb_evals = len([eval for eval in results['bohb_evaluations'] if eval['cv_fold'] == 'final'])
    print(f"\n📈 Training Summary:")
    print(f"   • RL Training Iterations: {len(results.get('rl_training_iterations', []))}")
    print(f"   • Total BOHB Evaluations: {total_bohb_evals}")
    print(f"   • Datasets Used: {len(pipeline.datasets)}")
    
    print("\n======================================================================")
    print("VISUALIZATION AND ANALYSIS")
    print("======================================================================")
    print("Add INFO here")

    return 0


if __name__ == "__main__":
    exit(main())
