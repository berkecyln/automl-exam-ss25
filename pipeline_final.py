"""
AutoML Pipeline with Leave-One-Out Cross-Validation
--------------------------------------------------

This module implements an enhanced AutoML pipeline with leave-one-out cross-validation
to evaluate transfer learning capabilities. The pipeline:

1. Extracts meta-features from all datasets
2. Performs leave-one-out cross-validation where:
   - One dataset is held out
   - RL+BOHB training occurs on remaining datasets
   - The trained RL agent selects a model for the held-out dataset
   - Performance is evaluated on the held-out dataset
3. After CV, trains a final RL+BOHB model on all datasets
4. Generates comprehensive evaluation metrics and visualizations

The architecture follows the critical separation between RL model selection and
BOHB hyperparameter optimization.
"""

import time
from automl.models import LSTMClassifier, SimpleFFNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import torch

from automl.meta_features import MetaFeatureExtractor
from automl.rl_agent import RLModelSelector
from automl.datasets import load_dataset
from automl.logging_utils import AutoMLLogger
from automl.bohb_optimization import BOHBOptimizer, BOHBConfig
from automl.constants import META_FEATURE_DIM, FEATURE_ORDER


class AutoMLPipeline:
    """Enhanced AutoML Pipeline with Leave-One-Out Cross-Validation."""
    
    def __init__(
        self, 
        max_runtime_hours: float = 1.0, 
        output_dir: str = "automl_pipeline_results",
    ):
        """Initialize AutoML pipeline.
        
        Args:
            max_runtime_hours: Maximum runtime in hours
            output_dir: Directory for results
            datasets: List of datasets to use (default: ['amazon', 'ag_news', 'dbpedia', 'imdb'])
        """
        self.max_runtime_hours = max_runtime_hours
        self.max_runtime_seconds = max_runtime_hours * 3600
        self.start_time = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            'cv_results': {}, # New for cross-validation results
            'detailed_logs': [],
        }
        
        # Available datasets
        self.datasets = ['amazon', 'ag_news', 'dbpedia', 'imdb']
        
    
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
        """Run the complete AutoML pipeline with leave-one-out cross-validation."""
        self.start_time = time.time()
        
        try:
            self._log_progress("STARTUP", "Initializing AutoML Pipeline with Leave-One-Out CV")
            
            # Step 1: Extract meta-features from all datasets
            self._log_progress("STEP_1", "Extracting meta-features")
            self._extract_meta_features()
            
            # Step 2: Run leave-one-out cross-validation
            self._log_progress("STEP_2", "Starting leave-one-out cross-validation")
            self.run_leave_one_out_cv()
            
            # Step 3: Run final training on all datasets
            self._log_progress("STEP_3", "Starting final training on all datasets")
            self._run_final_training()
            
            # Step 4: Predict Test Labels and save results
            predicted_labels = self._predict_on_test_split()
            output_path = Path("data/exam_dataset")
            output_path.mkdir(parents=True, exist_ok=True)

            np.save(output_path / "predictions.npy", predicted_labels)
            print("✅ Saved predictions to data/exam_dataset/predictions.npy")

            self._log_progress("COMPLETION", "Full pipeline completed successfully")
            

            final_results = {
                "cv_folds": len(self.results['cv_results']['folds']),
                "cv_mean_performance": self.results['cv_results']['summary'].get('mean_score', 0),
                "cv_by_dataset": self.results['cv_results']['performance'],
                "final_mean_performance": self.results["final_mean_performance"],
                "cv_vs_final_agreement": self.results["cv_vs_final_agreement"],
                "cv_best_configs": {
                    f['held_out']: f['best_config']
                    for f in self.results['cv_results']['folds']
                }
            }
            print("\n===== LOO-CV RESULTS =====")
            print(f"Folds: {final_results['cv_folds']}")
            print(f"Mean accuracy: {final_results['cv_mean_performance']:.4f}")
            for ds, sc in final_results["cv_by_dataset"].items():
                print(f"  {ds}: {sc:.4f}")
            print(f"Final Performance: {final_results['final_mean_performance']}")
            print(f"Final Agreement: {final_results['cv_vs_final_agreement']:.4f}")

            return final_results
            # Uncomment after exam dataset obtained
            #return final_results, exam_result

        except Exception as e:
            self._log_progress("CRITICAL_ERROR", f"Pipeline failed: {e}")
            import traceback
            self.logger.log_error(f"Pipeline error: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _extract_meta_features(self):
        """Extract meta-features from all datasets."""
        extractor = MetaFeatureExtractor()
        
        for dataset_name in self.datasets:
            try:
                self._log_progress("META_FEATURES", f"Processing {dataset_name}")
                
                # Load full dataset - important for accurate meta-features
                texts, labels = load_dataset(dataset_name, split='train')
                
                # Convert to DataFrame format expected by MetaFeatureExtractor
                import pandas as pd
                train_df = pd.DataFrame({
                    'text': texts,
                    'label': labels
                })
                
                # Extract meta-features using the existing extractor
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
    
    def _extract_meta_features_from_test(self, dataset_name: str):
        """Extract meta-features from test dataset."""
        extractor = MetaFeatureExtractor()
        
        try:
            self._log_progress("META_FEATURES", f"Processing {dataset_name}")
                
            # Load full dataset - important for accurate meta-features
            texts, labels = load_dataset(dataset_name, split='train')
                
            # Convert to DataFrame format expected by MetaFeatureExtractor
            import pandas as pd
            train_df = pd.DataFrame({
                'text': texts,
                'label': labels
            })
                
            # Extract meta-features using the existing extractor
            start_time = time.time()
            raw_meta_features = extractor.extract_features(train_df)
                
            # Check if extraction was successful
            if raw_meta_features is None:
                self._log_progress("ERROR", f"Failed to extract features for {dataset_name}: raw_meta_features is None") 

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
                
            self._log_progress(
                "META_FEATURES", 
                f"Extracted {len(meta_features)} features from {dataset_name}",
                {
                    'dataset': dataset_name,
                    'num_features': len(meta_features),
                    'extraction_time': extraction_time
                }
            )

            return meta_features
                
        except Exception as e:
            self._log_progress("ERROR", f"Failed to extract features from {dataset_name}: {e}")
                        

    def _prepare_training_datasets(self, exclude_dataset: Optional[str] = None) -> List[Tuple]:
        """Prepare training datasets for BOHB-enhanced RL training.
        
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
    
    def _run_iterative_rl_bohb_training(
            self,
            training_datasets: List[Tuple],
            prefix: str = ""
        ) -> RLModelSelector:
        """
        RL + BOHB training that learns only from real-data experience.
        """
        log_prefix = f"{prefix}_" if prefix else ""

        # 1. Initialise (no long synthetic pre‑train)
        rl_selector = RLModelSelector(
            meta_features_dim=META_FEATURE_DIM,
            model_save_path=self.output_dir / "models" / f"{log_prefix}rl_agent",
            logger=self.logger,
            random_state=42,
        )
        rl_selector.train(                # just enough steps to set up SB3 internals
            total_timesteps=1,            # <- minimal “bootstrap” call
            learning_rate=5e-4,
            exploration_fraction=0.0,     # we’ll act deterministically
            target_update_interval=500,
        )

        # Shared BOHB config
        bohb_cfg = BOHBConfig(
            max_budget=100.0,
            min_budget=10.0,
            n_trials=25,
            wall_clock_limit=720.0,
        )

        for iteration in range(1, 11):
            if self._check_time_remaining() < 600:
                self._log_progress(f"{log_prefix}TIME_LIMIT",
                                "Stopping - <10 min left")
                break

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
                    # bootstrap: pick a fixed/simple or random action
                    action = 0  # or: np.random.randint(rl_selector.env.action_space.n)
                else:
                    action, _ = rl_selector.agent.predict(obs, deterministic=True)
                action = int(action)
                model_type = rl_selector.env.action_to_model[int(action)]
                self._log_progress(f"{log_prefix}CHOICE",
                                f"{dname}: picked {model_type}")

                # ---- 3. BOHB hyper‑parameter search ---------------------------
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


                # bookkeeping ----------------------------------------------------
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


            # ---- iteration summary --------------------------------------------
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
            # simple convergence check
            if iteration > 1 and \
            abs(self.results['rl_training_iterations'][-1]['avg_performance']
                - self.results['rl_training_iterations'][-2]['avg_performance']) < 0.005:
                break

        self._log_progress(f"{log_prefix}TRAINING", "RL-BOHB finished")
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
            
            # Run BOHB on held-out dataset with selected model and our enhanced config
            # Create a generous config for final evaluation to avoid timeout issues
            cv_bohb_config = BOHBConfig(
                max_budget=100.0,
                min_budget=10.0,
                n_trials=30,
                wall_clock_limit=900.0  # 15 minutes per final evaluation - extremely generous
            )
            
            bohb_optimizer = BOHBOptimizer(
                model_type=model_type,
                random_state=42,
                config=cv_bohb_config
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

    def _run_final_training(self, dataset: str = "yelp") -> Dict[str, Any]:
        """Run final training on all datasets."""
        # Prepare all training datasets for final training
        training_datasets = self._prepare_training_datasets()
        
        # Run RL+BOHB training on all datasets
        self.final_rl_selector = self._run_iterative_rl_bohb_training(
            training_datasets, 
            prefix="final"
        )

        final_selection: Dict[str, Any] = {}
        # load meta‐features and data of exam data
        meta_feats = self._extract_meta_features_from_test(dataset)
        texts, labels = load_dataset(dataset, split='train')

        # call select_model_with_bohb to get both tier and its best config
        model_type, action, info = self.final_rl_selector.select_model_with_bohb(
            meta_features=meta_feats,
            training_data=(texts, labels),
            fidelity_mode="high",
            deterministic=True,
            bohb_config=BOHBConfig(
                max_budget=100.0,
                min_budget=10.0,
                n_trials=30,
                wall_clock_limit=900.0
            )
        )

        best_cfg = info.get('bohb_info', {}).get('best_config', {})
        best_score = info.get('bohb_info', {}).get('best_score', None)

        # log & store
        self._log_progress(
            "FINAL_SELECTION",
            f"{dataset}: {model_type} @ acc≈{best_score:.4f}",
            {"best_config": best_cfg}
        )
        final_selection[dataset] = {
            "model_type":   model_type,
            "best_config":  best_cfg,
            "bohb_score":   best_score,
            "confidence":   info.get("confidence"),
            "q_values":     info.get("q_values"),
        }

        # save into results
        self.results['final_selections'] = final_selection
        return final_selection

    def _predict_on_test_split(self, dataset: str = "yelp", data_path: str = "data") -> np.ndarray:
        """predict test split."""
        if dataset not in self.results.get('final_selections', {}):
            raise RuntimeError(f"No final selection found for dataset '{dataset}'. Did you run _run_final_training()?")

        # Extract best model info
        model_info = self.results['final_selections'][dataset]
        model_type = model_info['model_type']
        best_config = model_info['best_config']
        self._log_progress(f"Predicting on {dataset} using {model_type} with config: {best_config}")
        print(best_config)
        # Load train + test splits
        df_train = pd.read_csv(Path(data_path) / dataset /"train.csv")
        df_test = pd.read_csv(Path(data_path) / dataset /"test.csv")
        texts_train = df_train["text"].tolist()
        labels_train = df_train["label"].tolist()
        texts_test = df_test["text"].tolist()

        # Test using best config
        # THIS PART SHOULD BE CHANGED BECAUSE I WAS NOT SURE ABOUT THE MODEL STRUCTURES
        if model_type == "simple":
            preds = self._predict_simple(best_config, texts_train, labels_train, texts_test)

        elif model_type == "medium":
            preds = self._predict_medium(best_config, texts_train, labels_train, texts_test)

        elif model_type == "complex":
            preds = self._predict_complex(best_config, texts_train, labels_train, texts_test)

        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return np.asarray(preds, dtype=int)

    def _predict_simple(self, cfg, Xtr, ytr, Xte):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        vec = TfidfVectorizer(
            max_features=int(cfg["max_features"]),
            min_df=float(cfg["min_df"]),
            max_df=float(cfg["max_df"]),
            ngram_range=(1, int(cfg["ngram_max"])),
            stop_words="english",
        )
        Xtr_vec = vec.fit_transform(Xtr)
        Xte_vec = vec.transform(Xte)

        if cfg["algorithm"] == "logistic":
            clf = LogisticRegression(
                C=float(cfg["C"]), max_iter=int(cfg["max_iter"]), random_state=42
            )
        else:  # "svm"
            clf = SVC(C=float(cfg["C"]), kernel="linear", random_state=42)

        clf.fit(Xtr_vec, ytr)
        return clf.predict(Xte_vec)


# -----------------------------------------------------------------------------
#  Medium tier  (TF‑IDF ➜ TruncatedSVD ➜ LogisticRegression)
# -----------------------------------------------------------------------------
    def _predict_medium(self, cfg, Xtr, ytr, Xte):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        from sklearn.linear_model import LogisticRegression

        vec = TfidfVectorizer(
            max_features=int(cfg["max_features"]),
            stop_words="english",
            ngram_range=(1, 2),
        )
        Xtr_tf = vec.fit_transform(Xtr)
        Xte_tf = vec.transform(Xte)

        svd = TruncatedSVD(
            n_components=int(cfg["svd_components"]), random_state=42
        )
        lsa = make_pipeline(svd, Normalizer(copy=False))

        Xtr_lsa = lsa.fit_transform(Xtr_tf)
        Xte_lsa = lsa.transform(Xte_tf)

        clf = LogisticRegression(
            C=float(cfg["C"]), max_iter=int(cfg["max_iter"]), random_state=42
        )
        clf.fit(Xtr_lsa, ytr)
        return clf.predict(Xte_lsa)


# -----------------------------------------------------------------------------
#  Complex tier  (TF‑IDF ➜ 1‑hidden‑layer MLP)
# -----------------------------------------------------------------------------
    def _predict_complex(self, cfg, Xtr, ytr, Xte):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier

        vec = TfidfVectorizer(
            max_features=int(cfg["max_features"]),
            ngram_range=(1, 3),
            stop_words=None,          # include stopwords for richer vocab
        )
        Xtr_vec = vec.fit_transform(Xtr)
        Xte_vec = vec.transform(Xte)

        mlp = MLPClassifier(
            hidden_layer_sizes=(int(cfg["hidden_units"]),),
            alpha=float(cfg["alpha"]),
            max_iter=int(cfg["max_iter"]),
            early_stopping=True,
            random_state=42,
        )
        mlp.fit(Xtr_vec, ytr)
        return mlp.predict(Xte_vec)


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
    """Run AutoML pipeline with leave-one-out cross-validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML Pipeline with Leave-One-Out CV")
    parser.add_argument('--time', type=float, default=1.0, help="Maximum runtime in hours")
    parser.add_argument('--output', type=str, default="automl_pipeline_results", 
                        help="Output directory")
    
    args = parser.parse_args()
    
    pipeline = AutoMLPipeline(
        max_runtime_hours=args.time,
        output_dir=args.output,
    )
    
    results, exam_results = pipeline.run_pipeline()
    
    print("\n======================================================================")
    print("AUTOML PIPELINE WITH LEAVE-ONE-OUT CV COMPLETED")
    print("======================================================================")
    print(f"🔄 Folds: {results['cv_folds']}")
    print(f"📈 Mean CV accuracy: {results['cv_mean_performance']:.4f}")
    print("📊 Per-dataset CV scores:")
    for ds, sc in results["cv_by_dataset"].items():
        print(f"  {ds}: {sc:.4f}")
    print(f"\n📂 Results saved to: {args.output}")

    # Uncomment to print detailed exam dataset selection results after exam dataset obtained
    '''
    # Exam dataset selection results
    print("\n===== FINAL EXAM DATASET SELECTION =====")
    for ds, sel in exam_results.items():
        print(f"Dataset: {ds}")
        print(f"  • Model tier    : {sel['model_type']}")
        print(f"  • BOHB accuracy : {sel['bohb_score']:.4f}")
        print(f"  • Confidence    : {sel['confidence']:.4f}")
        print(f"  • Q‑values      : {sel['q_values']}")
        print(f"  • Best config   :")
        for k, v in sel['best_config'].items():
            print(f"      - {k}: {v}")
    '''
    return 0


if __name__ == "__main__":
    exit(main())
