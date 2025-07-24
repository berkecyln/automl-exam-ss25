"""BOHB Hyperparameter Optimization using SMAC3.

This module implements multi-fidelity Bayesian optimization using SMAC3's BOHB
implementation for hyperparameter tuning. It integrates with the RL agent to
provide feedback for model selection decisions.
"""

from __future__ import annotations

_TOKENIZER_CACHE: dict[str, Any] = {}
_MODEL_CACHE: dict[str, Any] = {}

import os
import copy
from pathlib import Path

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import time
from dataclasses import dataclass

# SMAC3 imports
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from smac.scenario import Scenario
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.runhistory import RunHistory

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from setfit import SetFitModel

# Local imports
from .logging_utils import AutoMLLogger

CACHE_ROOT = Path.home() / ".automl_cache"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("FLAIR_CACHE_ROOT", str(CACHE_ROOT / "flair"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

def get_tokenizer(name: str):
    if name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER_CACHE[name]

def get_hf_model(name: str, num_labels: int):
    key = f"{name}_{num_labels}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    # return a fresh copy so each trial starts untrained
    return copy.deepcopy(_MODEL_CACHE[key])

def get_setfit_model(name: str):
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SetFitModel.from_pretrained(name)
    return copy.deepcopy(_MODEL_CACHE[name])

@dataclass
class BOHBConfig:
    """Configuration for BOHB optimization."""

    max_budget: float = 100.0  # Maximum fidelity budget
    min_budget: float = 10.0  # Minimum fidelity budget
    n_trials: int = 30  # Number of optimization trials (reduced from 50)
    n_workers: int = 1  # Number of parallel workers
    seed: int = 42  # Random seed
    deterministic: bool = True  # Deterministic mode
    wall_clock_limit: float = (
        600.0  # Wall clock time limit in seconds (10 minutes default)
    )
    num_initial_design: int = 10  # Number of initial design configurations
    max_ratio: float = 1.0  # Maximum ratio for initial design (1.0 = no reduction)
    
    def __post_init__(self):
        """Validate configuration values."""
        # Ensure all parameters are positive
        if self.max_budget <= 0:
            self.max_budget = 100.0
        if self.min_budget <= 0:
            self.min_budget = 10.0
        
        # Ensure max_budget is greater than min_budget
        if self.max_budget <= self.min_budget:
            self.max_budget = self.min_budget * 3.0  # Make max budget 3x min budget
            
        if self.n_trials <= 0:
            self.n_trials = 30
        if self.wall_clock_limit <= 0:
            self.wall_clock_limit = 300.0


class TextDataset(Dataset):
    """Simple text dataset for PyTorch models."""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        if self.tokenizer:
            # For transformer models (if implemented)
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long),
            }
        else:
            # For simple models, return text and label
            return {"text": text, "label": label}


class BOHBOptimizer:
    """BOHB optimizer using SMAC3 for hyperparameter optimization."""

    def __init__(
        self,
        model_type: str,
        logger: Optional[AutoMLLogger] = None,
        config: Optional[BOHBConfig] = None,
        random_state: int = 42,
    ):
        """Initialize BOHB optimizer.

        Args:
            model_type: Type of model ('simple', 'medium', 'complex')
            logger: AutoML logger instance
            config: BOHB configuration
            random_state: Random seed
        """
        self.model_type = model_type.lower()
        self.logger = logger
        self.config = config or BOHBConfig()
        self.random_state = random_state

        # Training data placeholders
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        # Optimization history
        self.optimization_history = []
        self.best_config = None
        self.best_score = 0.0

        # Performance tracking
        self.evaluation_count = 0
        self.total_time = 0.0

    def _create_config_space(self) -> ConfigurationSpace:
        """Create configuration space based on model type."""
        cs = ConfigurationSpace()

        if self.model_type == "simple":
            # TF-IDF + Classical ML configuration space
            cs.add_hyperparameter(CategoricalHyperparameter("algorithm", ["logistic", "svm"]))
            # TF-IDF parameters
            cs.add_hyperparameter(UniformIntegerHyperparameter("max_features", 1000, 50000, default_value=10000))
            cs.add_hyperparameter(UniformFloatHyperparameter("min_df", 0.001, 0.1, default_value=0.01))
            cs.add_hyperparameter(UniformFloatHyperparameter("max_df", 0.7, 0.95, default_value=0.85))
            cs.add_hyperparameter(UniformIntegerHyperparameter("ngram_max", 1, 3, default_value=2))
            # Algorithm-specific parameters
            cs.add_hyperparameter(UniformFloatHyperparameter("C", 0.001, 100, log=True, default_value=1.0))
            cs.add_hyperparameter(UniformIntegerHyperparameter("max_iter", 100, 2000, default_value=1000))

        elif self.model_type == "medium":
            # Medium Model: SetFit
            # Embedding parameters
            cs.add_hyperparameter(CategoricalHyperparameter("algorithm", ["setFit"]))
            cs.add_hyperparameter(UniformIntegerHyperparameter("num_iterations", 1, 20, default_value=10))
            cs.add_hyperparameter(UniformIntegerHyperparameter("num_epochs", 1, 5, default_value=3))
            cs.add_hyperparameter(UniformIntegerHyperparameter("batch_size", 8, 64, default_value=16))
            cs.add_hyperparameter(UniformFloatHyperparameter("lr", 1e-6, 1e-3, log=True, default_value=2e-5))

        elif self.model_type == "complex":
            # Transformer configuration space
            cs.add_hyperparameter(CategoricalHyperparameter("algorithm", ["distilBert"]))
            cs.add_hyperparameter(UniformFloatHyperparameter("learning_rate", 1e-6, 5e-4, log=True, default_value=2e-5))
            cs.add_hyperparameter(UniformIntegerHyperparameter("batch_size", 8, 32, default_value=16))
            cs.add_hyperparameter(UniformFloatHyperparameter("warmup_ratio", 0.0, 0.2, default_value=0.1))
            cs.add_hyperparameter(UniformFloatHyperparameter("weight_decay", 0.0, 0.3, default_value=0.01))
            cs.add_hyperparameter(UniformIntegerHyperparameter("max_length", 128, 512, default_value=256))

        return cs

    def _evaluate_config(self, config: Configuration, seed: int = 0) -> float:
        """Evaluate a configuration (simplified without fidelity for SMAC compatibility).

        Args:
            config: Configuration to evaluate
            seed: Random seed

        Returns:
            Negative accuracy (SMAC minimizes)
        """
        start_time = time.time()
        self.evaluation_count += 1

        # Simple timeout check - if we're past our timeout, stop the optimization
        if hasattr(self, 'optimization_timeout') and time.time() > self.optimization_timeout:
            if self.evaluation_count % 3 == 0:  # Only log occasionally to avoid spam
                print(f"Stopping BOHB evaluation #{self.evaluation_count} due to timeout")
            return float("inf")  # Return bad score to stop optimization

        # Log progress every few evaluations to track progress without spam
        if self.evaluation_count % 3 == 1:  # Log every 3rd evaluation
            print(
                f"   BOHB evaluation #{self.evaluation_count}/{getattr(self.config, 'n_trials', '?')} in progress..."
            )

        try:
            # Use default budget since fidelity is causing warnings
            raw_budget = getattr(self, "current_budget", None)
            
            if raw_budget is None:
                # No budget provided -> treat as full fidelity
                n_trials = getattr(self.config, "n_trials", 20)
                budget_fraction = min(1.0, self.evaluation_count / float(n_trials))
            else:
                max_b = float(self.config.max_budget)
                min_b = float(self.config.min_budget)
                denom = max(1e-9, (max_b - min_b))
                budget_fraction = (float(raw_budget) - min_b) / denom
                budget_fraction = max(0.0, min(1.0, budget_fraction))  # clamp


            # Sample data based on budget
            if budget_fraction < 1.0:
                n_samples = max(100, int(len(self.X_train) * budget_fraction))
                X_train_sample = self.X_train[:n_samples]
                y_train_sample = self.y_train[:n_samples]
            else:
                X_train_sample = self.X_train
                y_train_sample = self.y_train

            # Train and evaluate model
            if self.model_type == "simple":
                score = self._evaluate_simple_model(config, X_train_sample, y_train_sample)
            elif self.model_type == "medium":
                score = self._evaluate_medium_model(config, X_train_sample, y_train_sample)
            else:  # complex
                epochs = max(1, int(5 * budget_fraction))
                score = self._evaluate_complex_model(config, X_train_sample, y_train_sample, epochs)

            # Track evaluation time
            eval_time = time.time() - start_time
            self.total_time += eval_time

            # Log evaluation with proper hyperparameters
            if self.logger:
                try:
                    # Convert SMAC Configuration to dictionary
                    hyperparams_dict = (
                        dict(config)
                        if hasattr(config, "__iter__")
                        else config.get_dictionary()
                    )
                    fid = f"budget_{raw_budget:.0f}" if raw_budget is not None else "budget_full"

                    self.logger.log_bohb_iteration(
                        iteration=self.evaluation_count,
                        model_type=self.model_type,
                        hyperparams=hyperparams_dict,
                        performance=score,
                        fidelity=fid,
                    )
                except Exception as logging_error:
                    # Don't let logging failures break optimization
                    self.logger.log_warning(f"BOHB logging failed: {logging_error}")
                
                bud_str = f"{raw_budget:.1f}" if raw_budget is not None else "full"
                self.logger.log_debug(
                    f"BOHB eval #{self.evaluation_count}: "
                    f"budget={bud_str}, score={score:.4f}, time={eval_time:.2f}s"
                )

            # Store in history
            b = raw_budget if raw_budget is not None else self.config.max_budget
            self.optimization_history.append(
                {
                    "config": dict(config),
                    "score": score,
                    "budget": b,
                    "time": eval_time,
                    "evaluation": self.evaluation_count,
                }
            )

            # Update best score
            if score > self.best_score:
                self.best_score = score
                self.best_config = dict(config)

            # Return negative score (SMAC minimizes)
            return -score

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"BOHB evaluation failed: {e}")
            print(f"BOHB evaluation error: {e}")
            return float('inf')  # Indicate failure without fallback

    def _evaluate_simple_model(
        self, config: Configuration, X_train: List[str], y_train: List[int]
    ) -> float:
        """Evaluate simple model configuration."""
        # Extract parameters
        algorithm = config["algorithm"]
        max_features = config["max_features"]
        min_df = config["min_df"]
        max_df = config["max_df"]
        ngram_max = config["ngram_max"]
        C = config["C"]
        max_iter = config["max_iter"]

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, ngram_max),
            stop_words="english",
        )

        # Transform data
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(self.X_val)

        # Train model
        if algorithm == "logistic":
            model = LogisticRegression(
                C=C, max_iter=max_iter, random_state=self.random_state
            )
        elif algorithm == "svm":
            model = SVC(C=C, kernel="linear", random_state=self.random_state)

        model.fit(X_train_vec, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val_vec)
        return accuracy_score(self.y_val, y_pred)

    def _evaluate_medium_model(
        self,
        config: Configuration,
        X_train: List[str],
        y_train: List[int],
    ) -> float:
        """Evaluate medium complexity model (SetFit)"""
        # Used model for medium complexity is SetFit
        # SetFit Architecture: SentenceTransformer + linear head
        try:
            from setfit import SetFitModel, SetFitTrainer
            from sentence_transformers.losses import CosineSimilarityLoss
            from datasets import Dataset as HFDataset
            import numpy as np

            algorithm = config["algorithm"] # Keep for future competibility
            num_iter = int(config["num_iterations"])
            num_epochs = int(config["num_epochs"])
            batch_size = int(config["batch_size"])
            lr = float(config.get("lr", 2e-5))

            # Build HF datasets
            train_ds = HFDataset.from_dict({"text": X_train, "label": y_train})
            val_ds   = HFDataset.from_dict({"text": self.X_val, "label": self.y_val})

            # Load model (SetFit)
            # NOTE: If multiple models are used, this part should be in if-else block 
            model = get_setfit_model("sentence-transformers/all-MiniLM-L6-v2")

            # Trainer - updated to use CosineSimilarityLoss from sentence_transformers
            trainer = SetFitTrainer(
                model=model,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                loss_class=CosineSimilarityLoss,  # Using from sentence_transformers now
                num_iterations=num_iter,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=lr,
            )

            trainer.train()
            metrics = trainer.evaluate() 
            return float(metrics.get("accuracy", 0.0))

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Medium model training failed: {e}.")
            print(f"Medium model evaluation error: {e}", flush=True)
            raise


    def _evaluate_complex_model(
        self,
        config: Configuration,
        X_train: List[str],
        y_train: List[int],
        epochs: int = 3,
    ) -> float:
        """Evaluate complex model DistilBERT"""
        # UUsed model for complex complexity is DistilBERT
        # DistilBERT Architecture: "Transformer"
        try:
            import numpy as np
            from datasets import Dataset as HFDataset
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                DataCollatorWithPadding,
            )
            from sklearn.metrics import accuracy_score

            # Hyperparams
            algorithm   = config["algorithm"] # Keep for future competibility
            lr           = float(config["learning_rate"])
            batch_size   = int(config["batch_size"])
            warmup_ratio = float(config["warmup_ratio"])
            weight_decay = float(config["weight_decay"])
            max_length   = int(config["max_length"])

            # epochs passed from BOHB (budget→epochs); cap a bit
            num_epochs = max(1, min(epochs, 3))

            # datasets
            train_ds = HFDataset.from_dict({"text": X_train, "label": y_train})
            val_ds   = HFDataset.from_dict({"text": self.X_val, "label": self.y_val})

            # Setup distilBert
            # NOTE: If multiple models are used, this part should be in if-else block
            # tokenizer & tokenization
            tokenizer = get_tokenizer("distilbert-base-uncased")

            def _tok(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )

            train_ds = train_ds.map(_tok, batched=True)
            val_ds   = val_ds.map(_tok, batched=True)

            # set format for torch
            cols = ["input_ids", "attention_mask", "label"]
            train_ds.set_format(type="torch", columns=cols)
            val_ds.set_format(type="torch", columns=cols)

            # model
            num_labels = len(set(y_train))
            model = get_hf_model("distilbert-base-uncased", num_labels)

            # steps for warmup
            total_steps   = (len(train_ds) // batch_size) * num_epochs
            warmup_steps  = int(total_steps * warmup_ratio)

            # tmp output dir
            import tempfile, os, shutil
            out_dir = tempfile.mkdtemp()

            args = TrainingArguments(
                output_dir=out_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=lr,
                weight_decay=weight_decay,
                num_train_epochs=num_epochs,
                warmup_steps=warmup_steps,
                logging_steps=max(1, total_steps),   
                save_strategy="no",          
                report_to="none",
                disable_tqdm=True,
            )

            data_collator = DataCollatorWithPadding(tokenizer)

            # define metric fn manually
            def compute_metrics(pred):
                preds = np.argmax(pred.predictions, axis=1)
                return {"accuracy": accuracy_score(pred.label_ids, preds)}

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            eval_out = trainer.evaluate()
            acc = float(eval_out.get("eval_accuracy", 0.0))

            # cleanup
            shutil.rmtree(out_dir, ignore_errors=True)
            return acc

           
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Complex model training failed: {e}.")
            print(f"Complex model evaluation error: {e}")
            raise

    def optimize(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: Optional[List[str]] = None,
        y_val: Optional[List[int]] = None,
        fidelity_mode: str = "low",
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Run BOHB optimization.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            fidelity_mode: "low" or "high" fidelity optimization

        Returns:
            Tuple of (best_config, best_score, optimization_stats)
        """
        self.evaluation_count = 0
        self.total_time = 0.0
        self.best_score = 0.0
        start_time = time.time()

        # Store training data
        self.X_train = X_train
        self.y_train = y_train

        # Create validation split if not provided
        if X_val is None or y_val is None:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train,
            )
        else:
            self.X_val = X_val
            self.y_val = y_val

        # Adjust configuration based on fidelity mode and time constraints
        if fidelity_mode == "high":
            max_budget = self.config.max_budget
        else:
            # For low fidelity, reduce the max budget but ensure it's greater than min_budget
            max_budget = max(self.config.min_budget * 1.5, self.config.max_budget * 0.3)
        n_trials = (
            self.config.n_trials
            if fidelity_mode == "high"
            else min(self.config.n_trials // 2, 10)
        )  # Limit trials
        
        # Set more generous wall clock limits based on model type and fidelity mode
        base_wall_clock_limit = self.config.wall_clock_limit
        model_type_multiplier = 1.0
        if self.model_type == "medium":
            model_type_multiplier = 2.0  # Double time for medium models
        elif self.model_type == "complex":
            model_type_multiplier = 3.0  # Triple time for complex models
        
        # Get actual time limit in seconds - much more generous
        actual_time_limit = (
            base_wall_clock_limit * model_type_multiplier
            if fidelity_mode == "high"
            else base_wall_clock_limit * 0.5  # Half time for low fidelity, but still generous
        )
        
        # Set SMAC time limit to be slightly less than our actual limit
        smac_time_limit = actual_time_limit * 0.9  # 90% of our actual time limit

        # Store the timeout for our manual enforcement
        self.optimization_timeout = start_time + actual_time_limit
        
        # Create configuration space
        config_space = self._create_config_space()

        # Update the Scenario creation:
        min_budget = self.config.min_budget
        effective_max_budget = max_budget
        
        # Ensure effective max budget is strictly greater than min budget
        # For low fidelity mode, don't let it go below min_budget
        if effective_max_budget <= min_budget:
            print(f"Warning: Adjusting max budget ({effective_max_budget}) to be greater than min budget ({min_budget})")
            effective_max_budget = min_budget * 3.0  # Make max budget 3x min budget
            
        scenario = Scenario(
            configspace    = config_space,
            deterministic  = self.config.deterministic,
            n_trials       = n_trials,
            seed           = self.config.seed,
            walltime_limit = smac_time_limit,
            min_budget = float(min_budget),
            max_budget = float(effective_max_budget)
        )

        # Create Hyperband intensifier
        intensifier = Hyperband(
            scenario         = scenario,
            eta              = 3,
            incumbent_selection = "highest_budget",   # optional but good
        )
        
        # TODO: Check if we really need budget as input
        def target_fn(config: Configuration, seed: int | None = None, budget: float | None = None):                
            # Store the budget so _evaluate_config can read it
            self.current_budget = budget
            try:
                return self._evaluate_config(config, seed)
            finally:
                self.current_budget = None

        # BOHB‑enabled facade
        smac = BlackBoxFacade(
            scenario         = scenario,
            target_function  = target_fn,
            intensifier      = intensifier,  # Use our Hyperband intensifier
            overwrite        = True,
        )

        if self.logger:
            self.logger.logger.info(
                f"🔧 Starting BOHB optimization: {self.model_type} model, "
                f"fidelity={fidelity_mode}, trials={n_trials}, time_limit={actual_time_limit}s"
            )

        # Add progress tracking variables
        self.start_optimization_time = start_time

        # Run optimization with time monitoring
        try:
            # Track the start time to calculate actual elapsed time
            optimization_start_time = time.time()
            
            
            try:
                # Run the optimization
                incumbent = smac.optimize()
            except Exception as e:
                print(f"BOHB optimization exception: {e}")
                incumbent = smac.solver.incumbent if hasattr(smac, "solver") else None

                
            actual_time_used = time.time() - optimization_start_time
            print(f"BOHB optimization completed in {actual_time_used:.2f}s (time limit was {actual_time_limit:.1f}s)")
            if self.logger:
                self.logger.logger.info(f"BOHB optimization completed in {actual_time_used:.2f}s (time limit was {actual_time_limit:.1f}s)")
                    
        except Exception as e:
            print(f"BOHB optimization error: {e}")
            if self.logger:
                self.logger.logger.warning(f"BOHB optimization error: {e}")
            # Get incumbent if available
            incumbent = (
                smac.intensifier.get_incumbent()
                if hasattr(smac, "intensifier") and smac.intensifier.get_incumbent() is not None
                else None
            )

        # Calculate optimization statistics
        optimization_time = time.time() - start_time

        # Get best configuration and score
        if incumbent is not None:
            best_config = dict(incumbent)
            neg_cost = smac.runhistory.get_cost(incumbent)
            best_score = -neg_cost                          # back to accuracy
        else:
            best_config = dict(config_space.get_default_configuration())
            best_score = max((h["score"] for h in self.optimization_history), default=0.0)

        # Update both best_score and best_config to ensure consistency
        self.best_score = best_score
        self.best_config = best_config  # Make sure this is set from SMAC's incumbent

        optimization_stats = {
            "total_time": optimization_time,
            "evaluation_time": self.total_time,
            "num_evaluations": self.evaluation_count,
            "fidelity_mode": fidelity_mode,
            "model_type": self.model_type,
            "convergence_history": self.optimization_history[
                -10:
            ],  # Last 10 evaluations
            "improvement_over_default": best_score
            - 0.5,  # Assuming 0.5 random baseline
        }

        if self.logger:
            self.logger.logger.info(
                f"✅ BOHB optimization complete: best_score={best_score:.4f}, "
                f"time={optimization_time:.1f}s, evaluations={self.evaluation_count}"
            )

        return best_config, best_score, optimization_stats

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        if not self.optimization_history:
            return {}

        scores = [eval_info["score"] for eval_info in self.optimization_history]
        budgets = [eval_info["budget"] for eval_info in self.optimization_history]
        times = [eval_info["time"] for eval_info in self.optimization_history]

        return {
            "best_score": self.best_score,
            "best_config": self.best_config,  # This is set properly during optimize()
            "total_evaluations": len(self.optimization_history),
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
            },
            "budget_statistics": {
                "mean": np.mean(budgets),
                "min": np.min(budgets),
                "max": np.max(budgets),
            },
            "time_statistics": {
                "total": self.total_time,
                "mean_per_eval": np.mean(times),
                "std_per_eval": np.std(times),
            },
            "convergence_trend": scores[-10:] if len(scores) >= 10 else scores,
        }
