"""BOHB Hyperparameter Optimization using SMAC3.

This module implements multi-fidelity Bayesian optimization using SMAC3's BOHB
implementation for hyperparameter tuning. It integrates with the RL agent to
provide feedback for model selection decisions.
"""

from __future__ import annotations


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
)

from smac.scenario import Scenario
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.intensifier.hyperband import Hyperband

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Local imports
from .logging_utils import AutoMLLogger
from .models import create_model


@dataclass
class BOHBConfig:
    """Configuration for BOHB optimization.
    
    Handles budget parameters for Bayesian Optimization with HyperBand.
    
    Parameters:
        min_budget: Minimum fidelity (e.g., dataset percentage)
        max_budget: Maximum fidelity (e.g., full dataset)
        n_trials: Total configurations to evaluate (must be > max_budget)
        wall_clock_limit: Maximum runtime in seconds
    """
    # Budget parameters
    max_budget: float = 100.0
    min_budget: float = 10.0
    n_trials: int = 150
    
    # Time constraint
    wall_clock_limit: float = 600.0  # 10 minutes default
    
    # SMAC parameters
    n_workers: int = 1
    seed: int = 42
    deterministic: bool = True
    num_initial_design: int = 10
    max_ratio: float = 1.0

    def __post_init__(self):
        """Validate configuration values."""
        # Ensure parameters are positive and consistent
        if self.max_budget <= 0:
            self.max_budget = 100.0
        if self.min_budget <= 0:
            self.min_budget = 10.0
        if self.max_budget <= self.min_budget:
            self.max_budget = self.min_budget * 3.0
        if self.n_trials <= self.max_budget:
            self.n_trials = int(self.max_budget * 1.5)
        if self.wall_clock_limit <= 0:
            self.wall_clock_limit = 600.0
            


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
        
        # Create a config with adjusted settings for this model type
        if config is None:
            self.config = BOHBConfig()
        else:
            self.config = config
            
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

    def _calculate_budget(self, fidelity_mode: str) -> Dict[str, float]:
        """Calculate budget parameters based on fidelity mode.
        
        Args:
            fidelity_mode: "low" or "high" fidelity optimization
            
        Returns:
            Dictionary with budget parameters
        """
        # Start with base parameters
        min_budget = self.config.min_budget
        
        # Adjust max_budget based on fidelity mode
        if fidelity_mode == "high":
            max_budget = self.config.max_budget
        else:
            # For low fidelity, reduce the max budget
            max_budget = max(min_budget * 1.5, self.config.max_budget * 0.3)
            
        # Adjust n_trials based on fidelity mode
        n_trials = (
            self.config.n_trials if fidelity_mode == "high" 
            else min(self.config.n_trials // 2, 10)
        )
        
        # Adjust time limit based on fidelity mode
        time_limit = (
            self.config.wall_clock_limit if fidelity_mode == "high"
            else self.config.wall_clock_limit * 0.5
        )
        
        # Ensure parameters meet SMAC requirements
        if max_budget <= min_budget:
            max_budget = min_budget * 3.0
            
        if n_trials <= max_budget:
            n_trials = int(max_budget * 1.5)
            
        return {
            "min_budget": min_budget,
            "max_budget": max_budget,
            "n_trials": n_trials,
            "time_limit": time_limit
        }

    def _create_config_space(self) -> ConfigurationSpace:
        """Create configuration space based on model type."""
        cs = ConfigurationSpace()

        if self.model_type == "simple":
            cs.add(CategoricalHyperparameter("algorithm", ["logistic", "svm"]))
            # TF-IDF parameters
            cs.add(UniformIntegerHyperparameter("max_features", 1000, 50000, default_value=10000))
            cs.add(UniformFloatHyperparameter("min_df", 0.001, 0.1, default_value=0.01))
            cs.add(UniformFloatHyperparameter("max_df", 0.7, 0.95, default_value=0.85))
            cs.add(UniformIntegerHyperparameter("ngram_max", 1, 3, default_value=2))

            cs.add(UniformFloatHyperparameter("C", 0.001, 100, log=True, default_value=1.0))
            cs.add(UniformIntegerHyperparameter("max_iter", 100, 2000, default_value=1000))

        elif self.model_type == "medium":
            cs.add(CategoricalHyperparameter("algorithm", ["lsa_lr"]))
            cs.add(UniformIntegerHyperparameter("max_features", 2000, 80000, default_value=20000))
            cs.add(UniformIntegerHyperparameter("svd_components", 100, 1000, default_value=300))
            cs.add(UniformFloatHyperparameter("C", 1e-3, 1e2, log=True, default_value=1.0))
            cs.add(UniformIntegerHyperparameter("max_iter", 200, 2000, default_value=1000))

        elif self.model_type == "complex":
            cs.add(CategoricalHyperparameter("algorithm", ["tfidf_mlp"]))
            cs.add(UniformIntegerHyperparameter("max_features", 5000, 100000, default_value=30000))
            cs.add(UniformIntegerHyperparameter("hidden_units", 50, 500, default_value=200))
            cs.add(UniformFloatHyperparameter("alpha", 1e-6, 1e-2, log=True, default_value=1e-4))
            cs.add(UniformIntegerHyperparameter("max_iter", 100, 500, default_value=200))

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
        if (
            hasattr(self, "optimization_timeout")
            and time.time() > self.optimization_timeout
        ):
            if self.evaluation_count % 3 == 0:  # Only log occasionally to avoid spam
                print(
                    f"Stopping BOHB evaluation #{self.evaluation_count} due to timeout"
                )
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

            # Train and evaluate model using unified model classes
            if self.model_type == "simple":
                score = self._evaluate_model(config, X_train_sample, y_train_sample)
            elif self.model_type == "medium":
                score = self._evaluate_model(config, X_train_sample, y_train_sample)
            else:  # complex
                score = self._evaluate_model(config, X_train_sample, y_train_sample, budget_fraction)

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
                    fid = (
                        f"budget_{raw_budget:.0f}"
                        if raw_budget is not None
                        else "budget_full"
                    )

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
            return float("inf")  # Indicate failure without fallback

    def _evaluate_model(
        self, 
        config: Configuration, 
        X_train: List[str], 
        y_train: List[int],
        budget_fraction: float = 1.0
    ) -> float:
        """Unified model evaluation using the model classes from models.py"""
        try:
            # Convert configuration to dictionary
            config_dict = dict(config)
            
            # Create model using the factory function
            model = create_model(
                model_type=self.model_type,
                config=config_dict,
                random_state=self.random_state,
                budget_fraction=budget_fraction
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            score = model.evaluate(self.X_val, self.y_val)
            
            return score
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"{self.model_type.title()} model evaluation failed: {e}")
            print(f"{self.model_type.title()} model evaluation error: {e}")
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

        # Calculate budget parameters based on fidelity mode
        budget_config = self._calculate_budget(fidelity_mode)
        
        # Extract the budget values
        min_budget = budget_config["min_budget"]
        max_budget = budget_config["max_budget"]
        n_trials = budget_config["n_trials"]
        actual_time_limit = budget_config["time_limit"]
        
        # Set SMAC time limit to be slightly less than our actual limit
        smac_time_limit = actual_time_limit * 0.9  # 90% of our actual time limit

        # Store the timeout for our manual enforcement
        self.optimization_timeout = start_time + actual_time_limit

        # Create configuration space
        config_space = self._create_config_space()

        # Create the SMAC scenario with our calculated budget parameters
        scenario = Scenario(
            configspace=config_space,
            deterministic=self.config.deterministic,
            n_trials=int(n_trials),  # Ensure n_trials is an integer
            seed=self.config.seed,
            walltime_limit=smac_time_limit,
            min_budget=float(min_budget),
            max_budget=float(max_budget),
        )

        # Create Hyperband intensifier
        intensifier = Hyperband(
            scenario=scenario,
            eta=3,
            incumbent_selection="highest_budget",  # optional but good
        )

        # TODO: Check if we really need budget as input
        def target_fn(
            config: Configuration, seed: int | None = None, budget: float | None = None
        ):
            # Store the budget so _evaluate_config can read it
            self.current_budget = budget
            try:
                return self._evaluate_config(config, seed)
            finally:
                self.current_budget = None

        # BOHB‑enabled facade
        smac = BlackBoxFacade(
            scenario=scenario,
            target_function=target_fn,
            intensifier=intensifier,  # Use our Hyperband intensifier
            overwrite=True,
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
            print(
                f"BOHB optimization completed in {actual_time_used:.2f}s (time limit was {actual_time_limit:.1f}s)"
            )
            if self.logger:
                self.logger.logger.info(
                    f"BOHB optimization completed in {actual_time_used:.2f}s (time limit was {actual_time_limit:.1f}s)"
                )

        except Exception as e:
            print(f"BOHB optimization error: {e}")
            if self.logger:
                self.logger.logger.warning(f"BOHB optimization error: {e}")
            # Get incumbent if available
            incumbent = (
                smac.intensifier.get_incumbent()
                if hasattr(smac, "intensifier")
                and smac.intensifier.get_incumbent() is not None
                else None
            )

        # Calculate optimization statistics
        optimization_time = time.time() - start_time

        # Get best configuration and score
        if incumbent is not None:
            best_config = dict(incumbent)
            neg_cost = smac.runhistory.get_cost(incumbent)
            best_score = -neg_cost  # back to accuracy
        else:
            best_config = dict(config_space.get_default_configuration())
            best_score = max(
                (h["score"] for h in self.optimization_history), default=0.0
            )

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
