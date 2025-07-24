"""RL-driven Model Selection for AutoML Text Classification.

This module implements a reinforcement learning agent that selects between different
model complexity tiers based on dataset meta-features. The RL agent learns to choose
the most appropriate model type (Simple/Medium/Complex) given dataset characteristics.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import pickle
import torch
import time

from stable_baselines3 import DQN

# Import constants for feature consistency across the system
from .constants import FEATURE_ORDER, META_FEATURE_DIM, MODEL_TYPES

logger = logging.getLogger(__name__)

# Import BOHB after logger to avoid circular imports
try:
    from .bohb_optimization import BOHBOptimizer, BOHBConfig
except ImportError:
    # Handle case where BOHB is not available
    BOHBOptimizer = None
    BOHBConfig = None


class ModelSelectionEnv(gym.Env):
    """Gymnasium environment for RL-based model selection.

    State: Normalized meta-features from dataset
    Action: Model complexity choice (0=Simple, 1=Medium, 2=Complex)
    Reward: Performance-based reward with complexity penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        meta_features_dim: int = 10,
        reward_type: str = "tier_based",
        complexity_penalty: float = 0.1,
        random_state: int = 42,
    ):
        """Initialize the model selection environment.

        Args:
            meta_features_dim: Dimension of meta-features state space
            reward_type: Type of reward function ('tier_based', 'performance_based')
            complexity_penalty: Penalty for choosing complex models
            random_state: Random seed for reproducibility
        """
        super().__init__()

        # Use META_FEATURE_DIM as default if available, otherwise use provided value
        self.meta_features_dim = meta_features_dim
        self.reward_type = reward_type
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.bohb_accuracy = None
        self.np_random = np.random.RandomState(random_state)
        self._pending_feats = None

        # Define action space based on number of model types
        self.action_space = spaces.Discrete(len(MODEL_TYPES))

        # Define observation space: normalized meta-features [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(meta_features_dim,), dtype=np.float32
        )

        # Model complexity definitions using MODEL_TYPES from constants
        self.action_names = {
            0: "Simple (TF-IDF + LR/SVM)",
            1: "Medium (CNN/LSTM)",
            2: "Complex (Transformer)",
        }
        
        # Create mapping from action index to model type
        self.action_to_model = {i: model_type for i, model_type in enumerate(MODEL_TYPES)}

        # Current state
        self.current_meta_features = None
        self.episode_step = 0
        self.max_episode_steps = 1  # Single decision per episode

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.episode_step = 0

        if options and "meta_features" in options:
            feats = np.array(options["meta_features"], dtype=np.float32)
        elif getattr(self, "_pending_feats", None) is not None:
            # you stored them before calling reset()
            feats = np.array(self._pending_feats, dtype=np.float32)
            self._pending_feats = None
        else:
            # SB3 internal reset -> give a harmless dummy vector
            feats = np.zeros(self.meta_features_dim, dtype=np.float32)
                # or feats = self._generate_synthetic_meta_features()

        # pad/truncate
        if len(feats) < self.meta_features_dim:
            feats = np.concatenate([feats, np.zeros(self.meta_features_dim - len(feats))])
        else:
            feats = feats[: self.meta_features_dim]

        self.current_meta_features = feats.astype(np.float32)
        return self.current_meta_features, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, and termination info."""
        self.episode_step += 1
        if self.bohb_accuracy is None:
            # bootstrap / dummy rollout for SB3 -> zero reward, still terminate
            reward = 0.0
        else:
            # Calculate reward based on action and current meta-features
            reward = self._calculate_enhanced_reward(action, self.current_meta_features, bohb_accuracy=self.bohb_accuracy)

        # Episode terminates after one decision
        terminated = True
        truncated = False

        info = {
            "action_name": self.action_names[action],
            "meta_features": self.current_meta_features.copy(),
            "reward_components": self._get_reward_components(
                action, self.current_meta_features
            ),
        }

        return (
            self.current_meta_features.astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def _calculate_enhanced_reward(
            self,
            action: int,
            meta_features: np.ndarray,
            bohb_accuracy: Optional[float] = None,
        ) -> float:
        """
        Calculate reward using ONLY the real BOHB accuracy plus your original
        penalty terms. Heuristic accuracy estimation and blending are removed.

        reward = w1 * normalized_accuracy
                - w2 * model_complexity_penalty
                + w3 * generalization_margin
                - w4 * resource_usage_penalty

        Requires: bohb_accuracy is not None.
        """
        if bohb_accuracy is None:
            raise ValueError("bohb_accuracy must be provided for reward calculation")

        # --- Extract meta info -------------------------------------------------
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5

        # --- Original weights --------------------------------------------------
        w1 = 0.5  # normalized_accuracy weight
        w2 = 0.2  # model_complexity_penalty weight
        w3 = 0.2  # generalization_margin weight
        w4 = 0.1  # resource_usage_penalty weight

        # 1. Normalized accuracy (real BOHB score only)
        normalized_accuracy = min(1.0, max(0.0, float(bohb_accuracy)))

        # 2. Model complexity penalty
        complexity_penalties = {
            0: 0.0,  # Simple
            1: 0.2,  # Medium
            2: 0.4,  # Complex
        }
        model_complexity_penalty = complexity_penalties[action]

        # 3. Generalization margin (your existing helper)
        generalization_margin = self._calculate_generalization_margin(action, meta_features)

        # 4. Resource usage penalty (your existing helper)
        resource_usage_penalty = self._calculate_resource_penalty(action, dataset_size)

        # Final reward (no blending)
        reward = (
            w1 * normalized_accuracy
            - w2 * model_complexity_penalty
            + w3 * generalization_margin
            - w4 * resource_usage_penalty
        )

        # Clamp as before
        reward = np.clip(reward, -1.0, 1.0)
        return float(reward)


    def _estimate_model_accuracy(self, action: int, meta_features: np.ndarray) -> float:
        """Estimate model accuracy based on type and dataset characteristics."""
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5
        vocab_size = meta_features[2] if len(meta_features) > 2 else 0.5

        # Base accuracy expectations for each model type
        base_accuracies = {
            0: 0.65,  # Simple models
            1: 0.75,  # Medium models
            2: 0.85,  # Complex models
        }

        base_acc = base_accuracies[action]

        # Adjust based on dataset characteristics
        # Simple models work better with small, simple datasets
        if action == 0:
            size_factor = 1.0 - dataset_size * 0.2  # Better with smaller datasets
            vocab_factor = 1.0 - vocab_size * 0.1
        # Medium models are versatile
        elif action == 1:
            size_factor = 0.9 + dataset_size * 0.2
            vocab_factor = 0.9 + vocab_size * 0.2
        # Complex models need large datasets
        else:
            size_factor = 0.7 + dataset_size * 0.4  # Much better with larger datasets
            vocab_factor = 0.8 + vocab_size * 0.3

        # Combine factors
        estimated_accuracy = base_acc * size_factor * vocab_factor

        # Ensure accuracy is reasonable and properly scaled
        return np.clip(estimated_accuracy, 0.3, 0.98)

    def _calculate_generalization_margin(
        self, action: int, meta_features: np.ndarray
    ) -> float:
        """Calculate how well the model type generalizes for this dataset."""
        baseline_acc = meta_features[0] if len(meta_features) > 0 else 0.5
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5
        vocab_size = meta_features[2] if len(meta_features) > 2 else 0.5

        # Calculate dataset complexity score
        complexity_score = 0.4 * baseline_acc + 0.4 * dataset_size + 0.2 * vocab_size

        # Define optimal ranges for each model type
        if action == 0:  # Simple models
            # Best for low-medium complexity (0.2-0.6)
            optimal_range = (0.2, 0.6)
        elif action == 1:  # Medium models
            # Best for medium complexity (0.4-0.8)
            optimal_range = (0.4, 0.8)
        else:  # Complex models
            # Best for high complexity (0.6-1.0)
            optimal_range = (0.6, 1.0)

        # Calculate distance from optimal range
        if complexity_score < optimal_range[0]:
            distance = optimal_range[0] - complexity_score
        elif complexity_score > optimal_range[1]:
            distance = complexity_score - optimal_range[1]
        else:
            distance = 0.0

        # Convert distance to margin (higher is better)
        generalization_margin = 1.0 - distance

        return np.clip(generalization_margin, 0.0, 1.0)

    def _calculate_resource_penalty(self, action: int, dataset_size: float) -> float:
        """Calculate resource usage penalty based on model complexity and data size."""
        # Resource usage multipliers for each model type
        resource_multipliers = {
            0: 1.0,  # Simple models: low resource usage
            1: 3.0,  # Medium models: moderate resource usage
            2: 10.0,  # Complex models: high resource usage
        }

        base_resource_cost = resource_multipliers[action]

        # Larger datasets increase resource usage
        dataset_factor = 1.0 + dataset_size * 2.0

        total_resource_cost = base_resource_cost * dataset_factor

        # Normalize to [0, 1] penalty (higher cost = higher penalty)
        # Max expected cost is 10.0 * 3.0 = 30.0 for complex model + large dataset
        resource_penalty = min(1.0, total_resource_cost / 30.0)

        return resource_penalty

    def evaluate_with_bohb(
        self,
        action: int,
        meta_features: np.ndarray,
        training_data: Tuple[List[str], List[int]],
        fidelity_mode: str = "low",
        bohb_config: Optional[Any] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate action using BOHB optimization.

        Args:
            action: Model choice (0=Simple, 1=Medium, 2=Complex)
            meta_features: Dataset meta-features
            training_data: (texts, labels) for training
            fidelity_mode: "low" or "high" fidelity optimization
            bohb_config: Optional custom BOHB configuration

        Returns:
            Tuple of (reward, optimization_info)
        """
        if BOHBOptimizer is None:
            logger.warning("BOHB optimizer is not available")
            return 0.0, {"method": "failed", "bohb_available": False}

        # Map action to model type
        model_type = self.action_to_model[action]

        try:
            # Initialize BOHB optimizer with custom config if provided
            bohb_optimizer = BOHBOptimizer(
                model_type=model_type,
                logger=None,  # Avoid circular logging
                config=bohb_config,  # Use custom config if provided
                random_state=self.random_state,
            )

            # Run BOHB optimization
            X_train, y_train = training_data
            best_config, best_score, opt_stats = bohb_optimizer.optimize(
                X_train=X_train, y_train=y_train, fidelity_mode=fidelity_mode
            )

            # Calculate enhanced reward with BOHB accuracy
            # Ensure best_score is converted to float
            reward = self._calculate_enhanced_reward(
                action, meta_features, bohb_accuracy=float(best_score)
            )

            optimization_info = {
                "method": "bohb",
                "best_config": best_config,
                "best_score": best_score,
                "optimization_stats": opt_stats,
                "fidelity_mode": fidelity_mode,
            }

            return reward, optimization_info

        except Exception as e:
            # Just log the error instead of falling back to heuristic
            logger.warning(f"BOHB evaluation failed: {e}")
            return 0.0, {"method": "failed", "error": str(e)}

    def _get_reward_components(
        self, action: int, meta_features: np.ndarray
    ) -> Dict[str, float]:
        """Get detailed breakdown of reward components for logging."""
        baseline_acc = meta_features[0] if len(meta_features) > 0 else 0.5
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5

        complexity_score = 0.6 * baseline_acc + 0.4 * dataset_size

        return {
            "complexity_score": complexity_score,
            "baseline_accuracy": baseline_acc,
            "dataset_size": dataset_size,
            "action": action,
            "action_name": self.action_names[action],
        }

    def render(self, mode="human"):
        """Render the environment state."""
        if self.current_meta_features is not None:
            print(f"Meta-features: {self.current_meta_features[:5]}...")
            print(f"Episode step: {self.episode_step}")


class RLModelSelector:
    """RL-based model selector for AutoML pipeline."""

    def __init__(
        self,
        meta_features_dim: int = 10,
        model_save_path: Optional[Path] = None,
        logger=None,
        random_state: int = 42,
    ):
        """Initialize RL model selector.

        Args:
            meta_features_dim: Dimension of meta-features
            model_save_path: Path to save/load trained models
            logger: AutoML logger instance
            random_state: Random seed
        """
        self.meta_features_dim = meta_features_dim
        self.model_save_path = (
            Path(model_save_path) if model_save_path else Path("models/rl_agent")
        )
        self.logger = logger
        self.random_state = random_state

        # Create environment
        self.env = ModelSelectionEnv(
            meta_features_dim=meta_features_dim, random_state=random_state
        )

        # RL agent
        self.agent = None
        self.is_trained = False

        # Action mapping - use environment's mapping
        self.action_to_model = self.env.action_to_model

    def train(
        self,
        total_timesteps: int = 10000,
        learning_rate: float = 1e-3,
        exploration_fraction: float = 0.3,
        target_update_interval: int = 1000,
    ) -> None:
        """Train the RL agent for model selection.

        Args:
            total_timesteps: Number of training timesteps
            learning_rate: Learning rate for DQN
            exploration_fraction: Fraction of timesteps for exploration
            target_update_interval: Update interval for target network
        """
        if self.logger:
            self.logger.log_debug(
                "Starting RL agent training",
                {
                    "total_timesteps": total_timesteps,
                    "learning_rate": learning_rate,
                    "exploration_fraction": exploration_fraction,
                },
            )

        start_time = time.time()

        # Create DQN agent
        self.agent = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.95,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            max_grad_norm=10,
            tensorboard_log=(
                str(self.model_save_path.parent / "tensorboard")
                if self.logger
                else None
            ),
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0,
            seed=self.random_state,
        )

        # Setup callback - simplified for compatibility
        callback = None  # Skip callback for now to avoid compatibility issues

        # Train the agent
        self.agent.learn(
            total_timesteps=total_timesteps, callback=callback, progress_bar=False
        )

        training_time = time.time() - start_time
        self.is_trained = True

        if self.logger:
            self.logger.log_debug(
                "RL training completed",
                {
                    "training_time": f"{training_time:.2f}s",
                    "timesteps": total_timesteps,
                },
            )

    def select_model(
        self, meta_features: Dict[str, float], deterministic: bool = True
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Select model based on meta-features.

        Args:
            meta_features: Normalized meta-features from dataset
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (model_type, action, decision_info)
        """
        if not self.is_trained and self.agent is None:
            # Just log that model is not trained, use a default action
            logger.warning("RL agent not trained, using default action (medium)")
            return "medium", 1, {"method": "default", "reason": "agent_not_trained"}

        # Prepare observation
        obs = self._prepare_observation(meta_features)

        # Get action from trained agent
        action, _states = self.agent.predict(obs, deterministic=deterministic)
        action = int(action)

        model_type = self.action_to_model[action]

        # Get Q-values for confidence estimation
        obs_tensor = torch.from_numpy(obs.reshape(1, -1).astype(np.float32))
        q_values = self.agent.q_net(obs_tensor).detach().numpy()[0]
        confidence = float(np.max(q_values) - np.mean(q_values))

        decision_info = {
            "action": action,
            "model_type": model_type,
            "confidence": confidence,
            "q_values": q_values.tolist(),
            "exploration": not deterministic,
        }

        return model_type, action, decision_info

    def _prepare_observation(self, meta_features: Dict[str, float]) -> np.ndarray:
        """Convert meta-features dict to observation array using consistent feature order."""
        # Use the global feature order from constants.py
        obs = []
        for feature in FEATURE_ORDER:
            if feature in meta_features:
                obs.append(meta_features[feature])
            else:
                obs.append(0.0)  # Default value for missing features

        # Pad or truncate to match expected dimension
        if len(obs) < self.meta_features_dim:
            obs.extend([0.0] * (self.meta_features_dim - len(obs)))
        else:
            obs = obs[: self.meta_features_dim]

        return np.array(obs, dtype=np.float32)

    # Removed heuristic selection method

    def select_model_with_bohb(
        self,
        meta_features: Dict[str, float],
        training_data: Optional[Tuple[List[str], List[int]]] = None,
        fidelity_mode: str = "low",
        deterministic: bool = True,
        bohb_config: Optional[Any] = None,
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Select model with BOHB-enhanced reward evaluation.

        Args:
            meta_features: Normalized meta-features from dataset
            training_data: (texts, labels) for BOHB evaluation
            fidelity_mode: "low" or "high" fidelity BOHB optimization
            deterministic: Whether to use deterministic policy
            bohb_config: Optional custom BOHB configuration

        Returns:
            Tuple of (model_type, action, decision_info)
        """
        if training_data is None:
            logger.warning("No training data provided for BOHB evaluation")
            return self.select_model(meta_features, deterministic)
            
        if not self.is_trained:
            logger.warning("RL agent not trained for BOHB evaluation")
            return "medium", 1, {"method": "default", "reason": "agent_not_trained"}

        # Prepare observation
        obs = self._prepare_observation(meta_features)

        if self.agent is None:
            logger.warning("RL agent is None")
            return "medium", 1, {"method": "default", "reason": "agent_is_none"}

        # Get Q-values for all actions
        obs_tensor = torch.from_numpy(obs.reshape(1, -1).astype(np.float32))
        q_values = self.agent.q_net(obs_tensor).detach().numpy()[0]

        # Evaluate top actions with BOHB
        top_actions = np.argsort(q_values)[::-1]  # Sort in descending order

        best_action = None
        best_reward = -np.inf
        best_info = {}

        # Evaluate top 2-3 actions with BOHB (limited for efficiency)
        max_evaluations = min(2, len(top_actions))

        for i in range(max_evaluations):
            action = top_actions[i]

            # Use environment's BOHB evaluation
            reward, optimization_info = self.env.evaluate_with_bohb(
                action=action,
                meta_features=obs,
                training_data=training_data,
                fidelity_mode=fidelity_mode,
                bohb_config=bohb_config,
            )

            if self.logger:
                self.logger.log_debug(
                    f"BOHB evaluation: action={action}, reward={reward:.4f}, "
                    f"method={optimization_info.get('method', 'unknown')}"
                )

            if reward > best_reward:
                best_reward = reward
                best_action = action
                best_info = optimization_info

        # Select best action
        if best_action is None:
            best_action = top_actions[0]  # Fall back to highest Q-value

        model_type = self.action_to_model[best_action]

        decision_info = {
            "action": best_action,
            "model_type": model_type,
            "confidence": float(best_reward),
            "q_values": q_values.tolist(),
            "bohb_info": best_info,
            "fidelity_mode": fidelity_mode,
            "method": "rl_with_bohb",
        }

        return model_type, best_action, decision_info

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save trained RL model."""
        save_path = path or self.model_save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.agent is not None:
            self.agent.save(str(save_path))

            # Also save metadata
            metadata = {
                "meta_features_dim": self.meta_features_dim,
                "is_trained": self.is_trained,
                "action_to_model": self.action_to_model,
                "random_state": self.random_state,
            }

            with open(save_path.with_suffix(".metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)

        return save_path

    def load_model(self, path: Optional[Path] = None) -> bool:
        """Load trained RL model."""
        load_path = path or self.model_save_path

        # DQN saves as .zip files, so check for that
        if not load_path.exists():
            zip_path = load_path.with_suffix(".zip")
            if zip_path.exists():
                load_path = zip_path
            else:
                return False

        try:
            self.agent = DQN.load(str(load_path), env=self.env)

            # Load metadata if available
            metadata_path = load_path.with_suffix(".metadata.pkl")
            if load_path.suffix == ".zip":
                # If we loaded from .zip, metadata is at base_path.metadata.pkl
                metadata_path = load_path.with_suffix("").with_suffix(".metadata.pkl")

            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    self.is_trained = metadata.get("is_trained", True)

            return True
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to load RL model: {e}")
            return False
