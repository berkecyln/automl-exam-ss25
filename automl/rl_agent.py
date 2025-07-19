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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

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
        random_state: int = 42
    ):
        """Initialize the model selection environment.
        
        Args:
            meta_features_dim: Dimension of meta-features state space
            reward_type: Type of reward function ('tier_based', 'performance_based')
            complexity_penalty: Penalty for choosing complex models
            random_state: Random seed for reproducibility
        """
        super().__init__()
        
        self.meta_features_dim = meta_features_dim
        self.reward_type = reward_type
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.np_random = np.random.RandomState(random_state)
        
        # Define action space: 3 model complexity tiers
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: normalized meta-features [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(meta_features_dim,), 
            dtype=np.float32
        )
        
        # Model complexity definitions
        self.action_names = {
            0: "Simple (TF-IDF + LR/SVM)",
            1: "Medium (CNN/LSTM)",
            2: "Complex (Transformer)"
        }
        
        # Current state
        self.current_meta_features = None
        self.episode_step = 0
        self.max_episode_steps = 1  # Single decision per episode
        
        # Reward function parameters
        self.reward_params = {
            'baseline_accuracy_weight': 0.6,
            'dataset_size_weight': 0.2,
            'vocab_size_weight': 0.1,
            'complexity_weight': 0.1
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        self.episode_step = 0
        
        # Generate or use provided meta-features
        if options and 'meta_features' in options:
            self.current_meta_features = np.array(
                options['meta_features'], dtype=np.float32
            )
        else:
            # Generate synthetic meta-features for training
            self.current_meta_features = self._generate_synthetic_meta_features()
        
        # Ensure meta-features are the right dimension
        if len(self.current_meta_features) != self.meta_features_dim:
            # Pad or truncate to match expected dimension
            if len(self.current_meta_features) < self.meta_features_dim:
                padding = np.zeros(self.meta_features_dim - len(self.current_meta_features))
                self.current_meta_features = np.concatenate([self.current_meta_features, padding])
            else:
                self.current_meta_features = self.current_meta_features[:self.meta_features_dim]
        
        return self.current_meta_features.astype(np.float32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, and termination info."""
        self.episode_step += 1
        
        # Calculate reward based on action and current meta-features
        reward = self._calculate_enhanced_reward(action, self.current_meta_features)
        
        # Episode terminates after one decision
        terminated = True
        truncated = False
        
        info = {
            'action_name': self.action_names[action],
            'meta_features': self.current_meta_features.copy(),
            'reward_components': self._get_reward_components(action, self.current_meta_features)
        }
        
        return self.current_meta_features.astype(np.float32), reward, terminated, truncated, info
    
    def _generate_synthetic_meta_features(self) -> np.ndarray:
        """Generate synthetic meta-features for training.
        
        Creates realistic meta-feature combinations representing different
        dataset types to help the RL agent learn general patterns.
        """
        # Define different dataset archetypes
        archetypes = [
            # Small, simple dataset
            {
                'baseline_accuracy': self.np_random.uniform(0.6, 0.8),
                'dataset_size': self.np_random.uniform(0.1, 0.3),
                'vocab_size': self.np_random.uniform(0.1, 0.4),
                'avg_char_length': self.np_random.uniform(0.01, 0.05),
                'num_classes': self.np_random.uniform(0.1, 0.3),
            },
            # Medium dataset
            {
                'baseline_accuracy': self.np_random.uniform(0.7, 0.9),
                'dataset_size': self.np_random.uniform(0.3, 0.7),
                'vocab_size': self.np_random.uniform(0.4, 0.7),
                'avg_char_length': self.np_random.uniform(0.03, 0.08),
                'num_classes': self.np_random.uniform(0.2, 0.6),
            },
            # Large, complex dataset
            {
                'baseline_accuracy': self.np_random.uniform(0.8, 0.95),
                'dataset_size': self.np_random.uniform(0.7, 1.0),
                'vocab_size': self.np_random.uniform(0.7, 1.0),
                'avg_char_length': self.np_random.uniform(0.05, 0.15),
                'num_classes': self.np_random.uniform(0.3, 1.0),
            }
        ]
        
        # Choose random archetype
        archetype = self.np_random.choice(archetypes)
        
        # Convert to array (first 5 most important features)
        features = [
            archetype['baseline_accuracy'],
            archetype['dataset_size'],
            archetype['vocab_size'],
            archetype['avg_char_length'],
            archetype['num_classes'],
        ]
        
        # Add some noise and additional synthetic features
        while len(features) < self.meta_features_dim:
            features.append(self.np_random.uniform(0.0, 1.0))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, action: int, meta_features: np.ndarray) -> float:
        """Calculate reward for the given action and meta-features.
        
        Reward is based on:
        1. Expected performance gain from model choice
        2. Complexity penalty for overly complex models
        3. Dataset characteristics alignment
        """
        if self.reward_type == "tier_based":
            return self._tier_based_reward(action, meta_features)
        else:
            return self._performance_based_reward(action, meta_features)
    
    def _tier_based_reward(self, action: int, meta_features: np.ndarray) -> float:
        """Calculate reward based on model tier suitability."""
        # Extract key features (first 5 are most important)
        baseline_acc = meta_features[0] if len(meta_features) > 0 else 0.5
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5
        vocab_size = meta_features[2] if len(meta_features) > 2 else 0.5
        avg_char_length = meta_features[3] if len(meta_features) > 3 else 0.5
        num_classes = meta_features[4] if len(meta_features) > 4 else 0.5
        
        # Calculate complexity score (0-1, higher = more complex dataset)
        complexity_score = (
            0.3 * baseline_acc +
            0.3 * dataset_size + 
            0.2 * vocab_size +
            0.1 * avg_char_length +
            0.1 * num_classes
        )
        
        # Optimal action based on complexity
        if complexity_score < 0.4:
            optimal_action = 0  # Simple
            expected_performance = 0.7 + 0.1 * complexity_score
        elif complexity_score < 0.7:
            optimal_action = 1  # Medium
            expected_performance = 0.75 + 0.15 * complexity_score
        else:
            optimal_action = 2  # Complex
            expected_performance = 0.8 + 0.2 * complexity_score
        
        # Base reward from expected performance
        base_reward = expected_performance
        
        # Bonus for choosing optimal action
        if action == optimal_action:
            base_reward += 0.1
        
        # Penalty for suboptimal choices
        action_penalty = abs(action - optimal_action) * 0.05
        
        # Complexity penalty (encourage simpler models when possible)
        complexity_penalties = [0.0, 0.02, 0.05]  # Simple, Medium, Complex
        complexity_penalty = complexity_penalties[action] * self.complexity_penalty
        
        # Calculate final reward
        reward = base_reward - action_penalty - complexity_penalty
        
        # Ensure reward is in reasonable range
        return np.clip(reward, 0.0, 1.0)
    
    def _performance_based_reward(self, action: int, meta_features: np.ndarray) -> float:
        """Calculate reward based on expected performance improvement."""
        # This will be used when integrating with BOHB
        # For now, use simplified version
        return self._tier_based_reward(action, meta_features)
    
    def _calculate_enhanced_reward(
        self, 
        action: int, 
        meta_features: np.ndarray,
        bohb_accuracy: Optional[float] = None,
        training_data: Optional[Tuple] = None
    ) -> float:
        """Calculate enhanced reward using the specified formula:
        reward = w1 * normalized_accuracy - w2 * model_complexity_penalty + 
                 w3 * generalization_margin - w4 * resource_usage_penalty
        
        Args:
            action: Model choice (0=Simple, 1=Medium, 2=Complex)
            meta_features: Dataset meta-features
            bohb_accuracy: Accuracy from BOHB optimization (if available)
            training_data: Training data for BOHB evaluation (if available)
        """
        # Extract key meta-features
        baseline_acc = meta_features[0] if len(meta_features) > 0 else 0.5
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5
        vocab_size = meta_features[2] if len(meta_features) > 2 else 0.5
        avg_char_length = meta_features[3] if len(meta_features) > 3 else 0.5
        
        # Reward component weights (sum to 1.0 for proper scaling)
        w1 = 0.5   # normalized_accuracy weight
        w2 = 0.2   # model_complexity_penalty weight  
        w3 = 0.2   # generalization_margin weight
        w4 = 0.1   # resource_usage_penalty weight
        
        # 1. Normalized Accuracy Component
        if bohb_accuracy is not None:
            # Use actual BOHB accuracy, normalized to [0, 1]
            normalized_accuracy = min(1.0, max(0.0, bohb_accuracy))
        else:
            # Estimate accuracy based on model type and meta-features
            normalized_accuracy = self._estimate_model_accuracy(action, meta_features)
        
        # 2. Model Complexity Penalty
        complexity_penalties = {
            0: 0.0,   # Simple models have no complexity penalty
            1: 0.3,   # Medium models have moderate penalty
            2: 0.6    # Complex models have higher penalty
        }
        model_complexity_penalty = complexity_penalties[action]
        
        # 3. Generalization Margin (how well model type fits dataset)
        generalization_margin = self._calculate_generalization_margin(action, meta_features)
        
        # 4. Resource Usage Penalty
        resource_usage_penalty = self._calculate_resource_penalty(action, dataset_size)
        
        # Calculate final reward with careful scaling
        reward = (
            w1 * normalized_accuracy - 
            w2 * model_complexity_penalty + 
            w3 * generalization_margin - 
            w4 * resource_usage_penalty
        )
        
        # Ensure reward is in reasonable range [0, 1]
        reward = np.clip(reward, 0.0, 1.0)
        
        return reward
    
    def _estimate_model_accuracy(self, action: int, meta_features: np.ndarray) -> float:
        """Estimate model accuracy based on type and dataset characteristics."""
        baseline_acc = meta_features[0] if len(meta_features) > 0 else 0.5
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5
        vocab_size = meta_features[2] if len(meta_features) > 2 else 0.5
        
        # Base accuracy expectations for each model type
        base_accuracies = {
            0: 0.65,  # Simple models
            1: 0.75,  # Medium models
            2: 0.85   # Complex models
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
    
    def _calculate_generalization_margin(self, action: int, meta_features: np.ndarray) -> float:
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
            0: 1.0,   # Simple models: low resource usage
            1: 3.0,   # Medium models: moderate resource usage
            2: 10.0   # Complex models: high resource usage
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
        bohb_config: Optional[Any] = None
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
            # Fall back to heuristic if BOHB not available
            reward = self._calculate_enhanced_reward(action, meta_features)
            return reward, {'method': 'heuristic', 'bohb_available': False}
        
        # Map action to model type
        model_types = {0: "simple", 1: "medium", 2: "complex"}
        model_type = model_types[action]
        
        try:
            # Initialize BOHB optimizer with custom config if provided
            bohb_optimizer = BOHBOptimizer(
                model_type=model_type,
                logger=None,  # Avoid circular logging
                config=bohb_config,  # Use custom config if provided
                random_state=self.random_state
            )
            
            # Run BOHB optimization
            X_train, y_train = training_data
            best_config, best_score, opt_stats = bohb_optimizer.optimize(
                X_train=X_train,
                y_train=y_train,
                fidelity_mode=fidelity_mode
            )
            
            # Calculate enhanced reward with BOHB accuracy
            reward = self._calculate_enhanced_reward(
                action, meta_features, bohb_accuracy=best_score
            )
            
            optimization_info = {
                'method': 'bohb',
                'bohb_available': True,
                'best_config': best_config,
                'best_score': best_score,
                'optimization_stats': opt_stats,
                'fidelity_mode': fidelity_mode
            }
            
            return reward, optimization_info
            
        except Exception as e:
            # Fall back to heuristic if BOHB fails
            logger.warning(f"BOHB evaluation failed: {e}")
            reward = self._calculate_enhanced_reward(action, meta_features)
            return reward, {'method': 'heuristic_fallback', 'error': str(e)}
    
    def _get_reward_components(self, action: int, meta_features: np.ndarray) -> Dict[str, float]:
        """Get detailed breakdown of reward components for logging."""
        baseline_acc = meta_features[0] if len(meta_features) > 0 else 0.5
        dataset_size = meta_features[1] if len(meta_features) > 1 else 0.5
        
        complexity_score = 0.6 * baseline_acc + 0.4 * dataset_size
        
        return {
            'complexity_score': complexity_score,
            'baseline_accuracy': baseline_acc,
            'dataset_size': dataset_size,
            'action': action,
            'action_name': self.action_names[action]
        }
    
    def render(self, mode="human"):
        """Render the environment state."""
        if self.current_meta_features is not None:
            print(f"Meta-features: {self.current_meta_features[:5]}...")
            print(f"Episode step: {self.episode_step}")


class RLTrainingCallback(BaseCallback):
    """Callback for monitoring RL training progress."""
    
    def __init__(self, automl_logger, check_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.automl_logger = automl_logger
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log training progress
            ep_infos = self.locals.get('infos', [])
            if ep_infos and any('episode' in info for info in ep_infos):
                episode_rewards = [info['episode']['r'] for info in ep_infos if 'episode' in info]
                if episode_rewards:
                    mean_reward = np.mean(episode_rewards)
                    
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.automl_logger:
                            self.automl_logger.log_debug(f"New best mean reward: {mean_reward:.4f}")
        
        return True


class RLModelSelector:
    """RL-based model selector for AutoML pipeline."""
    
    def __init__(
        self,
        meta_features_dim: int = 10,
        model_save_path: Optional[Path] = None,
        logger=None,
        random_state: int = 42
    ):
        """Initialize RL model selector.
        
        Args:
            meta_features_dim: Dimension of meta-features
            model_save_path: Path to save/load trained models
            logger: AutoML logger instance
            random_state: Random seed
        """
        self.meta_features_dim = meta_features_dim
        self.model_save_path = Path(model_save_path) if model_save_path else Path("models/rl_agent")
        self.logger = logger
        self.random_state = random_state
        
        # Create environment
        self.env = ModelSelectionEnv(
            meta_features_dim=meta_features_dim,
            random_state=random_state
        )
        
        # RL agent
        self.agent = None
        self.is_trained = False
        
        # Action mapping
        self.action_to_model = {
            0: "simple",
            1: "medium", 
            2: "complex"
        }
    
    def train(
        self, 
        total_timesteps: int = 10000,
        learning_rate: float = 1e-3,
        exploration_fraction: float = 0.3,
        target_update_interval: int = 1000
    ) -> None:
        """Train the RL agent for model selection.
        
        Args:
            total_timesteps: Number of training timesteps
            learning_rate: Learning rate for DQN
            exploration_fraction: Fraction of timesteps for exploration
            target_update_interval: Update interval for target network
        """
        if self.logger:
            self.logger.log_debug("Starting RL agent training", {
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "exploration_fraction": exploration_fraction
            })
        
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
            tensorboard_log=str(self.model_save_path.parent / "tensorboard") if self.logger else None,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0,
            seed=self.random_state
        )
        
        # Setup callback - simplified for compatibility
        callback = None  # Skip callback for now to avoid compatibility issues
        
        # Train the agent
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        if self.logger:
            self.logger.log_debug("RL training completed", {
                "training_time": f"{training_time:.2f}s",
                "timesteps": total_timesteps
            })
    
    def select_model(
        self, 
        meta_features: Dict[str, float],
        deterministic: bool = True
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Select model based on meta-features.
        
        Args:
            meta_features: Normalized meta-features from dataset
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (model_type, action, decision_info)
        """
        if not self.is_trained and self.agent is None:
            # If not trained, use heuristic fallback
            return self._heuristic_selection(meta_features)
        
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
            'action': action,
            'model_type': model_type,
            'confidence': confidence,
            'q_values': q_values.tolist(),
            'exploration': not deterministic
        }
        
        return model_type, action, decision_info
    
    def _prepare_observation(self, meta_features: Dict[str, float]) -> np.ndarray:
        """Convert meta-features dict to observation array."""
        # Use most important features in order
        feature_order = [
            'baseline_accuracy', 'dataset_size', 'avg_char_length', 
            'avg_word_length', 'vocab_size', 'num_classes',
            'type_token_ratio', 'class_imbalance_ratio', 'entropy',
            'std_char_length'
        ]
        
        obs = []
        for feature in feature_order:
            if feature in meta_features:
                obs.append(meta_features[feature])
            else:
                obs.append(0.0)  # Default value for missing features
        
        # Pad or truncate to match expected dimension
        if len(obs) < self.meta_features_dim:
            obs.extend([0.0] * (self.meta_features_dim - len(obs)))
        else:
            obs = obs[:self.meta_features_dim]
        
        return np.array(obs, dtype=np.float32)
    
    def _heuristic_selection(
        self, 
        meta_features: Dict[str, float]
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Fallback heuristic-based model selection."""
        baseline_acc = meta_features.get('baseline_accuracy', 0.5)
        dataset_size = meta_features.get('dataset_size', 0.5)
        vocab_size = meta_features.get('vocab_size', 0.5)
        
        # Simple heuristic rules
        complexity_score = 0.4 * baseline_acc + 0.4 * dataset_size + 0.2 * vocab_size
        
        if complexity_score > 0.7:
            action = 2  # Complex
            model_type = "complex"
        elif complexity_score > 0.4:
            action = 1  # Medium
            model_type = "medium"
        else:
            action = 0  # Simple
            model_type = "simple"
        
        decision_info = {
            'action': action,
            'model_type': model_type,
            'confidence': 0.5,  # Neutral confidence for heuristic
            'method': 'heuristic',
            'complexity_score': complexity_score
        }
        
        return model_type, action, decision_info
    
    def select_model_with_bohb(
        self,
        meta_features: Dict[str, float],
        training_data: Optional[Tuple[List[str], List[int]]] = None,
        fidelity_mode: str = "low",
        deterministic: bool = True,
        bohb_config: Optional[Any] = None
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
        if training_data is None or not self.is_trained:
            # Fall back to regular selection without BOHB
            return self.select_model(meta_features, deterministic)
        
        # Prepare observation
        obs = self._prepare_observation(meta_features)
        
        if self.agent is None:
            # Use heuristic if agent not available
            return self._heuristic_selection(meta_features)
        
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
                bohb_config=bohb_config
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
            'action': best_action,
            'model_type': model_type,
            'confidence': float(best_reward),
            'q_values': q_values.tolist(),
            'bohb_info': best_info,
            'fidelity_mode': fidelity_mode,
            'method': 'rl_with_bohb'
        }
        
        return model_type, best_action, decision_info
    
    def update_reward(
        self,
        meta_features: Dict[str, float],
        action: int,
        bohb_score: float,
        baseline_score: float = 0.5
    ) -> None:
        """Update RL agent with BOHB feedback reward.
        
        Args:
            meta_features: Meta-features used for model selection
            action: Action taken (0=simple, 1=medium, 2=complex)
            bohb_score: BOHB optimization score (accuracy)
            baseline_score: Baseline score for normalization
        """
        if not self.is_trained or self.agent is None:
            # Can't update if agent not trained
            if self.logger:
                self.logger.log_debug("RL agent not trained, skipping reward update")
            return
        
        try:
            # Prepare observation from meta-features dict
            obs = self._prepare_observation(meta_features)
            
            # Ensure action is integer
            action = int(action)
            
            # Normalize BOHB score to reward range [-1, 1]
            # Scale relative to baseline with some bounds
            if baseline_score > 0:
                relative_improvement = (bohb_score - baseline_score) / max(baseline_score, 0.1)
                # Clip to reasonable range and scale
                normalized_reward = np.clip(relative_improvement * 2.0, -1.0, 1.0)
            else:
                # If no baseline, use absolute performance with scaling
                normalized_reward = np.clip((bohb_score - 0.5) * 4.0, -1.0, 1.0)
            
            # For now, let's simplify and just log the reward information
            # The actual RL learning integration can be enhanced later
            if self.logger:
                self.logger.log_debug("RL reward calculated", {
                    "action": action,
                    "model_type": self.action_to_model[action],
                    "bohb_score": bohb_score,
                    "baseline_score": baseline_score,
                    "normalized_reward": normalized_reward,
                    "relative_improvement": (bohb_score - baseline_score) / max(baseline_score, 0.1) if baseline_score > 0 else None
                })
            
            # Store reward information for potential future learning
            if not hasattr(self, 'reward_history'):
                self.reward_history = []
            
            self.reward_history.append({
                'meta_features': meta_features,
                'action': action,
                'bohb_score': bohb_score,
                'baseline_score': baseline_score,
                'normalized_reward': normalized_reward,
                'timestamp': time.time()
            })
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error in update_reward: {e}")
            raise

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save trained RL model."""
        save_path = path or self.model_save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.agent is not None:
            self.agent.save(str(save_path))
            
            # Also save metadata
            metadata = {
                'meta_features_dim': self.meta_features_dim,
                'is_trained': self.is_trained,
                'action_to_model': self.action_to_model,
                'random_state': self.random_state
            }
            
            with open(save_path.with_suffix('.metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
        
        return save_path
    
    def load_model(self, path: Optional[Path] = None) -> bool:
        """Load trained RL model."""
        load_path = path or self.model_save_path
        
        # DQN saves as .zip files, so check for that
        if not load_path.exists():
            zip_path = load_path.with_suffix('.zip')
            if zip_path.exists():
                load_path = zip_path
            else:
                return False
        
        try:
            self.agent = DQN.load(str(load_path), env=self.env)
            
            # Load metadata if available
            metadata_path = load_path.with_suffix('.metadata.pkl')
            if load_path.suffix == '.zip':
                # If we loaded from .zip, metadata is at base_path.metadata.pkl
                metadata_path = load_path.with_suffix('').with_suffix('.metadata.pkl')
            
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.is_trained = metadata.get('is_trained', True)
            
            return True
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to load RL model: {e}")
            return False


class BOHBRewardEnv(gym.Wrapper):
    """Environment wrapper that uses BOHB for reward calculation.
    
    This wrapper enhances the base ModelSelectionEnv by using actual BOHB
    optimization results to calculate rewards, providing more accurate feedback
    for RL training.
    """
    
    def __init__(
        self,
        base_env: ModelSelectionEnv,
        training_datasets: List[Tuple[List[str], List[int], Dict[str, float]]],
        fidelity_mode: str = "low",
        logger=None
    ):
        """Initialize BOHB reward environment.
        
        Args:
            base_env: Base ModelSelectionEnv
            training_datasets: List of (texts, labels, meta_features) for training
            fidelity_mode: BOHB fidelity mode ("low" or "high")
            logger: AutoML logger instance
        """
        super().__init__(base_env)
        self.training_datasets = training_datasets
        self.fidelity_mode = fidelity_mode
        self.logger = logger
        self.dataset_index = 0
        
    def reset(self, **kwargs):
        """Reset environment with next training dataset."""
        # Cycle through training datasets
        if self.training_datasets:
            dataset_idx = self.dataset_index % len(self.training_datasets)
            texts, labels, meta_features = self.training_datasets[dataset_idx]
            
            # Store current dataset for BOHB evaluation
            self.current_training_data = (texts, labels)
            
            # Convert meta_features dict to array
            meta_features_array = []
            feature_order = [
                'baseline_accuracy', 'dataset_size', 'avg_char_length', 
                'avg_word_length', 'vocab_size', 'num_classes',
                'type_token_ratio', 'class_imbalance_ratio', 'entropy',
                'std_char_length'
            ]
            
            for feature in feature_order:
                if feature in meta_features:
                    meta_features_array.append(meta_features[feature])
                else:
                    meta_features_array.append(0.0)
            
            # Ensure correct dimension
            while len(meta_features_array) < self.env.meta_features_dim:
                meta_features_array.append(0.0)
            meta_features_array = meta_features_array[:self.env.meta_features_dim]
            
            kwargs['options'] = {'meta_features': meta_features_array}
            self.dataset_index += 1
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step with BOHB-enhanced reward calculation."""
        # Get base step results
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate BOHB-enhanced reward if training data available
        if hasattr(self, 'current_training_data'):
            try:
                bohb_reward, bohb_info = self.env.evaluate_with_bohb(
                    action=action,
                    meta_features=obs,
                    training_data=self.current_training_data,
                    fidelity_mode=self.fidelity_mode
                )
                
                # Use BOHB reward instead of base reward
                reward = bohb_reward
                info['bohb_evaluation'] = bohb_info
                info['base_reward'] = base_reward
                
                if self.logger:
                    self.logger.log_debug(
                        f"BOHB reward: {bohb_reward:.4f} vs base: {base_reward:.4f}"
                    )
                
            except Exception as e:
                # Fall back to base reward if BOHB fails
                reward = base_reward
                info['bohb_error'] = str(e)
                
                if self.logger:
                    self.logger.log_warning(f"BOHB reward calculation failed: {e}")
        else:
            reward = base_reward
        
        return obs, reward, terminated, truncated, info
