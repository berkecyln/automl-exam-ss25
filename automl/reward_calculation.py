"""
Advanced Reward Calculation for AutoML Text Classification RL Agent.

This module implements sophisticated reward functions that consider:
- Performance vs baseline improvement
- Adaptive complexity penalties based on dataset characteristics  
- Time efficiency and budget management
- Confidence and learning progress
- Smart exploration vs exploitation balance
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AutoMLRewardCalculator:
    """
    Advanced reward calculator for AutoML text classification RL agent.
    
    Provides context-aware, multi-objective reward calculation that adapts
    to dataset characteristics, training progress, and resource constraints.
    """
    
    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        enable_adaptive_weighting: bool = True,
        enable_nonlinear_scaling: bool = True
    ):
        """
        Initialize the reward calculator.
        
        Args:
            base_weights: Base weights for reward components
            enable_adaptive_weighting: Whether to adapt weights based on context
            enable_nonlinear_scaling: Whether to apply non-linear scaling
        """
        self.base_weights = base_weights or {
            'performance': 0.50,
            'complexity': 0.20,
            'time': 0.15,
            'confidence': 0.10,
            'exploration': 0.05
        }
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.enable_nonlinear_scaling = enable_nonlinear_scaling
        
        # Track history for learning progress calculation
        self.performance_history = []
        
    def calculate_enhanced_reward(
        self,
        action: int,
        meta_features: np.ndarray,
        bohb_accuracy: float,
        time_taken: Optional[float] = None,
        iteration: int = 1,
        remaining_budget: float = 1.0
    ) -> float:
        """
        Calculate comprehensive reward for AutoML model selection.
        
        Args:
            action: Model choice (0=Simple, 1=Medium, 2=Complex)
            meta_features: Dataset meta-features array
            bohb_accuracy: Actual BOHB optimization result
            time_taken: Time spent on this evaluation (optional)
            q_values: Current Q-values for confidence estimation (optional)
            iteration: Current training iteration
            remaining_budget: Fraction of time budget remaining
            
        Returns:
            Float reward value in range [-1.0, 1.0]
        """
        if bohb_accuracy is None:
            raise ValueError("bohb_accuracy must be provided for reward calculation")
        
        # Extract key meta-features with safe indexing
        baseline_acc = self._safe_get_feature(meta_features, 0, 0.5)
        dataset_size = self._safe_get_feature(meta_features, 1, 0.5)
        vocab_size = self._safe_get_feature(meta_features, 2, 0.5)
        num_classes = self._safe_get_feature(meta_features, 5, 0.5)
        class_imbalance = self._safe_get_feature(meta_features, 7, 0.5)
        
        # 1. Core Performance Reward
        performance_reward = self.calculate_performance_reward(
            bohb_accuracy, baseline_acc, action
        )
        
        # 2. Adaptive Complexity Penalty
        complexity_penalty = self.calculate_adaptive_complexity_penalty(
            action, dataset_size, vocab_size, num_classes
        )
        
        # 3. Time Efficiency Bonus
        time_bonus = self.calculate_time_efficiency_bonus(
            time_taken, remaining_budget, action
        )
        
        # 4. Confidence & Learning Bonus
        confidence_bonus = self.calculate_confidence_bonus(
            bohb_accuracy, baseline_acc, iteration
        )
        
        # 5. Exploration Bonus
        exploration_bonus = self.calculate_exploration_bonus(
            action, meta_features, iteration, remaining_budget
        )
        
        # Get weights (adaptive or static)
        if self.enable_adaptive_weighting:
            weights = self.get_adaptive_weights(
                iteration, remaining_budget, dataset_size, class_imbalance
            )
        else:
            weights = self.base_weights.copy()
        
        # Calculate total reward
        total_reward = (
            weights['performance'] * performance_reward +
            weights['complexity'] * (-complexity_penalty) +  # Negative for penalty
            weights['time'] * time_bonus +
            weights['confidence'] * confidence_bonus +
            weights['exploration'] * exploration_bonus
        )
        
        # Apply non-linear scaling if enabled
        if self.enable_nonlinear_scaling:
            total_reward = self.apply_nonlinear_scaling(total_reward, bohb_accuracy)
        
        # Store for learning progress tracking
        self.performance_history.append(bohb_accuracy)
        if len(self.performance_history) > 20:  # Keep last 20 results
            self.performance_history.pop(0)
        
        return np.clip(total_reward, -1.0, 1.0)
    
    def calculate_performance_reward(
        self, bohb_accuracy: float, baseline_acc: float, action: int
    ) -> float:
        """Calculate performance-based reward with improvement weighting."""
        
        # Base performance score
        normalized_acc = min(1.0, max(0.0, bohb_accuracy))
        
        # Improvement over baseline bonus (crucial for AutoML)
        improvement = max(0, bohb_accuracy - baseline_acc)
        improvement_bonus = min(0.3, improvement * 2.0)  # Cap at 0.3, scale by 2x
        
        # Diminishing returns for very high accuracy (avoid overfitting reward)
        if normalized_acc > 0.95:
            diminishing_factor = 0.95 + 0.05 * (normalized_acc - 0.95) * 0.5
            normalized_acc = diminishing_factor
        
        # Model-specific performance expectations
        model_expectations = {0: 0.70, 1: 0.80, 2: 0.88}  # Simple, Medium, Complex
        expectation_bonus = max(0, bohb_accuracy - model_expectations[action]) * 0.5
        
        return normalized_acc + improvement_bonus + expectation_bonus
    
    def calculate_adaptive_complexity_penalty(
        self, action: int, dataset_size: float, vocab_size: float, num_classes: float
    ) -> float:
        """Complexity penalty that adapts to dataset characteristics."""
        
        # Base complexity costs
        base_penalties = {0: 0.05, 1: 0.15, 2: 0.35}
        base_penalty = base_penalties[action]
        
        # Dataset-adaptive scaling
        complexity_score = (dataset_size + vocab_size + num_classes) / 3.0
        
        if action == 0:  # Simple models
            # Penalize simple models on complex datasets
            penalty_multiplier = 1.0 + complexity_score * 0.5
        elif action == 1:  # Medium models  
            # Medium models are versatile, less penalty variation
            penalty_multiplier = 1.0 + abs(complexity_score - 0.5) * 0.3
        else:  # Complex models
            # Penalize complex models on simple datasets
            penalty_multiplier = 1.0 + (1.0 - complexity_score) * 0.8
        
        return base_penalty * penalty_multiplier
    
    def calculate_time_efficiency_bonus(
        self, time_taken: Optional[float], remaining_budget: float, action: int
    ) -> float:
        """Reward efficient use of time budget."""
        
        if time_taken is None:
            return 0.0
        
        # Expected time costs for different model types
        expected_times = {0: 0.1, 1: 0.3, 2: 0.8}  # Relative time costs
        expected_time = expected_times[action]
        
        # Efficiency ratio (lower is better)
        if expected_time > 0:
            efficiency_ratio = min(2.0, time_taken / expected_time)
            time_bonus = max(0, 1.0 - efficiency_ratio * 0.5)
        else:
            time_bonus = 0.0
        
        # Budget conservation bonus (important late in training)
        if remaining_budget < 0.3:  # Less than 30% budget remaining
            conservation_bonus = (1.0 - remaining_budget) * 0.2
            time_bonus += conservation_bonus
        
        return time_bonus
    
    def calculate_confidence_bonus(
        self, bohb_accuracy: float, baseline_acc: float, iteration: int
    ) -> float:
        """Reward confident decisions and learning progress WITHOUT Q-values."""
        
        confidence_bonus = 0.0
        
        # Improvement confidence (how much better than baseline)
        improvement = max(0, bohb_accuracy - baseline_acc)
        if improvement > 0.05:  # Significant improvement (5%+)
            improvement_confidence = min(0.2, improvement * 2.0)
            confidence_bonus += improvement_confidence
        
        # Absolute performance confidence
        if bohb_accuracy > 0.85:  # Very good absolute performance
            performance_confidence = (bohb_accuracy - 0.85) * 0.5
            confidence_bonus += performance_confidence
        
        # Learning progress bonus (reward improvement over iterations)
        if iteration > 1:
            learning_bonus = min(0.2, (iteration - 1) * 0.02)
            confidence_bonus += learning_bonus
        
        # Historical improvement bonus
        if len(self.performance_history) >= 3:
            recent_avg = np.mean(self.performance_history[-3:])
            earlier_avg = np.mean(self.performance_history[:-3]) if len(self.performance_history) > 3 else recent_avg
            if recent_avg > earlier_avg:
                improvement_bonus = min(0.1, (recent_avg - earlier_avg) * 2.0)
                confidence_bonus += improvement_bonus
        
        # Performance consistency bonus
        if len(self.performance_history) >= 5:
            recent_scores = self.performance_history[-5:]
            consistency = 1.0 - np.std(recent_scores)  # Lower std = higher consistency
            consistency_bonus = max(0, consistency * 0.1)
            confidence_bonus += consistency_bonus
        
        return confidence_bonus
    
    def calculate_exploration_bonus(
        self, action: int, meta_features: np.ndarray, iteration: int, remaining_budget: float
    ) -> float:
        """Encourage smart exploration early, exploitation later."""
        
        # Exploration is more valuable early in training
        exploration_value = max(0, 1.0 - iteration * 0.1)
        
        # Exploration is less valuable when budget is low
        budget_factor = min(1.0, remaining_budget * 2.0)
        
        # Encourage trying different models for diverse datasets
        dataset_novelty = self.estimate_dataset_novelty(meta_features)
        
        exploration_bonus = exploration_value * budget_factor * dataset_novelty * 0.1
        
        return exploration_bonus
    
    def estimate_dataset_novelty(self, meta_features: np.ndarray) -> float:
        """Estimate if this dataset is different from previously seen ones."""
        # Simple heuristic - can be improved with more sophisticated measures
        if len(meta_features) < 5:
            return 0.5  # Default novelty
        
        feature_variance = np.var(meta_features[:5])  # Use first 5 features
        return min(1.0, feature_variance * 10.0)
    
    def get_adaptive_weights(
        self, iteration: int, remaining_budget: float, dataset_size: float, class_imbalance: float
    ) -> Dict[str, float]:
        """Adapt reward component weights based on context."""
        
        # Start with base weights
        weights = self.base_weights.copy()
        
        # Early iterations: emphasize exploration
        if iteration <= 3:
            weights['exploration'] += 0.05
            weights['performance'] -= 0.05
        
        # Low budget: emphasize time efficiency
        if remaining_budget < 0.3:
            weights['time'] += 0.10
            weights['exploration'] -= 0.05
            weights['performance'] -= 0.05
        
        # Large datasets: emphasize efficiency over exploration
        if dataset_size > 0.7:
            weights['time'] += 0.05
            weights['complexity'] += 0.05
            weights['exploration'] -= 0.10
        
        # Imbalanced datasets: emphasize performance more
        if class_imbalance > 0.8:
            weights['performance'] += 0.10
            weights['complexity'] -= 0.05
            weights['time'] -= 0.05
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def apply_nonlinear_scaling(self, reward: float, accuracy: float) -> float:
        """Apply non-linear scaling for edge cases."""
        
        # Boost reward for exceptional performance
        if accuracy > 0.90:
            exceptional_bonus = (accuracy - 0.90) * 2.0
            reward += exceptional_bonus
        
        # Penalize very poor performance more heavily
        if accuracy < 0.50:
            poor_performance_penalty = (0.50 - accuracy) * 1.5
            reward -= poor_performance_penalty
        
        # Smooth saturation at extremes
        if reward > 0.8:
            reward = 0.8 + (reward - 0.8) * 0.5
        elif reward < -0.8:
            reward = -0.8 + (reward + 0.8) * 0.5
        
        return reward
    
    def _safe_get_feature(self, meta_features: np.ndarray, index: int, default: float) -> float:
        """Safely extract feature value with fallback."""
        if len(meta_features) > index:
            return float(meta_features[index])
        return default
    
    def get_reward_breakdown(
        self,
        action: int,
        meta_features: np.ndarray,
        bohb_accuracy: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components for debugging/logging.
        
        Returns:
            Dictionary with individual reward component values
        """
        baseline_acc = self._safe_get_feature(meta_features, 0, 0.5)
        dataset_size = self._safe_get_feature(meta_features, 1, 0.5)
        vocab_size = self._safe_get_feature(meta_features, 2, 0.5)
        num_classes = self._safe_get_feature(meta_features, 5, 0.5)
        
        return {
            "performance_reward": self.calculate_performance_reward(bohb_accuracy, baseline_acc, action),
            "complexity_penalty": self.calculate_adaptive_complexity_penalty(action, dataset_size, vocab_size, num_classes),
            "time_bonus": self.calculate_time_efficiency_bonus(kwargs.get('time_taken'), kwargs.get('remaining_budget', 1.0), action),
            "confidence_bonus": self.calculate_confidence_bonus(bohb_accuracy, baseline_acc, kwargs.get('iteration', 1)),
            "exploration_bonus": self.calculate_exploration_bonus(action, meta_features, kwargs.get('iteration', 1), kwargs.get('remaining_budget', 1.0)),
            "bohb_accuracy": bohb_accuracy,
            "baseline_accuracy": baseline_acc,
        }