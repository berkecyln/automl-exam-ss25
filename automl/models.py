"""
Unified Model Classes for AutoML Pipeline

This module provides unified model classes that handle both training/evaluation
and prediction for simple, medium, and complex model types.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class BaseTextModel:
    """Base class for all text classification models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.preprocessor = None  # For models that need additional preprocessing (e.g., SVD)
    
    def fit(self, X_train: List[str], y_train: List[int]) -> 'BaseTextModel':
        """Train the model on training data."""
        raise NotImplementedError
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        """Make predictions on test data."""
        raise NotImplementedError
    
    def evaluate(self, X_val: List[str], y_val: List[int]) -> float:
        """Evaluate the model and return accuracy."""
        predictions = self.predict(X_val)
        return accuracy_score(y_val, predictions)


class SimpleModel(BaseTextModel):
    """Simple model: TF-IDF + LogisticRegression/SVM"""
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        super().__init__(random_state)
        self.config = config
    
    def fit(self, X_train: List[str], y_train: List[int]) -> 'SimpleModel':
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config["max_features"],
            min_df=self.config["min_df"],
            max_df=self.config["max_df"],
            ngram_range=(1, self.config["ngram_max"]),
            stop_words="english",
        )
        
        # Transform training data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Create and train model
        if self.config["algorithm"] == "logistic":
            self.model = LogisticRegression(
                C=self.config["C"],
                max_iter=self.config["max_iter"],
                random_state=self.random_state
            )
        elif self.config["algorithm"] == "svm":
            self.model = SVC(
                C=self.config["C"],
                kernel="linear",
                random_state=self.random_state
            )
        
        self.model.fit(X_train_vec, y_train)
        return self
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)


class MediumModel(BaseTextModel):
    """Medium model: TF-IDF + TruncatedSVD + LogisticRegression"""
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        super().__init__(random_state)
        self.config = config
    
    def fit(self, X_train: List[str], y_train: List[int]) -> 'MediumModel':
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config["max_features"],
            stop_words="english",
            ngram_range=(1, 2)
        )
        
        # Transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Create SVD preprocessor with adjusted components
        svd_components = min(self.config["svd_components"], X_train_tfidf.shape[1] - 1)
        svd = TruncatedSVD(
            n_components=svd_components,
            random_state=self.random_state
        )
        self.preprocessor = make_pipeline(svd, Normalizer(copy=False))
        
        # Apply SVD transformation
        X_train_svd = self.preprocessor.fit_transform(X_train_tfidf)
        
        # Create and train model
        self.model = LogisticRegression(
            C=self.config["C"],
            max_iter=self.config["max_iter"],
            solver="lbfgs",
            random_state=self.random_state
        )
        
        self.model.fit(X_train_svd, y_train)
        return self
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        if self.vectorizer is None or self.preprocessor is None or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        X_test_svd = self.preprocessor.transform(X_test_tfidf)
        return self.model.predict(X_test_svd)


class ComplexModel(BaseTextModel):
    """Complex model: TF-IDF + MLPClassifier"""
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42, budget_fraction: float = 1.0):
        super().__init__(random_state)
        self.config = config
        self.budget_fraction = budget_fraction
    
    def fit(self, X_train: List[str], y_train: List[int]) -> 'ComplexModel':
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config["max_features"],
            stop_words=None,  # Include stopwords for richer representation
            ngram_range=(1, 3)  # Include trigrams
        )
        
        # Transform training data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Adjust parameters based on budget
        max_iter = max(50, int(self.budget_fraction * self.config["max_iter"]))
        hidden_units = max(50, int(self.budget_fraction * self.config["hidden_units"]))
        
        # Create and train model
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation="relu",
            alpha=self.config["alpha"],
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=1e-4,
            random_state=self.random_state
        )
        
        self.model.fit(X_train_vec, y_train)
        return self
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)


# Factory function to create models
def create_model(model_type: str, config: Dict[str, Any], random_state: int = 42, **kwargs) -> BaseTextModel:
    """Factory function to create the appropriate model type."""
    if model_type == "simple":
        return SimpleModel(config, random_state)
    elif model_type == "medium":
        return MediumModel(config, random_state)
    elif model_type == "complex":
        budget_fraction = kwargs.get('budget_fraction', 1.0)
        return ComplexModel(config, random_state, budget_fraction)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
