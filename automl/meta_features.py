"""Meta-feature extraction module for text classification datasets.

This module extracts dataset-specific meta-features that will be used as input
to the RL agent for dynamic model selection.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Import constants for feature consistency
from .constants import FEATURE_ORDER
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class MetaFeatureExtractor:
    """Extracts meta-features from text classification datasets."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the meta-feature extractor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
    def extract_features(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame = None,
        num_classes: int = None
    ) -> Dict[str, float]:
        """Extract comprehensive meta-features from the dataset.
        
        Args:
            train_df: Training DataFrame with 'text' and 'label' columns
            val_df: Optional validation DataFrame
            num_classes: Number of classes in the dataset
            
        Returns:
            Dictionary containing all extracted meta-features
        """
        logger.info("Starting meta-feature extraction...")
        
        # Combine train and val for comprehensive feature extraction
        if val_df is not None:
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            combined_df = train_df.copy()
            
        texts = combined_df['text'].tolist()
        labels = combined_df['label'].tolist() if 'label' in combined_df.columns else None
        
        meta_features = {}
        
        # 1. Basic dataset statistics
        meta_features.update(self._extract_basic_stats(texts, labels, num_classes))
        
        # 2. Text length statistics
        meta_features.update(self._extract_length_stats(texts))
        
        # 3. Vocabulary and lexical richness
        meta_features.update(self._extract_vocabulary_stats(texts))
        
        # 4. Baseline performance using TF-IDF + Logistic Regression
        if labels is not None:
            meta_features.update(self._extract_baseline_performance(texts, labels))
        
        # 5. Text complexity features
        meta_features.update(self._extract_complexity_features(texts))
        
        logger.info(f"Extracted {len(meta_features)} meta-features")
        return meta_features
    
    def _extract_basic_stats(
        self, 
        texts: list, 
        labels: list = None, 
        num_classes: int = None
    ) -> Dict[str, float]:
        """Extract basic dataset statistics."""
        stats = {
            'dataset_size': len(texts),
            'num_classes': num_classes or len(set(labels)) if labels else 0,
        }
        
        if labels is not None:
            # Class distribution statistics
            label_counts = Counter(labels)
            class_probs = np.array(list(label_counts.values())) / len(labels)
            
            stats.update({
                'class_imbalance_ratio': max(class_probs) / min(class_probs),
                'entropy': -np.sum(class_probs * np.log2(class_probs + 1e-10)),
                'gini_coefficient': 1 - np.sum(class_probs ** 2),
            })
        
        return stats
    
    def _extract_length_stats(self, texts: list) -> Dict[str, float]:
        """Extract text length statistics (character and word level)."""
        char_lengths = [len(text) for text in texts]
        word_lengths = [len(text.split()) for text in texts]
        
        return {
            # Character-level statistics
            'avg_char_length': np.mean(char_lengths),
            'max_char_length': np.max(char_lengths),
            'min_char_length': np.min(char_lengths),
            'std_char_length': np.std(char_lengths),
            'median_char_length': np.median(char_lengths),
            'char_length_skewness': self._calculate_skewness(char_lengths),
            
            # Word-level statistics
            'avg_word_length': np.mean(word_lengths),
            'max_word_length': np.max(word_lengths),
            'min_word_length': np.min(word_lengths),
            'std_word_length': np.std(word_lengths),
            'median_word_length': np.median(word_lengths),
            'word_length_skewness': self._calculate_skewness(word_lengths),
        }
    
    def _extract_vocabulary_stats(self, texts: list) -> Dict[str, float]:
        """Extract vocabulary and lexical richness statistics."""
        # Tokenize all texts
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        vocab = set(all_words)
        word_counts = Counter(all_words)
        
        # Basic vocabulary stats
        vocab_stats = {
            'vocab_size': len(vocab),
            'total_tokens': len(all_words),
            'avg_word_freq': len(all_words) / len(vocab) if len(vocab) > 0 else 0,
        }
        
        # Lexical richness measures
        if len(all_words) > 0:
            vocab_stats.update({
                'type_token_ratio': len(vocab) / len(all_words),
                'hapax_legomena_ratio': sum(1 for count in word_counts.values() if count == 1) / len(vocab),
                'top_10_word_freq_ratio': sum(sorted(word_counts.values(), reverse=True)[:10]) / len(all_words),
            })
        
        # Character-level vocabulary
        all_chars = ''.join(texts)
        char_vocab = set(all_chars)
        vocab_stats['char_vocab_size'] = len(char_vocab)
        
        return vocab_stats
    
    def _extract_baseline_performance(self, texts: list, labels: list) -> Dict[str, float]:
        """Extract baseline performance using TF-IDF + Logistic Regression."""
        try:
            # Create a simple baseline pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )),
                ('classifier', LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                ))
            ])
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                pipeline, texts, labels, 
                cv=5, scoring='accuracy',
                n_jobs=-1
            )
            
            return {
                'baseline_accuracy': np.mean(cv_scores),
                'baseline_accuracy_std': np.std(cv_scores),
                'baseline_accuracy_min': np.min(cv_scores),
                'baseline_accuracy_max': np.max(cv_scores),
            }
            
        except Exception as e:
            logger.warning(f"Could not compute baseline performance: {e}")
            return {
                'baseline_accuracy': 0.0,
                'baseline_accuracy_std': 0.0,
                'baseline_accuracy_min': 0.0,
                'baseline_accuracy_max': 0.0,
            }
    
    def _extract_complexity_features(self, texts: list) -> Dict[str, float]:
        """Extract text complexity features."""
        # Punctuation and special character stats (improved)
        punct_counts = [len(re.findall(r'[^\w\s]', text)) for text in texts]
        digit_counts = [len(re.findall(r'\d', text)) for text in texts]
        uppercase_counts = [len(re.findall(r'[A-Z]', text)) for text in texts]
        
        # Average word length in characters
        word_char_lengths = []
        for text in texts:
            words = text.split()
            if words:
                word_char_lengths.extend([len(word) for word in words])
        
        complexity_features = {
            'avg_punctuation_per_text': np.mean(punct_counts) if punct_counts else 0.0,
            'avg_digits_per_text': np.mean(digit_counts) if digit_counts else 0.0,
            'avg_uppercase_per_text': np.mean(uppercase_counts) if uppercase_counts else 0.0,
        }
        
        if word_char_lengths:
            complexity_features.update({
                'avg_word_char_length': np.mean(word_char_lengths),
                'std_word_char_length': np.std(word_char_lengths),
            })
        else:
            complexity_features.update({
                'avg_word_char_length': 0.0,
                'std_word_char_length': 0.0,
            })
        
        # Sentence-level features (improved detection)
        sentence_counts = []
        for text in texts:
            # Count sentence-ending punctuation
            sentences = text.count('.') + text.count('!') + text.count('?')
            # If no sentence punctuation found, assume it's one sentence if text exists
            if sentences == 0 and len(text.strip()) > 0:
                sentences = 1
            sentence_counts.append(sentences)
            
        complexity_features.update({
            'avg_sentences_per_text': np.mean(sentence_counts) if sentence_counts else 0.0,
            'std_sentences_per_text': np.std(sentence_counts) if sentence_counts else 0.0,
        })
        
        return complexity_features
    
    def _calculate_skewness(self, data: list) -> float:
        """Calculate skewness of a distribution."""
        if len(data) < 3:
            return 0.0
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def get_feature_importance_ranking(self, meta_features: Dict[str, float]) -> list:
        """Return a ranking of meta-features by expected importance for model selection.
        
        Args:
            meta_features: Dictionary of extracted meta-features
            
        Returns:
            List of feature names ranked by importance (most important first)
        """
        # Define importance ranking based on domain knowledge
        importance_ranking = [
            'baseline_accuracy',
            'dataset_size',
            'avg_char_length',
            'avg_word_length',
            'vocab_size',
            'num_classes',
            'type_token_ratio',
            'class_imbalance_ratio',
            'entropy',
            'std_char_length',
            'std_word_length',
            'char_length_skewness',
            'word_length_skewness',
            'baseline_accuracy_std',
            'hapax_legomena_ratio',
            'avg_word_char_length',
            'avg_punctuation_per_text',
        ]
        
        # Return only features that exist in the meta_features dictionary
        return [feature for feature in importance_ranking if feature in meta_features]
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to [0, 1] range for RL input."""
        normalized = features.copy()
        
        # First, normalize accuracy
        if 'baseline_accuracy' in normalized:
            normalized['baseline_accuracy'] = min(normalized['baseline_accuracy'], 1.0)
        
        # Normalize size features by scaling
        size_normalizers = {
            'dataset_size': 100000,  # 100k samples
            'avg_char_length': 1000,  # Extremely long texts
            'max_char_length': 10000,  # Very long max length
            'std_char_length': 1000,  # High variation
            'avg_word_length': 20,  # Long average word length
            'max_word_length': 50,  # Very long words
        }
        
        for feature, normalizer in size_normalizers.items():
            if feature in normalized:
                normalized[feature] = min(normalized[feature] / normalizer, 1.0)
                
        return normalized
                
    def ensure_feature_order(self, features: Dict[str, float]) -> Dict[str, float]:
        """Ensure the feature dictionary follows the order specified in FEATURE_ORDER.
        
        This is crucial to maintain consistency between meta-feature extraction and RL input.
        """
        ordered_features = {}
        for feature in FEATURE_ORDER:
            ordered_features[feature] = features.get(feature, 0.0)
        return ordered_features
    
    def get_ordered_features(self, meta_features: Dict[str, float], normalize: bool = True) -> Dict[str, float]:
        """Get features ordered according to the global FEATURE_ORDER.
        
        Args:
            meta_features: Raw meta-features dictionary
            normalize: Whether to normalize features
            
        Returns:
            Dictionary with features in the correct order
        """
        if meta_features is None:
            return {}
            
        if normalize:
            normalized = self.normalize_features(meta_features)
        else:
            normalized = meta_features.copy()
            
        return self.ensure_feature_order(normalized)