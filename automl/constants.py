"""
Constants and configuration values shared across the AutoML system.
"""

# Feature order for meta-features across the AutoML system
# This ensures consistency between feature extraction and model usage
FEATURE_ORDER = [
    "baseline_accuracy",
    "baseline_accuracy_std",
    "dataset_size",
    "avg_char_length",
    "avg_word_length",
    "vocab_size",
    "num_classes",
    "type_token_ratio",
    "class_imbalance_ratio",
    "entropy",
    "std_char_length",
    "num_samples",
    "num_features",
    "num_categorical",
    "num_numerical",
    "num_binary",
    "ratio_categorical",
    "ratio_numerical",
    "ratio_binary",
    "missing_ratio",
    "symbol_ratio",
    "max_char_length",
    "min_char_length",
    "median_char_length",
    "hapax_legomena_ratio",
    "uppercase_ratio",
    "lowercase_ratio",
    "digit_ratio",
    "avg_sentence_length",
    "avg_sentence_count",
    "flesch_reading_ease",
    "avg_word_count",
    "unique_word_count",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
]

# Number of meta-features used in the system
META_FEATURE_DIM = len(FEATURE_ORDER)

# Model types available for selection
MODEL_TYPES = ["simple", "medium", "complex"]

# Random seed for reproducibility
DEFAULT_RANDOM_SEED = 42
