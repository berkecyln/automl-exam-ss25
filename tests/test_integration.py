"""Test the integrated model system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.models import create_model, SimpleModel, MediumModel, ComplexModel
from automl.bohb_optimization import BOHBOptimizer

def test_model_integration():
    """Test that the model integration works correctly."""
    
    # Sample data
    texts = ["This is good", "This is bad", "Great product", "Terrible service"] * 10
    labels = [1, 0, 1, 0] * 10
    
    # Test simple model
    print("Testing Simple Model...")
    simple_config = {
        "algorithm": "logistic",
        "max_features": 1000,
        "min_df": 0.01,
        "max_df": 0.85,
        "ngram_max": 2,
        "C": 1.0,
        "max_iter": 100
    }
    
    simple_model = create_model("simple", simple_config, random_state=42)
    simple_model.fit(texts[:30], labels[:30])
    simple_pred = simple_model.predict(texts[30:])
    simple_score = simple_model.evaluate(texts[30:], labels[30:])
    print(f"Simple model accuracy: {simple_score:.3f}")
    
    # Test medium model
    print("Testing Medium Model...")
    medium_config = {
        "algorithm": "lsa_lr",
        "max_features": 100,  # Reduced for small test data
        "svd_components": 10,  # Much smaller for test
        "C": 1.0,
        "max_iter": 100
    }
    
    medium_model = create_model("medium", medium_config, random_state=42)
    medium_model.fit(texts[:30], labels[:30])
    medium_pred = medium_model.predict(texts[30:])
    medium_score = medium_model.evaluate(texts[30:], labels[30:])
    print(f"Medium model accuracy: {medium_score:.3f}")
    
    # Test complex model
    print("Testing Complex Model...")
    complex_config = {
        "algorithm": "tfidf_mlp",
        "max_features": 100,  # Reduced for test
        "hidden_units": 20,   # Smaller for test
        "alpha": 1e-4,
        "max_iter": 20        # Fewer iterations for test
    }
    
    complex_model = create_model("complex", complex_config, random_state=42, budget_fraction=0.5)
    complex_model.fit(texts[:30], labels[:30])
    complex_pred = complex_model.predict(texts[30:])
    complex_score = complex_model.evaluate(texts[30:], labels[30:])
    print(f"Complex model accuracy: {complex_score:.3f}")
    
    print("All model integrations work correctly!")

def test_bohb_integration():
    """Test that BOHB integration works with new model classes."""
    
    print("\nTesting BOHB Integration...")
    
    # Sample data
    texts = ["This is good", "This is bad", "Great product", "Terrible service"] * 50
    labels = [1, 0, 1, 0] * 50
    
    # Test with simple model
    optimizer = BOHBOptimizer("simple", random_state=42)
    
    # Run a quick optimization with minimal settings
    optimizer.config.n_trials = 5
    optimizer.config.wall_clock_limit = 30.0  # 30 seconds
    
    best_config, best_score, stats = optimizer.optimize(
        texts, labels, fidelity_mode="low"
    )
    
    print(f"BOHB Simple model - Best score: {best_score:.3f}")
    print(f"BOHB completed {stats['num_evaluations']} evaluations")
    
    print("BOHB integration works correctly!")

if __name__ == "__main__":
    test_model_integration()
    test_bohb_integration()
