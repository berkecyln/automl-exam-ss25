#!/usr/bin/env python3
"""
Test script to verify the updated pipeline prediction functionality.
This ensures the unified model approach works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from automl.models import create_model

def test_unified_prediction():
    """Test the unified model prediction approach."""
    print("🧪 Testing Unified Model Prediction")
    print("=" * 40)
    
    # Sample data for testing - more samples to avoid train_test_split issues
    sample_texts = [
        "This is a great product!",
        "I hate this terrible service.",
        "The movie was okay, nothing special.",
        "Absolutely amazing experience!",
        "Not worth the money.",
        "Excellent quality and fast delivery.",
        "Poor customer service and bad quality.",
        "Average product, nothing remarkable.",
        "Outstanding performance, highly recommended!",
        "Disappointing experience, would not buy again.",
        "Good value for money.",
        "Terrible quality, waste of time.",
    ]
    sample_labels = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    test_texts = [
        "This product is fantastic!",
        "Very disappointing quality.",
    ]
    
    model_types = ["simple", "medium", "complex"]
    
    for model_type in model_types:
        print(f"\n🎯 Testing {model_type.upper()} model")
        print("-" * 30)
        
        try:
            # Create model with complete config for each type based on BOHB config space
            if model_type == "simple":
                config = {
                    "algorithm": "logistic",
                    "max_features": 1000,
                    "min_df": 0.01,
                    "max_df": 0.85,
                    "ngram_max": 2,
                    "C": 1.0,
                    "max_iter": 1000,
                    "random_state": 42
                }
            elif model_type == "medium":
                config = {
                    "algorithm": "lsa_lr",
                    "max_features": 2000,
                    "svd_components": 100,
                    "C": 1.0,
                    "max_iter": 1000,
                    "random_state": 42
                }
            else:  # complex
                config = {
                    "algorithm": "tfidf_mlp",
                    "max_features": 5000,
                    "hidden_units": 200,
                    "alpha": 1e-4,
                    "max_iter": 200,
                    "random_state": 42
                }
            
            model = create_model(
                model_type=model_type,
                config=config,
                random_state=42
            )
            
            # Train model
            print(f"  📝 Training {model_type} model...")
            model.fit(sample_texts, sample_labels)
            
            # Make predictions
            print(f"  🔮 Making predictions...")
            predictions = model.predict(test_texts)
            
            # Evaluate
            print(f"  📊 Evaluating...")
            accuracy = model.evaluate(sample_texts, sample_labels)
            
            print(f"  ✅ {model_type.title()} Model Results:")
            print(f"     Training accuracy: {accuracy:.4f}")
            print(f"     Test predictions: {predictions}")
            print(f"     Model class: {type(model).__name__}")
            
        except Exception as e:
            print(f"  ❌ Error with {model_type} model: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Unified model prediction test completed!")

if __name__ == "__main__":
    test_unified_prediction()
