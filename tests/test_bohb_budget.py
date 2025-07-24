"""Test the BOHB budget management."""

import sys
import os
import numpy as np
from typing import Dict

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.bohb_optimization import BOHBOptimizer, BOHBConfig


def test_bohb_budget_calculation():
    """Test that the BOHB budget parameters are calculated correctly."""
    
    # Create optimizer instances for each model type
    simple_optimizer = BOHBOptimizer("simple")
    medium_optimizer = BOHBOptimizer("medium")
    complex_optimizer = BOHBOptimizer("complex")
    
    # Calculate budgets for each model type in both fidelity modes
    simple_low = simple_optimizer._calculate_budget("low")
    simple_high = simple_optimizer._calculate_budget("high")
    
    medium_low = medium_optimizer._calculate_budget("low")
    medium_high = medium_optimizer._calculate_budget("high")
    
    complex_low = complex_optimizer._calculate_budget("low")
    complex_high = complex_optimizer._calculate_budget("high")
    
    # Print the results in a table format
    print("\n--- BOHB Budget Parameters ---\n")
    
    headers = ["Model", "Fidelity", "min_budget", "max_budget", "n_trials", "time_limit"]
    row_format = "{:<10} {:<10} {:<12} {:<12} {:<10} {:<12}"
    
    print(row_format.format(*headers))
    print("-" * 70)
    
    print(row_format.format("Simple", "Low", 
                          simple_low["min_budget"], 
                          simple_low["max_budget"],
                          simple_low["n_trials"],
                          simple_low["time_limit"]))
    
    print(row_format.format("Simple", "High", 
                          simple_high["min_budget"], 
                          simple_high["max_budget"],
                          simple_high["n_trials"],
                          simple_high["time_limit"]))
    
    print(row_format.format("Medium", "Low", 
                          medium_low["min_budget"], 
                          medium_low["max_budget"],
                          medium_low["n_trials"],
                          medium_low["time_limit"]))
    
    print(row_format.format("Medium", "High", 
                          medium_high["min_budget"], 
                          medium_high["max_budget"],
                          medium_high["n_trials"],
                          medium_high["time_limit"]))
    
    print(row_format.format("Complex", "Low", 
                          complex_low["min_budget"], 
                          complex_low["max_budget"],
                          complex_low["n_trials"],
                          complex_low["time_limit"]))
    
    print(row_format.format("Complex", "High", 
                          complex_high["min_budget"], 
                          complex_high["max_budget"],
                          complex_high["n_trials"],
                          complex_high["time_limit"]))
    
    # Verify that the budget parameters satisfy the requirements
    for budget in [simple_low, simple_high, medium_low, medium_high, complex_low, complex_high]:
        assert budget["min_budget"] > 0, "min_budget must be positive"
        assert budget["max_budget"] > budget["min_budget"], "max_budget must be > min_budget"
        assert budget["n_trials"] > budget["max_budget"], "n_trials must be > max_budget (SMAC requirement)"
        assert budget["time_limit"] > 0, "time_limit must be positive"
    
    print("\nAll budget parameters satisfy the requirements.")
    

def test_custom_bohb_config():
    """Test custom BOHB configuration with different parameters."""
    
    # Create a custom configuration with different parameters
    custom_config = BOHBConfig(
        max_budget=50.0,
        min_budget=5.0,
        n_trials=100,
        wall_clock_limit=300.0,
    )
    
    # Create optimizer with the custom config
    optimizer = BOHBOptimizer("simple", config=custom_config)
    
    # Calculate budgets for both fidelity modes
    low_budget = optimizer._calculate_budget("low")
    high_budget = optimizer._calculate_budget("high")
    
    # Print the results
    print("\n--- Custom BOHB Budget Parameters ---\n")
    
    headers = ["Fidelity", "min_budget", "max_budget", "n_trials", "time_limit"]
    row_format = "{:<10} {:<12} {:<12} {:<10} {:<12}"
    
    print(row_format.format(*headers))
    print("-" * 60)
    
    print(row_format.format("Low", 
                          low_budget["min_budget"], 
                          low_budget["max_budget"],
                          low_budget["n_trials"],
                          low_budget["time_limit"]))
    
    print(row_format.format("High", 
                          high_budget["min_budget"], 
                          high_budget["max_budget"],
                          high_budget["n_trials"],
                          high_budget["time_limit"]))
    
    # Verify that the budget parameters satisfy the requirements
    for budget in [low_budget, high_budget]:
        assert budget["min_budget"] > 0, "min_budget must be positive"
        assert budget["max_budget"] > budget["min_budget"], "max_budget must be > min_budget"
        assert budget["n_trials"] > budget["max_budget"], "n_trials must be > max_budget (SMAC requirement)"
        assert budget["time_limit"] > 0, "time_limit must be positive"
    
    print("\nAll custom budget parameters satisfy the requirements.")


if __name__ == "__main__":
    test_bohb_budget_calculation()
    test_custom_bohb_config()
