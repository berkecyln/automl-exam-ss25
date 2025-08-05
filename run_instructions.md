# AutoML Text Classification - Reproduction Instructions

## Environment Setup

### Step 1: Create Virtual Environment

**Conda Environment** (Recommended)

```bash
conda create -n automl-env python=3.10
conda activate automl-env
```

**OR using venv**

```bash
# Create virtual environment
python3 -m venv automl-env

# Activate environment
# On Linux/Mac:
source automl-env/bin/activate
# On Windows:
# automl-env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install the package and all dependencies
pip install -e .
```

**Verify Installation:**
```bash
python -c "import automl; print(' AutoML package installed successfully')"
```

## Data Setup

### Automatic Data Download (Recommended)

Run the automated data setup script to download both Phase 1 and Phase 2 datasets:

```bash
python setup_data.py
```

This will automatically:
- Download Phase 1 datasets (amazon, ag_news, dbpedia, imdb)
- Download Phase 2 dataset (yelp)
- Extract and organize files in the correct structure
- Verify data integrity

**Expected Directory Structure:**
```
data/
├── yelp/           # Phase 2 (Exam dataset)
│   ├── train.csv
│   └── test.csv
├── amazon/         # Phase 1 (Training datasets)
│   ├── train.csv
│   └── test.csv
├── ag_news/
│   ├── train.csv
│   └── test.csv
├── dbpedia/
│   ├── train.csv
│   └── test.csv
└── imdb/
    ├── train.csv
    └── test.csv
```

### Manual Data Download (Alternative)

If the automatic script fails, manually download:
1. **Phase 1**: [text-phase1.zip](https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase1.zip)
2. **Phase 2**: [text-phase2.zip](https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase2.zip)

Extract both to the `data/` directory following the structure above.

## Execution Instructions

### Complete AutoML Pipeline (24 hours maximum)

**Standard Execution:**
```bash
# Run the complete AutoML pipeline
python run.py --time 24.0 --output final_submission --max_iterations 10 
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--time` | 1.0 | Maximum runtime in hours (float) |
| `--output` | test_run | Experiment name (saved in `experiments/` folder) |
| `--max_iterations` | 10 | Maximum RL training iterations |
| `--random_state` | 42 | Random seed for reproducibility |

## Pipeline Architecture

### Training Phase (Phase 1 Datasets)
1. **Meta-feature extraction** from amazon, ag_news, dbpedia, imdb datasets
2. **Leave-one-out cross-validation**:
   - For each dataset (e.g., amazon):
     - Train RL+BOHB on remaining 3 datasets  
     - Evaluate on held-out dataset
     - Save trained RL agent
3. **Result**: 4 trained RL agents, each specialized for different dataset characteristics

### Application Phase (Phase 2 Dataset)
1. **Extract meta-features** from yelp training data
2. **Apply all 4 CV agents** to yelp dataset with BOHB optimization
3. **Select best performing** agent and configuration
4. **Train final model** on yelp training data
5. **Generate predictions** for yelp test data

## Output Files

### Generated Results (after pipeline execution)
- `data/exam_dataset/predictions.npy` - **Test predictions for GitHub auto-evaluation**

### Additional Results (for analysis)
- `experiments/final_submission/run_*/final_results.pkl` - Complete pipeline results
- `experiments/final_submission/run_*/logs/` - Detailed execution logs
- `experiments/final_submission/run_*/visualizations/` - Analysis plots
- `experiments/final_submission/run_*/models/` - Trained RL agents

## Key Features

### Meta-Learning Approach
- **35+ Meta-features**: Text statistics, complexity measures, baseline performance
- **Cross-dataset Learning**: Learns patterns from multiple datasets to generalize

### RL-Based Model Selection  
- **3 Model Complexity Tiers**: Simple (TF-IDF+LR), Medium (TF-IDF+SVD+LR), Complex (TF-IDF+MLP)
- **Intelligent Selection**: RL agent chooses optimal complexity based on dataset characteristics
- **Reward Function**: Balances accuracy, complexity penalty, confidence gap, and feature richness

### BOHB Optimization
- **Multi-fidelity Search**: Efficient hyperparameter optimization with budget allocation
- **Adaptive Resource Allocation**: More time for promising configurations
- **Model-Specific Search Spaces**: Tailored hyperparameter ranges for each model type

### Interpretability & Analysis
- **LIME Explanations**: Model decision explanations for sample predictions
- **Comprehensive Visualizations**: Performance trends, meta-feature analysis, BOHB progress
- **Detailed Logging**: Complete audit trail of all decisions and evaluations



## Technical Details

### Model Implementations
- **Simple Models**: Scikit-learn TF-IDF vectorization + Logistic Regression/SVM
- **Medium Models**: TF-IDF + Truncated SVD + Logistic Regression  
- **Complex Models**: TF-IDF + Multi-layer Perceptron with early stopping

### RL Environment
- **State Space**: 35-dimensional normalized meta-features
- **Action Space**: 3 discrete actions (Simple/Medium/Complex model selection)
- **Reward**: Multi-component function balancing performance and efficiency

### BOHB Configuration
- **Multi-fidelity**: Dataset size and training iterations as budget dimensions
- **Search Spaces**: Model-specific hyperparameter ranges
- **Stopping Criteria**: Time-based and convergence-based termination


## AutoML Concepts Used

This solution incorporates the following concepts from the lecture:

- **Multi-fidelity Optimization** - BOHB with budget allocation
- **Meta-learning** - Learning from Phase 1 datasets to generalize to new datasets  
- **Algorithm Selection** - RL-based dynamic model selection
- **Hyperparameter Optimization** - Bayesian optimization with Hyperband
- **Transfer Learning** - Knowledge transfer across datasets via meta-features