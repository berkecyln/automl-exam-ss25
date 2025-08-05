# AutoML Text Classification - Reproduction Instructions

## Environment Setup

### Step 1: Create Virtual Environment
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
# Install the package and dependencies
pip install -e .
```

## Data Setup

### Download Phase 2 Dataset
1. Download the yelp dataset from the provided Phase 2 link
2. Extract and place the files as follows:
```
data/
├── yelp/
│   ├── train.csv
│   └── test.csv
├── amazon/
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

## Execution Instructions

### One-Click AutoML Pipeline (24 hours maximum)

```bash
# Run the complete AutoML pipeline
python run.py --time 24.0 --output final_submission --max_iterations 10
```

This command will:
1. **Extract meta-features** from all Phase 1 datasets (amazon, ag_news, dbpedia, imdb)
2. **Run leave-one-out cross-validation** on the 4 training datasets
3. **Train RL agents** with BOHB optimization for each CV fold
4. **Apply learned knowledge** to the yelp exam dataset
5. **Generate final predictions** and save to required locations

### Quick Test Run (6 minutes)
```bash
# Short test to verify everything works
python run.py --time 0.1 --output test_run --max_iterations 3
```

## Output Files

After successful execution, the following files will be created:

### Required Submission Files
- `data/exam_dataset/predictions.npy` 
### Additional Results
- `experiments/final_submission/run_*/` - Detailed pipeline results
- `experiments/final_submission/run_*/logs/` - Execution logs
- `experiments/final_submission/run_*/visualizations/` - Analysis plots
- `experiments/final_submission/run_*/final_results.pkl` - Complete results

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

## Key Features

- **Meta-learning approach**: Learns from multiple datasets to generalize to new ones
- **RL-based model selection**: Automatically chooses between Simple/Medium/Complex models
- **BOHB optimization**: Multi-fidelity hyperparameter optimization
- **Interpretability**: LIME explanations and feature importance analysis
- **Time management**: Respects 24-hour budget with adaptive resource allocation

