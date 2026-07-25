# HIGGS Boson Classification

A machine learning pipeline for detecting Higgs bosons from particle collider data using ensemble methods and gradient boosting.

## Overview

This project implements a complete classification workflow to distinguish between signal processes that produce Higgs bosons and background processes that do not. The dataset comes from the ATLAS experiment at the Large Hadron Collider (LHC), where machine learning techniques are crucial for identifying rare particle events amidst massive background noise.

## What This Project Does

- **Data Processing**: Loads and preprocesses 500K samples from the HIGGS dataset with 28 kinematic features
- **Exploratory Analysis**: Generates comprehensive visualizations including correlation heatmaps, distribution plots, and pair plots
- **Model Training**: Trains 12+ classifiers with hyperparameter optimization:
  - XGBoost (GPU-accelerated)
  - LightGBM
  - CatBoost
  - Random Forest
  - Neural Network (MLP)
  - Support Vector Machine
  - Logistic Regression
  - Decision Trees
  - K-Nearest Neighbors
  - AdaBoost
  - Naive Bayes
  - Ensemble Methods (Voting/Stacking)
- **Evaluation**: Produces classification reports, confusion matrices, and performance metrics
- **Model Persistence**: Saves trained models as serialized pickle files for inference

## Dataset

The HIGGS dataset was produced using Monte Carlo simulations and contains:
- **11 million total examples** (we use a 500K subset for training)
- **28 features**: 21 low-level kinematic properties + 7 high-level derived features
- **Binary classification**: Signal (1) vs Background (0)

**Features include**: lepton pT, lepton eta, lepton phi, missing energy magnitude, jet properties (pt, eta, phi, b-tag), and invariant masses (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA (optional, for XGBoost acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd HIGGS-UCI-DATASET

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost
- joblib

## Usage

### Quick Start

Run the complete pipeline (data loading, EDA, and model training):

```bash
python run.py
```

Or run individual components:

```bash
# Run EDA analysis only
python EDA/eda_analysis.py

# Run data preprocessing
python Helpers/data_extract_and_preprocessing.py
```

### Project Structure

```
HIGGS-UCI-DATASET/
├── Dataset/                 # Raw data files
│   └── HIGGS_500K.csv
├── Definitions/             # Configuration constants
│   └── constants.py
├── EDA/                     # Exploratory data analysis
│   └── eda_analysis.py
├── EDA Plots/              # Generated visualizations
├── Helpers/                # Utility modules
│   ├── IOdata.py          # Data I/O and train/test split
│   ├── data_extract_and_preprocessing.py
│   └── plotters.py        # Visualization functions
├── Model/                  # Model implementations
│   ├── xgb_boost.py
│   ├── lightgbm.py
│   ├── catboost.py
│   ├── Random_forest.py
│   ├── neural_network.py
│   ├── svm.py
│   ├── logistic_regression.py
│   ├── Decision_tree.py
│   ├── knn.py
│   ├── ada_boost.py
│   ├── naive_bayes.py
│   └── ensemble.py
├── ML_Model/              # Saved trained models
│   ├── XGBoost/
│   ├── LGB/
│   ├── CB/
│   ├── RF/
│   ├── NN/
│   ├── SVM/
│   ├── LR/
│   ├── DT/
│   ├── KNN/
│   ├── ADA/
│   ├── NB/
│   └── ENS/
├── Plots/                 # Model evaluation plots
│   ├── XG/
│   ├── LGB/
│   ├── CB/
│   ├── RF/
│   ├── NN/
│   ├── SVM/
│   ├── LR/
│   ├── DT/
│   ├── KN/
│   ├── ADA/
│   ├── NB/
│   └── ENS/
├── Results/               # Training results and metrics
├── app.py                 # Main application orchestrator
├── run.py                 # Entry point
└── requirements.txt
```

## Model Details

### XGBoost
- GPU-accelerated training with CUDA
- Hyperparameter tuning via GridSearchCV
- Parameters optimized: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

### LightGBM
- Fast gradient boosting framework
- Efficient handling of large datasets
- Leaf-wise tree growth strategy

### CatBoost
- Gradient boosting with ordered boosting
- Handles categorical features natively
- Reduces overfitting with robust boosting

### Random Forest
- Ensemble of decision trees
- Configurable number of estimators and max depth
- Bootstrap aggregation for variance reduction

### Neural Network (MLP)
- Multi-layer perceptron architecture
- Automatic feature learning
- Configurable hidden layers and activation functions

### Support Vector Machine
- Maximum margin classifier
- Multiple kernel options (linear, RBF, polynomial)
- Effective in high-dimensional spaces

### Logistic Regression
- Linear baseline classifier
- Regularization parameters tuned
- Interpretable feature coefficients

### Decision Tree
- Interpretable single-tree model
- Depth and split criteria optimized
- Feature importance extraction

### K-Nearest Neighbors
- Instance-based learning algorithm
- Multiple distance metrics available
- Configurable neighbor count and weighting

### AdaBoost
- Adaptive boosting algorithm
- Combines weak learners into strong classifier
- Handles imbalanced data effectively

### Naive Bayes
- Probabilistic classifier based on Bayes theorem
- Fast training and prediction
- Assumes feature independence

### Ensemble Methods
- Voting Classifier: Combines predictions from multiple models
- Stacking Classifier: Uses meta-learner to optimize base model combinations
- Often achieves superior performance through diversity

## Results

After training, you'll find:
- **Confusion matrices** in `Plots/` subdirectories
- **Trained models** in `ML_Model/` as `.pkl` files
- **Performance metrics** printed to console (accuracy, precision, recall, F1-score)

## Citation

**Dataset Source**: Daniel Whiteson (daniel@uci.edu), Assistant Professor, Physics & Astronomy, University of California Irvine

**Reference Paper**:  
Baldi, P., P. Sadowski, and D. Whiteson. "Searching for Exotic Particles in High-energy Physics with Deep Learning." Nature Communications 5 (July 2, 2014).

## License

This project uses the HIGGS dataset from the UCI Machine Learning Repository. Please refer to the original source for dataset licensing terms.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for:
- Additional model implementations
- Feature engineering improvements
- Visualization enhancements
- Performance optimizations
