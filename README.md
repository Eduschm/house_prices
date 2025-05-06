# House Price Prediction

A machine learning pipeline for predicting house prices using the Kaggle House Prices: Advanced Regression Techniques dataset.

## Overview

This project implements a complete machine learning workflow for house price prediction:
- Data download and preprocessing
- Feature engineering
- Model training with hyperparameter optimization
- Model evaluation and comparison
- Prediction generation for submissions

The pipeline supports multiple machine learning models including Random Forest, XGBoost, and Stacking Ensemble techniques.

## Features

- **Automated Data Download**: Integrates with Kaggle API to fetch the competition dataset
- **Robust Preprocessing**: Handles missing values, categorical features, and feature scaling
- **Multiple Models**: Implements and compares various regression algorithms
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize model parameters
- **Cross-Validation**: Ensures reliable model performance evaluation
- **Submission Generation**: Creates properly formatted files for Kaggle submission

## Project Structure

```
.
├── data/                    # Data directory
│   ├── train.csv            # Training dataset
│   ├── test.csv             # Test dataset
│   └── preprocessed/        # Preprocessed data files
├── models/                  # Trained model files
├── main.py                  # Main script with CLI interface
├── preprocessing.py         # Data preprocessing functions
├── train.py                 # Model training functions
├── predict.py               # Prediction and evaluation functions
└── comp_results.py          # Competition submission generator
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- kaggle (optional, for data download)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Configure Kaggle API:
   - Install the Kaggle package: `pip install kaggle`
   - Download your Kaggle API token from your account page
   - Place the `kaggle.json` file in `~/.kaggle/` directory

## Usage

### Basic Usage

```bash
# Download data and train models
python main.py --mode train --skip_download

# Generate predictions with trained models
python main.py --mode predict --use_cached

# Run both training and prediction
python main.py --mode both
```

### Command Line Arguments

- `--mode`: Operation mode (`train`, `predict`, or `both`)
- `--use_cached`: Use cached preprocessed data if available
- `--skip_download`: Skip data download from Kaggle

### Competition Submission

```bash
# After training models, generate a submission file
python comp_results.py
```

## Model Pipeline

1. **Data Preprocessing**
   - Feature selection and dropping problematic columns
   - Missing value imputation
   - Categorical encoding
   - Feature standardization

2. **Model Training**
   - XGBoost Regressor
   - Random Forest Regressor
   - Stacking Ensemble (XGBoost + Random Forest with Ridge meta-learner)

3. **Evaluation Metrics**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² (Coefficient of Determination)
   - MSLE (Mean Squared Logarithmic Error)

## Adding New Models

To add a new model to the pipeline:

1. Add the model configuration to the `models` dictionary in `train.py`
2. Add a parameter grid to `param_grids` for hyperparameter tuning
3. Retrain using the command: `python main.py --mode train --use_cached`

