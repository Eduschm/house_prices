# House Price Prediction

A machine learning pipeline for predicting house prices using the Kaggle House Prices: Advanced Regression Techniques dataset, featuring advanced stacking ensemble techniques.

## Overview

This project implements a comprehensive machine learning workflow for house price prediction:
- Data download and preprocessing
- Feature engineering with advanced transformations
- Two-phase model training with hyperparameter optimization
- Stacking ensemble for improved performance
- Model evaluation and comparison
- Prediction generation for Kaggle submissions

The pipeline supports multiple state-of-the-art regression models including XGBoost, LightGBM, CatBoost, and Stacking Ensembles.

## Features

- **Automated Data Download**: Integrates with Kaggle API to fetch the competition dataset
- **Robust Preprocessing**: Handles missing values with KNN imputation, categorical features, and feature scaling
- **Advanced Feature Engineering**: Creates interaction terms and domain-specific features
- **Multiple Models**: Implements and compares various regression algorithms
- **Two-Phase Training**: First optimizes base models, then creates a stacking ensemble
- **Cross-Validation**: Ensures reliable model performance evaluation
- **Model Blending**: Combines predictions from multiple models for improved performance
- **Submission Generation**: Creates properly formatted files for Kaggle submission

## Project Structure

```
.
├── data/                    # Data directory
│   ├── train.csv            # Training dataset
│   ├── test.csv             # Test dataset
│   ├── blended_submission.csv # Final submission file
│   └── preprocessed/        # Preprocessed data files
├── models/                  # Trained model files
├── main.py                  # Main script with CLI interface
├── src/                     # Source code
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── train.py             # Model training functions
│   ├── predict.py           # Prediction and evaluation functions
│   ├── advanced_training.py # Two-phase training approach
│   ├── optimized_stacking.py # Stacking ensemble implementation
│   └── blend_models.py      # Model blending functionality
└── comp_results.py          # Competition submission generator
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- joblib
- kaggle (optional, for data download)

See `requirements.txt` for the complete list of dependencies.

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
# Download data and train models with advanced approach
python main.py --mode train --skip_download

# Generate predictions with trained models
python main.py --mode predict --use_cached

# Run both training and prediction
python main.py --mode all --use_cached

# Generate a blended submission from multiple models
python main.py --mode blend --use_cached
```

### Command Line Arguments

- `--mode`: Operation mode (`train`, `predict`, `blend`, or `all`)
- `--use_cached`: Use cached preprocessed data and models if available
- `--skip_download`: Skip data download from Kaggle

### Competition Submission

```bash
# After training models, generate a submission file
python comp_results.py
```

## Model Pipeline

1. **Data Preprocessing**
   - Feature selection and dropping problematic columns
   - Missing value imputation using KNN for numeric features
   - Categorical encoding with OneHotEncoder
   - Feature standardization

2. **Feature Engineering**
   - Creating total square footage features
   - Adding quality-related interaction terms
   - Transforming skewed numeric features
   - Creating domain-specific features like house age and remodeling indicators

3. **Two-Phase Training**
   - Phase 1: Train and optimize individual base models
   - Phase 2: Create a stacking ensemble using the optimized base models

4. **Model Training**
   - XGBoost Regressor
   - LightGBM Regressor
   - CatBoost Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - Stacking Ensemble with ElasticNet meta-learner

5. **Evaluation Metrics**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² (Coefficient of Determination)
   - MSLE (Mean Squared Logarithmic Error)

## Performance

The model achieves an RMSE of approximately 0.22 on the Kaggle competition leaderboard, placing it in a solid position among the ~5,400 teams. The stacking ensemble typically provides the best performance among all models.

## Adding New Models

To add a new model to the pipeline:

1. Add the model configuration to the `models` dictionary in `src/train.py`
2. Add a parameter grid to `param_grids` for hyperparameter tuning
3. Update the stacking ensemble in `src/optimized_stacking.py` to include the new model
4. Retrain using the command: `python main.py --mode train --use_cached`

## Customization

The pipeline can be customized in several ways:

- Modify feature engineering in `src/preprocessing.py`
- Adjust hyperparameter search spaces in `src/train.py`
- Update stacking configuration in `src/optimized_stacking.py`
- Change model blending weights in `src/blend_models.py`