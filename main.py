
import pandas as pd
import numpy as np
import os
import joblib
import argparse
import warnings
import zipfile
from sklearn.model_selection import train_test_split

# Try to import Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Kaggle API not available. If you need to download data, install it with: pip install kaggle")

from src.preprocessing import preprocess_data
from src.train import train
from src.predict import predict
# Import blend_predictions if available
try:
    from src.blend_models import blend_predictions
    BLEND_AVAILABLE = True
except ImportError:
    BLEND_AVAILABLE = False
    print("Blend functionality not available. Create src/blend_models.py to enable blending.")

def download_data():
    """Download dataset from Kaggle if not already present"""
    # Check if data already exists
    if os.path.exists('data/train.csv') and os.path.exists('data/test.csv'):
        print("Data files already exist. Skipping download.")
        return True
        
    if not KAGGLE_AVAILABLE:
        print("ERROR: Kaggle API not available. Please install it with: pip install kaggle")
        print("Make sure your Kaggle API credentials are configured.")
        return False
        
    try:
        print("Downloading data from Kaggle...")
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Authenticate with Kaggle
        api = KaggleApi()
        api.authenticate()
        
        # Download competition files
        api.competition_download_files(
            'house-prices-advanced-regression-techniques',
            path='data/'
        )
        
        # Unzip the downloaded file
        with zipfile.ZipFile('data/house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
            zip_ref.extractall('data/')
            
        print("Data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please download the data manually and place train.csv and test.csv in the data/ directory.")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train or predict using ML model")
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'predict', 'blend', 'all'], 
        required=True, 
        help='Select operation mode: train, predict, blend, or all'
    )
    
    parser.add_argument(
        '--use_cached', 
        action='store_true',
        help='Use cached preprocessed data if available'
    )
    
    parser.add_argument(
        '--skip_download', 
        action='store_true',
        help='Skip data download from Kaggle (assume data is already present)'
    )
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/preprocessed', exist_ok=True)
    
    # Download data from Kaggle if needed
    if not args.skip_download:
        if not download_data():
            print("Unable to download data. Exiting.")
            return
    
    # Check if preprocessed data already exists and use_cached is True
    preprocessed_exists = (
        os.path.exists('data/preprocessed/X_train.npy') and
        os.path.exists('data/preprocessed/y_train.npy') and
        os.path.exists('data/preprocessed/X_test.npy') and
        os.path.exists('data/preprocessed/y_test.npy') and
        os.path.exists('data/preprocessed/feature_names.joblib')
    )
    
    if args.use_cached and preprocessed_exists:
        print("Loading cached preprocessed data...")
        X_train = np.load('data/preprocessed/X_train.npy')
        y_train = np.load('data/preprocessed/y_train.npy')
        X_test = np.load('data/preprocessed/X_test.npy')
        y_test = np.load('data/preprocessed/y_test.npy')
        feature_names = joblib.load('data/preprocessed/feature_names.joblib')
    else:
        print("Loading and preprocessing data...")
        # Load raw data
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        
        # If test_df doesn't have SalePrice, use train_test_split
        if 'SalePrice' not in test_df.columns:
            # Create a split from training data
            train_df, holdout_df = train_test_split(
                train_df, test_size=0.2, random_state=42
            )
            # Preprocess both sets of data
            X_train, y_train, X_test, feature_names = preprocess_data(train_df, holdout_df)
            y_test = np.log1p(holdout_df['SalePrice']).values
        else:
            # If test_df has SalePrice, preprocess both
            X_train, y_train, X_test, feature_names = preprocess_data(train_df, test_df)
            y_test = np.log1p(test_df['SalePrice']).values
        
        # Save preprocessed data for future use
        np.save('data/preprocessed/X_train.npy', X_train)
        np.save('data/preprocessed/y_train.npy', y_train)
        np.save('data/preprocessed/X_test.npy', X_test)
        np.save('data/preprocessed/y_test.npy', y_test)
        joblib.dump(feature_names, 'data/preprocessed/feature_names.joblib')
    
    # Execute requested mode
    if args.mode in ['train', 'all']:
        print("\n===== TRAINING MODELS =====")
        models, cv_results = train(X_train, y_train, use_preprocessed=True)
        
        # Save cross-validation results
        joblib.dump(cv_results, 'models/cv_results.joblib')
    
    if args.mode in ['predict', 'all']:
        print("\n===== EVALUATING MODELS =====")
        test_results = predict(X_test, y_test)
        
        # Save evaluation results
        joblib.dump(test_results, 'models/test_results.joblib')
    
    if args.mode in ['blend', 'all']:
        if BLEND_AVAILABLE:
            print("\n===== BLENDING MODEL PREDICTIONS =====")
            submission = blend_predictions('data/test.csv')
        else:
            print("\nBlending functionality not available. Create src/blend_models.py to enable blending.")
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    # Filter some warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()