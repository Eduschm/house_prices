import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
)

def predict(X_test, y_test, X_test_preprocessed=None):
    """
    Evaluate model performance on test data
    
    Parameters:
    -----------
    X_test : DataFrame
        Raw test features
    y_test : Series or ndarray
        True target values
    X_test_preprocessed : ndarray, optional
        Preprocessed test features (if None, X_test is assumed to be preprocessed)
        
    Returns:
    --------
    results : dict
        Dictionary of model results with various evaluation metrics
    """
    # If no preprocessed data is provided, assume X_test is already preprocessed
    test_data = X_test_preprocessed if X_test_preprocessed is not None else X_test
    
    # Ensure y_test has the correct form (assuming SalePrice was log-transformed)
    y_test_original = np.expm1(y_test) if np.max(y_test) < 20 else y_test
    
    results = {}
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("No models directory found. Please train models first.")
        return results
    
    # For each model file in the models directory
    for file in os.listdir('models'):
        if not file.endswith('.pkl'):
            continue
            
        model_name = file.replace('.pkl', '')
        print(f"Evaluating {model_name}...")
        
        # Load model
        try:
            model = joblib.load(f"models/{file}")
        except Exception as e:
            print(f"Error loading model {file}: {e}")
            continue
        
        # Predict with trained model
        try:
            y_pred = model.predict(test_data)
            
            # If predictions are log-transformed, convert back
            if np.max(y_pred) < 20:  # Heuristic to detect log-transformed values
                y_pred = np.expm1(y_pred)
            
            # Check for NaN or Inf values
            print(f"NaN in y_test: {np.isnan(y_test_original).sum()}")
            print(f"NaN in y_pred: {np.isnan(y_pred).sum()}")
            print(f"Inf in y_test: {np.isinf(y_test_original).sum()}")
            print(f"Inf in y_pred: {np.isinf(y_pred).sum()}")
            
            # Calculate metrics
            mse = mean_squared_error(y_test_original, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            
            # MSLE can fail if there are negative values
            try:
                msle = mean_squared_log_error(y_test_original, y_pred)
            except ValueError:
                msle = None
                print(f"Could not calculate MSLE for {model_name} (negative values)")
            
            # Store results
            results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MSLE': msle
            }
            
            # Print metrics
            print(f"RMSE: {rmse:.2f}")
            print(f"R²: {r2:.4f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error predicting with {file}: {e}")
            continue
    
    # Print summary of all models
    print("\n===== MODEL COMPARISON =====")
    for model_name, metrics in results.items():
        print(f"{model_name}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
    
    return results