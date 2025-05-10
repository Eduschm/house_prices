#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model blending script for Kaggle House Prices competition
"""

import os
import numpy as np
import pandas as pd
import joblib

def blend_predictions(test_csv='data/test.csv', output_file='blended_submission.csv'):
    """
    Create an optimally weighted blend of model predictions
    
    Parameters:
    -----------
    test_csv : str
        Path to test CSV file with IDs
    output_file : str
        Path to save the output submission file
        
    Returns:
    --------
    submission_df : DataFrame
        Submission-ready DataFrame with IDs and predictions
    """
    print("\n===== CREATING BLENDED SUBMISSION =====")
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("Error: No models directory found. Train models first.")
        return None
    
    # Check if test data exists
    if not os.path.exists(test_csv):
        print(f"Error: Test file {test_csv} not found.")
        return None
    
    # Load test IDs from original file
    test_df = pd.read_csv(test_csv)
    ids = test_df['Id'].values
    
    # Check if preprocessed test data exists
    if not os.path.exists('data/preprocessed/X_test.npy'):
        print("Error: Preprocessed test data not found. Run preprocessing first.")
        return None
    
    # Load preprocessed test data
    X_test = np.load('data/preprocessed/X_test.npy')
    
    # Load all model files
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    predictions = {}
    weights = {}
    
    # Define default model weights - adjust based on cross-validation scores
    default_weights = {
        'XGBRegressor_best': 0.30,
        'CatBoost_best': 0.25,
        'LightGBM_best': 0.15,
        'GradientBoosting_best': 0.10,
        'RandomForest_best': 0.05,
        'Stacking_best': 0.40,  # Give higher weight to stacking
    }
    
    # Make predictions with each model
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        
        # Skip CV results and other non-model files
        if 'results' in model_name or 'cv' in model_name or 'base_models' in model_name:
            continue
            
        try:
            # Load the model
            model = joblib.load(f"models/{model_file}")
            
            # Make predictions
            preds = model.predict(X_test)
            
            # Store predictions and weight
            predictions[model_name] = preds
            weights[model_name] = default_weights.get(model_name, 0.1)
            
            print(f"Generated predictions from {model_name}")
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        weights = {k: v/weight_sum for k, v in weights.items()}
    
    # Check if we have any valid predictions
    if not predictions:
        print("Error: No valid predictions generated.")
        return None
    
    # Create weighted blend
    blend = np.zeros(len(ids))
    
    for model_name, preds in predictions.items():
        # Convert log predictions back if needed
        if np.max(preds) < 20:  # Heuristic to detect log-transformed values
            print(f"Converting {model_name} predictions from log scale")
            preds = np.expm1(preds)
        
        # Add weighted predictions
        blend += weights[model_name] * preds
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'Id': ids,
        'SalePrice': blend
    })
    
    # Clean up predictions
    # Ensure predictions are positive and reasonable
    submission['SalePrice'] = submission['SalePrice'].clip(lower=10000, upper=2000000)
    
    # Save submission
    submission.to_csv(output_file, index=False)
    print(f"Blended submission saved to {output_file}")
    
    # Print model contributions
    print("\nModel weights in final blend:")
    for model_name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"{model_name}: {weight:.4f}")
    
    return submission

if __name__ == "__main__":
    blend_predictions()