#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced training module with two-phase approach for house price prediction
This file should be placed in the src/ directory
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_advanced(X_train, y_train, test_size=0.2, use_cached=True):
    """
    Strategic two-phase training for house price prediction:
    1. Train and tune base models
    2. Use tuned base models in a stacking ensemble
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Target values
    test_size : float
        Proportion to hold out for internal validation
    use_cached : bool
        Whether to use cached models if available
        
    Returns:
    --------
    all_models : dict
        Dictionary containing all trained models
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create a validation split to evaluate stacking performance
    X_train_base, X_valid, y_train_base, y_valid = train_test_split(
        X_train, y_train, test_size=test_size, random_state=42
    )
    
    # Check if we can use cached models
    base_models_path = 'models/base_models.pkl'
    stacking_path = 'models/Stacking_best.pkl'
    
    all_models = {}
    
    # PHASE 1: Train base models
    if use_cached and os.path.exists(base_models_path):
        print("Loading cached base models...")
        all_models = joblib.load(base_models_path)
    else:
        print("\n===== PHASE 1: TRAINING BASE MODELS =====")
        # Import train function
        from src.train import train
        
        # Train base models
        base_models, cv_results = train(X_train_base, y_train_base, use_preprocessed=True)
        
        # Save base models for later use
        all_models = base_models
        joblib.dump(all_models, base_models_path)
    
    # PHASE 2: Create and train stacking ensemble
    if use_cached and os.path.exists(stacking_path):
        print("Loading cached stacking model...")
        all_models['Stacking'] = joblib.load(stacking_path)
    else:
        print("\n===== PHASE 2: TRAINING STACKING ENSEMBLE =====")
        
        # Import the stacking function
        from src.optimized_stacking import get_optimized_stacking
        
        # Create stacking model with tuned base models
        stacking_model = get_optimized_stacking(all_models)
        
        print("Training stacking ensemble on full dataset...")
        # Train stacking on the full dataset
        stacking_model.fit(X_train, y_train)
        
        # Add to models dictionary
        all_models['Stacking'] = stacking_model
        
        # Save stacking model
        joblib.dump(stacking_model, stacking_path)
        
        # Evaluate stacking on validation set
        y_pred = stacking_model.predict(X_valid)
        
        # If predictions were log-transformed
        if np.max(y_pred) < 20:  # Heuristic to detect log transform
            y_pred_original = np.expm1(y_pred)
            y_valid_original = np.expm1(y_valid)
            
            mse = mean_squared_error(y_valid_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_valid_original, y_pred_original)
        else:
            mse = mean_squared_error(y_valid, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_valid, y_pred)
        
        print(f"Stacking Ensemble Validation RMSE: {rmse:.2f}")
        print(f"Stacking Ensemble Validation R²: {r2:.4f}")
    
    print("\n===== ALL MODELS READY =====")
    print(f"Total models trained: {len(all_models)}")
    
    return all_models