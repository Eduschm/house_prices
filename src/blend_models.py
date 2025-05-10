import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing import preprocess_data, engineer_features

def load_models(model_dir='models'):
    """
    Load all trained models from the specified directory
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model files
        
    Returns:
    --------
    models : dict
        Dictionary of model name -> model object
    """
    models = {}
    
    for file in os.listdir(model_dir):
        if file.endswith('_best.pkl'):
            model_name = os.path.basename(file)
            try:
                model = joblib.load(os.path.join(model_dir, file))
                models[model_name] = model
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
    
    return models

def generate_predictions(model, X_test, model_name):
    """
    Generate predictions with error handling and feature compatibility
    
    Parameters:
    -----------
    model : trained model object
        The model to use for prediction
    X_test : ndarray
        Test features
    model_name : str
        Name of the model for logging
        
    Returns:
    --------
    preds : ndarray
        Log-scale predictions
    """
    # Get expected feature count for this model
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
    elif hasattr(model, 'feature_importances_'):
        expected_features = len(model.feature_importances_)
    else:
        # For StackingRegressor or other complex models
        try:
            expected_features = model.estimators_[0].n_features_in_
        except:
            expected_features = X_test.shape[1]  # Assume it's correct
            
    print(f"Model {model_name} expects {expected_features} features, got {X_test.shape[1]}")
    
    # Adjust features if needed
    if X_test.shape[1] != expected_features:
        if X_test.shape[1] > expected_features:
            print(f"Trimming features for {model_name}: {X_test.shape[1]} -> {expected_features}")
            X_test_adjusted = X_test[:, :expected_features]
        else:
            padding = expected_features - X_test.shape[1]
            print(f"Padding features for {model_name}: {X_test.shape[1]} -> {expected_features} (+{padding})")
            X_test_adjusted = np.pad(
                X_test,
                ((0, 0), (0, padding)),
                mode='constant',
                constant_values=0
            )
    else:
        X_test_adjusted = X_test
        
    try:
        print(f"Generating predictions from {model_name}")
        preds = model.predict(X_test_adjusted)
        
        # Check if predictions seem to be in log scale
        if np.max(preds) < 20:  # Heuristic to detect log-transformed values
            print(f"Converting {model_name} predictions from log scale")
            preds = np.expm1(preds)
        
        return preds
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return None

def blend_predictions(test_csv_path, weights=None):
    """
    Blend predictions from multiple models
    
    Parameters:
    -----------
    test_csv_path : str
        Path to the test CSV file
    weights : dict, optional
        Dictionary of model name -> weight
        If None, use equal weights for all models
        
    Returns:
    --------
    submission : DataFrame
        DataFrame with Id and SalePrice columns
    """
    # Load and preprocess test data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv(test_csv_path)
    ids = test_df["Id"].copy()
    
    # Clean and preprocess data
    _, _, X_test_transformed, _ = preprocess_data(train_df, test_df)
    
    # Load models
    models = load_models()
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            'XGBRegressor_best.pkl': 0.40,
            'CatBoost_best.pkl': 0.25,
            'RandomForest_best.pkl': 0.15,
            'GradientBoosting_best.pkl': 0.15,
            'LightGBM_best.pkl': 0.05
        }
        
    # Generate predictions
    predictions = {}
    valid_predictions = {}
    
    for model_name, model in models.items():
        preds = generate_predictions(model, X_test_transformed, model_name)
        if preds is not None:
            predictions[model_name] = preds
            # Only include models with weights in the blending
            if model_name in weights:
                valid_predictions[model_name] = preds
    
    # Normalize weights for valid models
    total_weight = sum(weights.get(model, 0) for model in valid_predictions.keys())
    if total_weight == 0:
        # If no weights are valid, use equal weights
        norm_weights = {model: 1.0 / len(valid_predictions) for model in valid_predictions}
    else:
        norm_weights = {model: weights.get(model, 0) / total_weight 
                        for model in valid_predictions}
    
    # Blend predictions
    blend = np.zeros(len(ids))
    for model_name, preds in valid_predictions.items():
        weight = norm_weights.get(model_name, 0)
        if weight > 0:
            blend += weight * preds
            print(f"Added {model_name} with weight {weight:.2f}")
    
    # Create submission DataFrame
    submission = pd.DataFrame({"Id": ids, "SalePrice": blend})
    submission_path = Path("data/blended_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path} ({len(submission)} rows)")
    
    return submission