# -*- coding: utf-8 -*-
import joblib
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


def train(X_train_data, y_train_data, use_preprocessed=True):
    """
    Train models with consistent preprocessing
    
    Parameters:
    -----------
    X_train_data : DataFrame or ndarray
        Training data features, either raw or preprocessed 
    y_train_data : Series or ndarray
        Target variable
    use_preprocessed : bool
        Whether X_train_data is already preprocessed (True) or needs preprocessing (False)
    
    Returns:
    --------
    best_models : dict
        Dictionary of fitted best models
    cv_results : dict
        Cross-validation results
    """
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Load pre-trained models into a list
    model_files = [
        "models/CatBoost_best.pkl",
        "models/GradientBoosting_best.pkl",
        "models/RandomForest_best.pkl",
        "models/XGBRegressor_best.pkl"
    ]

    loaded_models = [joblib.load(model_file) for model_file in model_files]

    # Define models
    models = {
        'XGBRegressor': XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.6,
            colsample_bytree=0.6,
            gamma=0,
            min_child_weight=1,
            objective='reg:squarederror',
            n_jobs=1,
            random_state=42
        ),
       
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=0
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            max_features='sqrt',
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        ),
        'Stacking': StackingRegressor(
            estimators=loaded_models,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
    }
    
    # Parameter grids for GridSearchCV
    param_grids = param_grids = {
        'RandomForest': {
            'n_estimators': [200, 300],              # Increase range: best value was at upper limit (200)
            'max_depth': [30, 40, 50],               # Focus around best value (40) with higher options
            'max_features': ['sqrt'],                # Keep only winner ('sqrt')
            'min_samples_split': [2],                # Keep only winner (2)
            'min_samples_leaf': [1, 2]               # Keep winner (1) and test one close value
        },
        'GradientBoosting': {
            'n_estimators': [300, 400],              # Increase range: best value was at upper limit (300)
            'learning_rate': [0.03, 0.05, 0.07],     # Explore around best value (0.05) with finer granularity
            'max_depth': [2, 3, 4],                  # Explore around best value (3)
            'subsample': [0.6, 0.7, 0.8],            # Explore around best value (0.7)
            'min_samples_split': [2]                 # Keep only winner (2)
        },
        'CatBoost': {
            'iterations': [300, 400],                # Increase range: best value was at upper limit (300)
            'learning_rate': [0.1, 0.15, 0.2],       # Explore around best value (0.15)
            'depth': [3, 4, 5],                      # Explore around best value (4)
            'l2_leaf_reg': [0.5, 1, 3]               # Explore around best value (1)
        },
        'XGBRegressor': {
            'n_estimators': [400, 500],              # Explore around best value (400)
            'max_depth': [3, 4, 5],                  # Explore around best value (4)
            'learning_rate': [0.05, 0.08, 0.1],      # Explore around best value (0.08)
            'subsample': [0.7, 0.8, 0.9],            # Explore around best value (0.8)
            'colsample_bytree': [0.5, 0.6, 0.7],     # Explore around best value (0.6)
            'gamma': [0],                            # Keep only winner (0)
            'min_child_weight': [1, 2]               # Explore around best value (1)
        },
        'Stacking': {}  # No hyperparameter tuning for stacking
    }
    # Store trained models and results
    best_models = {}
    cv_results = {}
    # Setup k-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        param_grid = param_grids.get(name, {})
        
        
        if param_grid:
           
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=kf,
                verbose=3,
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
        
            search.fit(X_train_data, y_train_data)
            final_model = search.best_estimator_
            
            cv_results[name] = {
                'best_params': search.best_params_,
                'best_score_cv': search.best_score_,
                'all_cv_results': search.cv_results_
            }
            
            print(f"Best parameters for {name}: {search.best_params_}")
            print(f"Best cross-validation score: {search.best_score_:.4f}")
        else:
            model.fit(X_train_data, y_train_data)
            final_model = model
            
            cv_results[name] = {
                'best_params': None,
                'best_score_cv': None,
                'all_cv_results': None
            }
            
            print(f"{name} was trained without hyperparameter tuning.")
        
        best_models[name] = final_model
        
        # Save the model
        joblib.dump(final_model, f"models/{name}_best.pkl")
        print("-" * 50)
    
    return best_models, cv_results