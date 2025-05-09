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
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
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
            estimators=[
                ('xgb', XGBRegressor(random_state=42)),
                ('rf', RandomForestRegressor(random_state=42)),
                ('lgbm', LGBMRegressor(random_state=42)),
                ('gb', GradientBoostingRegressor(random_state=42))
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
    }
    
    # Parameter grids for GridSearchCV
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 20, 40],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 6],
            'min_samples_leaf': [1, 4]
        },
        'LightGBM': {
            'n_estimators': [100, 300],       # Reduced options
            'learning_rate': [0.05, 0.15],    # Reduced options
            'num_leaves': [31, 51],           # Reduced options
            'max_depth': [-1, 10],            # Reduced options
            'min_child_samples': [20, 40]     # Reduced options
        },
        'GradientBoosting': {
            'n_estimators': [100, 300],
            'learning_rate': [0.05, 0.15],
            'max_depth': [3, 5],
            'subsample': [0.7, 0.9],
            'min_samples_split': [2, 6]
        },
        'CatBoost': {
            'iterations': [100, 300],
            'learning_rate': [0.05, 0.15],
            'depth': [4, 8],
            'l2_leaf_reg': [1, 7]
        },
        'XGBRegressor': {
            'n_estimators': [400, 600],
            'max_depth': [4, 6],
            'learning_rate': [0.08, 0.12],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'gamma': [0, 0.2],
            'min_child_weight': [1, 3]
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
            # Use RandomizedSearchCV for LightGBM to speed up the process
           
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=kf,
                verbose=1,
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