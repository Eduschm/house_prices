# -*- coding: utf-8 -*-
import joblib
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.linear_model import Ridge, ElasticNet
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
    
    # Define models with optimized parameters
    models = {
        'XGBRegressor': XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            min_child_weight=1,
            reg_alpha=0.01,
            reg_lambda=0.1,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
            min_split_gain=0,
            reg_alpha=0.01,
            reg_lambda=0.01,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_split=2,
            random_state=42
        ),
        'CatBoost': CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_state=42,
            verbose=0,
            allow_writing_files=False
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=40,
            max_features='sqrt',
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )
        # Note: Stacking moved to separate optimized_stacking.py module
    }
    
    # Parameter grids for hyperparameter tuning
    param_grids = {
        'RandomForest': {
            'n_estimators': [200, 300],              # Increase range: best value was at upper limit (200)
            'max_depth': [30, 40, 50],               # Focus around best value (40) with higher options
            'max_features': ['sqrt'],                # Keep only winner ('sqrt')
            'min_samples_split': [2],                # Keep only winner (2)
            'min_samples_leaf': [1, 2]               # Keep winner (1) and test one close value
        },
        'LightGBM': {
            'n_estimators': [100, 200],              # Conservative number of trees to avoid loops
            'learning_rate': [0.05, 0.1],            # Standard rates that work well
            'num_leaves': [15, 31],                  # 31 is default, 15 is more conservative
            'min_child_samples': [5, 20],            # 20 is default, 5 allows more splits
            'subsample': [0.7, 0.9],                 # Subsampling for robustness
            'colsample_bytree': [0.7, 0.9],          # Feature subsampling
            'reg_alpha': [0.1],                      # L1 regularization to prevent overfitting
            'reg_lambda': [0.1],                     # L2 regularization to prevent overfitting
            'min_split_gain': [0]                    # Allow splits with minimal gain
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
        }
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
        
        # Special case for LightGBM to avoid infinite loop issues
        if name == 'LightGBM':
            print(f"Training {name} with RandomizedSearchCV to avoid potential issues...")
            
            # Use RandomizedSearchCV instead of GridSearchCV
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=6,               # Try only 6 combinations instead of all
                cv=kf,
                verbose=1,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                random_state=42
            )
            
            try:
                # Train with early stopping if possible
                import lightgbm as lgb
                from sklearn.model_selection import train_test_split
                
                # Create a small validation set
                X_train_lgbm, X_valid_lgbm, y_train_lgbm, y_valid_lgbm = train_test_split(
                    X_train_data, y_train_data, test_size=0.2, random_state=42
                )
                
                # Train with early stopping
                search.fit(
                    X_train_lgbm, 
                    y_train_lgbm,
                    eval_set=[(X_valid_lgbm, y_valid_lgbm)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50)],
                    eval_metric='rmse'
                )
                
                final_model = search.best_estimator_
                
                cv_results[name] = {
                    'best_params': search.best_params_,
                    'best_score_cv': search.best_score_,
                    'all_cv_results': search.cv_results_
                }
                
                print(f"Best parameters for {name}: {search.best_params_}")
                print(f"Best cross-validation score: {search.best_score_:.4f}")
                
            except Exception as e:
                print(f"LightGBM training failed with error: {str(e)}")
                print("Falling back to default LightGBM model...")
                
                # Create a simpler model as fallback
                final_model = LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=15,
                    max_depth=5,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                final_model.fit(X_train_data, y_train_data)
                
                cv_results[name] = {
                    'best_params': None,
                    'best_score_cv': None,
                    'all_cv_results': None
                }
                
                print("Fallback LightGBM model trained successfully.")
        
        # Standard training for other models
        elif param_grid:
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