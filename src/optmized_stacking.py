#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized stacking implementation for house price prediction
This file should be placed in the src/ directory
"""

def get_optimized_stacking(best_models=None):
    """
    Create a stacking regressor with hyperparameter-optimized base estimators.
    If best_models is provided, uses already-tuned models from grid search.
    Otherwise, uses pre-tuned estimators based on competition best practices.
    
    Parameters:
    -----------
    best_models : dict, optional
        Dictionary of already-tuned models from grid search
        
    Returns:
    --------
    stacking : StackingRegressor
        Optimized stacking regressor with properly configured estimators
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge, ElasticNet, Lasso
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    
    # If we have already tuned models, use them
    if best_models and isinstance(best_models, dict):
        estimators = []
        
        # Models to include in stacking (exclude any existing stacking models)
        model_keys = [k for k in best_models.keys() if k != 'Stacking']
        
        # Use the best tuned models as estimators
        for name in model_keys:
            if name in best_models:
                estimators.append((name.lower(), best_models[name]))
        
        # If we don't have enough models, add defaults
        if len(estimators) < 3:
            print("Warning: Not enough tuned models found. Adding default models to stacking.")
            
            # Add default models that might not be in best_models
            if not any(name.lower().startswith('xgb') for name, _ in estimators):
                estimators.append(('xgb', XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.01,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=1,
                    reg_alpha=0.01,
                    reg_lambda=0.1,
                    n_jobs=-1,
                    random_state=42
                )))
            
            if not any(name.lower().startswith('lgbm') for name, _ in estimators):
                estimators.append(('lgbm', LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.01,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    n_jobs=-1,
                    force_col_wise=True,
                    random_state=42
                )))
                
            if not any(name.lower().startswith('rf') for name, _ in estimators):
                estimators.append(('rf', RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    max_features='sqrt',
                    min_samples_split=2,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42
                )))
    
    # If no tuned models provided, create optimized estimators from scratch
    else:
        print("Creating stacking ensemble with pre-tuned estimators")
        estimators = [
            # XGBoost - tuned for regression problems
            ('xgb', XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                reg_alpha=0.01,
                reg_lambda=0.1,
                n_jobs=-1,
                random_state=42
            )),
            
            # LightGBM - complementary to XGBoost with different algorithm
            ('lgbm', LGBMRegressor(
                n_estimators=500,
                learning_rate=0.01,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.01,
                reg_lambda=0.01,
                n_jobs=-1,
                force_col_wise=True,
                random_state=42
            )),
            
            # CatBoost - handles categorical features well
            ('cat', CatBoostRegressor(
                iterations=500,
                learning_rate=0.01,
                depth=6,
                l2_leaf_reg=3,
                random_state=42,
                verbose=0,
                allow_writing_files=False
            )),
            
            # Random Forest - different algorithm family for diversity
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                max_features='sqrt',
                min_samples_split=2,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )),
            
            # Gradient Boosting - sklearn implementation for diversity
            ('gbm', GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=4,
                subsample=0.8,
                min_samples_split=2,
                random_state=42
            ))
        ]
    
    # Use a tuned ElasticNet as meta-learner (better than Ridge for feature diversity)
    final_estimator = ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
        random_state=42,
        max_iter=10000  # Ensure convergence
    )
    
    # Create the stacking regressor with optimized parameters
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,                  # 5-fold CV is standard practice
        passthrough=True,      # Include original features for more signal
        n_jobs=-1              # Parallelize computation
    )
    
    return stacking