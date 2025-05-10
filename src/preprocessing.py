import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

def engineer_features(df):
    """
    Add high-value feature engineering while preserving original features
    """
    # Create a copy to avoid modifying the original
    df_eng = df.copy()
    
    # Total square footage (consistently a top feature)
    if all(col in df_eng.columns for col in ['1stFlrSF', '2ndFlrSF']):
        df_eng['TotalSF'] = df_eng['1stFlrSF'] + df_eng['2ndFlrSF']
        # Add basement area if available
        if 'TotalBsmtSF' in df_eng.columns:
            df_eng['TotalSF'] += df_eng['TotalBsmtSF']
    
    # Overall quality squared (exponential price relationship)
    if 'OverallQual' in df_eng.columns:
        df_eng['OverallQual2'] = df_eng['OverallQual'] ** 2
    
    # Quality and area interactions
    if 'OverallQual' in df_eng.columns and 'GrLivArea' in df_eng.columns:
        df_eng['QualXArea'] = df_eng['OverallQual'] * df_eng['GrLivArea']
    
    # Age features
    if all(col in df_eng.columns for col in ['YrSold', 'YearBuilt']):
        df_eng['Age'] = df_eng['YrSold'] - df_eng['YearBuilt']
        
        # Remodeled indicator (1 if house was remodeled)
        if 'YearRemodAdd' in df_eng.columns:
            df_eng['IsRemodeled'] = (df_eng['YearRemodAdd'] != df_eng['YearBuilt']).astype(int)
            df_eng['AgeRemod'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
    
    # Recently built indicator
    if all(col in df_eng.columns for col in ['YrSold', 'YearBuilt']):
        df_eng['IsNew'] = (df_eng['YrSold'] - df_eng['YearBuilt'] <= 3).astype(int)
    
    # Missing value indicators for important features
    for col in ['GarageType', 'GarageFinish', 'BsmtQual', 'BsmtExposure']:
        if col in df_eng.columns:
            df_eng[f'{col}_Missing'] = df_eng[col].isna().astype(int)
    
    # FIXED: Always create these log-transformed features for consistency
    # This ensures both train and test have the same features
    always_log_features = ['LotFrontage', 'TotalBsmtSF', 'LotArea', 'GrLivArea']
    
    for feat in always_log_features:
        if feat in df_eng.columns:
            # Handle NaN values by filling with 0 before log transform
            df_eng[f'{feat}_Log'] = np.log1p(df_eng[feat].fillna(0))
    
    # Apply log transform to other highly skewed numeric features
    numeric_feats = df_eng.dtypes[df_eng.dtypes != "object"].index
    skewed_feats = df_eng[numeric_feats].apply(lambda x: x.dropna().skew()).abs()
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    
    for feat in skewed_feats.index:
        if (feat in df_eng.columns and 
            feat not in always_log_features and  # Skip already transformed features
            feat not in ['SalePrice', 'Id', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'OverallQual']):
            # Add 1 to handle zeros
            df_eng[f'{feat}_Log'] = np.log1p(df_eng[feat].fillna(0))
    
    return df_eng

def create_preprocessor(X_train, fit_encoder=True):
    """
    Creates a consistent preprocessing pipeline that handles both training and test data identically.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training data to fit the preprocessor on
    fit_encoder : bool
        Whether to fit the encoder on X_train (True) or just create the transformer (False)
        
    Returns:
    --------
    preprocessor : ColumnTransformer
        Fitted preprocessor that can transform both training and test data
    """
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Create categorical transformer with proper missing value handling
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    # Create enhanced numerical transformer with KNN imputation
    numerical_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    if fit_encoder:
        preprocessor.fit(X_train)
        
    return preprocessor


def preprocess_data(train_df, test_df=None):
    """
    Preprocess both training and test data consistently
    
    Parameters:
    -----------
    train_df : DataFrame
        Training data with target variable
    test_df : DataFrame, optional
        Test data without target variable
        
    Returns:
    --------
    X_train, y_train, X_test : DataFrames
        Preprocessed training features, target, and test features (if test_df provided)
    feature_names : list
        Names of features after preprocessing
    """
    # Add engineered features first
    train_df_eng = engineer_features(train_df)
    test_df_eng = engineer_features(test_df) if test_df is not None else None
    
    # Initial data cleanup - same for both train and test
    def clean_df(df):
        df_clean = df.copy()
        
        # Remove outliers - but only from training data
        if 'SalePrice' in df_clean.columns and 'GrLivArea' in df_clean.columns:
            outliers_idx = df_clean[(df_clean['GrLivArea'] > 4000) & 
                                   (df_clean['SalePrice'] < 300000)].index
            df_clean = df_clean.drop(outliers_idx, errors='ignore')
        
        # Drop columns that were causing issues in your original code
        columns_to_drop = ['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
                          'MasVnrType', 'Alley']
        
        if 'Id' in df_clean.columns:
            columns_to_drop.append('Id')
            
        df_clean = df_clean.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Convert categorical columns
        cols_to_convert = ['MSSubClass', 'YrSold', 'MoSold', 'YearBuilt']
        for col in cols_to_convert:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
        
        # Drop features with high correlation - keeping the original structure
        columns_to_drop = ['GarageCars', 'GarageYrBlt']  # Keep TotalBsmtSF for the log feature
        df_clean = df_clean.drop(columns_to_drop, axis=1, errors='ignore')
        
        return df_clean
    
    # Clean training data
    train_clean = clean_df(train_df_eng)
    
    # Extract target if available
    if 'SalePrice' in train_clean.columns:
        y_train = np.log1p(train_clean['SalePrice'])  # Log transform
        X_train = train_clean.drop('SalePrice', axis=1)
    else:
        X_train = train_clean
        y_train = None
    
    # Critical fix: If test data is provided, ensure column consistency
    if test_df_eng is not None:
        test_clean = clean_df(test_df_eng)
        
        # Ensure test_clean has the same columns as X_train
        missing_cols = set(X_train.columns) - set(test_clean.columns)
        # Add missing columns
        for col in missing_cols:
            test_clean[col] = 0
        # Ensure the order of columns is the same
        test_clean = test_clean[X_train.columns]
    
    # Create and fit preprocessor on training data
    preprocessor = create_preprocessor(X_train, fit_encoder=True)
    
    # Transform training data
    X_train_transformed = preprocessor.transform(X_train)
    
    # Get feature names
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            # Extract OneHotEncoder from the Pipeline
            encoder = transformer.named_steps['encoder']
            # Get the categories
            for i, col in enumerate(columns):
                for cat in encoder.categories_[i][1:]:  # Skip first category (drop='first')
                    feature_names.append(f"{col}_{cat}")
    
    # If test data is provided, process it too
    if test_df_eng is not None:
        X_test_transformed = preprocessor.transform(test_clean)
        return X_train_transformed, y_train, X_test_transformed, feature_names
    
    return X_train_transformed, y_train, None, feature_names