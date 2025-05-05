import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

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
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    # Create numerical transformer
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
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
    # Initial data cleanup - same for both train and test
    def clean_df(df):
        df_clean = df.copy()
        
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
        
        # Drop features with high correlation
        columns_to_drop = ['GarageCars', 'TotalBsmtSF', 'GrLivArea', 'GarageYrBlt']
        df_clean = df_clean.drop(columns_to_drop, axis=1, errors='ignore')
        
        return df_clean
    
    # Clean training data
    train_clean = clean_df(train_df)
    
    # Extract target if available
    if 'SalePrice' in train_clean.columns:
        y_train = np.log1p(train_clean['SalePrice'])  # Log transform
        X_train = train_clean.drop('SalePrice', axis=1)
    else:
        X_train = train_clean
        y_train = None
    
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
    if test_df is not None:
        test_clean = clean_df(test_df)
        X_test_transformed = preprocessor.transform(test_clean)
        return X_train_transformed, y_train, X_test_transformed, feature_names
    
    return X_train_transformed, y_train, None, feature_names


# Example usage:
# train_df = pd.read_csv('data/train.csv')
# test_df = pd.read_csv('data/test.csv')
# X_train, y_train, X_test, feature_names = preprocess_data(train_df, test_df)