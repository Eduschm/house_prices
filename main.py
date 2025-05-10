print("Data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please download the data manually and place train.csv and test.csv in the data/ directory.")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train or predict using ML model")
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'predict', 'both', 'advanced'], 
        required=True, 
        help='Select operation mode: train, predict, both, or advanced'
    )
    
    parser.add_argument(
        '--use_cached', 
        action='store_true',
        help='Use cached preprocessed data and models if available'
    )
    
    parser.add_argument(
        '--skip_download', 
        action='store_true',
        help='Skip data download from Kaggle (assume data is already present)'
    )
    
    parser.add_argument(
        '--blend',
        action='store_true',
        help='Create a blended submission from all trained models'
    )
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/preprocessed', exist_ok=True)
    
    # Download data from Kaggle if needed
    if not args.skip_download:
        if not download_data():
            print("Unable to download data. Exiting.")
            return
    
    # Check if preprocessed data already exists and use_cached is True
    preprocessed_exists = (
        os.path.exists('data/preprocessed/X_train.npy') and
        os.path.exists('data/preprocessed/y_train.npy') and
        os.path.exists('data/preprocessed/X_test.npy') and
        os.path.exists('data/preprocessed/y_test.npy') and
        os.path.exists('data/preprocessed/feature_names.joblib')
    )
    
    if args.use_cached and preprocessed_exists:
        print("Loading cached preprocessed data...")
        X_train = np.load('data/preprocessed/X_train.npy')
        y_train = np.load('data/preprocessed/y_train.npy')
        X_test = np.load('data/preprocessed/X_test.npy')
        y_test = np.load('data/preprocessed/y_test.npy')
        feature_names = joblib.load('data/preprocessed/feature_names.joblib')
    else:
        print("Loading and preprocessing data...")
        # Load raw data
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        
        # If test_df doesn't have SalePrice, use train_test_split
        if 'SalePrice' not in test_df.columns:
            # Create a split from training data
            train_df, holdout_df = train_test_split(
                train_df, test_size=0.2, random_state=42
            )
            # Preprocess both sets of data
            X_train, y_train, X_test, feature_names = preprocess_data(train_df, holdout_df)
            y_test = np.log1p(holdout_df['SalePrice']).values
        else:
            # If test_df has SalePrice, preprocess both
            X_train, y_train, X_test, feature_names = preprocess_data(train_df, test_df)
            y_test = np.log1p(test_df['SalePrice']).values
        
        # Save preprocessed data for future use
        np.save('data/preprocessed/X_train.npy', X_train)
        np.save('data/preprocessed/y_train.npy', y_train)
        np.save('data/preprocessed/X_test.npy', X_test)
        np.save('data/preprocessed/y_test.npy', y_test)
        joblib.dump(feature_names, 'data/preprocessed/feature_names.joblib')
    
    # Execute requested mode
    if args.mode == 'advanced':
        print("\n===== RUNNING ADVANCED TRAINING PIPELINE =====")
        all_models = train_advanced(X_train, y_train, use_cached=args.use_cached)
        
        print("\n===== EVALUATING MODELS =====")
        test_results = predict(X_test, y_test)
        
        if args.blend or True:  # Always blend in advanced mode
            print("\n===== CREATING BLENDED SUBMISSION =====")
            submission = blend_predictions('data/test.csv')
    
    elif args.mode in ['train', 'both']:
        print("\n===== TRAINING MODELS =====")
        models, cv_results = train(X_train, y_train, use_preprocessed=True)
        
        # Save cross-validation results
        joblib.dump(cv_results, 'models/cv_results.joblib')
    
    if args.mode in ['predict', 'both']:
        print("\n===== EVALUATING MODELS =====")
        test_results = predict(X_test, y_test)
        
        # Save evaluation results
        joblib.dump(test_results, 'models/test_results.joblib')
        
        if args.blend:
            print("\n===== CREATING BLENDED SUBMISSION =====")
            submission = blend_predictions('data/test.csv')
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    # Filter some warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()