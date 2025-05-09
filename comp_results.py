import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.preprocessing import preprocess_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_CSV      = Path("data/train.csv")
TEST_CSV       = Path("data/test.csv")
MODEL_PKL      = Path("models/XGBRegressor_best.pkl")
SUBMISSION_CSV = Path("data/stacking_submission.csv")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
ids      = test_df["Id"].copy()

# ---------------------------------------------------------------------------
# 2. Preprocess test set
# ---------------------------------------------------------------------------
print("Applying preprocessing pipeline...")
# returns: X_train_trans, y_train, X_test_trans, feature_names
_, _, X_test_transformed, feature_names = preprocess_data(train_df, test_df)
print(f"Preprocessed test shape: {X_test_transformed.shape}")

# ---------------------------------------------------------------------------
# 3. Load pipeline and extract regressor
# ---------------------------------------------------------------------------
print("Loading pipeline and extracting regressor...")
pipeline = joblib.load(MODEL_PKL)
if hasattr(pipeline, "named_steps") and "regressor" in pipeline.named_steps:
    regressor = pipeline.named_steps["regressor"]
else:
    regressor = pipeline
print(f"Regressor expects {regressor.n_features_in_} features.")

# ---------------------------------------------------------------------------
# 4. Align features
# ---------------------------------------------------------------------------

n_expected = regressor.n_features_in_
n_actual   = X_test_transformed.shape[1]
if n_actual > n_expected:
    print(f"Trimming features: {n_actual} -> {n_expected}")
    X_test_transformed = X_test_transformed[:, :n_expected]
elif n_actual < n_expected:
    pad = n_expected - n_actual
    print(f"Padding features: {n_actual} -> {n_expected} (+{pad})")
    X_test_transformed = np.pad(
        X_test_transformed,
        ((0, 0), (0, pad)),
        mode='constant',
        constant_values=0
    )
print(f"Aligned test shape: {X_test_transformed.shape}")

# ---------------------------------------------------------------------------
# 5. Predict & inverse transform
# ---------------------------------------------------------------------------
print("Predicting log-prices...")
log_preds   = regressor.predict(X_test_transformed)
price_preds = np.expm1(log_preds)  # invert log1p

# ---------------------------------------------------------------------------
# 6. Save submission
# ---------------------------------------------------------------------------
print("Saving submission...")
submission = pd.DataFrame({"Id": ids, "SalePrice": price_preds})
submission.to_csv(SUBMISSION_CSV, index=False)
print(f"Submission written to {SUBMISSION_CSV} ({len(submission)} rows)")
