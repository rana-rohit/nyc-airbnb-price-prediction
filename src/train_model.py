# src/train_model.py
import argparse
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import __version__ as sklearn_version
from src.data_prep import load_listings, quick_clean


def build_preprocessor(df):
    cat_cols = ['neighbourhood', 'room_type']
    num_cols = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
                'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    # numeric pipeline
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    # create OneHotEncoder compatible with installed sklearn
    def make_ohe():
        """Create OneHotEncoder compatible with installed sklearn version."""
        try:
            # try newest API first (sparse_output for sklearn >= 1.2)
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            # older sklearn uses 'sparse' argument
            return OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', make_ohe())
    ])
    preproc = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')
    return preproc, num_cols, cat_cols

def train(input_csv, model_out='models/rf_model.joblib', preproc_out='models/preproc.joblib', sample_size=None):
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    df = load_listings(input_csv)
    df = quick_clean(df)
    if sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    # target and features
    y = np.log1p(df['price'].values)  # log(1+price)
    preproc, num_cols, cat_cols = build_preprocessor(df)
    X = preproc.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {'n_estimators': [100], 'max_depth': [8, 12]}  # small grid for speed
    gs = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    # eval
    y_pred = best.predict(X_test)
    mae_log = mean_absolute_error(y_test, y_pred)

# compute RMSE in a version-agnostic way
    mse_log = mean_squared_error(y_test, y_pred)        # mean squared error
    rmse_log = np.sqrt(mse_log)                         # root mean squared error

    y_test_d = np.expm1(y_test)
    y_pred_d = np.expm1(y_pred)
    mae_d = mean_absolute_error(y_test_d, y_pred_d)

    mse_d = mean_squared_error(y_test_d, y_pred_d)
    rmse_d = np.sqrt(mse_d)

    print("Log-target MAE:", mae_log)
    print(f"MAE (USD): {mae_d:.2f}, RMSE (USD): {rmse_d:.2f}")
    # save
    joblib.dump(best, model_out)
    joblib.dump(preproc, preproc_out)
    print("Saved model to", model_out)
    print("Saved preprocessor to", preproc_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='Path to listings.csv (if omitted, default paths will be tried)')
    parser.add_argument('--model_out', default='models/rf_model.joblib')
    parser.add_argument('--preproc_out', default='models/preproc.joblib')
    parser.add_argument('--sample', type=int, default=None, help='Sample size to speed up training')
    args = parser.parse_args()
    input_csv = args.input
    if input_csv is None:
        from src.data_prep import find_listings_path
        input_csv = find_listings_path()
        if input_csv is None:
            raise FileNotFoundError("Provide --input or place listings.csv in data/")
    train(input_csv, args.model_out, args.preproc_out, sample_size=args.sample)
