
# glacier_lake_model_training.py

import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# --- 1. Define model save path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
ROOT_DIR = os.path.dirname(CURRENT_DIR)  # go up one level
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'combined_data.geojson')
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'app', 'models', 'glacier_data_model')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_glacier_lake_models():
    """Train classification and regression models for glacier lake prediction using real dataset."""

    # --- 2. Load real glacier lake dataset ---
    data = gpd.read_file(DATA_PATH)
    print(f"✅ Data loaded with shape: {data.shape}")

    # Drop non-numeric or unnecessary columns
    # (You can adjust this list based on your dataset)
    drop_columns = ['glacier_id', 'country', 'datasource', 'geometry']
    data = data.drop(columns=[col for col in drop_columns if col in data.columns], errors='ignore')

    # Keep only numeric features
    features = data.select_dtypes(include=[np.number])

    # Handle missing values if any
    features = features.fillna(features.mean())

    # Fake targets for now (since real labels are missing)
    np.random.seed(42)
    num_samples = features.shape[0]

    climate_zone_target = np.random.choice(['Tropical', 'Temperate', 'Polar'], size=num_samples)
    extreme_event_target = np.random.choice(['Low', 'Medium', 'High'], size=num_samples)
    vulnerability_target = np.random.choice(['Low', 'Medium', 'High'], size=num_samples)
    impact_score = np.random.rand(num_samples) * 100

    # Split data
    X_train, X_test, y_climate_train, y_climate_test = train_test_split(features, climate_zone_target, test_size=0.2, random_state=42)
    _, _, y_extreme_train, y_extreme_test = train_test_split(features, extreme_event_target, test_size=0.2, random_state=42)
    _, _, y_vulnerability_train, y_vulnerability_test = train_test_split(features, vulnerability_target, test_size=0.2, random_state=42)
    _, _, y_impact_train, y_impact_test = train_test_split(features, impact_score, test_size=0.2, random_state=42)

    # --- 3. Train Classification Models ---
    random_forest_climate_zone = RandomForestClassifier()
    random_forest_climate_zone.fit(X_train, y_climate_train)

    svm_extreme_event = SVC(probability=True)
    svm_extreme_event.fit(X_train, y_extreme_train)

    gradient_boosting_vulnerability_level = GradientBoostingClassifier()
    gradient_boosting_vulnerability_level.fit(X_train, y_vulnerability_train)

    # --- 4. Train Regression Models ---
    linear_regression_impact_score = LinearRegression()
    linear_regression_impact_score.fit(X_train, y_impact_train)

    ridge_regression_impact_score = Ridge()
    ridge_regression_impact_score.fit(X_train, y_impact_train)

    lasso_regression_impact_score = Lasso()
    lasso_regression_impact_score.fit(X_train, y_impact_train)

    gradient_boosting_regression_impact_score = GradientBoostingRegressor()
    gradient_boosting_regression_impact_score.fit(X_train, y_impact_train)

    # --- 5. Save Models ---
    models_to_save = {
        'random_forest_climate_zone': random_forest_climate_zone,
        'svm_extreme_event': svm_extreme_event,
        'gradient_boosting_vulnerability_level': gradient_boosting_vulnerability_level,
        'linear_regression_impact_score': linear_regression_impact_score,
        'ridge_regression_impact_score': ridge_regression_impact_score,
        'lasso_regression_impact_score': lasso_regression_impact_score,
        'gradient_boosting_regression_impact_score': gradient_boosting_regression_impact_score
    }

    for model_name, model in models_to_save.items():
        save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.joblib")
        joblib.dump(model, save_path)

    # --- 6. Return Training Summary ---
    return {
        "classification_models_trained": list(models_to_save.keys())[:3],
        "regression_models_trained": list(models_to_save.keys())[3:],
        "models_saved_in": os.path.abspath(MODEL_SAVE_DIR)
    }

# Allow running this file directly
if __name__ == "__main__":
    result = train_glacier_lake_models()
    print(result)
