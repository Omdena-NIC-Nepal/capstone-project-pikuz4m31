# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# import joblib
# import os

# # Define the load_and_preprocess_data function
# def load_and_preprocess_data(file_path):
#     # Load and preprocess data
#     df = pd.read_csv(file_path)

#     # Drop rows with missing target labels
#     if 'disaster_type' in df.columns:
#         df = df.dropna(subset=['disaster_type'])

#     # Convert necessary columns
#     df['date'] = pd.to_datetime(df['date'])
#     df["disno"] = df["disno"].astype(str)  # Keep disno as string

#     # Encode target variable
#     label_mapping = {label: idx for idx, label in enumerate(df['disaster_type'].unique())}
#     df['target'] = df['disaster_type'].map(label_mapping)

#     # Features and target
#     feature_cols = [
#         'latitude', 'longitude', 'temperature_avg', 'precipitation',
#         'precip_rolling_30d_mean', 'precip_rolling_30d_std', 'spi_like',
#         'heat_stress_index', 'is_monsoon', 'temp_lag_1', 'precip_lag_1',
#         'temp_lag_3', 'precip_lag_3', 'temp_lag_7', 'precip_lag_7',
#         'temp_lag_30', 'precip_lag_30', 'distance_to_center_km',
#         'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5',
#         'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10'
#     ]
    
#     X = df[feature_cols]
#     y = df['target']

#     # Drop classes with fewer than 4 samples before SMOTE
#     class_counts = y.value_counts()
#     valid_classes = class_counts[class_counts >= 4].index  # Only keep classes with 4 or more samples
#     mask = y.isin(valid_classes)
#     X = X[mask]
#     y = y[mask]

#     # Balance the dataset using SMOTE
#     smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
#     X, y = smote.fit_resample(X, y)

#     return X, y


# def get_save_dir():
#     # Get the absolute path of the current file
#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     # Go up one directory (to the root of the project)
#     parent_dir = os.path.dirname(current_dir)

#     # Build the save directory path
#     save_dir = os.path.join(parent_dir, 'app', 'models', 'weather_climate_model')

#     return save_dir

# def save_models(models, save_dir):
#     # Ensure the directory exists
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # Save models
#     for model_name, model in models.items():
#         model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
#         joblib.dump(model, model_path)
#         print(f"Model {model_name} saved at {model_path}")

# def train_and_evaluate_models(X_train, X_test, y_train, y_test):
#     # Scale the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     models = {
#         "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
#         "Logistic Regression": LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto'),
#         "SVM": SVC(kernel='linear', C=0.5, probability=True)
#     }

#     results = {}
#     for name, model in models.items():
#         print(f"\nTraining and evaluating {name}...")
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)

#         # Classification Report and Confusion Matrix
#         target_names = [str(label) for label in sorted(y_test.unique())]
#         class_report = classification_report(y_test, y_pred, target_names=target_names)
#         conf_matrix = confusion_matrix(y_test, y_pred)

#         # Store results
#         results[name] = {
#             "classification_report": class_report,
#             "confusion_matrix": conf_matrix
#         }

#         # Print the results
#         print(f"\n{name} Classification Report:")
#         print(class_report)

#         print(f"\n{name} Confusion Matrix:")
#         print(conf_matrix)

#     # Get the appropriate save directory
#     save_dir = get_save_dir()

#     # Save models
#     save_models(models, save_dir)

#     return results

# # Example of running the full process
# # file_path = '../../feature_engineering/weather_and_temp_feature_engineering.csv'  # Provide the correct file path to the dataset

# # Get the absolute path to the data file relative to the current script
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.abspath(os.path.join(current_dir, '..', 'feature_engineering', 'weather_and_temp_feature_engineering.csv'))


# # Load and preprocess data
# X, y = load_and_preprocess_data(file_path)

# # Split data into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train and evaluate models
# results = train_and_evaluate_models(X_train, X_test, y_train, y_test)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# Define the load_and_preprocess_data function
def load_and_preprocess_data(file_path):
    # Load and preprocess data
    df = pd.read_csv(file_path)

    # Drop rows with missing target labels
    if 'disaster_type' in df.columns:
        df = df.dropna(subset=['disaster_type'])

    # Convert necessary columns
    df['date'] = pd.to_datetime(df['date'])
    df["disno"] = df["disno"].astype(str)  # Keep disno as string

    # Encode target variable
    label_mapping = {label: idx for idx, label in enumerate(df['disaster_type'].unique())}
    df['target'] = df['disaster_type'].map(label_mapping)

    # Features and target
    feature_cols = [
        'latitude', 'longitude', 'temperature_avg', 'precipitation',
        'precip_rolling_30d_mean', 'precip_rolling_30d_std', 'spi_like',
        'heat_stress_index', 'is_monsoon', 'temp_lag_1', 'precip_lag_1',
        'temp_lag_3', 'precip_lag_3', 'temp_lag_7', 'precip_lag_7',
        'temp_lag_30', 'precip_lag_30', 'distance_to_center_km',
        'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5',
        'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10'
    ]
    
    X = df[feature_cols]
    y = df['target']

    # Drop classes with fewer than 4 samples before SMOTE
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 4].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]

    # Balance the dataset using SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
    X, y = smote.fit_resample(X, y)

    return X, y

def get_save_dir():
    # Get the absolute path of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one directory (to the root of the project)
    parent_dir = os.path.dirname(current_dir)

    # Build the save directory path
    save_dir = os.path.join(parent_dir, 'app', 'models', 'weather_climate_model')

    return save_dir

def save_models(models, save_dir):
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save models
    for model_name, model in models.items():
        model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Model {model_name} saved at {model_path}")

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto'),
        "SVM": SVC(kernel='linear', C=0.5, probability=True)
    }

    results = {}
    for name, model in models.items():
        print(f"\n🚀 Training and evaluating {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Classification Report and Confusion Matrix
        target_names = [str(label) for label in sorted(y_test.unique())]
        class_report = classification_report(y_test, y_pred, target_names=target_names)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Store results
        results[name] = {
            "classification_report": class_report,
            "confusion_matrix": conf_matrix
        }

        # Print the results
        print(f"\n📊 {name} Classification Report:")
        print(class_report)

        print(f"\n🧾 {name} Confusion Matrix:")
        print(conf_matrix)

    # Save trained models
    save_models(models, get_save_dir())

    return results

# ---------- MAIN RUNNING LOGIC ----------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(current_dir, '..', 'feature_engineering', 'weather_and_temp_feature_engineering.csv'))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Data file not found at expected location:\n{file_path}")

    print(f"📁 Data file found at: {file_path}")

    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
