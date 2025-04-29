# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import LabelEncoder
# import warnings

# warnings.filterwarnings("ignore")

# # Load data
# df = pd.read_csv('../data/feature_engineering/engineered_weather_data.csv')
# print("Available columns:", df.columns.tolist())

# # Configuration
# task_type = "classification"  # or "regression"

# # Set target column based on task type
# if task_type == "classification":
#     target_column = "disaster_type"
# elif task_type == "regression":
#     target_column = "temperature_avg"
# else:
#     raise ValueError("Invalid task type selected.")

# # Ensure target exists
# if target_column not in df.columns:
#     raise ValueError(f"Target column '{target_column}' not found.")

# # Drop known non-numeric / non-feature columns
# non_features = ['date', 'district', 'country', 'disno', 'disaster_type', 'region_id']
# X = df.drop(columns=[col for col in non_features if col in df.columns])

# # Ensure only numeric features
# X = X.select_dtypes(include=[np.number])

# # Target variable
# y = df[target_column]

# # Encode classification target if needed
# if task_type == "classification":
#     y = LabelEncoder().fit_transform(y.astype(str))

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model selection
# models = []

# if task_type == "classification":
#     models = [
#         ("Random Forest", RandomForestClassifier()),
#         ("SVM", SVC()),
#         ("Gradient Boosting", GradientBoostingClassifier())
#     ]
# else:
#     models = [
#         ("Linear Regression", LinearRegression()),
#         ("Ridge Regression", Ridge()),
#         ("Lasso Regression", Lasso()),
#         ("Gradient Boosting", GradientBoostingRegressor())
#     ]

# # Train and evaluate
# for name, model in models:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f"\n{name} Results:")
    
#     if task_type == "classification":
#         print("Accuracy:", accuracy_score(y_test, y_pred))
#         print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
#         scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
#         print("Cross-Validated F1 Score:", np.mean(scores))
#     else:
#         print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
#         print("MAE:", mean_absolute_error(y_test, y_pred))
#         scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
#         print("Cross-Validated RMSE:", np.mean(np.sqrt(-scores)))


# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# import warnings
# import joblib

# warnings.filterwarnings("ignore")

# # Load data
# df = pd.read_csv('../data/feature_engineering/engineered_weather_data.csv')
# print("Available columns:", df.columns.tolist())

# # Configuration
# task_type = "classification"  # or "regression"

# # Set target column based on task type
# if task_type == "classification":
#     target_column = "disaster_type"
# elif task_type == "regression":
#     target_column = "temperature_avg"
# else:
#     raise ValueError("Invalid task type selected.")

# # Ensure target column exists
# if target_column not in df.columns:
#     raise ValueError(f"Target column '{target_column}' not found.")

# # Drop non-numeric / irrelevant columns
# non_features = ['date', 'district', 'country', 'disno', 'disaster_type', 'region_id']
# X = df.drop(columns=[col for col in non_features if col in df.columns], errors='ignore')

# # Keep only numeric features
# X = X.select_dtypes(include=[np.number])

# # Define target variable
# y = df[target_column]

# # Encode classification targets
# if task_type == "classification":
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y.astype(str))

# # Print class distribution before SMOTE
# print("Class distribution for target before SMOTE:", Counter(y))

# # Remove classes with fewer than 3 samples to prevent SMOTE error
# if task_type == "classification":
#     class_counts = pd.Series(y).value_counts()
#     rare_classes = class_counts[class_counts < 3].index
#     mask = ~pd.Series(y).isin(rare_classes)
#     X = X[mask]
#     y = pd.Series(y)[mask]

#     # Apply SMOTE
#     smote = SMOTE(k_neighbors=1)
#     X, y = smote.fit_resample(X, y)
#     y = pd.Series(y)  # Convert y back to Series for value_counts

#     # Post-SMOTE class distribution
#     print("Class distribution after SMOTE:")
#     print(y.value_counts())

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Choose models based on task type
# models = []

# if task_type == "classification":
#     models = [
#         ("Random Forest", RandomForestClassifier()),
#         ("SVM", SVC()),
#         ("Gradient Boosting", GradientBoostingClassifier())
#     ]
# else:
#     models = [
#         ("Linear Regression", LinearRegression()),
#         ("Ridge Regression", Ridge()),
#         ("Lasso Regression", Lasso()),
#         ("Gradient Boosting Regressor", GradientBoostingRegressor())
#     ]

# # Model evaluation function
# def evaluate_model(name, model, X_train, y_train, X_test, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     results = {"Model": name}
    
#     if task_type == "classification":
#         results["Accuracy"] = accuracy_score(y_test, y_pred)
#         results["F1 Score"] = f1_score(y_test, y_pred, average='weighted')
#         results["Cross-Validated F1 Score"] = np.mean(cross_val_score(model, X, y, cv=5, scoring='f1_weighted'))
#     else:
#         results["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
#         results["MAE"] = mean_absolute_error(y_test, y_pred)
#         results["Cross-Validated RMSE"] = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')))
    
#     return results

# # Evaluate all models
# for name, model in models:
#     print(f"\nEvaluating {name}...")
#     results = evaluate_model(name, model, X_train, y_train, X_test, y_test)
#     for metric, value in results.items():
#         print(f"{metric}: {value}")

# # Optional: Hyperparameter tuning for Random Forest
# if task_type == "classification":
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2]
#     }
#     grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
#     grid_search.fit(X_train, y_train)

#     print(f"\nBest parameters for Random Forest: {grid_search.best_params_}")
#     best_rf_model = grid_search.best_estimator_

#     print("\nEvaluating tuned Random Forest model...")
#     tuned_results = evaluate_model("Tuned Random Forest", best_rf_model, X_train, y_train, X_test, y_test)
#     for metric, value in tuned_results.items():
#         print(f"{metric}: {value}")

#     # Save best model
#     joblib.dump(best_rf_model, 'best_random_forest_model.pkl')


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# import time

# from collections import Counter
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# warnings.filterwarnings("ignore")

# start_time = time.time()

# # Load dataset
# file_path = "../data/feature_engineering/engineered_weather_data.csv"
# df = pd.read_csv(file_path)

# print("Available columns:", list(df.columns))

# # Drop rows with missing target labels
# if 'disaster_type' in df.columns:
#     df = df.dropna(subset=['disaster_type'])

# # Convert necessary columns
# df['date'] = pd.to_datetime(df['date'])
# df["disno"] = df["disno"].astype(str)  # Fix: keep disno as string

# # Encode target variable
# label_mapping = {label: idx for idx, label in enumerate(df['disaster_type'].unique())}
# df['target'] = df['disaster_type'].map(label_mapping)

# # Features and target
# feature_cols = [
#     'latitude', 'longitude', 'temperature_avg', 'precipitation',
#     'precip_rolling_30d_mean', 'precip_rolling_30d_std', 'spi_like',
#     'heat_stress_index', 'is_monsoon', 'temp_lag_1', 'precip_lag_1',
#     'temp_lag_3', 'precip_lag_3', 'temp_lag_7', 'precip_lag_7',
#     'temp_lag_30', 'precip_lag_30', 'distance_to_center_km',
#     'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5',
#     'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10'
# ]

# X = df[feature_cols]
# y = df['target']

# print("Class distribution for target before SMOTE:", Counter(y))

# # Drop classes with fewer than 4 samples before SMOTE
# class_counts = y.value_counts()
# valid_classes = class_counts[class_counts >= 4].index  # Only keep classes with 4 or more samples
# mask = y.isin(valid_classes)
# X = X[mask]
# y = y[mask]

# print(f"Classes after filtering: {Counter(y)}")

# # Balance the dataset using SMOTE
# smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)  # k_neighbors set to 3
# X, y = smote.fit_resample(X, y)
# print("Class distribution after SMOTE:")
# print(y.value_counts())

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Models to evaluate
# models = {
#     "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
#     "Logistic Regression": LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto'),
#     "SVM": SVC(kernel='linear', C=0.5, probability=True)
# }

# # Train and evaluate models
# for name, model in models.items():
#     print(f"\nEvaluating {name}...")
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     print(confusion_matrix(y_test, y_pred))
    
#     # Use unique classes from y_test for target_names
#     target_names = [str(label) for label in sorted(y_test.unique())]
#     print(classification_report(y_test, y_pred, target_names=target_names))

# print("\n--- Execution Time: %s seconds ---" % round(time.time() - start_time, 2))



# weather_and_climate_model_and_evaluation.py

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# import joblib
# import os

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

#     return results


# # Example of running the full process

# file_path = '../data/feature_engineering/engineered_weather_data.csv'  # Provide the correct file path to the dataset

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
    valid_classes = class_counts[class_counts >= 4].index  # Only keep classes with 4 or more samples
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
        print(f"Model {model_name} saved at {model_path}")

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
        print(f"\nTraining and evaluating {name}...")
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
        print(f"\n{name} Classification Report:")
        print(class_report)

        print(f"\n{name} Confusion Matrix:")
        print(conf_matrix)

    # Get the appropriate save directory
    save_dir = get_save_dir()

    # Save models
    save_models(models, save_dir)

    return results

# Example of running the full process
file_path = '../data/feature_engineering/engineered_weather_data.csv'  # Provide the correct file path to the dataset

# Load and preprocess data
X, y = load_and_preprocess_data(file_path)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
