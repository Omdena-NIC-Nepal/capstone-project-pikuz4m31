# import streamlit as st
# import joblib
# import os
# import sys

# # --- Correct path to 'src' folder ---
# # Correctly locate 'src' folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = (os.path.dirname(current_dir))  # Go up 1 levels
# src_dir = os.path.join(root_dir, 'src')

# if src_dir not in sys.path:
#     sys.path.append(src_dir)

# # Now import
# try:
#     from glacier_lake_model_training import train_glacier_lake_models
# except ImportError as e:
#     import streamlit as st
#     st.error(f"Error loading module: {e}")
#     st.stop()

# # --- Configurations ---
# MODEL_SAVE_DIR = './app/models/glacier_data_model/'

# # --- Utility Function to Load Models ---
# def load_model(model_name):
#     """Load a trained model from disk."""
#     model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pkl")
#     if os.path.exists(model_path):
#         return joblib.load(model_path)
#     else:
#         st.error(f"Model `{model_name}` not found!")
#         return None

# # --- Streamlit Interface ---
# st.title("Glacier Lake Future Predictions")

# # --- Training Section ---
# if st.button("Train Glacier Lake Models"):
#     st.write("Training Glacier Lake Models... Please wait.")
#     result = train_glacier_lake_models()  # Trigger training
#     st.write(result)

# # --- Prediction Section ---
# st.header("Glacier Lake Future Predictions")

# # Load models button
# if st.button("Load Trained Models"):
#     # Model names for classification and regression
#     classification_model_names = [
#         'random_forest_climate_zone', 
#         'svm_extreme_event', 
#         'gradient_boosting_vulnerability_level'
#     ]
#     regression_model_names = [
#         'linear_regression_impact_score', 
#         'ridge_regression_impact_score', 
#         'lasso_regression_impact_score',
#         'gradient_boosting_regression_impact_score'
#     ]

#     # Load classification models
#     classification_models = {name: load_model(name) for name in classification_model_names}
    
#     # Load regression models
#     regression_models = {name: load_model(name) for name in regression_model_names}

#     # Check if models are successfully loaded
#     if all(classification_models.values()) and all(regression_models.values()):
#         st.success("Models loaded successfully!")

#         # --- User Input for Predictions ---
#         user_input = st.file_uploader("Upload Data for Predictions", type=["csv", "geojson"])

#         if user_input is not None:
#             # Process the uploaded data (you can add logic to parse and prepare it for prediction)
#             st.write(f"Uploaded data: {user_input.name}")

#             # Placeholder example of using loaded models for predictions
#             # Here you would add your prediction logic

#             # Example of making predictions using the loaded models
#             # (assuming the uploaded data is a DataFrame ready for prediction)
#             import pandas as pd

#             # Placeholder: Read data (this can be improved depending on the file format)
#             if user_input.name.endswith('.csv'):
#                 data = pd.read_csv(user_input)
#             elif user_input.name.endswith('.geojson'):
#                 import geopandas as gpd
#                 data = gpd.read_file(user_input)

#             # Assuming you have a feature extraction function (this should be customized based on your model)
#             # Example prediction logic:
#             # Extract features from the uploaded data and pass it to the models
#             features = data.drop(columns=['target_column'], errors='ignore')  # Customize based on your data

#             # --- Classification Predictions ---
#             if classification_models['random_forest_climate_zone']:
#                 climate_zone_predictions = classification_models['random_forest_climate_zone'].predict(features)
#                 st.write(f"Climate Zone Predictions: {climate_zone_predictions}")

#             if classification_models['svm_extreme_event']:
#                 extreme_event_predictions = classification_models['svm_extreme_event'].predict(features)
#                 st.write(f"Extreme Event Predictions: {extreme_event_predictions}")

#             if classification_models['gradient_boosting_vulnerability_level']:
#                 vulnerability_level_predictions = classification_models['gradient_boosting_vulnerability_level'].predict(features)
#                 st.write(f"Vulnerability Level Predictions: {vulnerability_level_predictions}")

#             # --- Regression Predictions ---
#             if regression_models['linear_regression_impact_score']:
#                 impact_score_predictions = regression_models['linear_regression_impact_score'].predict(features)
#                 st.write(f"Impact Score Predictions (Linear Regression): {impact_score_predictions}")

#             if regression_models['ridge_regression_impact_score']:
#                 ridge_impact_predictions = regression_models['ridge_regression_impact_score'].predict(features)
#                 st.write(f"Impact Score Predictions (Ridge Regression): {ridge_impact_predictions}")

#             if regression_models['lasso_regression_impact_score']:
#                 lasso_impact_predictions = regression_models['lasso_regression_impact_score'].predict(features)
#                 st.write(f"Impact Score Predictions (Lasso Regression): {lasso_impact_predictions}")

#             if regression_models['gradient_boosting_regression_impact_score']:
#                 gbr_impact_predictions = regression_models['gradient_boosting_regression_impact_score'].predict(features)
#                 st.write(f"Impact Score Predictions (Gradient Boosting Regression): {gbr_impact_predictions}")

#     else:
#         st.warning("Models are not loaded. Please train first.")


# glacier_lake_future_predictions.py

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import sys
import joblib

# --- Add src to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, 'src')

if src_dir not in sys.path:
    sys.path.append(src_dir)

# --- Import Training Module ---
try:
    from glacier_lake_model_training import train_glacier_lake_models
except ImportError as e:
    st.error(f"Error loading module: {e}")
    st.stop()

# --- Configurations ---
# MODEL_SAVE_DIR = './models/glacier_data_model/'

current_dir = os.path.dirname(os.path.abspath(__file__))  # app/
MODEL_SAVE_DIR = os.path.join(current_dir, 'models', 'glacier_data_model')

# --- Helper Functions ---
@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load uploaded data (.csv or .geojson)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.geojson'):
            data = gpd.read_file(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return None
        return data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def load_model(model_name):
    """Load a model from disk."""
    model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ Model `{model_name}` not found.")
        return None

@st.cache_resource
def load_all_models():
    """Load all classification and regression models."""
    classification_models = {
        name: load_model(name) for name in [
            'random_forest_climate_zone', 
            'svm_extreme_event', 
            'gradient_boosting_vulnerability_level'
        ]
    }
    regression_models = {
        name: load_model(name) for name in [
            'linear_regression_impact_score', 
            'ridge_regression_impact_score', 
            'lasso_regression_impact_score',
            'gradient_boosting_regression_impact_score'
        ]
    }
    return classification_models, regression_models

def make_predictions(models, data):
    """Use loaded models to make predictions."""
    results = {}
    features = data.drop(columns=['target_column'], errors='ignore')  # Adjust as needed

    for model_name, model in models.items():
        if model:
            try:
                results[model_name] = model.predict(features)
            except Exception as e:
                st.error(f"Prediction error with `{model_name}`: {e}")
    return results

# --- Streamlit App ---
st.title("🧊 Glacier Lake Future Predictions")

st.markdown("""
This app allows you to **train models**, **load saved models**, and **predict** on your own data (.csv or .geojson files).
""")

# --- Section 1: Training ---
st.header("⚙️ Model Training")

if st.button("🚀 Train Glacier Lake Models"):
    with st.spinner("Training models... please wait..."):
        result = train_glacier_lake_models()
        st.success("Training completed successfully!")
        st.write(result)

# --- Section 2: Model Loading and Prediction ---
st.header("🔍 Model Loading and Predictions")

if st.button("📂 Load Trained Models"):
    classification_models, regression_models = load_all_models()

    if all(classification_models.values()) and all(regression_models.values()):
        st.success("✅ All models loaded successfully!")
        st.session_state['classification_models'] = classification_models
        st.session_state['regression_models'] = regression_models
    else:
        st.warning("⚠️ Some models failed to load. Please retrain if needed.")

# --- Section 3: Data Upload and Predictions ---
if 'classification_models' in st.session_state and 'regression_models' in st.session_state:
    uploaded_file = st.file_uploader("📤 Upload your .csv or .geojson file for prediction", type=["csv", "geojson"])

    if uploaded_file:
        data = load_uploaded_data(uploaded_file)
        if data is not None:
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            st.dataframe(data.head())

            if st.button("🔮 Predict"):
                with st.spinner("Making predictions..."):
                    # --- Classification Predictions ---
                    st.subheader("🔵 Classification Predictions")
                    classification_results = make_predictions(st.session_state['classification_models'], data)
                    for model_name, preds in classification_results.items():
                        st.write(f"**{model_name}** predictions:")
                        st.write(preds)

                    # --- Regression Predictions ---
                    st.subheader("🟣 Regression Predictions")
                    regression_results = make_predictions(st.session_state['regression_models'], data)
                    for model_name, preds in regression_results.items():
                        st.write(f"**{model_name}** predictions:")
                        st.write(preds)
else:
    st.info("ℹ️ Please load models before uploading data.")

