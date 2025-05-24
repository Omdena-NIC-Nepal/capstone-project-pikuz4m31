import streamlit as st
import pandas as pd
import joblib

# Paths to your saved models (update if needed)
MODEL_PATH_RF = 'models/socio_eco_data_model/random_forest_classifier.pkl'
MODEL_PATH_SVM = 'models/socio_eco_data_model/svm_classifier.pkl'
MODEL_PATH_GB_CLASS = 'models/socio_eco_data_model/gradient_boosting_classifier.pkl'

MODEL_PATH_LR = 'models/socio_eco_data_model/linear_regression.pkl'
MODEL_PATH_RIDGE = 'models/socio_eco_data_model/ridge_regression.pkl'
MODEL_PATH_LASSO = 'models/socio_eco_data_model/lasso_regression.pkl'
MODEL_PATH_GB_REG = 'models/socio_eco_data_model/gradient_boosting_regressor.pkl'

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Load models once at the start
rf_model = load_model(MODEL_PATH_RF)
svm_model = load_model(MODEL_PATH_SVM)
gb_class_model = load_model(MODEL_PATH_GB_CLASS)

lr_model = load_model(MODEL_PATH_LR)
ridge_model = load_model(MODEL_PATH_RIDGE)
lasso_model = load_model(MODEL_PATH_LASSO)
gb_reg_model = load_model(MODEL_PATH_GB_REG)

st.title("🌍 Socio-Economic Impact Predictor")

st.markdown("""
Welcome! This tool predicts if your area might experience an extreme event and estimates potential financial loss.
Please provide some basic information about your area below.
""")

st.header("Basic Local Information")

altitude = st.number_input(
    "Altitude (meters above sea level)",
    min_value=0.0,
    max_value=8848.0,
    value=100.0,
    help="Height above sea level, e.g., 100 means 100 meters."
)

affected_families = st.number_input(
    "Number of affected families",
    min_value=0,
    max_value=10000,
    value=50,
    help="How many families are affected in this area?"
)

# Full list of 77 districts with placeholder
districts = ["-- Select a district --"] + [
    "Achham", "Arghakhanchi", "Baglung", "Baitadi", "Bajhang",
    "Bajura", "Banke", "Bara", "Bardiya", "Bhaktapur",
    "Bhojpur", "Chitwan", "Dadeldhura", "Dailekh", "Dang",
    "Darchula", "Dhading", "Dhankuta", "Dhanusha", "Dolakha",
    "Dolpa", "Doti", "Gorkha", "Gulmi", "Humla",
    "Ilam", "Jajarkot", "Jhapa", "Jumla", "Kailali",
    "Kalikot", "Kanchanpur", "Kapilvastu", "Kaski", "Kathmandu",
    "Kavrepalanchok", "Khotang", "Lalitpur", "Lamjung", "Mahottari",
    "Makwanpur", "Manang", "Morang", "Mugu", "Mustang",
    "Myagdi", "Nawalparasi", "Nuwakot", "Okhaldhunga", "Palpa",
    "Panchthar", "Parbat", "Parsa", "Ramechhap", "Rasuwa",
    "Rautahat", "Rolpa", "Rukum East", "Rukum West", "Rupandehi",
    "Salyan", "Sankhuwasabha", "Saptari", "Sarlahi", "Sindhuli",
    "Sindhupalchok", "Siraha", "Solukhumbu", "Sunsari", "Surkhet",
    "Syangja", "Tanahun", "Taplejung", "Terhathum", "Udayapur",
]

selected_district = st.selectbox(
    "Select your district",
    districts,
    help="Choose your district from the list (type to jump to match)."
)

if selected_district == "-- Select a district --":
    st.error("🚨 Please select a valid district before proceeding.")
else:
    district_code = districts.index(selected_district) - 1  # adjust for placeholder

    road_length = st.number_input(
        "Total Road Length in District (Km)",
        min_value=0.0,
        value=10.0,
        help="Length of all roads in your district."
    )

    district_area = st.number_input(
        "District Area (Km²)",
        min_value=0.0,
        value=100.0,
        help="Total land area of your district."
    )

    population = st.number_input(
        "Total Population (2011 Census)",
        min_value=0,
        value=50000,
        help="Total people living in your district."
    )

    male_population = st.number_input(
        "Male Population",
        min_value=0,
        value=25000,
        help="Number of males in the population."
    )

    female_population = st.number_input(
        "Female Population",
        min_value=0,
        value=25000,
        help="Number of females in the population."
    )

    # Explanation for beginners
    st.markdown("""
    ### What does this mean?

    - We use several computer models to predict if your area might face an extreme event like a flood, landslide, or drought.
    - If an event is predicted, we estimate the potential financial loss to help local authorities prepare better.
    - These predictions are based on historical data and machine learning models and may not be 100% exact but provide useful guidance.
    """)

    input_df = pd.DataFrame([{
        'Altitude in masl': altitude,
        'Number of Affected Family': affected_families,
        'district_matched_count': 0,
        'district_matched_second_round_count': 0,
        'district_matched': 0,
        'district': district_code,
        'Total Road Length (Km.)': road_length,
        'Total Area (Km2)': district_area,
        'Total Population (2011)': population,
        'Male': male_population,
        'Female': female_population
    }])

    if st.button("Predict Extreme Event Risk"):
        # Classification predictions
        pred_rf = rf_model.predict(input_df)[0]
        pred_svm = svm_model.predict(input_df)[0]
        pred_gb = gb_class_model.predict(input_df)[0]

        predictions = [pred_rf, pred_svm, pred_gb]
        likely_votes = sum(predictions)

        st.markdown("### 🔍 Decision Summary for Extreme Event Risk:")

        if likely_votes >= 2:
            st.error("⚠️ Based on model consensus, an **extreme event is likely** in your area.")
        else:
            st.success("✅ An extreme event is **not likely** in your area at this time.")

        # Always estimate financial loss (even if event is not likely)
        pred_lr = lr_model.predict(input_df)[0]
        pred_ridge = ridge_model.predict(input_df)[0]
        pred_lasso = lasso_model.predict(input_df)[0]
        pred_gb_reg = gb_reg_model.predict(input_df)[0]

        loss_estimates = [pred_lr, pred_ridge, pred_lasso, pred_gb_reg]
        avg_loss = sum(loss_estimates) / len(loss_estimates)

        st.markdown("### 💸 Estimated Financial Loss *If* an Extreme Event Occurs:")

        st.markdown(
            f"<span style='color:red; font-weight:bold;'>~ NPR {avg_loss:.2f} Million</span>",
            unsafe_allow_html=True,
        )

        st.markdown("> *This is a hypothetical loss estimate assuming an extreme event does occur.*")

st.markdown("""
---
*If you’re unsure about any input, use the default value or contact your local authority for data.*
""")
