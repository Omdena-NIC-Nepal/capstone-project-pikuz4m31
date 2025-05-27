import streamlit as st
import os
import importlib.util
import pandas as pd

# Initialize session state
if "main_section" not in st.session_state:
    st.session_state.main_section = "Select..."
if "sub_page" not in st.session_state:
    st.session_state.sub_page = "Select..."
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar Layout
st.sidebar.markdown("### Main Navigation")

# Main Sections
main_sections = ["Climate Sections", "Weather Sections", "Glacier Lake Data", "Socio-Economic Impact"]

# Subpages Mapping
subpages_mapping = {
    "Climate Sections": [
        "Climate Data - Vulnerability",
        "Climate Data - Analysis",
        "Climate Data - Predictions"
    ],
    "Weather Sections": [
        "Weather Data Visualization",
        "Weather Impact Assessment",
        "Weather Predictions"
    ],
    "Glacier Lake Data": [
        "Glacier Lake Mapping & Visualization",
        "Glacier Lake Impact Assessment",
        "Glacier Lake Future Predictions"
    ],
    "Socio-Economic Impact": [
        "Socio-Economic Impact - Predictions",
    ]
}

# NLP Sections
nlp_sections = [
    "Language Prediction",
    "NER Prediction",
    "Sentiment Analysis",
    "Summary Details",
]

# File Mapping
PAGES = {
    "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
    "Climate Data - Analysis": "climate_pages/2_Analysis.py",
    "Climate Data - Predictions": "climate_pages/3_Predictions.py",
    "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
    "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
    "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
    "Weather Data Visualization": "weather_pages/weather_data_visualization.py",
    "Weather Impact Assessment": "weather_pages/weather_impact_assesment.py",
    "Weather Predictions": "weather_pages/weather_predictions.py",
    "Socio-Economic Impact - Predictions": "socio_eco_pages/extrem_events.py",
    "Sentiment Analysis": "nlp_pages/sentiment_analysis.py",
    "Language Prediction": "nlp_pages/language_prediction.py",
    "NER Prediction": "nlp_pages/ner_prediction.py",
    "Summary Details": "nlp_pages/summary_details.py",
    "coming_soon": "coming_soon.py"
}

# Home Button
if st.sidebar.button("🏠 Home"):
    st.session_state.main_section = "Select..."
    st.session_state.sub_page = "Select..."
    st.session_state.page = "Home"

# Main Section Selector
selected_main = st.sidebar.selectbox(
    "Select Section",
    ["Select..."] + main_sections,
    index=0,
    key="main_section"
)

# Subpage Selector
if selected_main != "Select...":
    available_subpages = subpages_mapping[selected_main]
    selected_subpage = st.sidebar.selectbox(
        f"Select {selected_main} Page",
        ["Select..."] + available_subpages,
        index=0,
        key="sub_page"
    )

    if selected_subpage in PAGES:
        st.session_state.page = selected_subpage

# Show district dropdown for specific pages
if st.session_state.page in ["Weather Data Visualization", "Weather Impact Assessment"]:
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../feature_engineering/weather_and_temp_feature_engineering.csv'))
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        districts = df['district'].dropna().unique().tolist()
        selected_district = st.sidebar.selectbox("Select District", ['All'] + districts)

        if selected_district != 'All':
            df = df[df['district'] == selected_district]
    else:
        st.sidebar.error("District data file not found.")

# NLP Section
st.sidebar.markdown("---")
st.sidebar.markdown("### NLP Tools")
selected_nlp = st.sidebar.selectbox(
    "Select NLP Section",
    ["Select..."] + nlp_sections,
    index=0,
    key="nlp_section"
)

if selected_nlp != "Select...":
    st.session_state.page = selected_nlp

# Helper function to dynamically load a module from file
# def show_page(file_path):
#     try:
#         spec = importlib.util.spec_from_file_location("module.name", file_path)
#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)
#     except Exception as e:
#         st.error(f"❌ Failed to load page: `{st.session_state.page}`\n\n**Error:** {str(e)}")

def show_page(file_path):
    try:
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Explicitly call main()
        if hasattr(module, "main"):
            module.main()
        else:
            st.error("Module has no 'main' function.")
    except Exception as e:
        st.error(f"❌ Failed to load page: `{st.session_state.page}`\n\n**Error:** {str(e)}")

# Page Rendering
if st.session_state.page == "Home":
    st.write("""
    ### 🌍 Climate Prediction and Assessment App  
    Welcome to the app!  
    Navigate through the sections using the sidebar.

    **Key Features:**
    - Vulnerability Analysis
    - Climate Trend Analysis
    - Climate Predictions
    - Glacier Lake Mapping and Impact
    - Socio-Economic Impact Assessment (Coming Soon!)
    - NLP Sections (Language Prediction, NER Prediction, Sentiment Analysis, Summary Details)
    """)
    st.markdown("---")
    st.warning("⚠️ Important: If the page is not redirected properly, try refreshing the browser.")
else:
    page_path = PAGES.get(st.session_state.page, "coming_soon")
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    abs_path = os.path.join(base_dir, page_path)

    if os.path.exists(abs_path):
        show_page(abs_path)
    else:
        st.error(f"🔍 Page file not found at: `{abs_path}`")
