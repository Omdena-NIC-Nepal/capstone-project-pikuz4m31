import streamlit as st
import os

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "climate_selection" not in st.session_state:
    st.session_state.climate_selection = "Select..."
if "glacier_selection" not in st.session_state:
    st.session_state.glacier_selection = "Select..."
if "socio_economic_selection" not in st.session_state:
    st.session_state.socio_economic_selection = "Select..."

# Sidebar layout
st.sidebar.markdown("### Main")  # Title for the main section
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
    st.session_state.climate_selection = "Select..."
    st.session_state.glacier_selection = "Select..."
    st.session_state.socio_economic_selection = "Select..."

# Now we avoid the <hr> tag to minimize gaps between sections
st.sidebar.markdown("### Navigations")

# Climate Pages
climate_pages = [
    "Climate Data - Vulnerability",
    "Climate Data - Analysis",
    "Climate Data - Predictions"
]

# Create full list for dropdown
climate_dropdown_options = ["Select..."] + climate_pages

# Find the correct index for current selection
climate_current_index = climate_dropdown_options.index(st.session_state.climate_selection)

# Render selectbox for Climate
selected_climate_option = st.sidebar.selectbox(
    "Climate Sections",
    options=climate_dropdown_options,
    index=climate_current_index,
    key="climate_selection"
)

# If a valid page is selected for Climate
if selected_climate_option in climate_pages:
    st.session_state.page = selected_climate_option

# Glacier Pages
st.sidebar.markdown("### Glacier Data")  # Title for Glacier section
glacier_pages = [
    "Glacier Data - Overview",
    "Glacier Data - Trends"
]

# Create full list for dropdown
glacier_dropdown_options = ["Select..."] + glacier_pages

# Find the correct index for current selection
glacier_current_index = glacier_dropdown_options.index(st.session_state.glacier_selection)

# Render selectbox for Glacier
selected_glacier_option = st.sidebar.selectbox(
    "Glacier Data",
    options=glacier_dropdown_options,
    index=glacier_current_index,
    key="glacier_selection"
)

# Socio-Economic Pages
st.sidebar.markdown("### Socio-Economic Impact")  # Title for Socio-Economic section
socio_economic_pages = [
    "Socio-Economic Impact - Overview",
    "Socio-Economic Impact - Trends"
]

# Create full list for dropdown
socio_economic_dropdown_options = ["Select..."] + socio_economic_pages

# Find the correct index for current selection
socio_economic_current_index = socio_economic_dropdown_options.index(st.session_state.socio_economic_selection)

# Render selectbox for Socio-Economic
selected_socio_economic_option = st.sidebar.selectbox(
    "Socio-Economic Impact",
    options=socio_economic_dropdown_options,
    index=socio_economic_current_index,
    key="socio_economic_selection"
)

# File mapping for page execution (currently no real pages, just placeholders)
PAGES = {
    "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
    "Climate Data - Analysis": "climate_pages/2_Analysis.py",
    "Climate Data - Predictions": "climate_pages/3_Predictions.py",
    "Glacier Data - Overview": "",  # Dummy page
    "Glacier Data - Trends": "",  # Dummy page
    "Socio-Economic Impact - Overview": "",  # Dummy page
    "Socio-Economic Impact - Trends": ""  # Dummy page
}

# Render selected page
if st.session_state.page == "Home":
    st.write("""
    ### App Overview:
    This app is designed to help monitor, analyze, and predict climate impacts, with a focus on Nepal's climate data. 

    Key Features:
    - **Vulnerability Analysis**: Identifies regions most at risk due to extreme climate conditions like high temperatures and precipitation.
    - **Climate Data Analysis**: Analyzes trends and detects outliers in climate data such as temperature and precipitation.
    - **Temperature Predictions**: Uses machine learning to forecast future temperature trends based on historical data.
    - **Glacier Data**: (Coming soon) Explore glacier melt trends and their impact on water resources.
    - **Socio-Economic Impact**: (Coming soon) Analyze how climate change affects livelihoods, migration, and economy.

    Each section offers interactive charts, data analysis, and predictive models to better understand the climate change impact.
    """)
else:
    # For now, these pages are placeholders, so no actual code execution.
    if st.session_state.page in PAGES:
        if PAGES[st.session_state.page]:
            page_path = PAGES[st.session_state.page]
            with open(page_path, "r") as f:
                code = f.read()
                exec(code, globals())
        else:
            st.write(f"{st.session_state.page} is a dummy page.")
    else:
        st.error(f"Page `{st.session_state.page}` not found.")
