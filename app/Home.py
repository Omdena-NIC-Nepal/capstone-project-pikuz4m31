
# import streamlit as st
# import os

# # Initialize session state
# if "main_section" not in st.session_state:
#     st.session_state.main_section = "Select..."
# if "sub_page" not in st.session_state:
#     st.session_state.sub_page = "Select..."
# if "page" not in st.session_state:
#     st.session_state.page = "Home"

# # Sidebar Layout
# st.sidebar.markdown("### Main Navigation")

# # Main Sections
# main_sections = ["Climate Sections", "Glacier Lake Data", "Socio-Economic Impact"]

# # Subpages Mapping
# subpages_mapping = {
#     "Climate Sections": [
#         "Climate Data - Vulnerability",
#         "Climate Data - Analysis",
#         "Climate Data - Predictions"
#     ],
#     "Glacier Lake Data": [
#         "Glacier Lake Mapping & Visualization",
#         "Glacier Lake Impact Assessment",
#         "Glacier Lake Future Predictions"
#     ],
#     "Socio-Economic Impact": [
#         "Socio-Economic Impact - Overview",
#         "Socio-Economic Impact - Trends"
#     ]
# }

# # File Mapping
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
#     "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
#     "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
#     "Socio-Economic Impact - Overview": "",  # Dummy
#     "Socio-Economic Impact - Trends": ""  # Dummy
# }

# # Home button
# if st.sidebar.button("🏠 Home"):
#     st.session_state.main_section = "Select..."
#     st.session_state.sub_page = "Select..."
#     st.session_state.page = "Home"

# # Select Main Section
# selected_main = st.sidebar.selectbox(
#     "Select Section",
#     ["Select..."] + main_sections,
#     index=0,
#     key="main_section"
# )

# # Select Subpage if a Main Section is selected
# if selected_main != "Select...":
#     available_subpages = subpages_mapping[selected_main]
#     selected_subpage = st.sidebar.selectbox(
#         f"Select {selected_main} Page",
#         ["Select..."] + available_subpages,
#         index=0,
#         key="sub_page"
#     )
    
#     if selected_subpage in PAGES:
#         st.session_state.page = selected_subpage

# # Page Display
# if st.session_state.page == "Home":
#     st.write("""
#     ### 🌍 Climate Prediction and Assessment App
#     Welcome to the app!  
#     Navigate through the sections using the sidebar.
    
#     **Key Features:**
#     - Vulnerability Analysis
#     - Climate Trend Analysis
#     - Climate Predictions
#     - Glacier Lake Mapping and Impact
#     - Socio-Economic Impact Assessment (Coming Soon!)
#     """)
# else:
#     page_path = PAGES.get(st.session_state.page, None)
#     if page_path:
#         try:
#             # Ensure the page is executed properly
#             with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
#                 code = f.read()
#                 exec(code, globals())
                
#             # Call the page-specific display function if it exists
#             if 'display_page' in globals():
#                 display_page()  # Assuming 'display_page' exists in the pages
            
#         except Exception as e:
#             st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
#     else:
#         st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")


# base_dir = os.path.dirname(__file__)
# climate_pages_dir = os.path.join(base_dir, 'climate_pages')

# print("Absolute path to climate_pages directory:", climate_pages_dir)

# file_path = os.path.join(climate_pages_dir, '3_Predictions.py')
# print("Looking for file at:", file_path)

# if os.path.exists(file_path):
#     print(f"File found at: {file_path}")
# else:
#     print(f"File not found: {file_path}")


import streamlit as st
import os

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
main_sections = ["Climate Sections", "Glacier Lake Data", "Socio-Economic Impact"]

# Subpages Mapping
subpages_mapping = {
    "Climate Sections": [
        "Climate Data - Vulnerability",
        "Climate Data - Analysis",
        "Climate Data - Predictions"
    ],
    "Glacier Lake Data": [
        "Glacier Lake Mapping & Visualization",
        "Glacier Lake Impact Assessment",
        "Glacier Lake Future Predictions"
    ],
    "Socio-Economic Impact": [
        "Socio-Economic Impact - Overview",
        "Socio-Economic Impact - Trends"
    ]
}

# File Mapping
PAGES = {
    "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
    "Climate Data - Analysis": "climate_pages/2_Analysis.py",
    "Climate Data - Predictions": "climate_pages/3_Predictions.py",
    "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
    "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
    "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
    "Socio-Economic Impact - Overview": "",  # Dummy
    "Socio-Economic Impact - Trends": ""  # Dummy
}

# Home button
if st.sidebar.button("🏠 Home"):
    st.session_state.main_section = "Select..."
    st.session_state.sub_page = "Select..."
    st.session_state.page = "Home"

# Select Main Section
selected_main = st.sidebar.selectbox(
    "Select Section",
    ["Select..."] + main_sections,
    index=0,
    key="main_section"
)

# Select Subpage if a Main Section is selected
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

# Page Display
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
    """)
else:
    page_path = PAGES.get(st.session_state.page, None)
    if page_path:
        try:
            # Ensure the page is executed properly
            with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
                exec(code, globals())
                
            # Call the page-specific display function if it exists
            if 'display_page' in globals():
                display_page()  # Assuming 'display_page' exists in the pages
            
        except Exception as e:
            st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
    else:
        st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")

# Debugging paths for deployment
base_dir = os.path.dirname(__file__)  # This works locally, but for Streamlit deployment, consider using fixed paths
climate_pages_dir = os.path.join(base_dir, 'climate_pages')

print("Absolute path to climate_pages directory:", climate_pages_dir)

# Convert to absolute path for deployment
file_path = os.path.abspath(os.path.join(climate_pages_dir, '3_Predictions.py'))
print("Looking for file at:", file_path)

if os.path.exists(file_path):
    print(f"File found at: {file_path}")
else:
    print(f"File not found: {file_path}")
