

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
# main_sections = ["Climate Sections", "Weather Sections", "Glacier Lake Data", "Socio-Economic Impact"]

# # Subpages Mapping
# subpages_mapping = {
#     "Climate Sections": [
#         "Climate Data - Vulnerability",
#         "Climate Data - Analysis",
#         "Climate Data - Predictions"
#     ],
#     "Weather Sections": [
#         "Weather Data Visualization",
#         "Weather Impact Assessment",
#         "Weather Predictions"
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

# # NLP Sections (updated for NLP tasks)
# nlp_sections = [
#     "Language Prediction",
#     "NER Prediction",
#     "Sentiment Analysis",
#     "Summary Details",
#     "Topic Details"
# ]

# # File Mapping
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
#     "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
#     "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
#     "Weather Data Visualization": "weather_pages/weather_data_visualization.py",
#     "Weather Impact Assessment": "weather_pages/weather_impact_assesment.py",
#     "Weather Predictions": "weather_pages/weather_predictions.py",
#     "Socio-Economic Impact - Overview": "",  # Dummy
#     "Socio-Economic Impact - Trends": "",  # Dummy
#     "Sentiment Analysis": "nlp_pages/sentiment_analysis.py",  # Added Sentiment Analysis
#     "Language Prediction": "nlp_pages/language_prediction.py",  # Dummy
#     "NER Prediction": "nlp_pages/ner_prediction.py",  # Dummy
#     "Summary Details": "nlp_pages/summary_details.py",  # Dummy
#     "Topic Details": "nlp_pages/topic_details.py",
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

# # Separate NLP section dropdown menu
# selected_nlp = st.sidebar.selectbox(
#     "Select NLP Section",  # NLP Section Dropdown
#     ["Select..."] + nlp_sections,
#     index=0,
#     key="nlp_section"
# )

# # Page Display Logic (same as before)
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
#     - NLP Sections (Language Prediction, NER Prediction, Sentiment Analysis, Summary Details, Topic Details)
#     """)
# else:
#     page_path = PAGES.get(st.session_state.page, None)
#     if page_path:
#         try:
#             # Dynamically calculate file path (handling both local and deployed environments)
#             base_dir = os.path.dirname(__file__)  # This is where your script is located
#             abs_path = os.path.join(base_dir, page_path)  # Get absolute path to the page file

#             # Check if file exists at that location
#             if os.path.exists(abs_path):
#                 with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
#                     code = f.read()
#                     exec(code, globals())

#                 # If the page has a 'display_page' function, call it
#                 if 'display_page' in globals():
#                     display_page()
#             else:
#                 st.error(f"Error: File not found at {abs_path}")

#         except Exception as e:
#             st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
#     else:
#         st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")

# # Debugging paths for deployment
# if st.session_state.page == "Home":
#     base_dir = os.path.dirname(__file__)  # This works locally, but for Streamlit deployment, consider using fixed paths
#     climate_pages_dir = os.path.join(base_dir, 'climate_pages')
#     abs_climate_pages_dir = os.path.abspath(climate_pages_dir)

#     # Use Streamlit to display debug output, instead of print
#     st.write("Absolute path to climate_pages directory:", abs_climate_pages_dir)

#     file_path = os.path.join(abs_climate_pages_dir, '3_Predictions.py')
#     st.write("Looking for file at:", file_path)

#     if os.path.exists(file_path):
#         st.write(f"File found at: {file_path}")
#     else:
#         st.error(f"File not found: {file_path}")



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
# main_sections = ["Climate Sections", "Weather Sections", "Glacier Lake Data", "Socio-Economic Impact"]

# # Subpages Mapping
# subpages_mapping = {
#     "Climate Sections": [
#         "Climate Data - Vulnerability",
#         "Climate Data - Analysis",
#         "Climate Data - Predictions"
#     ],
#     "Weather Sections": [
#         "Weather Data Visualization",
#         "Weather Impact Assessment",
#         "Weather Predictions"
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

# # NLP Sections
# nlp_sections = [
#     "Language Prediction",
#     "NER Prediction",
#     "Sentiment Analysis",  # Ensure this is here
#     "Summary Details",
#     "Topic Details"
# ]




# # File Mapping
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
#     "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
#     "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
#     "Weather Data Visualization": "weather_pages/weather_data_visualization.py",
#     "Weather Impact Assessment": "weather_pages/weather_impact_assesment.py",
#     "Weather Predictions": "weather_pages/weather_predictions.py",
#     "Socio-Economic Impact - Overview": "",  # Dummy
#     "Socio-Economic Impact - Trends": "",  # Dummy
#     "Sentiment Analysis": "nlp_pages/sentiment_analysis.py",  # Ensure correct path
#     "Language Prediction": "nlp_pages/language_prediction.py",  # Dummy
#     "NER Prediction": "nlp_pages/ner_prediction.py",  # Dummy
#     "Summary Details": "nlp_pages/summary_details.py",  # Dummy
#     "Topic Details": "nlp_pages/topic_details.py",
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

# # Separate NLP section dropdown menu
# selected_nlp = st.sidebar.selectbox(
#     "Select NLP Section",  # NLP Section Dropdown
#     ["Select..."] + nlp_sections,
#     index=0,
#     key="nlp_section"
# )
# # Assign NLP selection to session state
# if selected_nlp != "Select...":
#     st.session_state.page = selected_nlp

# # Page Display Logic
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
#     - NLP Sections (Language Prediction, NER Prediction, Sentiment Analysis, Summary Details, Topic Details)
#     """)
# else:
#     page_path = PAGES.get(st.session_state.page, None)
#     if page_path:
#         try:
#             # Dynamically calculate file path
#             base_dir = os.path.dirname(__file__)  # This is where your script is located
#             abs_path = os.path.join(base_dir, page_path)  # Get absolute path to the page file

#             # Check if file exists at that location
#             if os.path.exists(abs_path):
#                 with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
#                     code = f.read()
#                     exec(code, globals())  # Execute the code for sentiment analysis

                
#             else:
#                 st.error(f"Error: File not found at {abs_path}")

#         except Exception as e:
#             st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
#     else:
#         st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")


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

# # Home button
# if st.sidebar.button("🏠 Home"):
#     st.session_state.main_section = "Select..."
#     st.session_state.sub_page = "Select..."
#     st.session_state.page = "Home"

# # Main Sections
# main_sections = ["Climate Sections", "Weather Sections", "Glacier Lake Data", "Socio-Economic Impact"]

# # Subpages Mapping
# subpages_mapping = {
#     "Climate Sections": [
#         "Climate Data - Vulnerability",
#         "Climate Data - Analysis",
#         "Climate Data - Predictions"
#     ],
#     "Weather Sections": [
#         "Weather Data Visualization",
#         "Weather Impact Assessment",
#         "Weather Predictions"
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

# # NLP Sections
# nlp_sections = [
#     "Language Prediction",
#     "NER Prediction",
#     "Sentiment Analysis",
#     "Summary Details",
#     "Topic Details"
# ]

# # File Mapping
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
#     "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
#     "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
#     "Weather Data Visualization": "weather_pages/weather_data_visualization.py",
#     "Weather Impact Assessment": "weather_pages/weather_impact_assesment.py",
#     "Weather Predictions": "weather_pages/weather_predictions.py",
#     "Socio-Economic Impact - Overview": "",  # Dummy
#     "Socio-Economic Impact - Trends": "",  # Dummy
#     "Sentiment Analysis": "nlp_pages/sentiment_analysis.py",
#     "Language Prediction": "nlp_pages/language_prediction.py",
#     "NER Prediction": "nlp_pages/ner_prediction.py",
#     "Summary Details": "nlp_pages/summary_details.py",
#     "Topic Details": "nlp_pages/topic_details.py",
# }

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

# # Example additional selector before NLP section (like district)
# st.sidebar.selectbox("Select District", ["All", "Kathmandu", "Pokhara", "Lalitpur"])  # Optional line

# # --- NLP Section Separated ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("### NLP Tools")

# selected_nlp = st.sidebar.selectbox(
#     "Select NLP Section",
#     ["Select..."] + nlp_sections,
#     index=0,
#     key="nlp_section"
# )

# if selected_nlp != "Select...":
#     st.session_state.page = selected_nlp

# # Page Display Logic
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
#     - NLP Sections (Language Prediction, NER Prediction, Sentiment Analysis, Summary Details, Topic Details)
#     """)
# else:
#     page_path = PAGES.get(st.session_state.page, None)
#     if page_path:
#         try:
#             # Dynamically calculate file path
#             base_dir = os.path.dirname(__file__)
#             abs_path = os.path.join(base_dir, page_path)

#             if os.path.exists(abs_path):
#                 with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
#                     code = f.read()
#                     exec(code, globals())
#             else:
#                 st.error(f"Error: File not found at {abs_path}")

#         except Exception as e:
#             st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
#     else:
#         st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")


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
# main_sections = ["Climate Sections", "Weather Sections", "Glacier Lake Data", "Socio-Economic Impact"]

# # Subpages Mapping
# subpages_mapping = {
#     "Climate Sections": [
#         "Climate Data - Vulnerability",
#         "Climate Data - Analysis",
#         "Climate Data - Predictions"
#     ],
#     "Weather Sections": [
#         "Weather Data Visualization",
#         "Weather Impact Assessment",
#         "Weather Predictions"
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

# # NLP Sections
# nlp_sections = [
#     "Language Prediction",
#     "NER Prediction",
#     "Sentiment Analysis",
#     "Summary Details",
#     "Topic Details"
# ]

# # File Mapping
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
#     "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
#     "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
#     "Weather Data Visualization": "weather_pages/weather_data_visualization.py",
#     "Weather Impact Assessment": "weather_pages/weather_impact_assesment.py",
#     "Weather Predictions": "weather_pages/weather_predictions.py",
#     "Socio-Economic Impact - Overview": "",  # Dummy
#     "Socio-Economic Impact - Trends": "",  # Dummy
#     "Sentiment Analysis": "nlp_pages/sentiment_analysis.py",
#     "Language Prediction": "nlp_pages/language_prediction.py",
#     "NER Prediction": "nlp_pages/ner_prediction.py",
#     "Summary Details": "nlp_pages/summary_details.py",
#     "Topic Details": "nlp_pages/topic_details.py",
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

    
# # --- NLP Section Separated at Bottom ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("### NLP Tools")

# selected_nlp = st.sidebar.selectbox(
#     "Select NLP Section",
#     ["Select..."] + nlp_sections,
#     index=0,
#     key="nlp_section"
# )

# if selected_nlp != "Select...":
#     st.session_state.page = selected_nlp

# # Page Display Logic
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
#     - NLP Sections (Language Prediction, NER Prediction, Sentiment Analysis, Summary Details, Topic Details)
#     """)
# else:
#     page_path = PAGES.get(st.session_state.page, None)
#     if page_path:
#         try:
#             base_dir = os.path.dirname(__file__)
#             abs_path = os.path.join(base_dir, page_path)

#             if os.path.exists(abs_path):
#                 with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
#                     code = f.read()
#                     exec(code, globals())
#             else:
#                 st.error(f"Error: File not found at {abs_path}")
#         except Exception as e:
#             st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
#     else:
#         st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")


import streamlit as st
import os
from model_loader import load_spacy_model  # Import model loader function

# Initialize the spaCy model
nlp = load_spacy_model()

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
        "Socio-Economic Impact - Overview",
        "Socio-Economic Impact - Trends"
    ]
}

# NLP Sections
nlp_sections = [
    "Language Prediction",
    "NER Prediction",
    "Sentiment Analysis",
    "Summary Details",
    "Topic Details"
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
    "Socio-Economic Impact - Overview": "",  # Dummy
    "Socio-Economic Impact - Trends": "",  # Dummy
    "Sentiment Analysis": "nlp_pages/sentiment_analysis.py",
    "Language Prediction": "nlp_pages/language_prediction.py",
    "NER Prediction": "nlp_pages/ner_prediction.py",
    "Summary Details": "nlp_pages/summary_details.py",
    "Topic Details": "nlp_pages/topic_details.py",
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

# Show the District Dropdown above the NLP section if the user selects "Weather Data Visualization"
if st.session_state.page == "Weather Data Visualization || Weather Impact Assesment":
    import pandas as pd
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../feature_engineering/weather_and_temp_feature_engineering.csv'))
    df = pd.read_csv(DATA_PATH)
    districts = df['district'].dropna().unique().tolist()
    selected_district = st.sidebar.selectbox("Select District", ['All'] + districts)

    if selected_district != 'All':
        df = df[df['district'] == selected_district]

# --- NLP Section Separated at Bottom ---
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

# Page Display Logic
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
    - NLP Sections (Language Prediction, NER Prediction, Sentiment Analysis, Summary Details, Topic Details)
    """)
else:
    page_path = PAGES.get(st.session_state.page, None)
    if page_path:
        try:
            base_dir = os.path.dirname(__file__)
            abs_path = os.path.join(base_dir, page_path)

            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                    exec(code, globals())
            else:
                st.error(f"Error: File not found at {abs_path}")
        except Exception as e:
            st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
    else:
        st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")
