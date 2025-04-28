# import streamlit as st
# import os

# # Initialize session state
# if "page" not in st.session_state:
#     st.session_state.page = "Home"
# if "climate_selection" not in st.session_state:
#     st.session_state.climate_selection = "Select..."
# if "glacier_selection" not in st.session_state:
#     st.session_state.glacier_selection = "Select..."
# if "socio_economic_selection" not in st.session_state:
#     st.session_state.socio_economic_selection = "Select..."

# # Sidebar layout
# st.sidebar.markdown("### Main")  # Title for the main section
# if st.sidebar.button("🏠 Home"):
#     st.session_state.page = "Home"
#     st.session_state.climate_selection = "Select..."
#     st.session_state.glacier_selection = "Select..."
#     st.session_state.socio_economic_selection = "Select..."

# # Now we avoid the <hr> tag to minimize gaps between sections
# st.sidebar.markdown("### Navigations")

# # Climate Pages
# climate_pages = [
#     "Climate Data - Vulnerability",
#     "Climate Data - Analysis",
#     "Climate Data - Predictions"
# ]

# # Create full list for dropdown
# climate_dropdown_options = ["Select..."] + climate_pages

# # Find the correct index for current selection
# climate_current_index = climate_dropdown_options.index(st.session_state.climate_selection)

# # Render selectbox for Climate
# selected_climate_option = st.sidebar.selectbox(
#     "Climate Sections",
#     options=climate_dropdown_options,
#     index=climate_current_index,
#     key="climate_selection"
# )

# # If a valid page is selected for Climate
# if selected_climate_option in climate_pages:
#     st.session_state.page = selected_climate_option

# # Glacier Pages
# st.sidebar.markdown("### Glacier Data")  # Title for Glacier section
# glacier_pages = [
#     "Glacier Data - Overview",
#     "Glacier Data - Trends"
# ]

# # Create full list for dropdown
# glacier_dropdown_options = ["Select..."] + glacier_pages

# # Find the correct index for current selection
# glacier_current_index = glacier_dropdown_options.index(st.session_state.glacier_selection)

# # Render selectbox for Glacier
# selected_glacier_option = st.sidebar.selectbox(
#     "Glacier Data",
#     options=glacier_dropdown_options,
#     index=glacier_current_index,
#     key="glacier_selection"
# )

# # Socio-Economic Pages
# st.sidebar.markdown("### Socio-Economic Impact")  # Title for Socio-Economic section
# socio_economic_pages = [
#     "Socio-Economic Impact - Overview",
#     "Socio-Economic Impact - Trends"
# ]

# # Create full list for dropdown
# socio_economic_dropdown_options = ["Select..."] + socio_economic_pages

# # Find the correct index for current selection
# socio_economic_current_index = socio_economic_dropdown_options.index(st.session_state.socio_economic_selection)

# # Render selectbox for Socio-Economic
# selected_socio_economic_option = st.sidebar.selectbox(
#     "Socio-Economic Impact",
#     options=socio_economic_dropdown_options,
#     index=socio_economic_current_index,
#     key="socio_economic_selection"
# )

# # File mapping for page execution (currently no real pages, just placeholders)
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Data - Overview": "",  # Dummy page
#     "Glacier Data - Trends": "",  # Dummy page
#     "Socio-Economic Impact - Overview": "",  # Dummy page
#     "Socio-Economic Impact - Trends": ""  # Dummy page
# }

# # Render selected page
# if st.session_state.page == "Home":
#     st.write("""
#     ### App Overview:
#     This app is designed to help monitor, analyze, and predict climate impacts, with a focus on Nepal's climate data. 

#     Key Features:
#     - **Vulnerability Analysis**: Identifies regions most at risk due to extreme climate conditions like high temperatures and precipitation.
#     - **Climate Data Analysis**: Analyzes trends and detects outliers in climate data such as temperature and precipitation.
#     - **Temperature Predictions**: Uses machine learning to forecast future temperature trends based on historical data.
#     - **Glacier Data**: (Coming soon) Explore glacier melt trends and their impact on water resources.
#     - **Socio-Economic Impact**: (Coming soon) Analyze how climate change affects livelihoods, migration, and economy.

#     Each section offers interactive charts, data analysis, and predictive models to better understand the climate change impact.
#     """)
# else:
#     # For now, these pages are placeholders, so no actual code execution.
#     if st.session_state.page in PAGES:
#         if PAGES[st.session_state.page]:
#             page_path = PAGES[st.session_state.page]
#             # with open(page_path, "r") as f:
#             #     code = f.read()
#             #     exec(code, globals())
#             with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
#               code = f.read()
#               exec(code, globals())

#         else:
#             st.write(f"{st.session_state.page} is a dummy page.")
#     else:
#         st.error(f"Page `{st.session_state.page}` not found.")


# # import streamlit as st
# # import os
# # import pandas as pd
# # # import from utils base  # Import your backend base.py for model training
# # from utils.base import load_and_preprocess_data, train_model, evaluate_model


# # # Initialize session state
# # if "page" not in st.session_state:
# #     st.session_state.page = "Home"
# # if "climate_selection" not in st.session_state:
# #     st.session_state.climate_selection = "Select..."
# # if "glacier_selection" not in st.session_state:
# #     st.session_state.glacier_selection = "Select..."
# # if "socio_economic_selection" not in st.session_state:
# #     st.session_state.socio_economic_selection = "Select..."

# # # Sidebar layout
# # st.sidebar.markdown("### Main")  
# # if st.sidebar.button("🏠 Home"):
# #     st.session_state.page = "Home"
# #     st.session_state.climate_selection = "Select..."
# #     st.session_state.glacier_selection = "Select..."
# #     st.session_state.socio_economic_selection = "Select..."

# # st.sidebar.markdown("### Navigations")

# # # Climate Pages
# # climate_pages = [
# #     "Climate Data - Vulnerability",
# #     "Climate Data - Analysis",
# #     "Climate Data - Predictions"
# # ]
# # climate_dropdown_options = ["Select..."] + climate_pages
# # climate_current_index = climate_dropdown_options.index(st.session_state.climate_selection)

# # selected_climate_option = st.sidebar.selectbox(
# #     "Climate Sections",
# #     options=climate_dropdown_options,
# #     index=climate_current_index,
# #     key="climate_selection"
# # )
# # if selected_climate_option in climate_pages:
# #     st.session_state.page = selected_climate_option

# # # Glacier Pages
# # st.sidebar.markdown("### Glacier Data")
# # glacier_pages = [
# #     "Glacier Data - Overview",
# #     "Glacier Data - Trends"
# # ]
# # glacier_dropdown_options = ["Select..."] + glacier_pages
# # glacier_current_index = glacier_dropdown_options.index(st.session_state.glacier_selection)
# # selected_glacier_option = st.sidebar.selectbox(
# #     "Glacier Data",
# #     options=glacier_dropdown_options,
# #     index=glacier_current_index,
# #     key="glacier_selection"
# # )

# # # Socio-Economic Pages
# # st.sidebar.markdown("### Socio-Economic Impact")
# # socio_economic_pages = [
# #     "Socio-Economic Impact - Overview",
# #     "Socio-Economic Impact - Trends"
# # ]
# # socio_economic_dropdown_options = ["Select..."] + socio_economic_pages
# # socio_economic_current_index = socio_economic_dropdown_options.index(st.session_state.socio_economic_selection)
# # selected_socio_economic_option = st.sidebar.selectbox(
# #     "Socio-Economic Impact",
# #     options=socio_economic_dropdown_options,
# #     index=socio_economic_current_index,
# #     key="socio_economic_selection"
# # )

# # # File mapping for page execution
# # PAGES = {
# #     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
# #     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
# #     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
# #     "Glacier Data - Overview": "",
# #     "Glacier Data - Trends": "",
# #     "Socio-Economic Impact - Overview": "",
# #     "Socio-Economic Impact - Trends": ""
# # }

# # # Render selected page
# # if st.session_state.page == "Home":
# #     st.write("""
# #     ### App Overview:
# #     This app is designed to monitor, analyze, and predict climate impacts, focusing on Nepal's climate data.
# #     """)

# #     st.markdown("---")
# #     st.header("🛠️ Train Temperature Prediction Model")

# #     uploaded_file = st.file_uploader("Upload your climate dataset (CSV with Date, Temperature)", type=["csv"])

# #     if uploaded_file:
# #         df = pd.read_csv(uploaded_file)
# #         df['Date'] = pd.to_datetime(df['Date'])

# #         st.subheader("Preview of Dataset")
# #         st.dataframe(df.head())

# #         st.sidebar.markdown("---")
# #         st.sidebar.header("🔧 Model Training Settings")

# #         feature_set = st.sidebar.radio(
# #             "Select Feature Engineering Level",
# #             ("Basic", "Rich", "Very Rich")
# #         )

# #         scaling = st.sidebar.checkbox("Apply Feature Scaling?", value=True)

# #         tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning?", value=False)

# #         model_choice = st.sidebar.selectbox(
# #             "Choose Model",
# #             ("LinearRegression", "Ridge", "Lasso", "GradientBoosting")
# #         )

# #         if st.sidebar.button("🚀 Train Model Now"):
# #             with st.spinner("Training model..."):
# #                 model, mse, scaler, features_used = base.train_model(
# #                     df,
# #                     feature_set=feature_set,
# #                     scaling=scaling,
# #                     tuning=tuning,
# #                     model_choice=model_choice
# #                 )

# #             st.success("✅ Model Trained Successfully!")
# #             st.write(f"### Results:")
# #             st.write(f"**Model:** {model_choice}")
# #             st.write(f"**Features Used:** {features_used}")
# #             st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")

# #             if scaler:
# #                 st.info("Feature Scaling was applied.")
# #             if tuning:
# #                 st.info("Hyperparameter Tuning was performed.")

# #     else:
# #         st.info("Upload a dataset to start training!")

# # else:
# #     if st.session_state.page in PAGES:
# #         if PAGES[st.session_state.page]:
# #             page_path = PAGES[st.session_state.page]
# #             with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
# #                 code = f.read()
# #                 exec(code, globals())
# #         else:
# #             st.write(f"{st.session_state.page} is a dummy page.")
# #     else:
# #         st.error(f"Page `{st.session_state.page}` not found.")


# import streamlit as st
# import os
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Initialize session state if it doesn't already exist
# if "page" not in st.session_state:
#     st.session_state.page = "Home"
# if "climate_selection" not in st.session_state:
#     st.session_state.climate_selection = "Select..."
# if "glacier_selection" not in st.session_state:
#     st.session_state.glacier_selection = "Select..."
# if "socio_economic_selection" not in st.session_state:
#     st.session_state.socio_economic_selection = "Select..."

# # Function to scale features
# def scale_features(X):
#     """Scales the features of X using StandardScaler."""
#     scaler = StandardScaler()
#     return scaler.fit_transform(X)

# # Sidebar layout
# st.sidebar.markdown("### Main")  # Title for the main section

# # Home button to reset to Home page
# if st.sidebar.button("🏠 Home"):
#     st.session_state.page = "Home"
#     st.session_state.climate_selection = "Select..."
#     st.session_state.glacier_selection = "Select..."
#     st.session_state.socio_economic_selection = "Select..."

# # Sidebar navigation sections
# st.sidebar.markdown("### Navigations")

# # Climate Pages
# climate_pages = [
#     "Climate Data - Vulnerability",
#     "Climate Data - Analysis",
#     "Climate Data - Predictions"
# ]

# # Fix: Correct index calculation for climate selectbox
# climate_index = 0 if st.session_state.climate_selection == "Select..." else climate_pages.index(st.session_state.climate_selection) + 1

# selected_climate = st.sidebar.selectbox(
#     "Climate Sections",
#     ["Select..."] + climate_pages,
#     index=climate_index,
#     key="climate_selection"
# )
# if selected_climate in climate_pages:
#     st.session_state.page = selected_climate

# # Glacier Pages
# st.sidebar.markdown("### Glacier Data")  # Title for Glacier section
# glacier_pages = [
#     "Glacier Data - Overview",
#     "Glacier Data - Trends"
# ]

# # Fix: Correct index calculation for glacier selectbox
# glacier_index = 0 if st.session_state.glacier_selection == "Select..." else glacier_pages.index(st.session_state.glacier_selection) + 1

# selected_glacier = st.sidebar.selectbox(
#     "Glacier Data",
#     ["Select..."] + glacier_pages,
#     index=glacier_index,
#     key="glacier_selection"
# )

# # Socio-Economic Pages
# st.sidebar.markdown("### Socio-Economic Impact")  # Title for Socio-Economic section
# socio_economic_pages = [
#     "Socio-Economic Impact - Overview",
#     "Socio-Economic Impact - Trends"
# ]

# # Fix: Correct index calculation for socio-economic selectbox
# socio_economic_index = 0 if st.session_state.socio_economic_selection == "Select..." else socio_economic_pages.index(st.session_state.socio_economic_selection) + 1

# selected_socio_economic = st.sidebar.selectbox(
#     "Socio-Economic Impact",
#     ["Select..."] + socio_economic_pages,
#     index=socio_economic_index,
#     key="socio_economic_selection"
# )

# # File mapping for page execution (currently no real pages, just placeholders)
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Data - Overview": "",  # Dummy page
#     "Glacier Data - Trends": "",  # Dummy page
#     "Socio-Economic Impact - Overview": "",  # Dummy page
#     "Socio-Economic Impact - Trends": ""  # Dummy page
# }

# # Render selected page
# if st.session_state.page == "Home":
#     st.write("""
#     ### App Overview:
#     This app is designed to help monitor, analyze, and predict climate impacts, with a focus on Nepal's climate data. 

#     Key Features:
#     - **Vulnerability Analysis**: Identifies regions most at risk due to extreme climate conditions like high temperatures and precipitation.
#     - **Climate Data Analysis**: Analyzes trends and detects outliers in climate data such as temperature and precipitation.
#     - **Temperature Predictions**: Uses machine learning to forecast future temperature trends based on historical data.
#     - **Glacier Data**: (Coming soon) Explore glacier melt trends and their impact on water resources.
#     - **Socio-Economic Impact**: (Coming soon) Analyze how climate change affects livelihoods, migration, and economy.

#     Each section offers interactive charts, data analysis, and predictive models to better understand the climate change impact.
#     """)
# else:
#     # For now, these pages are placeholders, so no actual code execution.
#     if st.session_state.page in PAGES:
#         if PAGES[st.session_state.page]:
#             page_path = PAGES[st.session_state.page]
#             try:
#                 with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
#                     code = f.read()
#                     print(f"Executing code for page: {st.session_state.page}")
#                     exec(code, globals())  # Execute the code
#             except Exception as e:
#                 st.error(f"Error in executing page `{st.session_state.page}`: {str(e)}")
#                 print(f"Error in executing {st.session_state.page}: {str(e)}")
#                 st.write(f"Traceback: {str(e)}")
#         else:
#             st.write(f"{st.session_state.page} is a dummy page.")
#     else:
#         st.error(f"Page `{st.session_state.page}` not found.")


# import streamlit as st
# import os
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Initialize session state if it doesn't already exist
# if "page" not in st.session_state:
#     st.session_state.page = "Home"
# if "climate_selection" not in st.session_state:
#     st.session_state.climate_selection = "Select..."
# if "glacier_selection" not in st.session_state:
#     st.session_state.glacier_selection = "Select..."
# if "socio_economic_selection" not in st.session_state:
#     st.session_state.socio_economic_selection = "Select..."

# # Function to scale features
# def scale_features(X):
#     """Scales the features of X using StandardScaler."""
#     scaler = StandardScaler()
#     return scaler.fit_transform(X)

# # Sidebar layout
# st.sidebar.markdown("### Main")  # Title for the main section

# # Home button to reset to Home page
# if st.sidebar.button("🏠 Home"):
#     st.session_state.page = "Home"
#     st.session_state.climate_selection = "Select..."
#     st.session_state.glacier_selection = "Select..."
#     st.session_state.socio_economic_selection = "Select..."

# # Sidebar navigation sections
# st.sidebar.markdown("### Navigations")

# # Climate Pages
# climate_pages = [
#     "Climate Data - Vulnerability",
#     "Climate Data - Analysis",
#     "Climate Data - Predictions"
# ]

# # Fix: Correct index calculation for climate selectbox
# climate_index = 0 if st.session_state.climate_selection == "Select..." else climate_pages.index(st.session_state.climate_selection) + 1

# selected_climate = st.sidebar.selectbox(
#     "Climate Sections",
#     ["Select..."] + climate_pages,
#     index=climate_index,
#     key="climate_selection"
# )
# if selected_climate in climate_pages:
#     st.session_state.page = selected_climate

# # Glacier Lake Pages
# st.sidebar.markdown("### Glacier Lake Data")  # Title for Glacier Lake section
# glacier_pages = [
#     "Glacier Lake Mapping & Visualization",
#     "Glacier Lake Impact Assessment",
#     "Glacier Lake Future Predictions"
# ]

# # Fix: Correct index calculation for glacier selectbox
# glacier_index = 0 if st.session_state.glacier_selection == "Select..." else glacier_pages.index(st.session_state.glacier_selection) + 1

# selected_glacier = st.sidebar.selectbox(
#     "Glacier Lake Data",
#     ["Select..."] + glacier_pages,
#     index=glacier_index,
#     key="glacier_selection"
# )
# if selected_glacier in glacier_pages:
#     st.session_state.page = selected_glacier

# # Socio-Economic Pages
# st.sidebar.markdown("### Socio-Economic Impact")  # Title for Socio-Economic section
# socio_economic_pages = [
#     "Socio-Economic Impact - Overview",
#     "Socio-Economic Impact - Trends"
# ]

# # Fix: Correct index calculation for socio-economic selectbox
# socio_economic_index = 0 if st.session_state.socio_economic_selection == "Select..." else socio_economic_pages.index(st.session_state.socio_economic_selection) + 1

# selected_socio_economic = st.sidebar.selectbox(
#     "Socio-Economic Impact",
#     ["Select..."] + socio_economic_pages,
#     index=socio_economic_index,
#     key="socio_economic_selection"
# )

# # File mapping for page execution (currently no real pages, just placeholders)
# PAGES = {
#     "Climate Data - Vulnerability": "climate_pages/1_Vulnerability.py",
#     "Climate Data - Analysis": "climate_pages/2_Analysis.py",
#     "Climate Data - Predictions": "climate_pages/3_Predictions.py",
#     "Glacier Lake Mapping & Visualization": "glacier_lake_pages/glacier_lake_mapping_visualization.py",
#     "Glacier Lake Impact Assessment": "glacier_lake_pages/glacier_lake_impact_assessment.py",
#     "Glacier Lake Future Predictions": "glacier_lake_pages/glacier_lake_future_predictions.py",
#     "Socio-Economic Impact - Overview": "",  # Dummy page
#     "Socio-Economic Impact - Trends": ""  # Dummy page
# }

# # Render selected page
# if st.session_state.page == "Home":
#     st.write("""
#     ### App Overview:
#     This app is designed to help monitor, analyze, and predict climate impacts, with a focus on Nepal's climate data. 

#     Key Features:
#     - **Vulnerability Analysis**: Identifies regions most at risk due to extreme climate conditions like high temperatures and precipitation.
#     - **Climate Data Analysis**: Analyzes trends and detects outliers in climate data such as temperature and precipitation.
#     - **Temperature Predictions**: Uses machine learning to forecast future temperature trends based on historical data.
#     - **Glacier Lake Data**: Explore glacier lake mapping, impact assessment, and future predictions.
#     - **Socio-Economic Impact**: (Coming soon) Analyze how climate change affects livelihoods, migration, and economy.

#     Each section offers interactive charts, data analysis, and predictive models to better understand the climate change impact.
#     """)
# else:
#     # For now, these pages are placeholders, so no actual code execution.
#     if st.session_state.page in PAGES:
#         if PAGES[st.session_state.page]:
#             page_path = PAGES[st.session_state.page]
#             try:
#                 with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
#                     code = f.read()
#                     print(f"Executing code for page: {st.session_state.page}")
#                     exec(code, globals())  # Execute the code
#             except Exception as e:
#                 st.error(f"Error in executing page `{st.session_state.page}`: {str(e)}")
#                 print(f"Error in executing {st.session_state.page}: {str(e)}")
#                 st.write(f"Traceback: {str(e)}")
#         else:
#             st.write(f"{st.session_state.page} is a dummy page.")
#     else:
#         st.error(f"Page `{st.session_state.page}` not found.")


# import streamlit as st
# import os
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Initialize session state
# if "main_section" not in st.session_state:
#     st.session_state.main_section = "Select..."
# if "sub_page" not in st.session_state:
#     st.session_state.sub_page = "Select..."
# if "page" not in st.session_state:
#     st.session_state.page = "Home"

# # Function to scale features
# def scale_features(X):
#     scaler = StandardScaler()
#     return scaler.fit_transform(X)

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
#             with open(page_path, "r", encoding="utf-8", errors="ignore") as f:
#                 code = f.read()
#                 print(f"Executing code for page: {st.session_state.page}")
#                 exec(code, globals())
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
