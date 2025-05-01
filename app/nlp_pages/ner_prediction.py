
# import pickle
# import streamlit as st
# import spacy
# import logging
# import os

# # Initialize logging
# logging.basicConfig(level=logging.INFO)

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Corrected path to look inside the 'ner' subdirectory for preloaded .pkl files
# def load_ner_outputs(base_path='../nlp/models/trained_model/ner'):
#     ner_models = {}

#     if not os.path.exists(base_path):
#         logging.warning(f"NER output folder '{base_path}' not found.")
#         return ner_models

#     for filename in os.listdir(base_path):
#         if filename.endswith('.pkl'):
#             file_path = os.path.join(base_path, filename)
#             try:
#                 with open(file_path, 'rb') as f:
#                     ner_data = pickle.load(f)
#                     # Inspecting the structure of ner_data
#                     logging.info(f"Loaded NER data from {filename} with structure: {type(ner_data)}")
#                     ner_models[filename] = ner_data
#                     logging.info(f"Successfully loaded NER data from {filename}")
#             except Exception as e:
#                 logging.error(f"Error loading {filename}: {e}")
    
#     return ner_models

# # Function to process text and perform NER
# def perform_ner_on_input(text):
#     # Process the input text with spaCy's NER model
#     doc = nlp(text)

#     # Extract and store entities
#     entities = {
#         "PERSON": [],
#         "ORG": [],
#         "GPE": [],
#         "LOC": [],
#         "NORP": [],  # Nationalities, religious, or political groups
#         "MONEY": [],
#         "DATE": [],
#         "TIME": [],
#         "PERCENT": [],
#         "FAC": []  # Facilities, like buildings
#     }

#     for ent in doc.ents:
#         if ent.label_ in entities:
#             entities[ent.label_].append(ent.text)

#     # Clean up and remove empty lists
#     entities = {key: value for key, value in entities.items() if value}

#     return entities

# # Function to display NER results from preloaded models
# def display_ner_from_saved_models():
#     st.title("NER from Preprocessed Files")

#     ner_outputs = load_ner_outputs()

#     if not ner_outputs:
#         st.warning("No NER output files found.")
#         return

#     selected_file = st.selectbox("Choose a file to display NER results", list(ner_outputs.keys()))

#     if selected_file:
#         entities = ner_outputs[selected_file]
        
#         # Check if entities are structured in a recognizable way
#         if isinstance(entities, dict):
#             st.write(f"### Named Entities in: {selected_file}")
#             for category, items in entities.items():
#                 st.subheader(category.capitalize())
#                 if items:
#                     for item in set(items):
#                         st.write(f"- {item}")
#                 else:
#                     st.write("None found.")
#         else:
#             st.write(f"Unable to interpret the structure of the NER data in {selected_file}.")
#             st.write(f"Data type: {type(entities)}")
#             st.write("Inspect the file manually for a detailed structure.")

# # Streamlit UI for user text input and NER predictions
# def display_ner_for_user_input():
#     st.title("NER Prediction for User Input")

#     # Text input for user to enter their text
#     user_input_text = st.text_area("Enter text for NER analysis", height=200)

#     if st.button("Analyze NER"):
#         if user_input_text.strip():
#             # Perform NER on user input
#             entities = perform_ner_on_input(user_input_text)

#             # Display the results
#             if entities:
#                 st.write("### Recognized Named Entities:")
#                 for category, items in entities.items():
#                     st.subheader(category.capitalize())
#                     for item in set(items):
#                         st.write(f"- {item}")
#             else:
#                 st.write("No named entities found in the text.")
#         else:
#             st.warning("Please enter some text for analysis.")

# if __name__ == "__main__" or st._is_running_with_streamlit:
#     display_ner_for_user_input()


# import pickle
# import streamlit as st
# import spacy
# import logging
# import os
# import pandas as pd  # Import pandas to help with tabular data

# # Initialize logging
# logging.basicConfig(level=logging.INFO)

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Corrected path to look inside the 'ner' subdirectory for preloaded .pkl files
# def load_ner_outputs(base_path='../nlp/models/trained_model/ner'):
#     ner_models = {}

#     if not os.path.exists(base_path):
#         logging.warning(f"NER output folder '{base_path}' not found.")
#         return ner_models

#     for filename in os.listdir(base_path):
#         if filename.endswith('.pkl'):
#             file_path = os.path.join(base_path, filename)
#             try:
#                 with open(file_path, 'rb') as f:
#                     ner_data = pickle.load(f)
#                     # Inspecting the structure of ner_data
#                     logging.info(f"Loaded NER data from {filename} with structure: {type(ner_data)}")
#                     ner_models[filename] = ner_data
#                     logging.info(f"Successfully loaded NER data from {filename}")
#             except Exception as e:
#                 logging.error(f"Error loading {filename}: {e}")
    
#     return ner_models

# # Function to process text and perform NER
# def perform_ner_on_input(text):
#     # Process the input text with spaCy's NER model
#     doc = nlp(text)

#     # Extract and store entities
#     entities = {
#         "PERSON": [],
#         "ORG": [],
#         "GPE": [],
#         "LOC": [],
#         "NORP": [],  # Nationalities, religious, or political groups
#         "MONEY": [],
#         "DATE": [],
#         "TIME": [],
#         "PERCENT": [],
#         "FAC": []  # Facilities, like buildings
#     }

#     for ent in doc.ents:
#         if ent.label_ in entities:
#             entities[ent.label_].append(ent.text)

#     # Clean up and remove empty lists
#     entities = {key: value for key, value in entities.items() if value}

#     return entities

# # Function to display NER results from preloaded models
# def display_ner_from_saved_models():
#     st.title("NER from Preprocessed Files")

#     ner_outputs = load_ner_outputs()

#     if not ner_outputs:
#         st.warning("No NER output files found.")
#         return

#     selected_file = st.selectbox("Choose a file to display NER results", list(ner_outputs.keys()))

#     if selected_file:
#         entities = ner_outputs[selected_file]
        
#         # Check if entities are structured in a recognizable way
#         if isinstance(entities, dict):
#             st.write(f"### Named Entities in: {selected_file}")
#             for category, items in entities.items():
#                 st.subheader(category.capitalize())
#                 if items:
#                     for item in set(items):
#                         st.write(f"- {item}")
#                 else:
#                     st.write("None found.")
#         else:
#             st.write(f"Unable to interpret the structure of the NER data in {selected_file}.")
#             st.write(f"Data type: {type(entities)}")
#             st.write("Inspect the file manually for a detailed structure.")

# # Streamlit UI for user text input and NER predictions
# def display_ner_for_user_input():
#     st.title("NER Prediction for User Input")

#     # Text input for user to enter their text
#     user_input_text = st.text_area("Enter text for NER analysis", height=200)

#     if st.button("Analyze NER"):
#         if user_input_text.strip():
#             # Perform NER on user input
#             entities = perform_ner_on_input(user_input_text)

#             # Display the results in a table format
#             if entities:
#                 st.write("### Recognized Named Entities:")

#                 # Prepare data for the table
#                 data = []
#                 for category, items in entities.items():
#                     for item in set(items):
#                         data.append({"SN": "", "Entity Type": category, "Entity": item})

#                 # Convert data into a DataFrame
#                 df = pd.DataFrame(data)

#                 # Reset the index to start from 1
#                 df.index = df.index + 1  # Start index from 1
#                 df["SN"] = df.index  # Assign the serial number to the SN column

#                 # Re-arrange columns so SN is the first column
#                 df = df[["SN", "Entity Type", "Entity"]]

#                 # Display the DataFrame as a table
#                 st.table(df)  # or use st.dataframe() for more interactivity
#             else:
#                 st.write("No named entities found in the text.")
#         else:
#             st.warning("Please enter some text for analysis.")

# if __name__ == "__main__" or st._is_running_with_streamlit:
#     display_ner_for_user_input()



# import pickle
# import streamlit as st
# import spacy
# import logging
# import os
# import pandas as pd  # Import pandas to help with tabular data

# # Initialize logging
# logging.basicConfig(level=logging.INFO)

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Corrected path to look inside the 'ner' subdirectory for preloaded .pkl files
# def load_ner_outputs(base_path='../nlp/models/trained_model/ner'):
#     ner_models = {}

#     if not os.path.exists(base_path):
#         logging.warning(f"NER output folder '{base_path}' not found.")
#         return ner_models

#     for filename in os.listdir(base_path):
#         if filename.endswith('.pkl'):
#             file_path = os.path.join(base_path, filename)
#             try:
#                 with open(file_path, 'rb') as f:
#                     ner_data = pickle.load(f)
#                     logging.info(f"Loaded NER data from {filename} with structure: {type(ner_data)}")
#                     ner_models[filename] = ner_data
#                     logging.info(f"Successfully loaded NER data from {filename}")
#             except Exception as e:
#                 logging.error(f"Error loading {filename}: {e}")
    
#     return ner_models

# # Function to process text and perform NER
# def perform_ner_on_input(text):
#     doc = nlp(text)

#     entities = {
#         "PERSON": [],
#         "ORG": [],
#         "GPE": [],
#         "LOC": [],
#         "NORP": [],
#         "MONEY": [],
#         "DATE": [],
#         "TIME": [],
#         "PERCENT": [],
#         "FAC": []
#     }

#     for ent in doc.ents:
#         if ent.label_ in entities:
#             entities[ent.label_].append(ent.text)

#     entities = {key: value for key, value in entities.items() if value}
#     return entities

# # Function to display NER results from preloaded models
# def display_ner_from_saved_models():
#     st.title("NER from Preprocessed Files")

#     ner_outputs = load_ner_outputs()

#     if not ner_outputs:
#         st.warning("No NER output files found.")
#         return

#     selected_file = st.selectbox("Choose a file to display NER results", list(ner_outputs.keys()))

#     if selected_file:
#         entities = ner_outputs[selected_file]
        
#         if isinstance(entities, dict):
#             st.write(f"### Named Entities in: {selected_file}")
#             for category, items in entities.items():
#                 st.subheader(category.capitalize())
#                 if items:
#                     for item in set(items):
#                         st.write(f"- {item}")
#                 else:
#                     st.write("None found.")
#         else:
#             st.write(f"Unable to interpret the structure of the NER data in {selected_file}.")
#             st.write(f"Data type: {type(entities)}")
#             st.write("Inspect the file manually for a detailed structure.")

# # Streamlit UI for user text input and NER predictions
# def display_ner_for_user_input():
#     st.title("NER Prediction for User Input")

#     user_input_text = st.text_area("Enter text for NER analysis", height=200)

#     if st.button("Analyze NER"):
#         if user_input_text.strip():
#             entities = perform_ner_on_input(user_input_text)

#             if entities:
#                 st.write("### Recognized Named Entities:")

#                 # Prepare data
#                 data = []
#                 for category, items in entities.items():
#                     for item in set(items):
#                         data.append({"Entity Type": category, "Entity": item})

#                 # Create DataFrame and add proper SN column
#                 df = pd.DataFrame(data)
#                 df.insert(0, "SN", range(1, len(df) + 1))

#                 # Convert to HTML with styling
#                 styled_table = df.to_html(index=False, classes="styled-table")

#                 table_css = """
#                 <style>
#                     .styled-table {
#                         width: 100%;
#                         border-collapse: collapse;
#                         margin: 10px 0;
#                         font-size: 16px;
#                         font-family: sans-serif;
#                     }
#                     .styled-table thead tr {
#                         background-color: #f2f2f2;
#                         text-align: center;
#                     }
#                     .styled-table th, .styled-table td {
#                         border: 1px solid #ddd;
#                         padding: 8px;
#                         text-align: center;
#                     }
#                 </style>
#                 """

#                 # Display in Streamlit
#                 st.markdown(table_css + styled_table, unsafe_allow_html=True)
#             else:
#                 st.write("No named entities found in the text.")
#         else:
#             st.warning("Please enter some text for analysis.")

# if __name__ == "__main__" or st._is_running_with_streamlit:
#     display_ner_for_user_input()


# import pickle
# import streamlit as st
# import spacy
# import logging
# import os
# import pandas as pd

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# # Load the spaCy model
# @st.cache_resource
# def load_spacy_model():
#     return spacy.load("en_core_web_sm")

# nlp = load_spacy_model()

# # Supported entity types
# SUPPORTED_ENTITY_LABELS = [
#     "PERSON", "ORG", "GPE", "LOC", "NORP", "MONEY", "DATE", "TIME", "PERCENT", "FAC"
# ]

# # Load preprocessed NER outputs from pickle files
# def load_ner_outputs(base_path='../nlp/models/trained_model/ner'):
#     ner_models = {}
#     if not os.path.exists(base_path):
#         logging.warning(f"NER output folder '{base_path}' not found.")
#         return ner_models

#     for filename in os.listdir(base_path):
#         if filename.endswith('.pkl'):
#             file_path = os.path.join(base_path, filename)
#             try:
#                 with open(file_path, 'rb') as f:
#                     ner_data = pickle.load(f)
#                     logging.info(f"Loaded NER data from {filename}")
#                     ner_models[filename] = ner_data
#             except Exception as e:
#                 logging.error(f"Error loading {filename}: {e}")
#     return ner_models

# # Perform NER using spaCy
# def perform_ner_on_input(text):
#     doc = nlp(text)
#     entities = {label: [] for label in SUPPORTED_ENTITY_LABELS}
#     for ent in doc.ents:
#         if ent.label_ in entities:
#             entities[ent.label_].append(ent.text)
#     return {k: v for k, v in entities.items() if v}  # Remove empty entries

# # Create styled HTML table from entity dict
# def render_entity_table(entities: dict):
#     if not entities:
#         st.info("No named entities found.")
#         return

#     data = [{"Entity Type": label, "Entity": item}
#             for label, items in entities.items() for item in set(items)]
#     df = pd.DataFrame(data)
#     df.insert(0, "SN", range(1, len(df) + 1))

#     # Convert to styled HTML table
#     styled_table = df.to_html(index=False, classes="styled-table", escape=False)

#     # CSS styling
#     table_css = """
#     <style>
#         .styled-table {
#             width: 100%;
#             border-collapse: collapse;
#             margin: 16px 0;
#             font-size: 16px;
#             font-family: sans-serif;
#         }
#         .styled-table thead tr {
#             background-color: #f2f2f2;
#             text-align: center;
#         }
#         .styled-table th, .styled-table td {
#             border: 1px solid #ddd;
#             padding: 8px;
#             text-align: center;
#         }
#         .styled-table tbody tr:nth-child(even) {
#             background-color: #f9f9f9;
#         }
#         .styled-table tbody tr:hover {
#             background-color: #f1f1f1;
#         }
#     </style>
#     """

#     # Display in Streamlit with styling
#     st.markdown(table_css, unsafe_allow_html=True)
#     st.markdown(styled_table, unsafe_allow_html=True)

# # Streamlit UI for processing user input
# def display_ner_for_user_input():
#     st.title("🧠 Named Entity Recognition (NER) - Live Input")

#     user_text = st.text_area("Enter text for NER analysis", height=200)

#     if st.button("Analyze NER"):
#         if user_text.strip():
#             entities = perform_ner_on_input(user_text)
#             st.subheader("🔍 Extracted Entities")
#             render_entity_table(entities)
#         else:
#             st.warning("Please enter some text to analyze.")

# # Streamlit UI for displaying preloaded NER results
# def display_ner_from_saved_models():
#     st.title("📁 NER from Preprocessed Files")

#     ner_outputs = load_ner_outputs()
#     if not ner_outputs:
#         st.warning("No preloaded NER outputs found.")
#         return

#     selected_file = st.selectbox("Choose a file:", list(ner_outputs.keys()))
#     if selected_file:
#         entities = ner_outputs[selected_file]
#         if isinstance(entities, dict):
#             st.subheader(f"🔍 Entities in: {selected_file}")
#             render_entity_table(entities)
#         else:
#             st.error("Invalid structure in the selected file.")

# # Entry point
# # Entry point
# def main():
#     st.title("📊 Named Entity Recognition (NER) App")
#     st.subheader("🧭 Choose Your Mode")
    
#     app_mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

#     if app_mode == "Live Input":
#         display_ner_for_user_input()
#     else:
#         display_ner_from_saved_models()


# if __name__ == "__main__" or st._is_running_with_streamlit:
#     main()



import pickle
import streamlit as st
import spacy
import logging
import os
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Supported entity types
SUPPORTED_ENTITY_LABELS = [
    "PERSON", "ORG", "GPE", "LOC", "NORP", "MONEY", "DATE", "TIME", "PERCENT", "FAC"
]

# Load preprocessed NER outputs from pickle files
@st.cache_data
def load_ner_outputs(base_path='../nlp/models/trained_model/ner'):
    ner_models = {}
    if not os.path.exists(base_path):
        logging.warning(f"NER output folder '{base_path}' not found.")
        return ner_models

    for filename in os.listdir(base_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(base_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    ner_data = pickle.load(f)
                    logging.info(f"Loaded NER data from {filename}")
                    ner_models[filename] = ner_data
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
    return ner_models

# Perform NER using spaCy
def perform_ner_on_input(text):
    doc = nlp(text)
    entities = {label: [] for label in SUPPORTED_ENTITY_LABELS}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return {k: v for k, v in entities.items() if v}  # Remove empty entries

# Create styled HTML table from entity dict
def render_entity_table(entities: dict):
    if not entities:
        st.info("No named entities found.")
        return

    data = [{"Entity Type": label, "Entity": item}
            for label, items in entities.items() for item in set(items)]
    df = pd.DataFrame(data)
    df.insert(0, "SN", range(1, len(df) + 1))

    # Convert to styled HTML table
    styled_table = df.to_html(index=False, classes="styled-table", escape=False)

    # CSS styling
    table_css = """
    <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 16px;
            font-family: sans-serif;
        }
        .styled-table thead tr {
            background-color: #f2f2f2;
            text-align: center;
        }
        .styled-table th, .styled-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .styled-table tbody tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .styled-table tbody tr:hover {
            background-color: #f1f1f1;
        }
    </style>
    """

    # Display in Streamlit with styling
    st.markdown(table_css, unsafe_allow_html=True)
    st.markdown(styled_table, unsafe_allow_html=True)

# Streamlit UI for processing user input
def display_ner_for_user_input():
    st.title("Named Entity Recognition (NER) - Live Input")

    user_text = st.text_area("Enter text for NER analysis", height=200)

    if st.button("Analyze NER"):
        if user_text.strip():
            entities = perform_ner_on_input(user_text)
            st.subheader(" Extracted Entities")
            render_entity_table(entities)
        else:
            st.warning("Please enter some text to analyze.")

# def display_ner_from_saved_models():
#     st.title(" NER from Preprocessed Files")

#     ner_outputs = load_ner_outputs()
#     if not ner_outputs:
#         st.warning("No preloaded NER outputs found.")
#         return

#     # Display the "Select File" dropdown with a placeholder
#     selected_file = st.selectbox("Select a preprocessed NER file:", ["SELECT HERE"] + list(ner_outputs.keys()))

#     # If the user selects a file (not the placeholder)
#     if selected_file != "SELECT HERE":
#         entities = ner_outputs[selected_file]
#         if isinstance(entities, dict):
#             st.subheader(f" Entities in: {selected_file}")
#             render_entity_table(entities)
#         else:
#             st.error("Invalid structure in the selected file.")
#     else:
#         # If the placeholder is selected, do nothing or display a message
#         st.info("Please select a file to view extracted NER entities.")

# Streamlit UI for displaying preloaded NER results
# Streamlit UI for displaying preloaded NER results
# Streamlit UI for displaying preloaded NER results
def display_ner_from_saved_models():
    st.title("📁 NER from Preprocessed Files")

    ner_outputs = load_ner_outputs()
    if not ner_outputs:
        st.warning("No preloaded NER outputs found.")
        return

    # Replace underscores with spaces, remove ".pkl" and "ner" for display purposes
    files_with_spaces = {
        file: file.replace('_', ' ').replace('.pkl', '').replace('ner', '').strip() 
        for file in ner_outputs.keys()
    }

    # Add a placeholder option
    files_with_spaces = {"SELECT HERE": "SELECT HERE"} | files_with_spaces

    # Display the "Select File" dropdown with a placeholder
    selected_file_display_name = st.selectbox("Select a preprocessed NER file:", list(files_with_spaces.values()))

    # Find the original filename from the selected display name
    selected_file = next(key for key, value in files_with_spaces.items() if value == selected_file_display_name)

    # If the user selects a file (not the placeholder)
    if selected_file != "SELECT HERE":
        entities = ner_outputs[selected_file]
        if isinstance(entities, dict):
            st.subheader(f"🔍 Entities in: {selected_file}")
            render_entity_table(entities)
        else:
            st.error("Invalid structure in the selected file.")
    else:
        # If the placeholder is selected, do nothing or display a message
        st.info("Please select a file to view extracted NER entities.")

# Enhanced UI with sections
def main():
    st.title(" Named Entity Recognition (NER) App")
    st.subheader(" Choose Your Mode")

    app_mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

    if app_mode == "Live Input":
        st.markdown("#### Enter text to analyze entities in real-time.")
        display_ner_for_user_input()
    else:
        st.markdown("#### Select a preprocessed file to view extracted NER entities.")
        display_ner_from_saved_models()

if __name__ == "__main__" or st._is_running_with_streamlit:
    main()
