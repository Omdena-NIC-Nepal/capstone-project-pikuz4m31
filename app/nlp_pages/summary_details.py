# import streamlit as st
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from sumy.utils import get_stop_words
# import logging

# # Set up logging for better error tracking
# logging.basicConfig(level=logging.INFO)

# # Summarize text using sumy
# def summarize_text_with_sumy(text):
#     try:
#         if not text.strip():
#             return "No input text provided."

#         # Create a parser from the input text
#         parser = PlaintextParser.from_string(text, Tokenizer("english"))
        
#         # Using LSA Summarizer
#         summarizer = LsaSummarizer()
#         summarizer.stop_words = get_stop_words("english")  # Set stop words
        
#         # Generate summary (3 sentences by default)
#         summary_sentences = summarizer(parser.document, 3)
        
#         # Join sentences from the summary and return the summarized text
#         return ' '.join([str(sentence) for sentence in summary_sentences])
#     except Exception as e:
#         logging.error(f"Error during summarization: {e}")
#         return f"An error occurred during summarization: {e}"

# # Summarization for preloaded summaries
# def load_summary_outputs():
#     # Load preprocessed summaries (for demo, it would simulate preloaded summaries)
#     return {
#         "summary_file_1.txt": "This is a sample preloaded summary 1.",
#         "summary_file_2.txt": "This is a sample preloaded summary 2."
#     }

# # Render summaries in styled HTML table
# def render_summary_text(summary_text):
#     if not summary_text:
#         st.info("No summary found.")
#         return

#     # Display the summary in a box
#     st.text_area("Generated Summary:", summary_text, height=200)

# # Main function to handle Streamlit interactions
# def main():
#     st.title("🧠 Text Summarization App")
#     st.subheader("Choose Your Mode")

#     app_mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

#     if app_mode == "Live Input":
#         st.markdown("#### Enter text to get a summary in real-time.")
#         user_text = st.text_area("Input text:", height=200)

#         if user_text:
#             if st.button("Summarize"):
#                 summary_text = summarize_text_with_sumy(user_text)
#                 st.subheader("📝 Summary:")
#                 render_summary_text(summary_text)
#             else:
#                 st.warning("Please enter text to summarize.")

#     else:
#         st.markdown("#### Select a preprocessed file to view the summary.")
#         summary_outputs = load_summary_outputs()
#         if not summary_outputs:
#             st.warning("No preloaded summaries found.")
#             return

#         # Prepare file list for selection
#         files_with_spaces = {
#             file: file.replace('_', ' ').replace('.txt', '').replace('summary', '').strip()
#             for file in summary_outputs.keys()
#         }

#         files_with_spaces = {"SELECT HERE": "SELECT HERE"} | files_with_spaces

#         # File selection widget
#         selected_file_display_name = st.selectbox("Select a preprocessed summary file:", list(files_with_spaces.values()))
#         selected_file = next(key for key, value in files_with_spaces.items() if value == selected_file_display_name)

#         if selected_file != "SELECT HERE":
#             summary_text = summary_outputs[selected_file]
#             if isinstance(summary_text, str):
#                 st.subheader(f"📝 Summary in: {selected_file}")
#                 render_summary_text(summary_text)
#             else:
#                 st.error("Invalid summary structure in the selected file.")
#         else:
#             st.info("Please select a file to view its summary.")

# if __name__ == "__main__":
#     main()


import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words
import logging
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# Summarize text using sumy
def summarize_text_with_sumy(text):
    try:
        if not text.strip():
            return "No input text provided."

        # Create a parser from the input text
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        
        # Using LSA Summarizer
        summarizer = LsaSummarizer()
        summarizer.stop_words = get_stop_words("english")  # Set stop words
        
        # Generate summary (3 sentences by default)
        summary_sentences = summarizer(parser.document, 3)
        
        # Join sentences from the summary and return the summarized text
        return ' '.join([str(sentence) for sentence in summary_sentences])
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return f"An error occurred during summarization: {e}"

# Summarization for preloaded summaries
def load_summary_outputs():
    # Load preprocessed summaries (for demo, it would simulate preloaded summaries)
    return {
        "summary_file_1.txt": "This is a sample preloaded summary 1.",
        "summary_file_2.txt": "This is a sample preloaded summary 2."
    }

# Render summaries in styled HTML table
def render_summary_text(summary_text):
    if not summary_text:
        st.info("No summary found.")
        return

    # Display the summary in a box
    st.text_area("Generated Summary:", summary_text, height=200)

# Main function to handle Streamlit interactions
def main():
    st.title("🧠 Text Summarization App")
    st.subheader("Choose Your Mode")

    app_mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

    if app_mode == "Live Input":
        st.markdown("#### Enter text to get a summary in real-time.")
        user_text = st.text_area("Input text:", height=200)

        if user_text:
            if st.button("Summarize"):
                summary_text = summarize_text_with_sumy(user_text)
                st.subheader("📝 Summary:")
                render_summary_text(summary_text)
            else:
                st.warning("Please enter text to summarize.")

    else:
        st.markdown("#### Select a preprocessed file to view the summary.")
        summary_outputs = load_summary_outputs()
        if not summary_outputs:
            st.warning("No preloaded summaries found.")
            return

        # Prepare file list for selection
        files_with_spaces = {
            file: file.replace('_', ' ').replace('.txt', '').replace('summary', '').strip()
            for file in summary_outputs.keys()
        }

        files_with_spaces = {"SELECT HERE": "SELECT HERE"} | files_with_spaces

        # File selection widget
        selected_file_display_name = st.selectbox("Select a preprocessed summary file:", list(files_with_spaces.values()))
        selected_file = next(key for key, value in files_with_spaces.items() if value == selected_file_display_name)

        if selected_file != "SELECT HERE":
            summary_text = summary_outputs[selected_file]
            if isinstance(summary_text, str):
                st.subheader(f"📝 Summary in: {selected_file}")
                render_summary_text(summary_text)
            else:
                st.error("Invalid summary structure in the selected file.")
        else:
            st.info("Please select a file to view its summary.")

if __name__ == "__main__":
    main()