import streamlit as st
from summa.summarizer import summarize
import nltk
import logging
import os

# -------------------- Setup NLTK --------------------
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)

# -------------------- Summarization Function --------------------
def summarize_text_with_summa(text):
    try:
        if not text.strip():
            return "No input text provided."
        
        summary = summarize(text, ratio=0.3)  # Keep 30% of the original text
        return summary if summary else "Text too short or unstructured to summarize effectively."
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return f"An error occurred during summarization: {e}"

# -------------------- Demo Preloaded Data --------------------
def load_summary_outputs():
    return {
        "summary_file_1.txt": "This is a sample preloaded summary 1.",
        "summary_file_2.txt": "This is a sample preloaded summary 2."
    }

# -------------------- Display Summary --------------------
def render_summary_text(summary_text):
    if not summary_text:
        st.info("No summary found.")
        return
    st.text_area("Generated Summary:", summary_text, height=200)

# -------------------- Streamlit App --------------------
def main():
    setup_nltk()

    st.title("🧠 Text Summarization App")
    st.subheader("Choose Your Mode")

    app_mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

    if app_mode == "Live Input":
        st.markdown("#### Enter text to get a summary in real-time.")
        user_text = st.text_area("Input text:", height=200)

        if user_text:
            if st.button("Summarize"):
                summary_text = summarize_text_with_summa(user_text)
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

        files_with_spaces = {
            file: file.replace('_', ' ').replace('.txt', '').replace('summary', '').strip()
            for file in summary_outputs.keys()
        }

        files_with_spaces = {"SELECT HERE": "SELECT HERE"} | files_with_spaces
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

# -------------------- Run App --------------------
if __name__ == "__main__":
    main()
