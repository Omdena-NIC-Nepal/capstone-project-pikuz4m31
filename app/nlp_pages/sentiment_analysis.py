

# sentimental_analysis.py in nlp_pages

import os
import streamlit as st
from textblob import TextBlob
import logging
import pickle
import spacy
from langdetect import detect
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import CountVectorizer

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load models function
def load_all_models(base_path='trained_model'):
    models = {}
    if not os.path.exists(base_path):
        logging.warning(f"Model directory '{base_path}' not found.")
        return models

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.pkl'):
                    model_name = os.path.splitext(file)[0]
                    file_path = os.path.join(folder_path, file)
                    try:
                        with open(file_path, 'rb') as f:
                            models[model_name] = pickle.load(f)
                            logging.info(f"Loaded model: {model_name} from {file_path}")
                    except Exception as e:
                        logging.error(f"Failed to load {model_name}: {e}")
    return models

# Extract sentiment scores
def extract_sentiment_scores(model_dict):
    try:
        if isinstance(model_dict, dict) and 'textblob' in model_dict and 'vader' in model_dict:
            textblob = model_dict.get('textblob', {})
            vader = model_dict.get('vader', {})
            return {
                "textblob": {
                    "polarity": textblob.get("polarity", 0.0),
                    "subjectivity": textblob.get("subjectivity", 0.0)
                },
                "vader": {
                    "neg": vader.get("neg", 0.0),
                    "neu": vader.get("neu", 1.0),
                    "pos": vader.get("pos", 0.0),
                    "compound": vader.get("compound", 0.0)
                }
            }
    except Exception as e:
        logging.error(f"Error extracting sentiment: {e}")
    return None

# Main sentiment analysis function
def sentiment_analysis(article_text):
    try:
        models = load_all_models()

        sentiment_scores = {}
        for model_name, model in models.items():
            scores = extract_sentiment_scores(model)
            if scores:
                sentiment_scores[model_name] = scores

        # Fallback if no models loaded
        if not sentiment_scores:
            polarity = TextBlob(article_text).sentiment.polarity
            subjectivity = TextBlob(article_text).sentiment.subjectivity
            sentiment_scores["textblob_fallback"] = {
                "textblob": {
                    "polarity": polarity,
                    "subjectivity": subjectivity
                },
                "vader": {
                    "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0
                }
            }

        return sentiment_scores

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

# Streamlit UI
def display_sentiment_analysis():
    st.title("Sentiment Analysis")

    article_text = st.text_area("Enter Article Text", "This is an example article about climate change.")

    if st.button("Analyze Sentiment"):
        try:
            sentiment_scores = sentiment_analysis(article_text)

            if sentiment_scores:
                st.write("### Sentiment Scores")
                for model, scores in sentiment_scores.items():
                    st.write(f"**{model}**")
                    st.write(f"- TextBlob Polarity: {scores['textblob']['polarity']}")
                    st.write(f"- TextBlob Subjectivity: {scores['textblob']['subjectivity']}")
                    st.write(f"- VADER Negative: {scores['vader']['neg']}")
                    st.write(f"- VADER Neutral: {scores['vader']['neu']}")
                    st.write(f"- VADER Positive: {scores['vader']['pos']}")
                    st.write(f"- VADER Compound: {scores['vader']['compound']}")
            else:
                st.error("No sentiment scores found or model loading failed.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logging.exception("Error during Streamlit display")

# Call UI function
if __name__ == "__main__" or st._is_running_with_streamlit:
    display_sentiment_analysis()
