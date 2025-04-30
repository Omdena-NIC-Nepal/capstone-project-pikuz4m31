# import os
# import spacy  # For NLP tasks like Named Entity Recognition
# from textblob import TextBlob  # For sentiment analysis
# from newspaper import Article

# # Load the NLP model (you can use a pre-trained model for Named Entity Recognition)
# nlp = spacy.load("en_core_web_sm")

# # Function to process the text: Clean up and perform NLP tasks
# def process_article_content(article_text):
#     # Clean the article text (remove unwanted characters, excessive whitespace, etc.)
#     # You can customize this function as per your cleaning needs
#     article_text = article_text.strip()

#     # Perform Named Entity Recognition (NER)
#     doc = nlp(article_text)
#     entities = [ent.text for ent in doc.ents]

#     # Perform Sentiment Analysis using TextBlob
#     sentiment = TextBlob(article_text).sentiment

#     # Create a summary (you can use different summarization models if needed)
#     summary = article_text[:500]  # Simple summary: First 500 characters of the article

#     return {
#         'entities': entities,
#         'sentiment': sentiment,
#         'summary': summary
#     }

# # Function to save processed article
# def save_processed_article(processed_data, original_title, directory='processed_articles'):
#     # Ensure the directory exists
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     # Create a filename based on the original article title
#     filename = f"{original_title[:50].replace(' ', '_').replace('/', '_')}_processed.txt"  # Clean title to avoid invalid file names
    
#     # Full file path for processed article
#     file_path = os.path.join(directory, filename)
    
#     try:
#         with open(file_path, 'w', encoding='utf-8') as file:
#             # Write the processed data (entities, sentiment, summary)
#             file.write(f"Title: {original_title}\n")
#             file.write(f"Entities: {', '.join(processed_data['entities'])}\n")
#             file.write(f"Sentiment: {processed_data['sentiment']}\n")
#             file.write(f"Summary: {processed_data['summary']}\n")
#         print(f"Processed article saved as: {file_path}")
#     except Exception as e:
#         print(f"Error saving processed article {original_title}: {e}")

# # Function to load raw articles, process them, and save the results
# def process_and_save_articles(raw_articles_folder='../news_articles', processed_articles_folder='processed_articles'):
#     # Counter to track number of processed articles
#     processed_count = 0
    
#     # Loop through the files in the news_articles folder
#     for filename in os.listdir(raw_articles_folder):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(raw_articles_folder, filename)
            
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     # Read the article content (you can extract full text or metadata from each file)
#                     content = file.read()

#                     # Process the content
#                     processed_data = process_article_content(content)

#                     # Save the processed data
#                     save_processed_article(processed_data, filename, processed_articles_folder)

#                     # Increment the counter after successfully processing an article
#                     processed_count += 1
#             except Exception as e:
#                 print(f"Error processing article {filename}: {e}")
    
#     # After processing all articles, display the total count
#     print(f"Total articles processed: {processed_count}")

# # Example usage: Process and save all articles in the 'news_articles' folder
# process_and_save_articles()


# import os
# import spacy  # For NLP tasks like Named Entity Recognition
# from textblob import TextBlob  # For sentiment analysis
# from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.feature_extraction.text import CountVectorizer
# from transformers import pipeline  # For advanced text summarization
# from langdetect import detect  # For detecting language (for multilingual processing)
# import pandas as pd

# # Load spaCy model for NER
# nlp = spacy.load("en_core_web_sm")

# # Load a transformer model for text summarization
# summarizer = pipeline("summarization")

# # Function for Sentiment Analysis using TextBlob
# def get_sentiment(text):
#     return TextBlob(text).sentiment

# # Function for Named Entity Recognition (NER)
# def extract_entities(text):
#     doc = nlp(text)
#     return [ent.text for ent in doc.ents]

# # Function for Topic Modeling (LDA)
# def perform_topic_modeling(corpus, num_topics=5):
#     vectorizer = CountVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(corpus)
#     lda = LDA(n_components=num_topics, random_state=42)
#     lda.fit(X)
#     topics = lda.components_
#     return topics, vectorizer.get_feature_names_out()

# # Function for Text Summarization using Transformers
# def summarize_text(text):
#     summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
#     return summary[0]['summary_text']

# # Function for Multilingual Processing (Language Detection and Support)
# def process_multilingual_text(text):
#     lang = detect(text)
#     if lang == "ne":
#         # You can load a Nepali model or pre-trained NLP model here
#         pass  # For simplicity, just detect the language in this example
#     return lang

# # Function to Process and Save Data for Each Article
# def process_article_content(article_text):
#     # Clean the article text
#     article_text = article_text.strip()

#     # Sentiment Analysis
#     sentiment = get_sentiment(article_text)

#     # Named Entity Recognition (NER)
#     entities = extract_entities(article_text)

#     # Topic Modeling (LDA)
#     topics, features = perform_topic_modeling([article_text])

#     # Summarization
#     summary = summarize_text(article_text)

#     # Multilingual Processing (detect language and handle as needed)
#     language = process_multilingual_text(article_text)

#     return {
#         'entities': entities,
#         'sentiment': sentiment,
#         'topics': topics,
#         'summary': summary,
#         'language': language
#     }

# # Function to Save Processed Articles
# def save_processed_article(processed_data, original_title, directory='processed_articles'):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filename = f"{original_title[:50].replace(' ', '_').replace('/', '_')}_processed.txt"
#     file_path = os.path.join(directory, filename)
#     try:
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write(f"Title: {original_title}\n")
#             file.write(f"Entities: {', '.join(processed_data['entities'])}\n")
#             file.write(f"Sentiment: {processed_data['sentiment']}\n")
#             file.write(f"Topics: {processed_data['topics']}\n")
#             file.write(f"Summary: {processed_data['summary']}\n")
#             file.write(f"Language: {processed_data['language']}\n")
#         print(f"Processed article saved as: {file_path}")
#     except Exception as e:
#         print(f"Error saving processed article {original_title}: {e}")

# # Main Processing Function
# def process_and_save_articles(raw_articles_folder='../news_articles', processed_articles_folder='processed_articles'):
#     processed_count = 0
#     for filename in os.listdir(raw_articles_folder):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(raw_articles_folder, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     content = file.read()
#                     processed_data = process_article_content(content)
#                     save_processed_article(processed_data, filename, processed_articles_folder)
#                     processed_count += 1
#             except Exception as e:
#                 print(f"Error processing article {filename}: {e}")
#     print(f"Total articles processed: {processed_count}")

# # Example usage
# process_and_save_articles()





import os
import spacy
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from langdetect import detect
import streamlit as st

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load a transformer model for text summarization
summarizer = pipeline("summarization")

# Function for Sentiment Analysis using TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment

# Function for Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

# Function for Topic Modeling (LDA)
def perform_topic_modeling(corpus, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    lda = LDA(n_components=num_topics, random_state=42)
    lda.fit(X)
    topics = lda.components_
    return topics, vectorizer.get_feature_names_out()

# Function for Text Summarization using Transformers
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function for Multilingual Processing (Language Detection and Support)
def process_multilingual_text(text):
    lang = detect(text)
    if lang == "ne":
        # You can load a Nepali model or pre-trained NLP model here
        pass  # For simplicity, just detect the language in this example
    return lang

# Function to process and save data for one article
def process_article_content(article_text):
    # Clean the article text
    article_text = article_text.strip()

    # Sentiment Analysis
    sentiment = get_sentiment(article_text)

    # Named Entity Recognition (NER)
    entities = extract_entities(article_text)

    # Topic Modeling (LDA)
    topics, features = perform_topic_modeling([article_text])

    # Summarization
    summary = summarize_text(article_text)

    # Multilingual Processing (detect language and handle as needed)
    language = process_multilingual_text(article_text)

    return {
        'sentiment': sentiment,
        'entities': entities,
        'topics': topics,
        'summary': summary,
        'language': language
    }

# Function to display processed data for the article
def display_article_results(article_name, processed_data):
    st.subheader(f"Results for Article: {article_name}")

    # Display Sentiment
    st.subheader("Sentiment Analysis")
    st.write(f"Polarity: {processed_data['sentiment'].polarity}, Subjectivity: {processed_data['sentiment'].subjectivity}")

    # Display Named Entities
    st.subheader("Named Entities")
    st.write(", ".join(processed_data['entities']))

    # Display Topics
    st.subheader("Topic Modeling")
    for i, topic in enumerate(processed_data['topics']):
        st.write(f"Topic {i + 1}: {', '.join([processed_data['features'][i] for i in topic.argsort()[:-6:-1]])}")

    # Display Summary
    st.subheader("Summary")
    st.write(processed_data['summary'])

    # Display Language
    st.subheader("Detected Language")
    st.write(processed_data['language'])


# Streamlit App Interface
st.title("Climate Change Article Processing Tool")

# Define the path for raw articles
raw_articles_folder = 'news_articles'

# Process and display results for one specific article
article_name = "your_article_name.txt"  # Change this to the name of the article you want to process
file_path = os.path.join(raw_articles_folder, article_name)

# Read and process the article content
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        article_content = file.read()

        # Process the article content
        processed_data = process_article_content(article_content)

        # Display results using Streamlit
        display_article_results(article_name, processed_data)

except Exception as e:
    st.write(f"Error processing article {article_name}: {e}")
