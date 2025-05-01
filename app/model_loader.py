import spacy
from spacy.cli import download

# Function to load the spaCy model
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp
