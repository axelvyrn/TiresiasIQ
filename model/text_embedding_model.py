import spacy
import numpy as np

class TextEmbedder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")  # Download with: python -m spacy download en_core_web_md

    def embed_text(self, text):
        doc = self.nlp(text)
        return doc.vector

    def embed_texts(self, texts):
        return np.array([self.embed_text(text) for text in texts])
