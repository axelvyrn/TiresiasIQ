import spacy
from textblob import TextBlob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from typing import List

nlp = spacy.load('en_core_web_sm')

def get_action_vector(action: str):
    doc = nlp(action)
    if doc.vector_norm:
        return doc.vector
    else:
        # fallback: average token vectors
        vectors = [token.vector for token in doc if token.has_vector]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(nlp.vocab.vectors_length)

def action_similarity(action1: str, action2: str) -> float:
    v1 = get_action_vector(action1)
    v2 = get_action_vector(action2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(cosine_similarity([v1], [v2])[0][0])

class Predictor:
    def __init__(self, model, all_keywords: List[str], actions: List[str]):
        self.model = model
        self.all_keywords = all_keywords
        self.actions = actions

    def extract_action(self, text):
        doc = nlp(text)
        # Try to find the main verb or action
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                return token.lemma_.lower()
        # fallback: first noun
        for token in doc:
            if token.pos_ == 'NOUN' and not token.is_stop:
                return token.lemma_.lower()
        return ''

    def log_to_vector(self, keywords, polarity, subjectivity, action):
        kws = set([k.strip() for k in keywords.split(',') if k.strip()])
        keyword_vec = [1 if k in kws else 0 for k in self.all_keywords]
        # Action embedding (spaCy vector, reduced to mean for simplicity)
        action_vec = get_action_vector(action)
        # Use only first 10 dims for simplicity (or you can use all if model supports)
        action_vec = action_vec[:10] if len(action_vec) >= 10 else np.pad(action_vec, (0, 10-len(action_vec)))
        return np.array(keyword_vec + [polarity, subjectivity] + list(action_vec))

    def predict(self, query, query_action, all_logs):
        # Extract features from query
        doc = nlp(query)
        keywords = set()
        for ent in doc.ents:
            keywords.add(ent.text.lower())
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop:
                keywords.add(token.lemma_.lower())
        blob = TextBlob(query)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        # Use provided action or extract from context
        action = query_action.strip().lower() if query_action else self.extract_action(query)
        x_pred = self.log_to_vector(', '.join(keywords), polarity, subjectivity, action).reshape(1, -1)
        prob = float(self.model.predict(x_pred)[0][0])
        return int(prob * 100)

