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

    def log_to_vector(self, keywords, polarity, subjectivity, action, user_time):
        kws = set([k.strip() for k in keywords.split(',') if k.strip()])
        keyword_vec = [1 if k in kws else 0 for k in self.all_keywords]
        # Action embedding (spaCy vector, reduced to mean for simplicity)
        doc = nlp(action) if action else None
        if doc and doc.vector_norm:
            action_vec = doc.vector[:10] if len(doc.vector) >= 10 else np.pad(doc.vector, (0, 10-len(doc.vector)))
        else:
            action_vec = np.zeros(10)
        # Time features: extract hour and weekday from ISO string
        import dateutil.parser
        try:
            dt = dateutil.parser.isoparse(user_time)
            hour = dt.hour / 23.0  # normalize
            weekday = dt.weekday() / 6.0  # normalize (0=Mon, 6=Sun)
        except Exception:
            hour = 0.0
            weekday = 0.0
        return np.array(keyword_vec + [polarity, subjectivity] + list(action_vec) + [hour, weekday])

    def extract_features_from_text(self, text):
        doc = nlp(text)
        keywords = set()
        for ent in doc.ents:
            keywords.add(ent.text.lower())
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop:
                keywords.add(token.lemma_.lower())
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        action = self.extract_action(text)
        return ', '.join(keywords), polarity, subjectivity, action

    def predict(self, query, pred_time):
        keywords, polarity, subjectivity, action = self.extract_features_from_text(query)
        x_pred = self.log_to_vector(keywords, polarity, subjectivity, action, pred_time).reshape(1, -1)
        prob = float(self.model.predict(x_pred)[0][0])
        return int(prob * 100), action

    def prepare_training_data(self, data):
        X = []
        y = []
        for row in data:
            user_input, keywords, polarity, subjectivity, target_action, user_time = row
            action = target_action if target_action else self.extract_action(user_input)
            X.append(self.log_to_vector(keywords, polarity, subjectivity, action, user_time))
            y.append(1 if action in user_input.lower() or action in keywords else 0)
        return np.array(X), np.array(y)

