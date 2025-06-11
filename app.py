import streamlit as st
import sqlite3
import spacy
from datetime import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import re
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Connect to SQLite database (or create it)
conn = sqlite3.connect('behavior.db')
c = conn.cursor()

# Create table for storing logs and keywords
c.execute('''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_input TEXT,
    keywords TEXT,
    polarity REAL,
    subjectivity REAL,
    target_action TEXT,
    user_time TEXT
)
''')
conn.commit()

# --- UI Improvements ---
st.set_page_config(page_title="Behavior Prediction Engine", layout="centered")
st.markdown("""
<style>
.big-font {font-size:22px !important;}
</style>
""", unsafe_allow_html=True)

st.title('Behavior Prediction Engine (BPE 2.19)')
st.markdown('<div class="big-font">Predict future actions based on user logs, emotions, and context.</div>', unsafe_allow_html=True)

with st.expander('ℹ️ How to use this app', expanded=False):
    st.markdown('''
    1. **Log your feelings or actions** in the Logger Window. Optionally, specify a target action (e.g., "cry").
    2. **Train the model** after logging at least 5 entries.
    3. **Ask prediction questions** in natural English (e.g., "Will Jack cry in the next hour?").
    4. **View and manage your logs** in the Database Viewer.
    ''')

# --- Enhanced NLP & Labelling ---
def extract_keywords_and_sentiment(text):
    doc = nlp(text)
    keywords = set()
    for ent in doc.ents:
        keywords.add(ent.text.lower())
    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop:
            keywords.add(token.lemma_.lower())
    # Sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return ', '.join(keywords), polarity, subjectivity

# --- Logger Window ---
st.header('1. Logger Window')
col1, col2, col3 = st.columns([2,1,1])
with col1:
    user_log = st.text_area('How are you feeling or what are you doing?', key='log_input')
with col2:
    target_action = st.text_input('Target action to predict (e.g., cry, quit, call)', key='target_action')
with col3:
    user_time = st.text_input('Time and day (e.g., Monday 10AM)', key='user_time')

if st.button('Log Entry'):
    if user_log.strip() and user_time.strip():
        keywords, polarity, subjectivity = extract_keywords_and_sentiment(user_log)
        c.execute('''INSERT INTO logs (timestamp, user_input, keywords, polarity, subjectivity, target_action, user_time) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), user_log, keywords, polarity, subjectivity, target_action.strip().lower(), user_time.strip()))
        conn.commit()
        st.success(f'Logged! Extracted keywords: {keywords} | Sentiment: {polarity:.2f}')
    else:
        st.warning('Please enter something to log and specify the time/day.')

# --- DB Schema Upgrade ---
c.execute('''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_input TEXT,
    keywords TEXT,
    polarity REAL,
    subjectivity REAL,
    target_action TEXT,
    user_time TEXT
)
''')
conn.commit()

# --- Feature Engineering ---
def get_all_keywords():
    c.execute('SELECT keywords FROM logs')
    all_keywords = set()
    for row in c.fetchall():
        kws = [k.strip() for k in row[0].split(',') if k.strip()]
        all_keywords.update(kws)
    return sorted(list(all_keywords))

def log_to_vector(keywords, polarity, subjectivity, action, all_keywords):
    kws = set([k.strip() for k in keywords.split(',') if k.strip()])
    keyword_vec = [1 if k in kws else 0 for k in all_keywords]
    # Action embedding (spaCy vector, reduced to mean for simplicity)
    doc = nlp(action) if action else None
    if doc and doc.vector_norm:
        action_vec = doc.vector[:10] if len(doc.vector) >= 10 else np.pad(doc.vector, (0, 10-len(doc.vector)))
    else:
        action_vec = np.zeros(10)
    return np.array(keyword_vec + [polarity, subjectivity] + list(action_vec))

# --- Improved Label Generation ---
def generate_label(target_action, user_input, keywords):
    # 1 if user log is about the target action, else 0
    if not target_action:
        return 0
    action = target_action.strip().lower()
    if action in user_input.lower() or action in keywords:
        return 1
    return 0

# Model training and prediction
model = None
all_keywords = []

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# --- Model Training ---
st.header('2. Train/Update Prediction Model')
if st.button('Train Model'):
    c.execute('SELECT user_input, keywords, polarity, subjectivity, target_action, user_time FROM logs')
    data = c.fetchall()
    if len(data) < 5:
        st.warning('Need at least 5 logs to train the model.')
    else:
        all_keywords = get_all_keywords()
        actions = list(set([row[4] for row in data if row[4]]))
        X = np.array([log_to_vector(row[1], row[2], row[3], row[4], all_keywords) for row in data])
        y = np.array([generate_label(row[4], row[0], row[1]) for row in data])
        model = keras.Sequential([
            layers.Input(shape=(len(all_keywords)+2+10,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=30, verbose=0)
        st.session_state['model'] = model
        st.session_state['all_keywords'] = all_keywords
        st.session_state['actions'] = actions
        st.session_state['model_trained'] = True
        st.success('Model trained!')

# --- Prediction Window ---
from predictor import Predictor
st.header('3. Prediction Window')
pred_col1, pred_col2, pred_col3 = st.columns([2,1,1])
with pred_col1:
    prediction_query = st.text_input('Ask a prediction question (e.g., What is the chance of Jack crying over his breakup in the next 1 hour?)', key='pred_query')
with pred_col2:
    pred_action = st.text_input('Target action to predict (e.g., cry, quit, call)', key='pred_action')
with pred_col3:
    pred_time = st.text_input('Time and day (e.g., Monday 10AM)', key='pred_time')

if st.button('Predict'):
    if st.session_state.get('model_trained', False):
        c.execute('SELECT user_input, keywords, polarity, subjectivity, target_action, user_time FROM logs')
        all_logs = c.fetchall()
        predictor = Predictor(st.session_state['model'], st.session_state['all_keywords'], st.session_state['actions'])
        prob = predictor.predict(prediction_query, pred_action, all_logs)
        st.info(f'Prediction for "{pred_action or predictor.extract_action(prediction_query)}" at {pred_time}: {prob}%')
    else:
        st.warning('Please train the model first!')

# --- Database Viewer ---
st.header('4. Database Viewer')
if st.checkbox('Show logged entries'):
    c.execute('SELECT timestamp, user_input, keywords, polarity, subjectivity, target_action, user_time FROM logs ORDER BY id DESC')
    rows = c.fetchall()
    for row in rows:
        st.write(f"[{row[0]}] {row[1]} (Keywords: {row[2]}) | Sentiment: {row[3]:.2f} | Target: {row[5]} | Time: {row[6]}")
