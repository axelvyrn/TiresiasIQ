import streamlit as st
import sqlite3
import spacy
from datetime import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from textblob import TextBlob
# Use the adaptive predictor implementation
from predictor import Predictor, normalize_action

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Connect to SQLite database
conn = sqlite3.connect('behavior.db')
c = conn.cursor()

# Create logs table
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

# --- Streamlit Setup ---
st.set_page_config(page_title="Behavior Prediction Engine", layout="centered")
st.title('Behavior Prediction Engine (BPE 3.0)')

# --- Keyword + Sentiment ---
def extract_keywords_and_sentiment(text):
    doc = nlp(text)
    keywords = set()
    for ent in doc.ents:
        keywords.add(ent.text.lower())
    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop:
            keywords.add(token.lemma_.lower())
    blob = TextBlob(text)
    return ', '.join(keywords), blob.sentiment.polarity, blob.sentiment.subjectivity

# --- Predictor session init ---
if 'predictor' not in st.session_state:
    # start with an empty predictor; we'll re-initialize with stable keywords/actions when training
    st.session_state['predictor'] = Predictor(model=None, all_keywords=[], actions=[])
    st.session_state['model_trained'] = False

predictor = st.session_state['predictor']

# --- Logger Window ---
st.header('1. Logger Window')
col1, col2 = st.columns([2,1])
with col1:
    user_log = st.text_area('How are you feeling or what are you doing?', key='log_input')
with col2:
    user_time = st.text_input('Time and date (ISO format, e.g., 2025-06-13T10:00)', key='user_time')
    target_action_input = st.text_input('Optional target action (e.g., cry)', key='target_action')

if st.button('Log Entry'):
    if user_log.strip() and user_time.strip():
        # Extract keywords & sentiment
        keywords, polarity, subjectivity = extract_keywords_and_sentiment(user_log)
        final_action = target_action_input.strip().lower() if target_action_input.strip() else None

        # Save to DB
        c.execute('''INSERT INTO logs 
                     (timestamp, user_input, keywords, polarity, subjectivity, target_action, user_time) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), user_log, keywords, polarity, subjectivity,
                   final_action, user_time.strip()))
        conn.commit()
        log_row = (user_log, keywords, polarity, subjectivity, final_action, user_time.strip())
        predictor.update_with_log(log_row)

        st.success(f"Logged! Action: {final_action} | Keywords: {keywords} | Sentiment: {polarity:.2f}")
    else:
        st.warning('Please enter log and time in ISO format.')

# --- Helper: Collect all keywords ---
def get_all_keywords():
    c.execute('SELECT keywords FROM logs')
    all_keywords = set()
    for row in c.fetchall():
        if not row[0]:
            continue
        all_keywords.update([k.strip() for k in row[0].split(',') if k.strip()])
    return sorted(list(all_keywords))

# --- Training ---
st.header('2. Train/Update Prediction Model')
if st.button('Train Model'):
    c.execute('SELECT user_input, keywords, polarity, subjectivity, target_action, user_time FROM logs')
    data = c.fetchall()

    if len(data) < 5:
        st.warning('Need at least 5 logs to train the model.')
    else:
        # Build authoritative keyword/action set from DB
        all_keywords = get_all_keywords()
        actions = sorted(list({normalize_action(row[4]) for row in data if row[4]}))

        if not actions:
            st.warning('No labeled actions found in logs. Add some logs with a target action to train.')
        else:
            # Rebuild predictor with stable keyword/action set
            predictor = Predictor(None, all_keywords, actions)

            # Build training data: for each logged row, create a positive sample for the recorded action
            # and negative samples for the other actions. This leverages Predictor.log_to_vector which
            # already encodes the action into the feature vector.
            X_list = []
            y_list = []
            for (user_input, keywords, polarity, subjectivity, target_action, user_time) in data:
                target_action_norm = normalize_action((target_action or '').strip()) if target_action else ''
                # Create one sample per action (positive for the true action, negative otherwise)
                for a in actions:
                    feat = predictor.log_to_vector(keywords or '', float(polarity or 0.0), float(subjectivity or 0.0), a, user_time or datetime.now().isoformat())
                    X_list.append(feat)
                    y_list.append(1.0 if (a == target_action_norm) else 0.0)
                    # Update running stats so predictor has sensible scaler
                    try:
                        predictor._update_running_stats(feat)
                    except Exception:
                        pass

            X = np.array(X_list, dtype=float)
            y = np.array(y_list, dtype=float)

            # Shuffle and split
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Apply running scaler transform if available
            X_scaled = predictor._transform_with_running(X) if predictor.scaler_mean is not None else X

            # Build and train model (binary classifier: given feature vector, probability that action==label)
            model = keras.Sequential([
                layers.Input(shape=(X_scaled.shape[1],)),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Lightweight training; for small datasets this will be quick
            model.fit(X_scaled, y, epochs=40, batch_size=32, verbose=0)

            predictor.model = model
            st.session_state['predictor'] = predictor
            st.session_state['model_trained'] = True

            st.success("✅ Model trained and stored!\nTraining samples: {} | Actions: {}".format(len(y), len(actions)))

# --- Prediction ---
st.header('3. Prediction Window')
pred_query = st.text_input('Ask a prediction (e.g., Will Jack cry in the next hour?)', key='pred_query')
pred_time = st.text_input('Time (ISO format, e.g., 2025-06-13T23:00)', key='pred_time')

if st.button('Predict'):
    if st.session_state.get('model_trained', False):
        predictor = st.session_state['predictor']
        try:
            result = predictor.predict(pred_query, pred_time)
            # 1) Primary: prediction for the queried/extracted action at that time
            extracted_block = result.details.get('extracted_prediction')
            if extracted_block and extracted_block.get('action'):
                act_phrase = predictor.summarize_action_context(extracted_block['action'])
                st.info(f'Prediction for "{act_phrase}" at {pred_time}: {extracted_block["blended"]:.2%}')
            else:
                # fallback to best action if extraction failed
                st.info(f'Prediction for "{result.action}" at {pred_time}: {result.probability:.2%}')

            # 2) Additional block: the most probable action at that time with contextualized phrase
            top_action = result.details.get('top_action', result.action)
            top_prob = result.details.get('blended', result.probability)
            top_phrase = predictor.summarize_action_context(top_action)
            st.success(f'The user is likely to {top_phrase} at that time: {top_prob:.2%}')

            with st.expander("🔍 Breakdown"):
                # Pretty-print details with percentages
                details = result.details
                for k, v in details.items():
                    if k in ("model_score", "temporal_score", "semantic_score", "blended") and isinstance(v, (int, float)):
                        st.write(f"{k}: {v:.2%}")
                    elif k == "top_k" and isinstance(v, list):
                        pct_list = [(a, f"{p:.2%}") for (a, p) in v]
                        st.write(f"top_k: {pct_list}")
                    elif k == "extracted_prediction" and isinstance(v, dict):
                        ep = dict(v)
                        for fld in ("model_score", "temporal_score", "semantic_score", "blended"):
                            if fld in ep and isinstance(ep[fld], (int, float)):
                                ep[fld] = f"{ep[fld]:.2%}"
                        st.write("extracted_prediction:", ep)
                    else:
                        st.write(f"{k}: {v}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("⚠️ Please train the model first!")

# --- Database Viewer ---
st.header('4. Database Viewer')
if st.checkbox('Show logs'):
    c.execute('SELECT timestamp, user_input, keywords, polarity, subjectivity, target_action, user_time FROM logs ORDER BY id DESC')
    rows = c.fetchall()
    for row in rows:
        st.write(f"[{row[0]}] {row[1]} | Keywords: {row[2]} | Sentiment: {row[3]:.2f} | Target: {row[5]} | Time: {row[6]}")

    if st.button('Reset DB'):
        c.execute('DELETE FROM logs')
        conn.commit()
        st.session_state['predictor'] = Predictor(None, [], [])
        st.session_state['model_trained'] = False
        st.success("DB cleared and predictor reset.")
