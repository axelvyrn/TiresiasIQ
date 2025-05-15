import streamlit as st
import torch
import pandas as pd
from model.predictor import FeedforwardNN
from utils.data_preprocessing import load_and_prepare_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

st.title("🔮 Personal Action Predictor")

data_file = "data/personal_log.csv"
if not os.path.exists(data_file):
    st.error("No data found. Please log some actions first using log_entry.py.")
    st.stop()

# Load dataset and encoders
df = pd.read_csv(data_file)
X_raw = df[["mood", "location", "activity", "weather"]]

encoders = {}
for col in X_raw.columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col])
    encoders[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Input form
with st.form("prediction_form"):
    mood = st.text_input("Mood", "")
    location = st.text_input("Location", "")
    activity = st.text_input("Activity", "")
    weather = st.text_input("Weather", "")
    submitted = st.form_submit_button("Predict")

if submitted:
    def encode_input(value, le):
        try:
            return le.transform([value])[0]
        except ValueError:
            st.warning(f"⚠️ Unknown value '{value}'. Defaulting to 0.")
            return 0

    x = [
        encode_input(mood, encoders["mood"]),
        encode_input(location, encoders["location"]),
        encode_input(activity, encoders["activity"]),
        encode_input(weather, encoders["weather"]),
    ]

    x_scaled = scaler.transform([x])
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    input_size = x_tensor.shape[1]
    model = FeedforwardNN(input_size)
    model.load_state_dict(torch.load("model/feedforward_model.pt"))
    model.eval()

    with torch.no_grad():
        prediction = model(x_tensor).item()

    st.success(f"🧠 Prediction: You will do this action with **{prediction * 100:.2f}%** probability.")
