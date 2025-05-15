import torch
import pandas as pd
from model.predictor import FeedforwardNN
from utils.data_preprocessing import load_and_prepare_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Load original data to reuse encoders
data_file = "data/personal_log.csv"
df = pd.read_csv(data_file)
X_raw = df[["mood", "location", "activity", "weather"]]

# Fit encoders on full dataset
encoders = {}
for col in X_raw.columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col])
    encoders[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Ask for new input
print("🔮 Let's predict your next move:")
mood = input("Mood: ").strip()
location = input("Location: ").strip()
activity = input("Activity: ").strip()
weather = input("Weather: ").strip()

# Encode user input
def encode_input(value, le):
    try:
        return le.transform([value])[0]
    except ValueError:
        print(f"⚠️ Unknown value '{value}'. Defaulting to 0.")
        return 0

x = [
    encode_input(mood, encoders["mood"]),
    encode_input(location, encoders["location"]),
    encode_input(activity, encoders["activity"]),
    encode_input(weather, encoders["weather"]),
]

x_scaled = scaler.transform([x])
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

# Load model
input_size = x_tensor.shape[1]
model = FeedforwardNN(input_size)
model.load_state_dict(torch.load("model/feedforward_model.pt"))
model.eval()

# Predict
with torch.no_grad():
    prediction = model(x_tensor).item()

print(f"\n🧠 Prediction: You will do this action with {prediction * 100:.2f}% probability.")
