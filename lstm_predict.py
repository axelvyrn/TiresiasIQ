# lstm_predict.py
import torch
import pandas as pd
import joblib
from lstm_model import LSTMModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys
from datetime import datetime

def predict_manual():
    # Load encoders
    context_encoder = joblib.load("models/context_encoder.pkl")
    activity_encoder = joblib.load("models/activity_encoder.pkl")

    context = input("Enter your current context: ")
    activity = input("Enter the activity you want to predict: ")

    context_encoded = context_encoder.transform([context])
    activity_encoded = activity_encoder.transform([activity])

    X = np.hstack((context_encoded.reshape(1, -1), activity_encoded.reshape(1, -1)))
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    model = LSTMModel(input_size=X_tensor.shape[2])
    model.load_state_dict(torch.load("models/lstm_model.pt"))
    model.eval()

    with torch.no_grad():
        output = model(X_tensor)
        prob = torch.sigmoid(output).item()
        print(f"\n🔥 Prediction: {activity} will be completed within 2 hours with probability: {prob * 100:.2f}%")

if __name__ == "__main__":
    predict_manual()
