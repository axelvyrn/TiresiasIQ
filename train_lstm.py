# train_lstm.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
from lstm_model import LSTMModel

# Load the labeled data
df = pd.read_csv("data/personal_log_labeled.csv")

# Encode categorical features
context_encoder = LabelEncoder()
activity_encoder = LabelEncoder()

df['context_enc'] = context_encoder.fit_transform(df['context'])
df['activity_enc'] = activity_encoder.fit_transform(df['activity'])

# Save encoders
os.makedirs("models", exist_ok=True)
joblib.dump(context_encoder, "models/context_encoder.pkl")
joblib.dump(activity_encoder, "models/activity_encoder.pkl")

# Features and target
X = df[['context_enc', 'activity_enc']].values
y = df['completed_within_2hrs'].values

# Add dummy sequence dimension for LSTM: (batch, seq_len, features)
X = np.expand_dims(X, axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model
model = LSTMModel(input_size=X.shape[2])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(30):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/lstm_model.pt")
print("LSTM Model trained and saved.")
