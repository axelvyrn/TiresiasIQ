import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from model.lstm_predictor import LSTMPredictor
import joblib

df = pd.read_csv("data/personal_log_labeled.csv")

context_encoder = joblib.load("models/context_encoder.pkl")
activity_encoder = joblib.load("models/activity_encoder.pkl")

df['context_enc'] = context_encoder.transform(df['context'])
df['activity_enc'] = activity_encoder.transform(df['activity'])

X = df[['context_enc', 'activity_enc']].values
y = df['completed_in_window'].values

X_tensor = torch.tensor(np.expand_dims(X, axis=1), dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = LSTMPredictor(input_size=X.shape[1])
model.load_state_dict(torch.load("models/lstm_model.pt"))
model.eval()

with torch.no_grad():
    preds = torch.sigmoid(model(X_tensor)).squeeze().numpy()
    preds_binary = (preds > 0.5).astype(int)

precision = precision_score(y, preds_binary)
recall = recall_score(y, preds_binary)
f1 = f1_score(y, preds_binary)

print(f"📈 Precision: {precision:.2f}")
print(f"📈 Recall:    {recall:.2f}")
print(f"📈 F1 Score:  {f1:.2f}")
