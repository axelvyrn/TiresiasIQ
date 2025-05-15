import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop any rows with missing data
    df.dropna(inplace=True)

    # Separate features and target
    X = df[["mood", "location", "activity", "weather"]]
    y = df["target_action"]

    # Encode categorical features
    encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Scale features to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, X_scaled.shape[1]
