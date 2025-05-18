import os
import subprocess

def delete_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            os.rmdir(path)
        print(f"Deleted: {path}")

# STEP 0: RESET
print("🔄 Resetting LSTM pipeline...")
delete_if_exists("data/personal_log_labeled.csv")
delete_if_exists("models/lstm_model.pth")
delete_if_exists("label_encoders.pkl")

# STEP 1: LABEL
print("🏷️  Labeling data...")
subprocess.run(["python", "label_actions.py", "--window", "120"])

# STEP 2: TRAIN
print("🧠 Training LSTM model...")
subprocess.run(["python", "train_lstm.py"])

# STEP 3: EVALUATE
print("📈 Evaluating LSTM model...")
subprocess.run(["python", "evaluate_model.py"])

# STEP 4: PREDICT
print("🔮 Predicting action (LSTM)...")
subprocess.run(["python", "lstm_predict.py"])
