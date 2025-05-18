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
print("🔄 Resetting FFN pipeline...")
delete_if_exists("data/personal_log_labeled.csv")
delete_if_exists("models/ffn_model.pth")
delete_if_exists("label_encoders.pkl")

# STEP 1: LABEL
print("🏷️  Labeling data...")
subprocess.run(["python", "label_actions.py", "--window", "120"])

# STEP 2: TRAIN
print("🧠 Training Feedforward model...")
subprocess.run(["python", "train_ffn.py"])

# STEP 3: PREDICT
print("🔮 Predicting action (FFN)...")
subprocess.run(["python", "ffn_predict.py"])
