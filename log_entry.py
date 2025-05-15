# log_entry.py
import csv
from datetime import datetime
import os

def log_action():
    context = input("Enter current context (mood, time, environment, etc.): ")
    activity = input("What activity are you considering? ")
    taken = input("Did you actually do it? (yes/no): ").strip().lower()
    taken_binary = 1 if taken == "yes" else 0

    timestamp = datetime.now().isoformat()

    file_path = "data/personal_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "context", "activity", "action_taken"])
        writer.writerow([timestamp, context, activity, taken_binary])

    print("Action logged successfully.")

if __name__ == "__main__":
    log_action()
