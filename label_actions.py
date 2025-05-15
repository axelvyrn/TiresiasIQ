# label_actions.py
import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv("data/personal_log.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['completed_within_2hrs'] = 0

for i in range(len(df)):
    current_row = df.iloc[i]
    future_rows = df.iloc[i+1:]
    for _, row in future_rows.iterrows():
        if row['activity'] == current_row['activity'] and row['action_taken'] == 1:
            time_diff = row['timestamp'] - current_row['timestamp']
            if timedelta(minutes=0) < time_diff <= timedelta(hours=2):
                df.at[i, 'completed_within_2hrs'] = 1
                break

df.to_csv("data/personal_log_labeled.csv", index=False)
print("Labeling complete.")
