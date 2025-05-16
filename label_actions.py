# label_actions.py
import pandas as pd
from datetime import datetime, timedelta
import argparse

def label_actions(window_minutes=120):
    df = pd.read_csv("data/personal_log.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['completed_in_window'] = 0

    for i in range(len(df)):
        current = df.iloc[i]
        future = df.iloc[i + 1:]
        for _, row in future.iterrows():
            if row['activity'] == current['activity'] and row['action_taken'] == 1:
                delta = row['timestamp'] - current['timestamp']
                if timedelta(minutes=0) < delta <= timedelta(minutes=window_minutes):
                    df.at[i, 'completed_in_window'] = 1
                    break

    df.to_csv("data/personal_log_labeled.csv", index=False)
    print(f"Labeling done for window: {window_minutes} mins")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=120, help="Time window in minutes")
    args = parser.parse_args()
    label_actions(window_minutes=args.window)
