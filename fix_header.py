# fix_header.py
import pandas as pd

df = pd.read_csv("data/personal_log.csv", header=None)
df.columns = ["timestamp", "activity", "context", "action_taken"]
df.to_csv("data/personal_log.csv", index=False)

print("✅ Header fixed.")
