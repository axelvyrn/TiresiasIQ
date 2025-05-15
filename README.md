# 🧠 Personal Action Predictor : TiresiasIQ

This project uses neural networks (Feedforward & LSTM) to predict the probability of a personal action based on your context (mood, activity, etc.).

## 💻 Features

- CLI Logger: Fast logging from terminal.
- Feedforward Predictor: Predicts behavior from context.
- LSTM Model: Learns temporal sequences.
- Streamlit Dashboard: Visual, user-friendly interface.

## 📦 Installation
1. Go to https://python.org/downloads and install Python 3.10+
**Important: During install, check ✔️ “Add Python to PATH”**

2. Open CMD or terminal in your project folder and run:
```bash
git clone https://github.com/axelvyrn/noesis.git
cd TyresiasIQ
pip install -r requirements.txt
```

3. Run your CLI logger like this:
```bash
python log_entry.py
```
It will ask for:

- Mood
- Location
- Activity
- Weather
- Whether you actually did the action (yes/no)

It saves the data to `data/personal_log.csv`

Log at least 10+ entries to start.
