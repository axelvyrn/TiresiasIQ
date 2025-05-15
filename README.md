# 🧠 Personal Action Predictor : TiresiasIQ
![Tiresias](https://github.com/user-attachments/assets/c0afa6fd-661a-4b3a-a11d-00b14da32ec6)

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
**You can either use the CLI logger which will create another personal_log.csv file, or you can edit the already existing one by running the CLI logger once and checking the parameters to be updated respectively.**

4. You have two training scripts:

🔹 Train Feedforward:
```bash
python train_ffn.py
```
Saves to `model/feedforward_model.pt`

🔹 Train LSTM:
```bash
python train_lstm.py
```
Saves to `model/lstm_model.pt`

**Log at least 10+ entries to start.**
5. 🔹 With Feedforward (manual input):
bash
Copy
Edit
python predict.py
👉 It asks for your current state and predicts the probability you'll do the action.

🔹 With LSTM (auto last 5 logs):
```bash
python lstm_predict.py
```
Uses your last 5 entries to predict whether you’ll do the action again.

6. Optional: Use the Web Dashboard
Launch it in your browser with:
```bash
streamlit run dashboard.py
```
→ Enter context via form
→ Get prediction
→ Fancy and interactive

## 🔍 Troubleshooting
| Problem                   | Fix                                                |
| ------------------------- | -------------------------------------------------- |
| `ModuleNotFoundError`     | Run `pip install` for the missing module           |
| `lstm_model.pt` not found | Run `python train_lstm.py` to train and save model |
| Not enough logs for LSTM  | Log at least 5+ entries with `log_entry.py`        |
| Streamlit won’t launch    | Make sure `streamlit` is installed, then try again |

**Ensure you’re always in the root project folder `(TiresiasIQ/)` when running Python files. This avoids path issues.**
---
Made by @axelvyrn
