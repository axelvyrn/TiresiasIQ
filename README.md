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

- Context
- Activity
- Whether you actually did the action (yes/no)

It saves the data to `data/personal_log.csv`
**You can either use the CLI logger which will create another personal_log.csv file, or you can edit the already existing one by running the CLI logger once and checking the parameters to be updated respectively.**
LSTM uses your last 5 entries to predict whether you’ll do the action again. So do the process atleast 10 times for a good output.

4. Run `run_ffn_pipeline.py` to use the FFN Predictor (higher efficiency) or the `run_lstm_pipeline.py` to use the LSTM Predictor (higher accuracy)

## Optional: Use the Web Dashboard

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
| Not enough logs for LSTM  | Log at least 10+ entries with `log_entry.py`        |
| Streamlit won’t launch    | Make sure `streamlit` is installed, then try again |

**Ensure you’re always in the root project folder `(TiresiasIQ/)` when running Python files. This avoids path issues.**
---
Made by @axelvyrn
