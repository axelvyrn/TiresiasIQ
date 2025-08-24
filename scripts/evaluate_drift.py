import json
import os
import time
from datetime import datetime, timedelta

# Minimal drift evaluation script
# Loads drift counts written by Predictor._on_drift_detected and prints a summary.
# Optionally, if thresholds are exceeded, it exits with a non-zero code to signal a retraining recommendation.

# --- tiny .env loader (no external deps) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ENV_FILES = [os.path.join(PROJECT_ROOT, '.env'), os.path.join(PROJECT_ROOT, '.env.local')]

def load_dotenv(paths):
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if '=' not in s:
                        continue
                    k, v = s.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

load_dotenv(ENV_FILES)

THRESHOLD_PER_ACTION = int(os.getenv('TIQ_DRIFT_ACTION_THRESHOLD', '3'))
THRESHOLD_GLOBAL = int(os.getenv('TIQ_DRIFT_GLOBAL_THRESHOLD', '5'))
WINDOW_DAYS = int(os.getenv('TIQ_DRIFT_WINDOW_DAYS', '14'))
# data directory lives at project_root/data (predictor writes there)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COUNTS_PATH = os.path.join(DATA_DIR, 'drift_counts.json')


def load_counts():
    try:
        with open(COUNTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def recent_count(info, window_days):
    now = time.time()
    last_ts = float(info.get('last_ts', 0))
    if last_ts <= 0:
        return 0
    # simple heuristic: if last drift within window, count occurrences
    age_days = (now - last_ts) / 86400.0
    return info.get('count', 0) if age_days <= window_days else 0


def main():
    counts = load_counts()
    if not counts:
        print('No drift detected so far.')
        return 0
    window = WINDOW_DAYS
    flagged = []
    total_recent = 0
    for action, info in counts.items():
        c = recent_count(info, window)
        total_recent += c
        if c >= THRESHOLD_PER_ACTION:
            flagged.append((action, c))
    print('Drift summary (last %d days):' % window)
    for action, c in sorted(flagged, key=lambda x: -x[1]):
        print(f'  - {action}: {c} events')
    print(f'Global recent drift count: {total_recent}')
    if flagged or total_recent >= THRESHOLD_GLOBAL:
        print('Drift threshold exceeded — retraining recommended')
        return 2
    return 0


if __name__ == '__main__':
    code = main()
    raise SystemExit(code)

