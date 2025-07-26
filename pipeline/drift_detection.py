import pandas as pd
from scipy.stats import entropy
import os
import subprocess

TRAIN_PATH = "data/synthetic_messages.csv"
INCOMING_PATH = "data/incoming_messages.csv"

def simulate_new_data():
    print("Generating simulated incoming messages with drift...")
    subprocess.run(["python", "data/generate_synthetic.py", "drift", INCOMING_PATH])

def load_label_dist(path):
    df = pd.read_csv(path)
    return df["label"].value_counts(normalize=True)

def detect_label_shift(threshold=0.1):
    if not os.path.exists(TRAIN_PATH):
        print("No training data found. Asking system for new synthetic data...")
        subprocess.run(["python", "data/generate_synthetic.py", TRAIN_PATH])
        return False

    if not os.path.exists(INCOMING_PATH):
        simulate_new_data()

    p = load_label_dist(TRAIN_PATH)
    q = load_label_dist(INCOMING_PATH).reindex(p.index).fillna(0)

    kl = entropy(p, q)
    print(f"KL Divergence: {kl:.4f}")

    if kl > threshold:
        print("Drift detected in label distribution.")
        return True
    else:
        print("No significant drift detected.")
        return False

if __name__ == "__main__":
    detect_label_shift()
