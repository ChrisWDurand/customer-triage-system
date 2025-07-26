# pipeline/retrain.py

import subprocess
import pandas as pd
from pipeline import drift_detection

TRAIN_PATH = "data/synthetic_messages.csv"
INCOMING_PATH = "data/incoming_messages.csv"

def merge_training_data():
    print("Merging incoming data into training set...")
    train_df = pd.read_csv(TRAIN_PATH)
    incoming_df = pd.read_csv(INCOMING_PATH)

    combined = pd.concat([train_df, incoming_df], ignore_index=True)
    combined.to_csv(TRAIN_PATH, index=False)
    print(f"Training data updated with {len(incoming_df)} new records.")

def retrain_if_drift():
    print("Checking for drift...")
    if drift_detection.detect_label_shift():
        merge_training_data()
        print("Retraining model with updated data...")
        subprocess.run(["python", "-m", "pipeline.train"])
    else:
        print("No drift. Skipping retrain.")

if __name__ == "__main__":
    retrain_if_drift()

