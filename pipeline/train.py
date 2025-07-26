# pipeline/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os
from data.generate_synthetic import generate_messages

DATA_PATH = "data/synthetic_messages.csv"
MODEL_PATH = "pipeline/model.pkl"

# Ensure data exists
if not os.path.exists(DATA_PATH):
    print(f"Data file not found at {DATA_PATH}. Generating synthetic data...")
    os.makedirs("data", exist_ok=True)
    df = generate_messages()
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic data created at {DATA_PATH}")


DATA_PATH = "data/synthetic_messages.csv"
MODEL_PATH = "pipeline/model.pkl"

# Load data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['message', 'label'])
    return df['message'], df['label']

# Train model
def train():
    print("Loading data...")
    X, y = load_data()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Creating pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=200))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()