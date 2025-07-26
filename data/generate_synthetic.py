# data/generate_synthetic.py

from faker import Faker
import pandas as pd
import random
import sys

def generate_messages(n=600, drift=False):
    fake = Faker()
    Faker.seed(1234)

    # topics = {
    #     "Billing": [
    #         "I was charged twice for my last payment",
    #         "Can I get an invoice for April?",
    #         "Why is my bill higher this month?"
    #     ],
    #     "Technical Issue": [
    #         "App crashes when I try to open it",
    #         "My password reset link isn’t working",
    #         "The website is loading very slow"
    #     ],
    #     "Account": [
    #         "How do I change my email address?",
    #         "I want to close my account",
    #         "Need help updating my profile info"
    #     ]
    # }
    topics = {
        "Billing": ["Charged twice", "Invoice issue", "Unexpected bill"],
        "Technical Issue": ["App crashes", "Reset doesn't work", "Slow website"],
        "Account": ["Change email", "Close my account", "Update info"]
    }

    # Drifted distribution (e.g., more tech issues, fewer billing)
    topic_weights = {
        "Billing": 0.2 if drift else 0.33,
        "Technical Issue": 0.6 if drift else 0.33,
        "Account": 0.2 if drift else 0.34
    }

    all_labels = list(topics.keys())
    all_probs = [topic_weights[label] for label in all_labels]

    rows = []
    for _ in range(n * len(all_labels)):
        label = random.choices(all_labels, weights=all_probs)[0]
        message = random.choice(topics[label]) + ". " + fake.sentence(nb_words=8)
        urgency = random.choices(["Low", "Medium", "High"], weights=[0.5, 0.3, 0.2])[0]
        rows.append({"message": message, "label": label, "urgency": urgency})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    drift = False
    output_path = "data/synthetic_messages.csv"

    # Optional command-line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "drift":
            drift = True
        if len(sys.argv) > 2:
            output_path = sys.argv[2]

    df = generate_messages(drift=drift)
    df.to_csv(output_path, index=False)
    print(f"Data {'with drift' if drift else 'without drift'} saved to {output_path}")
