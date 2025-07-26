# customer-triage-system
AI application classifying customer messages by topic and urgency using Python, scikit-learn, FastAPI, and Docker.

## Problem/Situation
Some businesses receiving high volumes of customer messages through email, support portals, and chat must manually review and rout these messages. The process is slow prone to error.

## Solution
This system will simulate customer message triage service with:

- Incoming message classifying yields categories: **Billing**, **Technical Issue**, and **Account**
- Urgency level detecting produces: **Low**, **Medium**, **High**
- Training and inference Automating requires scripts and APIs
- Serving real-time usage requires a Dockerized FastAPI


## Features

System Simulating a production-ready NLP pipeline:
- Synthetic customer messages generating
- Classifier training using scikit-learn
- Message intent detecting (e.g., billing, technical issues, account help)
- Prediction serving using REST API
- Drift detecting
- Model validating and retraining

## Project Structure
```
customer-triage-system/
│
├── data/
│ ├── generate_synthetic.py
│ └── synthetic_messages.csv
│
├── pipeline/
│ ├── train.py
│ └── model.pkl
│
├── serve/
│ └── api.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech Stack

- Python 3.10+
- scikit-learn
- Pandas, NumPy
- FastAPI + Uvicorn
- Faker (for synthetic data)
- Docker

## Getting Started

### 1. Clone and set up
```bash
git clone https://github.com/ChrisWDurand/customer-triage-system.git
cd customer-triage-system
```
### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Generate data and train model
```
python data/generate_synthetic.py
python pipeline/train.py
```

### 4. Run the inference API
```
uvicorn serve.api:app --reload
```

Go to http://localhost:8000 to check the API is live.

## Example Request
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "App crashes when I open it. Please help!"}'
```

## Sample response
```
{
  "label": "Technical Issue"
}
```

## Docker Deployment
```
docker build -t customer-triage-app .
docker run -p 8000:8000 customer-triage-app
```
Visit http://localhost:8000

## Roadmap
- [ ] Add Model Drift detecting and Model retraining

- [ ] Add Weekly reports validating

- [ ] Implement GCP Cloud Run or AWS ECS services

- [ ] Add integration monitoring and alerting 

## Disclaimer
The project uses synthetic data and is simply a demonstration. It can simulate a real-world ML deployment pipeline without exposing proprietary data or internal systems.
