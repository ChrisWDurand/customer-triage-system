# serve/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

MODEL_PATH = "pipeline/model.pkl"

# Define request model
class MessageInput(BaseModel):
    message: str

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

model = joblib.load(MODEL_PATH)

# Init FastAPI
app = FastAPI(title="Customer Triage API")

@app.get("/")
def root():
    return {"message": "Customer Triage API is running"}

@app.post("/predict")
def predict(input: MessageInput):
    try:
        preds = model.predict([input.message])[0]
        return {"label": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
