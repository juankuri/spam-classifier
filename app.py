from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Spam or not spam prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load(pathlib.Path('model/spamornot.joblib'))
vectorizer = load(pathlib.Path('model/vectorizer.joblib'))

class InputData(BaseModel):
    message: str

class OutputData(BaseModel):
    score: float
    prediction: str  

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    vector = vectorizer.transform([data.message])
    probability = model.predict_proba(vector)[:, -1][0]
    
    print(f"Probability: {probability}")
    
    if probability > 0.149:
        prediction = "spam"
    else:
        prediction = "ham"
    
    print(f"Prediction: {prediction}")

    return {
        'score': probability,
        'prediction': prediction,
    }