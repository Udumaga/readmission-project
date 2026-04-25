from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join("models", "readmission_model.joblib")

app = FastAPI(title="Readmission Risk API")

model = joblib.load(MODEL_PATH)

class Patient(BaseModel):
    # Adjust fields to match your dataset
    age: float
    gender: str
    time_in_hospital: float
    num_lab_procedures: float
    num_medications: float

@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.dict()])
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}
