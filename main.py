from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Heart Disease Predictor", version="1.0")

model = joblib.load("model/heart_model.joblib")
feature_names = joblib.load("model/feature_names.joblib")

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "features": feature_names,
        "accuracy": 0.8833
    }

@app.post("/predict")
def predict(patient: PatientData):
    input_data = [[
        patient.age, patient.sex, patient.cp, patient.trestbps,
        patient.chol, patient.fbs, patient.restecg, patient.thalach,
        patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]]
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "heart_disease": bool(prediction),
        "probability": float(probability)
    }
