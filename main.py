from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Predictor", version="1.0")

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
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "accuracy": 0.8833
    }

@app.post("/predict")
def predict(patient: PatientData):
    intercept = -6.27744855
    coef = [-0.01141606, 1.14217783, 0.38345851, 0.02565861, 0.00569410, -0.92588075, 0.21539965, -0.02210521, 0.76186102, 0.32132278, 0.31875469, 1.15129144, 0.29962864]
    
    input_data = [
        patient.age, patient.sex, patient.cp, patient.trestbps,
        patient.chol, patient.fbs, patient.restecg, patient.thalach,
        patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]
    
    z = sum(coef[i] * input_data[i] for i in range(len(coef))) + intercept
    probability = 1 / (1 + 2.71828 ** (-z))
    prediction = 1 if probability > 0.5 else 0
    
    return {
        "heart_disease": bool(prediction),
        "probability": float(probability)
    }
