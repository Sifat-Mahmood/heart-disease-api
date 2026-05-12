<<<<<<< HEAD
# Heart Disease Prediction API

A FastAPI application that predicts heart disease using a trained machine learning model.

## Project Structure

├── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── model/
    ├── heart_model.joblib
    └── feature_names.joblib

## Local Testing

docker-compose build
docker-compose up

Then visit: http://localhost:8000/docs

## API Endpoints

- GET /health - Health check
- GET /info - Model information
- POST /predict - Make predictions

## Deployment

Push to GitHub, connect to Render, and deploy.
=======
# heart-disease-api
FastAPI Heart Disease Prediction
>>>>>>> 8aa0950a8ce67e5ac69299ca9d6b4ddb801d30ad
