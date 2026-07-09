from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import joblib
import os
import json
import pandas as pd
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Customer Attrition API", version="1.0", description="API-driven Cloud Native Solutions")

MODEL_PATH = "models/best_model.pkl"
PIPELINE_STATUS_PATH = "artifacts/pipeline_status.json"

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Application API. Navigate to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/application-details")
@app.get("/metadata")
def get_application_details():
    """Retrieves key application details and the latest pipeline status."""
    details = {
        "pipeline_status": "Active - Scheduled every 2 minutes",
        "deployment_info": "Dockerized Multi-Container Application (FastAPI, Prefect, MLflow)",
        "model_version": "v1.0 (Random Forest / Logistic Regression pipeline)",
        "dataset_info": "Telco Customer Churn Dataset",
        "flow_status": "Running via Prefect UI on port 4200",
        "metadata_last_checked": datetime.utcnow().isoformat(),
        "last_execution_time": None
    }

    if os.path.exists(PIPELINE_STATUS_PATH):
        try:
            with open(PIPELINE_STATUS_PATH, 'r', encoding='utf-8') as status_file:
                status_data = json.load(status_file)
                details.update(status_data)
                details["last_execution_time"] = status_data.get("last_pipeline_run", details["last_execution_time"])
        except Exception:
            details["status_read_error"] = "Unable to read pipeline status file"

    return details

@app.get("/model-info")
def model_info():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not trained yet.")
    return {"model_path": MODEL_PATH, "status": "Ready for inference"}

def _build_features(request: PredictionRequest, feature_names):
    feature_map = {
        'tenure': float(request.tenure),
        'MonthlyCharges': float(request.MonthlyCharges),
        'TotalCharges': float(request.TotalCharges),
        'SeniorCitizen': int(request.SeniorCitizen),
        'Partner': int(request.Partner),
    }

    if len(feature_names) != len(feature_map):
        raise HTTPException(
            status_code=500,
            detail=f"Loaded model expects {len(feature_names)} features, but API supports {len(feature_map)}. Please retrain the model with the current input schema."
        )

    return [[feature_map[name] for name in feature_names]]


def _churn_probability(model, features):
    probs = model.predict_proba(features)[0]
    classes = list(getattr(model, 'classes_', []))
    if 1 in classes:
        return float(probs[classes.index(1)])
    if len(probs) > 1:
        return float(probs[1])
    return float(probs[0])


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found. Run pipeline first.")

    model = joblib.load(MODEL_PATH)
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is not None:
        features = _build_features(request, feature_names)
    else:
        features = [[
            float(request.tenure),
            float(request.MonthlyCharges),
            float(request.TotalCharges),
            int(request.SeniorCitizen),
            int(request.Partner),
        ]]

    try:
        prediction = model.predict(features)[0]
        probability = _churn_probability(model, features)
    except ValueError:
        raise HTTPException(status_code=500, detail="Model input shape mismatch. Re-train the model with current API schema.")
    except AttributeError:
        raise HTTPException(status_code=500, detail="Loaded model does not support the expected prediction interface.")

    return PredictionResponse(
        churn_prediction=int(prediction),
        probability=float(probability),
        model_used=type(model).__name__
    )
