from fastapi.testclient import TestClient
from app.main import app
import os
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_application_details():
    response = client.get("/application-details")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_status" in data
    assert "last_execution_time" in data


def test_metadata_alias():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_status" in data
    assert "metadata_last_checked" in data

@pytest.mark.skipif(not os.path.exists("models/best_model.pkl"), reason="Model not trained")
def test_predict_endpoint():
    payload = {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "SeniorCitizen": 0,
        "Partner": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "probability" in data
    assert data["churn_prediction"] in (0, 1)
    assert 0.0 <= float(data["probability"]) <= 1.0
