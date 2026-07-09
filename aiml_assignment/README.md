# Telecom Customer Churn Prediction

## Overview
This repository contains an end-to-end churn prediction solution built with:
- **FastAPI** for serving real-time predictions
- **Streamlit** for dashboard visualization
- **MLflow** for experiment tracking
- **Prefect** for pipeline orchestration
- **Docker Compose** for containerized deployment

The project uses telecom customer data to predict whether a customer is likely to churn and stores prediction history for dashboard monitoring.

## Prerequisites
Before running the project, install or enable the following:
- **Docker** (Docker Desktop for macOS / Linux)
- **Docker Compose** (included with Docker Desktop)
- **GNU Make** (`make` command)
- Optional: **Python 3.10+** if you want to run tests or scripts locally outside Docker

## Quick Start
1. Open a terminal and change into the project folder:
   ```bash
   cd /Users/naveendeswal/Documents/Semester\ 3/API/Assign1/aiml_assignment
   ```
2. Start the application stack:
   ```bash
   make up
   ```
3. After the containers are up, run the data pipeline:
   ```bash
   make pipeline
   ```
4. Train the model:
   ```bash
   make train
   ```

## What the Make targets do
- `make up`
  - Builds and starts all services in detached mode using `docker compose up --build -d`
- `make pipeline`
  - Runs the data preprocessing pipeline inside the API container
  - Internally executes: `docker exec -it aiml_assignment-api-1 python pipelines/data_pipeline.py`
- `make train`
  - Runs the ML training pipeline inside the API container
  - Internally executes: `docker exec -it aiml_assignment-api-1 python pipelines/ml_pipeline.py`
- `make down`
  - Stops and removes the Docker Compose services
- `make test`
  - Runs the pytest suite for API and pipeline validation

## Application Services
Once `make up` is running, the following services are available:
- **Streamlit dashboard:** http://localhost:8501
- **FastAPI docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5010
- **Prefect UI:** http://localhost:4200

## Data Flow
1. **Data ingestion and preprocessing** are handled by `pipelines/data_pipeline.py`.
2. **Model training** is performed by `pipelines/ml_pipeline.py`.
3. **Inference** is served through `app/main.py` at `POST /predict`.
4. **Dashboard** saves request history to `dashboard/predictions.db` and visualizes it in Streamlit.

## Prediction API
Use the `/predict` endpoint with JSON input like:
```json
{
  "tenure": 12,
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0,
  "SeniorCitizen": 0,
  "Partner": 1
}
```
Response includes:
- `churn_prediction` (0 or 1)
- `probability` (0.0 - 1.0)
- `model_used`

## Where to find the data
- Raw dataset: `data/raw_data.csv`
- Processed dataset: `data/processed_data.csv`
- Trained model artifact: `models/best_model.pkl`
- Prediction history DB: `dashboard/predictions.db`
- MLflow experiment storage: `mlruns/`
- Preprocessing reports: `artifacts/preprocessing/`
- EDA reports and plots: `artifacts/eda/`
- Logs: `logs/`

## Viewing results in the dashboard
- Open the Streamlit app at `http://localhost:8501`
- Go to the **Prediction** page to run live churn scoring
- Go to the **ML Pipeline** page to inspect model metrics
- Go to the **Data Pipeline** page to inspect the raw and processed datasets
- The **Recent Predictions** section shows the last five saved scores

## Notes
- If the API or dashboard cannot reach the prediction service, ensure the Docker Compose stack is running with `make up`.
- Re-run `make pipeline` whenever data preprocessing changes.
- Re-run `make train` whenever training logic or the model should be updated.
- Use `make down` to stop containers when you are done.
