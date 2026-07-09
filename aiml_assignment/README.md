# Project Name: Telecom Customer Churn Prediction (Production-Grade)

## 1. Project Overview
This project establishes a production-quality, end-to-end MLOps and DataOps pipeline. It utilizes **Prefect** for workflow orchestration, **MLflow** for model tracking, **FastAPI** for low-latency inference, and **Kubernetes** for cloud-native deployment.

## 2. Business Understanding
**Objective:** Reduce customer churn by predicting which customers are likely to leave, enabling proactive marketing interventions.
- **Problem:** Identify churn drivers using historical telecom data.
- **Solution:** A containerized API that serves predictions based on a trained Random Forest model.

## 3. Architecture
- **Data Pipeline:** Prefect flow (Ingestion -> Binning -> Preprocessing -> Validation).
- **ML Pipeline:** MLflow (Experiment tracking -> Model Registry -> Versioning).
- **Service Layer:** FastAPI (Endpoint: `/predict`, `/health`, `/metadata`).
- **Deployment:** Dockerized containers orchestrated via Kubernetes.

## 4. Getting Started
1. **Clone the repo.**
2. **Setup Environment:** `pip install -r requirements.txt`
3. **Start Orchestration:** `prefect server start`
4. **Run Data Pipeline:** `python pipelines/data_pipeline.py`
5. **Deploy Model:** `python pipelines/ml_pipeline.py`
6. **Run API:** `uvicorn app.main:app --reload`

## 5. Generated Artifacts
The pipeline automatically generates preprocessing reports, EDA reports, and visualizations every time it runs.

### Preprocessing artifacts
- `artifacts/preprocessing/summary_statistics.csv`
- `artifacts/preprocessing/missing_values.csv`
- `artifacts/preprocessing/data_types.csv`

### EDA artifacts
- `artifacts/eda/correlation_matrix.csv`
- `artifacts/eda/categorical_correlation.csv`
- `artifacts/eda/binned_features.csv`
- `artifacts/eda/encoding_report.csv`
- `artifacts/eda/feature_importance.csv`
- `artifacts/pipeline_status.json`

### Visualization artifacts
- `artifacts/eda/heatmap.png`
- `artifacts/eda/histogram.png`
- `artifacts/eda/boxplot.png`
- `artifacts/eda/feature_importance.png`
- `artifacts/eda/countplot.png`
- `artifacts/eda/pairplot.png` (optional when dataset size permits)

## 6. API Documentation
- `GET /health`: Returns system status and model version.
- `GET /metadata`: Fetches live configuration from MLflow/Prefect.
- `POST /predict`: Accepts JSON payload, returns churn prediction.