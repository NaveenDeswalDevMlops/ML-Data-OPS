import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.config import (
    API_URL,
    EDA_DIR,
    LOG_FILES,
    MLFLOW_URL,
    MODEL_PATH,
    PREDICTION_DB,
    PREPROCESS_DIR,
    PIPELINE_STATUS_PATH,
    RAW_DATA_PATH,
)
from dashboard.services import (
    get_app_health,
    get_app_metadata,
    get_data_preview,
    get_mlflow_client,
    get_mlflow_experiments,
    get_mlflow_run_metrics,
    get_mlflow_run_params,
    get_mlflow_runs,
    get_monitoring_stats,
    get_pipeline_status,
    get_prefect_status,
    get_preprocessing_report,
    get_eda_report,
    get_image,
    get_prediction_history,
    init_prediction_db,
    make_prediction,
    read_log,
    save_prediction,
)

st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>body {background-color: #0e1117; color: #ffffff;} .stApp {background-color: #0e1117;} button {background-color: #0f62fe; color: white; border-radius: 6px; padding: 10px 18px; border: none; cursor: pointer;}</style>",
    unsafe_allow_html=True,
)

SERVICE_URLS = {
    "API Docs": f"{API_URL}/docs",
    "Prefect UI": "http://localhost:4200",
    "MLflow UI": MLFLOW_URL,
}

PAGES = [
    "Home",
    "Data Pipeline",
    "Exploratory Data Analysis",
    "ML Pipeline",
    "MLflow Experiments",
    "Prefect Workflows",
    "Prediction",
    "Monitoring",
    "Logs",
    "Application Details",
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

sidebar = st.sidebar
sidebar.title("MLOps Platform")
page = sidebar.radio("Navigation", PAGES, index=PAGES.index(st.session_state.selected_page))
st.session_state.selected_page = page
sidebar.markdown("---")
sidebar.markdown("Built with Streamlit, FastAPI, MLflow, Prefect")


def link_button(label: str, url: str) -> None:
    st.markdown(
        f'<a href="{url}" target="_blank"><button>{label}</button></a>',
        unsafe_allow_html=True,
    )

if not PREDICTION_DB.exists():
    init_prediction_db()

if page == "Home":
    st.title("🏠 MLOps Dashboard")
    st.subheader("Enterprise-grade model operations platform")
    metadata = get_app_metadata()
    health = get_app_health()
    pipeline_status = get_pipeline_status()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project", "Telecom Churn Prediction")
    c1.metric("Dataset", "Telco Customer Churn")
    c2.metric("Model", Path(MODEL_PATH).name)
    c2.metric("Model Version", metadata.get("model_version", "unknown"))
    c3.metric("API Status", health.get("status", "unknown"))
    c3.metric("Pipeline Status", pipeline_status.get("pipeline_status", "unknown"))
    c4.metric("Last Pipeline Run", pipeline_status.get("last_pipeline_run", "unknown"))
    c4.metric("Total Predictions", len(get_prediction_history()))

    st.markdown("---")
    st.write("### Quick Navigation")
    tile_cols = st.columns(3)
    if tile_cols[0].button("Data Pipeline", key="home_data_pipeline"):
        st.session_state.selected_page = "Data Pipeline"
        st.experimental_rerun()
    if tile_cols[0].button("EDA", key="home_eda"):
        st.session_state.selected_page = "Exploratory Data Analysis"
        st.experimental_rerun()
    if tile_cols[0].button("ML Pipeline", key="home_ml_pipeline"):
        st.session_state.selected_page = "ML Pipeline"
        st.experimental_rerun()
    if tile_cols[1].button("MLflow", key="home_mlflow"):
        st.session_state.selected_page = "MLflow Experiments"
        st.experimental_rerun()
    if tile_cols[1].button("Prefect", key="home_prefect"):
        st.session_state.selected_page = "Prefect Workflows"
        st.experimental_rerun()
    if tile_cols[1].button("Prediction", key="home_prediction"):
        st.session_state.selected_page = "Prediction"
        st.experimental_rerun()
    if tile_cols[2].button("Monitoring", key="home_monitoring"):
        st.session_state.selected_page = "Monitoring"
        st.experimental_rerun()
    if tile_cols[2].button("Logs", key="home_logs"):
        st.session_state.selected_page = "Logs"
        st.experimental_rerun()
    if tile_cols[2].button("App Details", key="home_app_details"):
        st.session_state.selected_page = "Application Details"
        st.experimental_rerun()

    st.markdown("---")
    st.write("### External Service Links")
    link_button("Open API Docs", SERVICE_URLS["API Docs"])
    st.write(" ")
    link_button("Open Prefect UI", SERVICE_URLS["Prefect UI"])
    st.write(" ")
    link_button("Open MLflow UI", SERVICE_URLS["MLflow UI"])

    st.markdown("---")
    expander = st.expander("Project Summary")
    expander.write(
        "This dashboard provides a unified interface for dataset validation, model training, monitoring, and prediction without requiring separate access to Swagger, Prefect, or MLflow UIs."
    )

elif page == "Data Pipeline":
    st.title("📥 Data Pipeline")
    st.write("### Jump to external tools")
    link_button("Open Prefect UI", SERVICE_URLS["Prefect UI"])
    st.write(" ")
    link_button("Open API Docs", SERVICE_URLS["API Docs"])

    df = get_data_preview()
    if not df.empty:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        st.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")
    else:
        st.warning("Raw data not available.")

    st.write("### Summary Statistics")
    st.dataframe(get_preprocessing_report("summary_statistics.csv"))
    st.write("### Missing Values")
    st.dataframe(get_preprocessing_report("missing_values.csv"))
    st.write("### Data Types")
    st.dataframe(get_preprocessing_report("data_types.csv"))
    st.write("### Processing Status")
    status = get_pipeline_status()
    st.json(status)

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    raw = get_data_preview()
    if raw.empty:
        st.warning("EDA data not available.")
    else:
        st.subheader("Histogram")
        fig = px.histogram(raw, x=raw.columns[0], title="Sample Histogram")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Correlation Matrix")
        st.dataframe(get_eda_report("correlation_matrix.csv"))
        images = {
            "Heatmap": Path(EDA_DIR) / "heatmap.png",
            "Histogram": Path(EDA_DIR) / "histogram.png",
            "Boxplot": Path(EDA_DIR) / "boxplot.png",
            "Countplot": Path(EDA_DIR) / "countplot.png",
            "Feature Importance": Path(EDA_DIR) / "feature_importance.png",
        }
        for title, img_path in images.items():
            if img_path.exists():
                st.subheader(title)
                st.image(img_path, use_column_width=True)
        binned = get_eda_report("binned_features.csv")
        if not binned.empty:
            st.subheader("Binned Features")
            st.dataframe(binned)
        categorical_corr = get_eda_report("categorical_correlation.csv")
        if not categorical_corr.empty:
            st.subheader("Categorical Correlation")
            st.dataframe(categorical_corr)

elif page == "ML Pipeline":
    st.title("🤖 ML Pipeline")
    st.write("### Jump to external tools")
    link_button("Open MLflow UI", SERVICE_URLS["MLflow UI"])
    st.write(" ")
    link_button("Open API Docs", SERVICE_URLS["API Docs"])

    st.subheader("Model and Training Overview")
    st.markdown(f"**Best Model:** {Path(MODEL_PATH).name}")
    st.markdown(f"**Model Path:** {MODEL_PATH}")
    processed = get_data_preview()
    if not processed.empty:
        st.markdown(f"**Dataset Rows:** {processed.shape[0]}")
    st.write("### Evaluation Metrics")
    st.write("Use the MLflow Experiments page for full metrics visualization.")

elif page == "MLflow Experiments":
    st.title("📜 MLflow Experiments")
    st.write("Connects to MLflow tracking server directly.")
    link_button("Open MLflow UI", SERVICE_URLS["MLflow UI"])
    st.write(" ")
    link_button("Open API Docs", SERVICE_URLS["API Docs"])

    client = get_mlflow_client()
    experiments = get_mlflow_experiments()
    exp_names = [exp.get("name") for exp in experiments if isinstance(exp, dict)]
    experiment_name = st.selectbox("Select Experiment", exp_names)
    selected = next((exp for exp in experiments if exp.get("name") == experiment_name), None)
    if selected:
        st.write(selected)
        runs = get_mlflow_runs(selected.get("experiment_id"))
        st.dataframe(pd.DataFrame(runs))
        if runs:
            selected_run = runs[0]
            run_id = selected_run.get("info", {}).get("run_id")
            if run_id:
                st.write("### Run Metrics")
                st.json(get_mlflow_run_metrics(run_id))
                st.write("### Run Parameters")
                st.json(get_mlflow_run_params(run_id))

elif page == "Prefect Workflows":
    st.title("🔄 Prefect Workflows")
    st.write("### Jump to Prefect")
    link_button("Open Prefect UI", SERVICE_URLS["Prefect UI"])
    st.write(" ")
    st.write("### Prefect Status")
    status = get_prefect_status()
    st.json(status)

elif page == "Prediction":
    st.title("🎯 Prediction")
    st.write("Use this form to call the existing FastAPI prediction endpoint and store results in SQLite.")
    with st.form("prediction_form"):
        cols = st.columns(3)
        age = cols[0].number_input("Age", min_value=18, max_value=100, value=35)
        tenure = cols[1].number_input("Tenure", min_value=0, max_value=100, value=12)
        monthly = cols[2].number_input("Monthly Charges", min_value=0.0, value=70.0)
        total = cols[0].number_input("Total Charges", min_value=0.0, value=1200.0)
        contract = cols[1].selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        partner = cols[2].selectbox("Partner", ["Yes", "No"])
        dependents = cols[0].selectbox("Dependents", ["Yes", "No"])
        internet = cols[1].selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
        payment = cols[2].selectbox("Payment Method", ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"])
        gender = cols[0].selectbox("Gender", ["Male", "Female"])
        senior = cols[1].selectbox("Senior Citizen", ["Yes", "No"])
        submitted = st.form_submit_button("Predict")
    if submitted:
        payload = {
            "Age": age,
            "MonthlyIncome": monthly * 100,
            "JobSatisfaction": 3,
            "YearsAtCompany": tenure,
            "EnvironmentSatisfaction": 3,
            "WorkLifeBalance": 3,
            "TotalWorkingYears": max(1, int(total / 100)),
        }
        try:
            result = make_prediction(payload)
            risk = "High" if result.get("probability", 0) > 0.5 else "Low"
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "age": age,
                "tenure": tenure,
                "monthly_charges": monthly,
                "total_charges": total,
                "contract": contract,
                "partner": partner,
                "dependents": dependents,
                "internet_service": internet,
                "payment_method": payment,
                "gender": gender,
                "senior_citizen": senior,
                "prediction": result.get("attrition_prediction"),
                "probability": result.get("probability"),
                "risk_level": risk,
            }
            save_prediction(record)
            st.success(f"Prediction: {record['prediction']} - Probability: {record['probability']:.2f} - Risk: {risk}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    history = get_prediction_history()
    if not history.empty:
        st.subheader("Prediction History")
        st.dataframe(history)

elif page == "Monitoring":
    st.title("📈 Monitoring")
    stats = get_monitoring_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU Usage", f"{stats['cpu_percent']}%")
    col2.metric("Memory Usage", f"{stats['memory_percent']}%")
    col3.metric("Disk Usage", f"{stats['disk_percent']}%")
    st.write("### Service Health")
    st.json({
        "API": stats['api_status'],
        "MLflow": stats['mlflow_status'],
        "Prefect": stats['prefect_status'],
    })

elif page == "Logs":
    st.title("📜 Logs")
    log_type = st.selectbox("Select log stream", list(LOG_FILES.keys()))
    level = st.selectbox("Level filter", ["", "INFO", "WARNING", "ERROR"])
    search = st.text_input("Search")
    lines = read_log(LOG_FILES[log_type], level_filter=level if level else None, search_text=search or None)
    st.text_area(f"{log_type}", "\n".join(lines[-100:]), height=400)

elif page == "Application Details":
    st.title("⚙ Application Details")
    metadata = get_app_metadata()
    st.json(metadata)

st.sidebar.caption("Auto-refreshes every 5 seconds")
