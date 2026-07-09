from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Make the `dashboard` package importable whether this file sits at the
# project root (next to the dashboard/ folder) or inside a subfolder.
CURRENT_DIR = Path(__file__).resolve().parent
for candidate in (CURRENT_DIR, CURRENT_DIR.parent, CURRENT_DIR.parent.parent):
    if (candidate / "dashboard").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Ensure these imports match your actual local file structure ---
from dashboard.config import (
    API_URL, EDA_DIR, EXTERNAL_API_URL, EXTERNAL_MLFLOW_URL,
    EXTERNAL_PREFECT_URL, LOG_FILES, MLFLOW_URL, MODEL_PATH,
    PREDICTION_DB, PREPROCESS_DIR, PIPELINE_STATUS_PATH, RAW_DATA_PATH,
)
from dashboard.services import (
    get_app_health, get_app_metadata, get_data_preview,
    get_mlflow_client, get_mlflow_experiments, get_mlflow_run_metrics,
    get_mlflow_run_params, get_mlflow_runs, get_model_info,
    get_monitoring_stats, get_pipeline_status, get_prefect_status,
    get_preprocessing_report, get_eda_report, get_image,
    get_prediction_history, init_prediction_db, make_prediction,
    read_log, save_prediction,
)

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="MLOps Control Center",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light, modern analytics theme — white surfaces, soft shadows, blue + coral
# accents, and high-contrast slate text (inspired by the Streamlit design system).
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #F5F7FB;
        --surface: #FFFFFF;
        --surface-2: #F8FAFD;
        --border: #E6EAF2;
        --text: #111827;
        --muted: #6B7280;
        --accent: #3B82F6;
        --accent-2: #FF4B4B;
        --accent-soft: rgba(59, 130, 246, 0.08);
        --success: #16A34A;
        --warning: #D97706;
        --danger: #DC2626;
        --shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px rgba(16, 24, 40, 0.06);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(900px 400px at 85% -10%, rgba(59,130,246,0.10), transparent 60%),
            radial-gradient(700px 350px at -10% 0%, rgba(255,75,75,0.06), transparent 55%),
            var(--bg);
        color: var(--text);
    }

    /* Force readable text everywhere (labels, markdown, widgets) */
    .stApp p, .stApp label, .stApp span, .stApp li,
    .stMarkdown, [data-testid="stWidgetLabel"] p {
        color: var(--text);
    }
    .stCaption, [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] .stRadio label p {
        font-size: 0.92rem;
        color: #374151;
    }
    .sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        padding: 4px 0 14px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 10px;
    }
    .sidebar-brand .mark {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 60%, #FF4B4B 130%);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: #fff; font-size: 15px;
        letter-spacing: -0.5px;
        box-shadow: 0 4px 12px rgba(59,130,246,0.35);
    }
    .sidebar-brand .name { font-weight: 700; font-size: 0.98rem; color: var(--text); line-height: 1.1; }
    .sidebar-brand .sub  { font-size: 0.72rem; color: var(--muted); }

    /* ---------- Page header ---------- */
    .page-header { margin-bottom: 4px; }
    .page-eyebrow {
        text-transform: uppercase; letter-spacing: 0.14em;
        font-size: 0.7rem; font-weight: 700; color: var(--accent);
        margin-bottom: 2px;
    }
    .page-title {
        font-size: 1.7rem; font-weight: 800; color: var(--text);
        letter-spacing: -0.02em; margin: 0;
    }
    .page-desc { color: var(--muted); font-size: 0.93rem; margin-top: 4px; }

    .section-label {
        text-transform: uppercase; letter-spacing: 0.12em;
        font-size: 0.72rem; font-weight: 700; color: #94A3B8;
        margin: 6px 0 10px 0;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: var(--shadow);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(16,24,40,0.06), 0 12px 32px rgba(16,24,40,0.10);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: var(--text);
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }

    /* ---------- Status badges ---------- */
    .badge {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 4px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
        border: 1px solid var(--border);
        background: var(--surface); color: var(--muted);
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }
    .badge .dot { width: 7px; height: 7px; border-radius: 50%; background: #9CA3AF; }
    .badge.ok      { color: #15803D; border-color: #BBE7C9; background: #F0FBF4; }
    .badge.ok .dot { background: var(--success); }
    .badge.warn      { color: #B45309; border-color: #F3DFB8; background: #FFFAEF; }
    .badge.warn .dot { background: var(--warning); }
    .badge.err      { color: #B91C1C; border-color: #F5C6C6; background: #FEF2F2; }
    .badge.err .dot { background: var(--danger); }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: var(--surface);
        color: #374151;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        font-size: 0.88rem;
        width: 100%;
        box-shadow: 0 1px 2px rgba(16,24,40,0.05);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--accent-soft);
        transform: translateY(-1px);
    }
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #3B82F6, #6366F1);
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 700;
        width: 100%;
        box-shadow: 0 6px 16px rgba(59,130,246,0.35);
        transition: all 0.15s ease;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(59,130,246,0.45);
        color: #FFFFFF !important;
    }
    .stFormSubmitButton > button p { color: #FFFFFF !important; }

    /* External link buttons */
    .ext-link {
        display: block; text-decoration: none;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: var(--text);
        font-weight: 600; font-size: 0.86rem;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        transition: all 0.15s ease;
    }
    .ext-link:hover { border-color: var(--accent); transform: translateY(-1px); box-shadow: var(--shadow); }
    .ext-link .arrow { float: right; color: var(--accent); }
    .ext-link .sub { display: block; color: var(--muted); font-size: 0.74rem; font-weight: 500; margin-top: 2px; }

    /* ---------- Panels / cards ---------- */
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: var(--shadow);
    }

    /* ---------- Data / inputs ---------- */
    [data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        background: #F8FAFC !important;
        color: #334155 !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
        background: var(--surface);
        border-radius: 8px;
    }

    h2, h3 { color: var(--text); font-weight: 700; letter-spacing: -0.01em; }
    hr { border-color: var(--border); }

    /* Expander (pipeline run logs) */
    [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }

    /* Alerts (info/success/error) — soft pastel look */
    [data-testid="stAlert"] { border-radius: 12px; }

    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3B82F6, #6366F1);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. UI HELPERS (presentation only — no service logic changed)
# ---------------------------------------------------------
def page_header(eyebrow: str, title: str, desc: str = "") -> None:
    desc_html = f'<div class="page-desc">{desc}</div>' if desc else ""
    st.markdown(
        f'<div class="page-header"><div class="page-eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>{desc_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def status_badge(label: str, state: str = "neutral") -> str:
    # state: ok | warn | err | neutral
    cls = {"ok": "ok", "warn": "warn", "err": "err"}.get(state, "")
    return f'<span class="badge {cls}"><span class="dot"></span>{label}</span>'


def infer_state(value: str) -> str:
    v = str(value).lower()
    if any(k in v for k in ("ok", "healthy", "running", "success", "complete", "up", "active")):
        return "ok"
    if any(k in v for k in ("fail", "error", "down", "crash")):
        return "err"
    if any(k in v for k in ("pending", "warn", "degraded", "partial", "unknown")):
        return "warn"
    return "warn"


def link_button(label: str, url: str, sub: str = "") -> None:
    sub_html = f'<span class="sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<a class="ext-link" href="{url}" target="_blank">{label}'
        f'<span class="arrow">↗</span>{sub_html}</a>',
        unsafe_allow_html=True,
    )


def extract_run_metrics(run: dict) -> dict:
    """Safely pull the metrics dict out of an MLflow run payload."""
    if not isinstance(run, dict):
        return {}
    metrics = run.get("data", {}).get("metrics") if isinstance(run.get("data"), dict) else None
    if not metrics:
        metrics = run.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def find_metric(metrics: dict, *candidates) -> float | None:
    """Find a metric by fuzzy key match, e.g. 'accuracy' matches 'val_accuracy'."""
    if not isinstance(metrics, dict):
        return None
    lowered = {str(k).lower(): v for k, v in metrics.items()}
    for cand in candidates:
        for key, val in lowered.items():
            if cand in key:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return None


def model_metric_cards(metrics: dict) -> None:
    """Render Accuracy / Precision / Recall / F1 cards from an MLflow metrics dict."""
    accuracy = find_metric(metrics, "accuracy", "acc")
    precision = find_metric(metrics, "precision")
    recall = find_metric(metrics, "recall")
    f1 = find_metric(metrics, "f1")
    roc_auc = find_metric(metrics, "roc_auc", "auc")

    cols = st.columns(5)
    values = [
        ("Accuracy", accuracy),
        ("Precision", precision),
        ("Recall", recall),
        ("F1 Score", f1),
        ("ROC AUC", roc_auc),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, f"{value:.3f}" if value is not None else "—")


def metrics_bar_chart(metrics: dict, title: str = "Run metrics") -> None:
    if not metrics:
        return
    numeric = {}
    for k, v in metrics.items():
        try:
            numeric[k] = float(v)
        except (TypeError, ValueError):
            continue
    if not numeric:
        return
    df = pd.DataFrame({"metric": list(numeric.keys()), "value": list(numeric.values())})
    fig = px.bar(df, x="metric", y="value", title=title)
    fig.update_traces(marker_color="#3B82F6", marker_line_width=0)
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)


def style_plotly(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#374151", size=12),
        title_font=dict(size=14, color="#111827"),
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(gridcolor="#EEF1F6", zerolinecolor="#EEF1F6"),
        yaxis=dict(gridcolor="#EEF1F6", zerolinecolor="#EEF1F6"),
        colorway=["#3B82F6", "#FF4B4B", "#8B5CF6", "#10B981", "#F59E0B"],
    )


def split_log_into_runs(lines: list[str]) -> list[list[str]]:
    """Group raw log lines into pipeline runs.

    A new run starts whenever a line contains a run-start marker
    (Prefect flow-run creation / pipeline start messages). Lines before
    the first marker form their own group.
    """
    markers = (
        "created flow run", "beginning flow run", "flow run",
        "pipeline started", "starting pipeline", "pipeline run",
    )
    runs: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        lowered = line.lower()
        is_start = any(m in lowered for m in markers) and any(
            s in lowered for s in ("created", "beginning", "started", "starting")
        )
        if is_start and current:
            runs.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        runs.append(current)
    return runs


def maybe_rerun() -> None:
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ---------------------------------------------------------
# 3. APP STATE & NAVIGATION
# ---------------------------------------------------------
SERVICE_URLS = {
    "API Docs": f"{EXTERNAL_API_URL}/docs",
    "Prefect UI": EXTERNAL_PREFECT_URL,
    "MLflow UI": EXTERNAL_MLFLOW_URL,
}

PAGES = [
    "Home", "Data Pipeline", "Exploratory Data Analysis", "ML Pipeline",
    "MLflow Experiments", "Prefect Workflows", "Prediction",
    "Monitoring", "Logs", "Application Details",
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

sidebar = st.sidebar
sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="mark">ML</div>
        <div>
            <div class="name">MLOps Control Center</div>
            <div class="sub">Telecom Churn · Production</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
selected_radio = sidebar.radio("Navigation", PAGES, index=PAGES.index(st.session_state.selected_page))
if selected_radio != st.session_state.selected_page:
    st.session_state.selected_page = selected_radio
page = st.session_state.selected_page

sidebar.markdown("---")
sidebar.markdown('<div class="section-label">External services</div>', unsafe_allow_html=True)
with sidebar:
    link_button("MLflow Tracking", SERVICE_URLS["MLflow UI"])
    link_button("Prefect Orchestration", SERVICE_URLS["Prefect UI"])
    link_button("API Reference", SERVICE_URLS["API Docs"])
sidebar.markdown("---")
sidebar.caption("Streamlit · FastAPI · MLflow · Prefect")

if not PREDICTION_DB.exists():
    init_prediction_db()

# ---------------------------------------------------------
# 4. PAGE ROUTING & UI LAYOUTS
# ---------------------------------------------------------
if page == "Home":
    page_header(
        "Overview",
        "MLOps Control Center",
        "End-to-end visibility across data pipelines, experiments, model serving and system health.",
    )

    metadata = get_app_metadata()
    health = get_app_health()
    pipeline_status = get_pipeline_status()

    # --- Live status strip ---
    api_state = health.get("status", "unknown")
    pipe_state = pipeline_status.get("pipeline_status", "unknown")
    st.markdown(
        "&nbsp;&nbsp;".join([
            status_badge(f"API · {api_state}", infer_state(api_state)),
            status_badge(f"Pipeline · {pipe_state}", infer_state(pipe_state)),
            status_badge(f"Model · {Path(MODEL_PATH).name if MODEL_PATH else 'none'}",
                         "ok" if MODEL_PATH else "err"),
        ]),
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    section("Key indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project", "Telecom Churn")
    c2.metric("Dataset", "Telco Customer")
    c3.metric("Model Version", metadata.get("model_version", "unknown"))
    c4.metric("Total Predictions", len(get_prediction_history()))

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Latest model quality from MLflow ---
    section("Latest model quality (MLflow)")
    experiments = get_mlflow_experiments()
    latest_metrics = {}
    if experiments:
        latest_experiment_id = experiments[0].get("experiment_id")
        if latest_experiment_id:
            runs = get_mlflow_runs(latest_experiment_id)
            if runs:
                latest_run = runs[0]
                latest_metrics = extract_run_metrics(latest_run)
                if not latest_metrics:
                    run_id = latest_run.get("info", {}).get("run_id")
                    if run_id:
                        latest_metrics = get_mlflow_run_metrics(run_id) or {}
    if latest_metrics:
        model_metric_cards(latest_metrics)
    else:
        st.info("No MLflow run metrics available yet. Train a model to populate this section.")

    st.markdown("---")

    col_nav, col_ext = st.columns([2, 1], gap="large")
    with col_nav:
        section("Quick actions")
        t1, t2, t3 = st.columns(3)
        if t1.button("Data Pipeline"):
            st.session_state.selected_page = "Data Pipeline"
        if t2.button("Explore EDA"):
            st.session_state.selected_page = "Exploratory Data Analysis"
        if t3.button("ML Pipeline"):
            st.session_state.selected_page = "ML Pipeline"
        t4, t5, t6 = st.columns(3)
        if t4.button("Prediction"):
            st.session_state.selected_page = "Prediction"
        if t5.button("Monitoring"):
            st.session_state.selected_page = "Monitoring"
        if t6.button("View Logs"):
            st.session_state.selected_page = "Logs"
        page = st.session_state.selected_page

    with col_ext:
        section("External tools")
        link_button("API Docs (Swagger)", SERVICE_URLS["API Docs"], "FastAPI service reference")
        link_button("Prefect Dashboard", SERVICE_URLS["Prefect UI"], "Workflow orchestration")
        link_button("MLflow Tracking", SERVICE_URLS["MLflow UI"], "Experiments & model registry")

elif page == "Data Pipeline":
    page_header("Data", "Data Pipeline",
                "Raw dataset intake, preprocessing reports and pipeline execution status.")

    status = get_pipeline_status()
    st.markdown(
        "&nbsp;&nbsp;".join([
            status_badge(f"Status · {status.get('pipeline_status', 'unknown')}",
                         infer_state(status.get('pipeline_status', 'unknown'))),
            status_badge(f"Last run · {status.get('last_pipeline_run', 'unknown')}"),
        ]),
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    df = get_data_preview()
    if not df.empty:
        section("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        st.info("Raw data not available.")

    col1, col2 = st.columns(2)
    with col1:
        section("Summary statistics")
        st.dataframe(get_preprocessing_report("summary_statistics.csv"), use_container_width=True)
    with col2:
        section("Missing values")
        st.dataframe(get_preprocessing_report("missing_values.csv"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("Data types")
        st.dataframe(get_preprocessing_report("data_types.csv"), use_container_width=True)
    with col4:
        section("Processing status")
        st.json(status)

elif page == "Exploratory Data Analysis":
    page_header("Data", "Exploratory Data Analysis",
                "Distributions, correlations and engineered feature views of the training data.")
    raw = get_data_preview()

    if raw.empty:
        st.warning("EDA data not available.")
    else:
        section("Distribution")
        fig = px.histogram(raw, x=raw.columns[0], title=f"Distribution · {raw.columns[0]}")
        fig.update_traces(marker_color="#3B82F6")
        style_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

        section("Correlation matrix")
        st.dataframe(get_eda_report("correlation_matrix.csv"), use_container_width=True)

        images = {
            "Heatmap": Path(EDA_DIR) / "heatmap.png",
            "Histogram": Path(EDA_DIR) / "histogram.png",
            "Boxplot": Path(EDA_DIR) / "boxplot.png",
            "Countplot": Path(EDA_DIR) / "countplot.png",
            "Feature Importance": Path(EDA_DIR) / "feature_importance.png",
        }
        available = [(t, p) for t, p in images.items() if p.exists()]
        if available:
            section("Visual reports")
            for i in range(0, len(available), 2):
                cols = st.columns(2)
                for col, (title, img_path) in zip(cols, available[i:i + 2]):
                    with col:
                        st.caption(title)
                        st.image(str(img_path), use_column_width=True)

        binned = get_eda_report("binned_features.csv")
        if not binned.empty:
            section("Binned features")
            st.dataframe(binned, use_container_width=True)

        categorical_corr = get_eda_report("categorical_correlation.csv")
        if not categorical_corr.empty:
            section("Categorical correlation")
            st.dataframe(categorical_corr, use_container_width=True)

elif page == "ML Pipeline":
    page_header("Model", "ML Pipeline",
                "Active model artefact, training summary and evaluation metrics from MLflow.")

    section("Active model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", Path(MODEL_PATH).name if MODEL_PATH else "None")
    processed = get_data_preview()
    c2.metric("Training Rows", f"{processed.shape[0]:,}" if not processed.empty else "—")
    c3.metric("Artefact Path", "Registered" if MODEL_PATH else "Missing")
    st.caption(f"Model path: `{MODEL_PATH}`")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Evaluation metrics (latest MLflow run)")
    experiments = get_mlflow_experiments()
    metrics = {}
    if experiments:
        exp_id = experiments[0].get("experiment_id")
        if exp_id:
            runs = get_mlflow_runs(exp_id)
            if runs:
                metrics = extract_run_metrics(runs[0])
                if not metrics:
                    run_id = runs[0].get("info", {}).get("run_id")
                    if run_id:
                        metrics = get_mlflow_run_metrics(run_id) or {}
    if metrics:
        model_metric_cards(metrics)
        st.markdown("<br>", unsafe_allow_html=True)
        metrics_bar_chart(metrics, "All logged metrics · latest run")
    else:
        st.info("No evaluation metrics found. Open the MLflow Experiments page for the full run history.")

elif page == "MLflow Experiments":
    page_header("Model", "MLflow Experiments",
                "Experiment tracking connected directly to the MLflow tracking server.")

    client = get_mlflow_client()
    experiments = get_mlflow_experiments()
    exp_names = [exp.get("name") for exp in experiments if isinstance(exp, dict)]

    if not exp_names:
        st.info("No MLflow experiments detected yet.")
    else:
        experiment_name = st.selectbox("Experiment", exp_names)
        selected = next((exp for exp in experiments if exp.get("name") == experiment_name), None)

        if selected:
            st.markdown(
                "&nbsp;&nbsp;".join([
                    status_badge(f"Experiment ID · {selected.get('experiment_id', '—')}"),
                    status_badge(f"Lifecycle · {selected.get('lifecycle_stage', '—')}",
                                 infer_state(selected.get('lifecycle_stage', ''))),
                ]),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            runs = get_mlflow_runs(selected.get("experiment_id"))
            if runs:
                # Build a readable run selector
                run_labels = []
                for r in runs:
                    info = r.get("info", {}) if isinstance(r, dict) else {}
                    label = info.get("run_name") or info.get("run_id", "run")
                    run_labels.append(str(label))
                idx = st.selectbox("Run", range(len(runs)),
                                   format_func=lambda i: run_labels[i])
                selected_run = runs[idx]
                run_id = selected_run.get("info", {}).get("run_id")

                metrics = {}
                if run_id:
                    metrics = get_mlflow_run_metrics(run_id) or {}
                if not metrics:
                    metrics = extract_run_metrics(selected_run)

                section("Model performance")
                model_metric_cards(metrics)

                st.markdown("<br>", unsafe_allow_html=True)
                col_a, col_b = st.columns([3, 2], gap="large")
                with col_a:
                    metrics_bar_chart(metrics, "All logged metrics")
                with col_b:
                    section("Run parameters")
                    params = {}
                    if run_id:
                        params = get_mlflow_run_params(run_id) or {}
                    if params:
                        params_df = pd.DataFrame(
                            {"parameter": list(params.keys()), "value": [str(v) for v in params.values()]}
                        )
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No parameters logged for this run.")

                section("All runs in experiment")
                st.dataframe(pd.DataFrame(runs), use_container_width=True)
            else:
                st.info("No runs recorded in this experiment yet.")

elif page == "Prefect Workflows":
    page_header("Orchestration", "Prefect Workflows",
                "Orchestrator health and execution logs for each data pipeline run.")

    status = get_prefect_status()
    state_text = status.get("status", status.get("state", "unknown")) if isinstance(status, dict) else "unknown"
    st.markdown(status_badge(f"Prefect · {state_text}", infer_state(state_text)),
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3], gap="large")
    with col1:
        section("Orchestrator status")
        st.json(status)
    with col2:
        section("Pipeline run logs")
        # Pick the most pipeline-relevant log stream, falling back to the first one.
        log_keys = list(LOG_FILES.keys())
        default_idx = 0
        for i, key in enumerate(log_keys):
            if any(k in key.lower() for k in ("pipeline", "prefect", "flow")):
                default_idx = i
                break
        log_choice = st.selectbox("Log stream", log_keys, index=default_idx)
        level = st.selectbox("Level filter", ["All", "INFO", "WARNING", "ERROR"])

        lines = read_log(
            LOG_FILES[log_choice],
            level_filter=None if level == "All" else level,
            search_text=None,
        )
        if lines:
            grouped = split_log_into_runs(lines)
            # Show most recent runs first
            for run_index, run_lines in enumerate(reversed(grouped)):
                run_number = len(grouped) - run_index
                first_line = run_lines[0][:110] if run_lines else ""
                has_error = any("error" in l.lower() for l in run_lines)
                icon = "🔴" if has_error else "🟢"
                with st.expander(f"{icon} Run {run_number} · {len(run_lines)} lines — {first_line}",
                                 expanded=(run_index == 0)):
                    st.text_area(
                        f"run_{run_number}_log",
                        "\n".join(run_lines[-200:]),
                        height=260,
                        label_visibility="collapsed",
                    )
        else:
            st.info("No log entries found for this stream yet.")

elif page == "Prediction":
    page_header("Serving", "Live Prediction",
                "Real-time churn scoring against the deployed model using the Telco feature schema.")

    with st.form("prediction_form", clear_on_submit=False):
        section("Customer profile")
        cols = st.columns(3)
        tenure = cols[0].number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = cols[1].number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = cols[2].number_input("Total Charges ($)", min_value=0.0, value=840.0)

        senior_citizen = cols[0].selectbox("Senior Citizen", ["No", "Yes"])
        partner = cols[1].selectbox("Partner", ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submit_col, _ = st.columns([1, 3])
        with submit_col:
            submitted = st.form_submit_button("Run prediction", use_container_width=True)

    if submitted:
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
        }
        with st.spinner("Scoring customer..."):
            try:
                result = make_prediction(payload)
                risk = "High" if result.get("probability", 0) > 0.5 else "Low"
                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "age": 0,
                    "tenure": tenure,
                    "monthly_charges": monthly_charges,
                    "total_charges": total_charges,
                    "contract": "N/A",
                    "partner": partner,
                    "dependents": "N/A",
                    "internet_service": "N/A",
                    "payment_method": "N/A",
                    "gender": "N/A",
                    "senior_citizen": senior_citizen,
                    "prediction": result.get("churn_prediction"),
                    "probability": result.get("probability"),
                    "risk_level": risk,
                }
                save_prediction(record)

                prob = record["probability"] or 0.0
                r1, r2 = st.columns([1, 3])
                with r1:
                    st.metric("Churn probability", f"{prob:.1%}")
                with r2:
                    if risk == "High":
                        st.error("High churn risk — consider retention outreach for this customer.")
                    else:
                        st.success("Low churn risk — no immediate action required.")
                    st.progress(min(max(prob, 0.0), 1.0))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    history = get_prediction_history()
    if not history.empty:
        st.markdown("---")
        section("Recent predictions")
        st.dataframe(history.tail(5), use_container_width=True)

elif page == "Monitoring":
    page_header("Operations", "System Monitoring",
                "Infrastructure utilisation, service health and current model performance.")

    stats = get_monitoring_stats()

    section("Resource utilisation")
    col1, col2, col3 = st.columns(3)
    cpu = float(stats.get("cpu_percent", 0) or 0)
    mem = float(stats.get("memory_percent", 0) or 0)
    disk = float(stats.get("disk_percent", 0) or 0)
    with col1:
        st.metric("CPU Usage", f"{cpu:.0f}%")
        st.progress(min(cpu / 100, 1.0))
    with col2:
        st.metric("Memory Usage", f"{mem:.0f}%")
        st.progress(min(mem / 100, 1.0))
    with col3:
        st.metric("Disk Usage", f"{disk:.0f}%")
        st.progress(min(disk / 100, 1.0))

    st.markdown("<br>", unsafe_allow_html=True)
    section("Service health")
    services = {
        "FastAPI": stats.get("api_status", "Unknown"),
        "MLflow": stats.get("mlflow_status", "Unknown"),
        "Prefect": stats.get("prefect_status", "Unknown"),
    }
    st.markdown(
        "&nbsp;&nbsp;".join(
            status_badge(f"{name} · {state}", infer_state(state))
            for name, state in services.items()
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        section("Model info")
        model_info = get_model_info()
        st.json(model_info)
    with col_b:
        section("Latest model performance (MLflow)")
        experiments = get_mlflow_experiments()
        if experiments:
            latest_experiment_id = experiments[0].get("experiment_id")
            if latest_experiment_id:
                runs = get_mlflow_runs(latest_experiment_id)
                if runs:
                    latest_run = runs[0]
                    metrics = extract_run_metrics(latest_run)
                    if not metrics:
                        run_id = latest_run.get("info", {}).get("run_id")
                        if run_id:
                            metrics = get_mlflow_run_metrics(run_id) or {}
                    if metrics:
                        model_metric_cards(metrics)
                    else:
                        st.info("No metrics found for the latest run.")
                else:
                    st.info("No MLflow runs available for the selected experiment.")
            else:
                st.info("No MLflow experiment id found.")
        else:
            st.info("No MLflow experiments detected yet.")

elif page == "Logs":
    page_header("Operations", "Logs",
                "Live service logs with level filtering and full-text search.")

    c1, c2, c3 = st.columns(3)
    with c1:
        log_type = st.selectbox("Log stream", list(LOG_FILES.keys()))
    with c2:
        level = st.selectbox("Level filter", ["", "INFO", "WARNING", "ERROR"],
                             format_func=lambda x: x or "All levels")
    with c3:
        search = st.text_input("Search")

    lines = read_log(LOG_FILES[log_type], level_filter=level if level else None,
                     search_text=search or None)

    total = len(lines)
    errors = sum(1 for l in lines if "error" in l.lower())
    warnings = sum(1 for l in lines if "warning" in l.lower())
    m1, m2, m3 = st.columns(3)
    m1.metric("Entries", f"{total:,}")
    m2.metric("Warnings", f"{warnings:,}")
    m3.metric("Errors", f"{errors:,}")

    st.text_area(f"{log_type}", "\n".join(lines[-100:]), height=420)

elif page == "Application Details":
    page_header("System", "Application Details",
                "Deployment metadata and runtime configuration of the platform.")
    metadata = get_app_metadata()
    st.json(metadata)