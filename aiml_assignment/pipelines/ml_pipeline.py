import os
import sys
import pandas as pd
import mlflow
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_logger import get_logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Silence the Git warning
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

LOG = get_logger("ml_pipeline", "training.log")
os.makedirs("models", exist_ok=True)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Customer_Attrition_Prediction")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)

LOG.info("MLflow tracking URI set to %s", mlflow_tracking_uri)
LOG.info("MLflow experiment set to %s", mlflow_experiment_name)

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probs)
    }
    
    print(f"--- {model_name} Evaluation ---")
    print(classification_report(y_test, predictions))
    return metrics

def run_ml_pipeline():
    LOG.info("Starting ML pipeline")
    LOG.info("Loading raw training data from data/raw_data.csv")
    if not os.path.exists("data/raw_data.csv"):
        raise FileNotFoundError("Raw data file not found. Run the data pipeline first to create data/raw_data.csv.")

    df = pd.read_csv("data/raw_data.csv")
    df = df.rename(columns={
        'Tenure in Months': 'tenure',
        'Total Charges': 'TotalCharges',
        'Monthly Charge': 'MonthlyCharges',
        'Senior Citizen': 'SeniorCitizen'
    })

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # --- THE FIX: FORCE BINARY LABELS ---
    # Convert whatever is in 'Churn' to a string, then re-encode to 0 and 1
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'].astype(str))
    print(f"Target labels identified: {df['Churn'].unique()}")
    # ------------------------------------

    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner']
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }
    
    best_f1 = 0
    best_model_name = ""
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model),
        ])
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test, name)
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model_name = name
                joblib.dump(pipeline, "models/best_model.pkl")
                
            LOG.info("Run %s completed with metrics: %s", name, metrics)
    LOG.info("Pipeline complete. Best Model: %s (F1 Score: %.4f)", best_model_name, best_f1)

if __name__ == "__main__":
    run_ml_pipeline()