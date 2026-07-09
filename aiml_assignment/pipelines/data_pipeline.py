import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import task, flow, get_run_logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_logger import get_logger

# The stable URL
DATA_URL = "https://huggingface.co/datasets/Brammi114/telco-customer-churn/resolve/main/train.csv"
DATA_DIR = "data"
PREPROCESS_DIR = "artifacts/preprocessing"
EDA_DIR = "artifacts/eda"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREPROCESS_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
LOG = get_logger("data_pipeline", "pipeline.log")


def compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    n = confusion.sum().sum()
    if n == 0:
        return 0.0

    row_totals = confusion.sum(axis=1).values.reshape(-1, 1)
    col_totals = confusion.sum(axis=0).values.reshape(1, -1)
    expected = row_totals.dot(col_totals) / n

    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = ((confusion.values - expected) ** 2 / expected)
        chi2 = np.nansum(chi2)

    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0


@task(retries=2)
def ingest_data() -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Starting Data Ingestion...")
    LOG.info("Starting Data Ingestion...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(f"{DATA_DIR}/raw_data.csv", index=False)
    logger.info(f"Rows Loaded: {df.shape[0]}")
    logger.info(f"Columns Loaded: {df.shape[1]}")
    logger.info(f"Ingested data shape: {df.shape}")
    return df


@task
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Pipeline Started")
    logger.info("Starting Data Pre-processing...")
    LOG.info("Starting Data Pre-processing")

    df = df.rename(columns={
        'Tenure in Months': 'tenure',
        'Total Charges': 'TotalCharges',
        'Monthly Charge': 'MonthlyCharges',
        'Senior Citizen': 'SeniorCitizen'
    })

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    summary_statistics = df.describe(include='all')
    summary_statistics.to_csv(f"{PREPROCESS_DIR}/summary_statistics.csv")
    logger.info("Summary Statistics Generated")

    dtypes = df.dtypes.reset_index()
    dtypes.columns = ['column', 'dtype']
    dtypes.to_csv(f"{PREPROCESS_DIR}/data_types.csv", index=False)
    logger.info("Data Types Generated")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        'missing_count': missing,
        'missing_percentage': missing_pct
    })
    missing_report.to_csv(f"{PREPROCESS_DIR}/missing_values.csv")
    logger.info(f"Missing Value Report Generated: total missing values={int(missing.sum())}")

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    logger.info(f"Numeric columns: {numeric_columns}")
    logger.info(f"Categorical columns: {categorical_columns}")

    raw_tenure = None
    if 'tenure' in df.columns:
        raw_tenure = df['tenure'].copy()

    if len(numeric_columns) > 0:
        if df[numeric_columns].isnull().any(axis=None):
            medians = df[numeric_columns].median()
            df[numeric_columns] = df[numeric_columns].fillna(medians)
            logger.info("Numeric imputation completed using median values")

        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        logger.info(f"Normalization Completed")
        logger.info(f"Columns normalized: {numeric_columns}")
        logger.info("Scaler used: StandardScaler")
        logger.info(f"Output shape: {df.shape}")
    else:
        logger.info("No numeric columns found for normalization")

    if raw_tenure is not None:
        bins = [0, 13, 25, 49, np.inf]
        labels = ['0-12', '13-24', '25-48', '49+']
        binned = pd.DataFrame({'tenure': raw_tenure})
        binned['tenure_bin'] = pd.cut(raw_tenure, bins=bins, labels=labels, right=False)
        binned.to_csv(f"{EDA_DIR}/binned_features.csv", index=False)
        logger.info("Binning Completed")
    else:
        logger.info("Tenure column not available for binning")

    if 'Churn' in df.columns:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'].astype(str))
        encoding_report = pd.DataFrame({
            'Original Value': le.classes_.astype(str),
            'Encoded Value': list(range(len(le.classes_)))
        })
        encoding_report.to_csv(f"{EDA_DIR}/encoding_report.csv", index=False)
        logger.info("Encoding Completed: Churn values encoded and report generated")
    else:
        logger.info("Churn column not found for encoding")

    df.to_csv(f"{DATA_DIR}/processed_data.csv", index=False)
    logger.info("Pre-processing complete. Processed data saved.")
    return df


@task
def perform_eda(df: pd.DataFrame):
    logger = get_run_logger()
    logger.info("Generating EDA reports...")
    LOG.info("Generating EDA reports")

    numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()
    correlation_matrix = numeric_df.corr()
    correlation_matrix.to_csv(f"{EDA_DIR}/correlation_matrix.csv")

    corr_values = correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool))
    strongest_positive = corr_values.stack().idxmax()
    strongest_negative = corr_values.stack().idxmin()
    logger.info(
        f"Correlation Analysis Completed: strongest positive={strongest_positive} ({corr_values.loc[strongest_positive]:.4f}), "
        f"strongest negative={strongest_negative} ({corr_values.loc[strongest_negative]:.4f})"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/heatmap.png")
    plt.close()

    if 'MonthlyCharges' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['MonthlyCharges'], kde=True)
        plt.title('Monthly Charges Distribution')
        plt.savefig(f"{EDA_DIR}/histogram.png")
        plt.close()
    else:
        logger.info("MonthlyCharges column not available for histogram")

    if 'Churn' in df.columns and 'MonthlyCharges' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
        plt.title('Monthly Charges by Churn')
        plt.savefig(f"{EDA_DIR}/boxplot.png")
        plt.close()
    else:
        logger.info("Required columns not available for boxplot")

    if 'Churn' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Churn', data=df)
        plt.title('Target Countplot')
        plt.savefig(f"{EDA_DIR}/countplot.png")
        plt.close()
    else:
        logger.info("Churn column not available for countplot")

    if not numeric_df.empty:
        sample = numeric_df.sample(n=min(200, len(numeric_df)), random_state=42)
        pairplot_vars = list(sample.columns[:6])
        logger.info(f"Pairplot variables selected: {pairplot_vars}")
        try:
            sns.pairplot(sample[pairplot_vars])
            plt.savefig(f"{EDA_DIR}/pairplot.png")
            plt.close()
            logger.info("Pairplot generated")
        except Exception as exc:
            logger.info(f"Pairplot skipped due to size or plotting issue: {exc}")
    else:
        logger.info("Skipping pairplot due to no numeric data")

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        categorical_pairs = []
        for i, col_x in enumerate(categorical_columns):
            for col_y in categorical_columns[i + 1:]:
                x_values = df[col_x].fillna('NA').astype(str)
                y_values = df[col_y].fillna('NA').astype(str)
                cramers_value = compute_cramers_v(x_values, y_values)
                categorical_pairs.append({
                    'feature_1': col_x,
                    'feature_2': col_y,
                    'cramers_v': cramers_value
                })
        categorical_corr_df = pd.DataFrame(categorical_pairs)
        categorical_corr_df.to_csv(f"{EDA_DIR}/categorical_correlation.csv", index=False)
        logger.info("Categorical Correlation Completed")
    else:
        pd.DataFrame(columns=['feature_1', 'feature_2', 'cramers_v']).to_csv(
            f"{EDA_DIR}/categorical_correlation.csv", index=False)
        logger.info("No categorical columns found for categorical correlation")

    if 'tenure' in df.columns:
        bins = [0, 13, 25, 49, np.inf]
        labels = ['0-12', '13-24', '25-48', '49+']
        binned = df[['tenure']].copy()
        binned['tenure_bin'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
        binned.to_csv(f"{EDA_DIR}/binned_features.csv", index=False)
        logger.info("Binning Completed")
    else:
        logger.info("Tenure column not available for binning")

    feature_importance_generated = False
    if 'Churn' in numeric_df.columns and numeric_df.shape[1] > 1:
        X = numeric_df.drop(columns=['Churn'])
        y = numeric_df['Churn']
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values(by='importance', ascending=False)
        importance_df.to_csv(f"{EDA_DIR}/feature_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{EDA_DIR}/feature_importance.png")
        plt.close()
        logger.info("Feature Importance Generated")
        feature_importance_generated = True
    else:
        pd.DataFrame(columns=['feature', 'importance']).to_csv(
            f"{EDA_DIR}/feature_importance.csv", index=False)
        logger.info("Feature Importance skipped due to missing numeric target or features")

    missing_count = int(df.isnull().sum().sum())
    status_payload = {
        'last_pipeline_run': datetime.utcnow().isoformat(),
        'rows_processed': int(df.shape[0]),
        'features_processed': int(df.shape[1]),
        'missing_values_found': missing_count,
        'correlation_matrix_generated': True,
        'eda_status': 'completed',
        'feature_importance_status': 'generated' if feature_importance_generated else 'skipped',
        'pipeline_status': 'completed'
    }
    with open('artifacts/pipeline_status.json', 'w', encoding='utf-8') as status_file:
        import json
        json.dump(status_payload, status_file, indent=2)

    logger.info("Visualizations Generated")
    logger.info("Pipeline Completed Successfully")
    LOG.info("Pipeline Completed Successfully")


@flow(name="DataOps-Pipeline", log_prints=True)
def data_pipeline_flow():
    df = ingest_data()
    processed_df = preprocess_data(df)
    perform_eda(processed_df)


if __name__ == "__main__":
    data_pipeline_flow.serve(name="dataops-deployment", interval=120)
