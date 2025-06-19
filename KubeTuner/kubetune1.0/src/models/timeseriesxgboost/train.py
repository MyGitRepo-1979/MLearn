
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import evaluate as evaluate
import joblib
from utils import create_features


def train_pod_usage_model():

    """
    Train and evaluate XGBoost regression models for CPU and memory usage prediction,
    and RandomForest classifiers for under-provisioning detection.

    Loads preprocessed pod metrics data, prepares features and targets, splits the data
    chronologically, trains models, saves them to disk, and evaluates their performance.

    Returns:
        None
    """

    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    # Path to your Json file
    base_dir_input = Path(__file__).resolve().parents[0]
    file_path = base_dir_input / 'output' / 'aks01_pod_metrics_processed.json'
    # Load the JSON Lines file
    df = pd.read_json(file_path, lines=True)

    # Flags for under-provisioning (usage exceeds request by margin)
    cpu_margin = 1.2
    mem_margin = 1.2
    df['cpu_under'] = (df['cpuUsage'] > df['cpuRequest'] * cpu_margin).astype(int)
    df['mem_under'] = (df['memUsage'] > df['memRequest'] * mem_margin).astype(int)

    # -------------------- Prepare Features --------------------
    # Select features for model training
    features = [
        'cpuRequest', 'cpuLimit', 
        'memRequest', 'memLimit', 
        'avg_cpu_5min', 'avg_mem_5min', 'max_cpu', 'max_mem',
        'hour', 'dayofweek', 'is_weekend',
        'cpu_utilization_ratio', 'mem_utilization_ratio',
        'cat_deployment', 'cat_namespace', 'cat_controllerName', 'cat_controllerKind'
    ]
    X = df[features]
    y_cpu = df['cpuUsage']  # Target: actual CPU usage
    y_mem = df['memUsage']  # Target: actual memory usage in MB
    y_cpu_class = df['cpu_under']           # Target: under-provisioning flag for CPU
    y_mem_class = df['mem_under']           # Target: under-provisioning flag for memory

    # -------------------- Train/Test Split (chronological) --------------------
    # Split data chronologically to avoid data leakage (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_cpu_train, y_cpu_test = y_cpu.iloc[:train_size], y_cpu.iloc[train_size:]
    y_mem_train, y_mem_test = y_mem.iloc[:train_size], y_mem.iloc[train_size:]
    y_cpu_class_train, y_cpu_class_test = y_cpu_class.iloc[:train_size], y_cpu_class.iloc[train_size:]
    y_mem_class_train, y_mem_class_test = y_mem_class.iloc[:train_size], y_mem_class.iloc[train_size:]

    # -------------------- Modeling --------------------
    # Define hyperparameters for XGBoost regression models
    param_grid = {
        'n_estimators': [100],
        'max_depth': [4],
        'learning_rate': [0.1]
    }
    # Train XGBoost regressor for CPU usage prediction
    cpu_model = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='r2', cv=2, n_jobs=-1).fit(X_train, y_cpu_train).best_estimator_
    # Train XGBoost regressor for memory usage prediction
    mem_model = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='r2', cv=2, n_jobs=-1).fit(X_train, y_mem_train).best_estimator_

    # Train RandomForest classifiers for under-provisioning detection
    clf_cpu = RandomForestClassifier(random_state=42).fit(X_train, y_cpu_class_train)
    clf_mem = RandomForestClassifier(random_state=42).fit(X_train, y_mem_class_train)
     
    # Save the  model
    cpu_model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_cpuusage.pkl'
    cpu_model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    joblib.dump(cpu_model, cpu_model_path)

    # Save the  model
    mem_model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_memusage.pkl'
    mem_model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    joblib.dump(mem_model, mem_model_path)

    # evaluate model
    evaluate.evaluate_model(cpu_model_path,X_test, y_cpu_test)
    evaluate.evaluate_model(mem_model_path,X_test, y_mem_test)

if __name__ == "__main__":

    print("------------------------------")
    print("Training & Evaluating xgboost  Models for Usage...")
    train_pod_usage_model()

    print("Training & Evaluating xgboost  Models for Usage... Done")
    print("------------------------------")



