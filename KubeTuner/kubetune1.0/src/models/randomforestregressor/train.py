import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib
import evaluate  # This refers to evaluate.py file (see below)

def train_memory_model():
    print("Training memory usage model...")

    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'aks_02_data_mb.xlsx'

    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Filter out zero values
    df = df[(df['memUsageMB'] > 0) & (df['memRequestMB'] > 0)]
    df['memUtilization'] = df['memUsageMB'] / df['memRequestMB']

    feature_cols = ['memRequestMB', 'memUtilization']
    X = df[feature_cols]
    y = df['memUsageMB']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                               scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_rfr_model_memoryusage.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    evaluate.evaluate_model(model_path, X_test, y_test)


def train_cpu_model():
    print("Training CPU usage model...")
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'aks_02_data_mb.xlsx'

    df = pd.read_excel(file_path, sheet_name="Sheet1")

    df = df[(df['cpuUsage'] > 0) & (df['cpuRequest'] > 0)]
    df['cpuUtilization'] = df['cpuUsage'] / df['cpuRequest']

    feature_cols = ['cpuRequest', 'cpuUtilization']
    X = df[feature_cols]
    y = df['cpuUsage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                               scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_rfr_model_cpuusage.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    evaluate.evaluate_model(model_path, X_test, y_test)


if __name__ == "__main__":
    print("------------------------------")
    print("Training & Evaluating Random Forest Regressor Models for Memory Usage...")
    train_memory_model()

    print("------------------------------")
    print("Training & Evaluating Random Forest Regressor Models for CPU Usage...")
    train_cpu_model()
    print("------------------------------")
