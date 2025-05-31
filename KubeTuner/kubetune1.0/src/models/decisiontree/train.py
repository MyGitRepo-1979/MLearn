import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import evaluate as evaluate
import joblib


def train_memory_model():
    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    # Path to your Excel file
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'aks_02_data_mb.xlsx'

    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Remove rows with zeros to avoid log errors
    df = df[(df['memUsageMB'] > 0) & (df['memRequestMB'] > 0)]
    # Feature engineering
    df['memUtilization'] = df['memUsageMB'] / df['memRequestMB']
    feature_cols = ['memRequestMB', 'memUtilization']  # Add more if available

    y = df['memUsageMB']
    X = df[feature_cols]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Save the  model
    model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_dt_model_memoryusage.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    joblib.dump(model, model_path)

    # evaluate model
    evaluate.evaluate_model(model_path,X_test, y_test)

def train_cpu_model():
    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    # Path to your Excel file
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'aks_02_data_mb.xlsx'

    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Remove rows with zeros to avoid log errors
    df = df[(df['cpuUsage'] > 0) & (df['cpuRequest'] > 0)]

    # Prepare features and target
    y = df['cpuUsage']
    X = df[['cpuRequest']]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Save the  model
    model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_dt_model_cpuusage.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    joblib.dump(model, model_path)

    # evaluate model
    evaluate.evaluate_model(model_path,X_test, y_test)


if __name__ == "__main__":

    print("------------------------------")
    print("Training & Evaluating Decision Tree Models for Memory Usage...")
    train_memory_model()

    print("------------------------------")
    print("Training & Evaluating Decision Tree Models for CPU Usage...")
    train_cpu_model()
    print("------------------------------")
    


