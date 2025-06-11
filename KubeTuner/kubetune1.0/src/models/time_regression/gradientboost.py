import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_aks_data():
    """
    Loads the AKS02-Data.xlsx file from the correct data directory,
    converts memory columns to MB if needed.
    Returns a pandas DataFrame.
    """
    base_dir = Path(__file__).resolve().parents[3]
    file_path = base_dir / 'data' / 'AKS02-Data.xlsx'
    print("Looking for data file at:", file_path)  # Debugging line
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Convert memory columns to MB if they look like bytes
    for col in ['memUsage', 'memRequest', 'memLimit']:
        if col in df.columns and df[col].max() > 1024 * 1024:
            df[col + 'MB'] = df[col] / (1024 * 1024)
        elif col in df.columns:
            df[col + 'MB'] = df[col]  # Already in MB or unknown, just copy

    # If you want to convert CPU units, add similar logic here

    return df

def train_memory_model():
    print("Training memory usage model...")
    df = load_aks_data()
    df['memUtilization'] = np.where(df['memRequestMB'] != 0, df['memUsageMB'] / df['memRequestMB'], np.nan)
    feature_cols = ['memRequestMB', 'memUtilization']
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['memUsageMB'].fillna(df['memUsageMB'].median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    model_path = Path(__file__).resolve().parent / 'output' / 'kubetune_gb_model_memoryusage.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    y_pred = model.predict(X_test)
    print("Best Params:", grid_search.best_params_)
    print(f"Memory Usage Model R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Memory Usage Model MAE: {mean_absolute_error(y_test, y_pred):.4f}")

def train_cpu_model():
    print("Training CPU usage model...")
    df = load_aks_data()
    df['cpuUtilization'] = np.where(df['cpuRequest'] != 0, df['cpuUsage'] / df['cpuRequest'], np.nan)
    feature_cols = ['cpuRequest', 'cpuUtilization']
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['cpuUsage'].fillna(df['cpuUsage'].median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    model_path = Path(__file__).resolve().parent / 'output' / 'kubetune_gb_model_cpuusage.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    y_pred = model.predict(X_test)
    print("Best Params:", grid_search.best_params_)
    print(f"CPU Usage Model R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"CPU Usage Model MAE: {mean_absolute_error(y_test, y_pred):.4f}")

def predict_and_export_all():
    df = load_aks_data()

    # Memory predictions
    df['memUtilization'] = np.where(df['memRequestMB'] != 0, df['memUsageMB'] / df['memRequestMB'], np.nan)
    mem_X = df[['memRequestMB', 'memUtilization']].fillna(df[['memRequestMB', 'memUtilization']].median())
    mem_model_file = Path(__file__).resolve().parent / 'output' / 'kubetune_gb_model_memoryusage.pkl'
    mem_model = joblib.load(mem_model_file)
    df['predicted_memrequest'] = mem_model.predict(mem_X)
    df['recommended_memrequest'] = np.where(
        df['predicted_memrequest'] < df['memUsageMB'],
        df['memUsageMB'] + 0.2 * df['memUsageMB'],
        df['predicted_memrequest'] + 0.2 * df['predicted_memrequest']
    )

    # CPU predictions
    df['cpuUtilization'] = np.where(df['cpuRequest'] != 0, df['cpuUsage'] / df['cpuRequest'], np.nan)
    cpu_X = df[['cpuRequest', 'cpuUtilization']].fillna(df[['cpuRequest', 'cpuUtilization']].median())
    cpu_model_file = Path(__file__).resolve().parent / 'output' / 'kubetune_gb_model_cpuusage.pkl'
    cpu_model = joblib.load(cpu_model_file)
    df['predicted_cpurequest'] = cpu_model.predict(cpu_X)
    df['recommended_cpurequest'] = np.where(
        df['predicted_cpurequest'] < df['cpuUsage'],
        df['cpuUsage'] + 0.2 * df['cpuUsage'],
        df['predicted_cpurequest'] + 0.2 * df['predicted_cpurequest']
    )

    # --- Only export the requested columns ---
    export_cols = [
        'key', 'pod', 'node', 'container', 'controllerKind', 'controllerName', 'deployment',
        'collectionTimestamp', 'cpuUsage', 'memUsage',
        'cpuRequest', 'memRequest', 'cpuLimit', 'memLimit',
        'memUsageMB', 'memRequestMB', 'memLimitMB', 'predicted_memrequest', 'recommended_memrequest'
        , 'predicted_cpurequest', 'recommended_cpurequest'
    ]
    export_cols = [col for col in export_cols if col in df.columns]

    output_file_path = Path(__file__).resolve().parent / 'output' / 'kubetune_combined_predictions.xlsx'
    df[export_cols].to_excel(output_file_path, index=False)
    print(f"Combined predictions saved to {output_file_path}")

if __name__ == "__main__":
    print("------------------------------")
    print("Training & Evaluating Gradient Boosting Models for Memory Usage...")
    train_memory_model()
    print("------------------------------")
    print("Training & Evaluating Gradient Boosting Models for CPU Usage...")
    train_cpu_model()
    print("------------------------------")
    print("Predicting and exporting recommendations for Memory and CPU (combined)...")
    predict_and_export_all()
    print("------------------------------")