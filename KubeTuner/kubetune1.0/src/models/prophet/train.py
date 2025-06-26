
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
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def fit_model(growth,df_train,regressor):
    m = Prophet(growth=growth, seasonality_mode="multiplicative", daily_seasonality=15)
    m.add_regressor("location_4", mode="multiplicative")
    m.fit(df_train)
    preds = pd.merge(
        test,
        m.predict(test),
        on="ds",
        how="inner"
    )

        # future = cpu_model.make_future_dataframe(periods=30)# Predict next 30 days
    preds = m.make_future_dataframe(periods=30, freq='min')

      
    # # Use a sample encoded value (e.g., mean or specific controller)
    preds['controller_encoded'] = df['controller_encoded']

    # # Forecast
    forecast = m.predict(future)
    # mape = ((preds["yhat"] - preds["y"]).abs() / preds_linear["y"]).mean()
    # return m, preds, mape

def train_pod_usage_model():

    """
    Train and evaluate prophet for CPU and memory usage prediction,
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
    df['ds'] = pd.to_datetime(df['timestamp'])
    df['y'] = df['cpuUsage']  # Assuming 'cpu_usage' is the target variable
    
    # Encode controllerName
    encoder = LabelEncoder()
    df['controller_encoded'] = encoder.fit_transform(df['controllerName'])

      
    # Step 2: Initialize and train the model
    # cpu_model = Prophet()
    # cpu_model.add_regressor('controller_encoded')
    # cpu_model.fit(df[['ds', 'y', 'controller_encoded']])
    df_train=df[['ds', 'y', 'controller_encoded']]
     
    # # Save the  model
    # cpu_model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_prophet_model_cpuusage.pkl'
    # cpu_model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # joblib.dump(cpu_model, cpu_model_path)
    print(f"CPU Usage model saved to {cpu_model_path}")

    # # Save the  model
    # mem_model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_memusage.pkl'
    # mem_model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # joblib.dump(mem_model, mem_model_path)

    # # evaluate model
    # evaluate.evaluate_model(cpu_model_path,X_test, y_cpu_test)
    # evaluate.evaluate_model(mem_model_path,X_test, y_mem_test)

if __name__ == "__main__":

    print("------------------------------")
    print("Training & Evaluating prophet  Models for CPU Usage...")
    train_pod_usage_model()




