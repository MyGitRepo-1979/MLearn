import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from utils import create_features
from sklearn.preprocessing import LabelEncoder

def plot_predict_usage_model(df_all):
    return df_all

def predict_usage_model():
    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    base_dir = Path(__file__).resolve().parents[0]
    file_path = base_dir / 'output' / 'aks01_pod_metrics_processed.json'
    df = pd.read_json(file_path, lines=True)

    # -----------------------------
    # 2. Feature Selection
    # -----------------------------
    features = [
        'cpuRequest', 'cpuLimit', 
        'memRequest', 'memLimit', 
        'avg_cpu_5min', 'avg_mem_5min', 'max_cpu', 'max_mem',
        'hour', 'dayofweek', 'is_weekend',
        'cpu_utilization_ratio', 'mem_utilization_ratio',
        'cat_deployment', 'cat_namespace', 'cat_controllerName', 'cat_controllerKind'
    ]
    X_all = df[features]

    # -----------------------------
    # 3. Load Trained Models
    # -----------------------------
    model_dir = Path(__file__).resolve().parents[0] / 'output'
    cpu_model_path = model_dir / 'kubetune_prophet_model_cpuusage.pkl'
    
    try:
        cpu_model = joblib.load(cpu_model_path)
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return

    df['ds'] = pd.to_datetime(df['timestamp'])
    df['y'] = df['cpuUsage']  # Assuming 'cpu_usage' is the target variable

    
    # Encode controllerName
    encoder = LabelEncoder()
    df['controller_encoded'] = encoder.fit_transform(df['controllerName'])


    # -----------------------------
    # 4. Make Predictions
    # -----------------------------
    
    # # Step 3: Create future dataframe
    # future = cpu_model.make_future_dataframe(periods=30)# Predict next 30 days
    future = cpu_model.make_future_dataframe(periods=30, freq='min')

      
    # # Use a sample encoded value (e.g., mean or specific controller)
    future['controller_encoded'] = df['controller_encoded']

    # # Forecast
    forecast = cpu_model.predict(future)
    # print(forecast[['ds', 'controller_encoded','yhat']].head(1000))

   

    # -----------------------------
    # 8. Export to Excel
    # -----------------------------
    output_path = model_dir / 'kubetune_recommended_usage.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        # Sheet1 - full data
        forecast.to_excel(writer, sheet_name='Full Recommendations', index=False)

    print(f"Predictions saved to {output_path}")

    
if __name__ == "__main__":
    predict_usage_model()
