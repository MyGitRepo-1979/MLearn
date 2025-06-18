import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from utils import create_features

def plot_predict_usage_model(df_all):
    return df_all
    

def predict_usage_model():

    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    # Path to your Json file
    base_dir_input = Path(__file__).resolve().parents[0]
    file_path = base_dir_input / 'output' / 'aks01_pod_metrics_processed.json'
    df = pd.read_json(file_path, lines=True)
    # -------------------- Predict for All Pods --------------------
    features = [
        'cpuRequest', 'cpuLimit', 
        'memRequest', 'memLimit', 'memUsage',
        'avg_cpu_5min', 'avg_mem_5min', 'max_cpu', 'max_mem',
        'hour', 'dayofweek', 'is_weekend',
        'cpu_utilization_ratio', 'mem_utilization_ratio',
        'cat_deployment', 'cat_namespace', 'cat_controllerName', 'cat_controllerKind'
    ]
    X_all = df[features]

        # Load the model
    cpu_model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_cpuusage.pkl'
    try:
        cpu_model = joblib.load(cpu_model_file_path)
    except FileNotFoundError:
        print(f"Model file not found: {cpu_model_file_path}")
        return
    
    mem_model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_memusage.pkl'
    try:
        mem_model = joblib.load(mem_model_file_path)
    except FileNotFoundError:
        print(f"Model file not found: {mem_model_file_path}")
        return
    
    # Predict CPU and Memory usage (in cores and MB)
    df['cpuRequest_predicted'] = (cpu_model.predict(X_all)).round(4)  # Ensure cpuRequest is float and rounded to 4 decimal places
    df['memRequest_predicted'] = (mem_model.predict(X_all)).round(2)  # Ensure memRequest is float and rounded to 2 decimal places


    # CPU recommendation logic:
    # If predicted request < actual usage, recommend usage + 20%; else, predicted + 20%
    df['recommended_cpuRequest'] = (np.where(
        df['cpuRequest_predicted'] < df['cpuUsage'],
        df['cpuUsage'] + 0.2 * df['cpuUsage'],
        df['cpuRequest_predicted'] + 0.2 * df['cpuRequest_predicted']
    )).round(4)  # Ensure cpuRequest is float and rounded to 4 decimal places

    # Memory recommendation logic:
    # If predicted request < actual usage, recommend usage + 20%; else, predicted + 20%
    df['recommended_memRequest'] = (np.where(
        df['memRequest_predicted'] < df['memUsage'],
        df['memUsage'] + 0.2 * df['memUsage'],
        df['memRequest_predicted'] + 0.2 * df['memRequest_predicted']
    )).round(2)  # Ensure memRequest is float and rounded to 2 decimal places
 
    df_max = df.loc[df.groupby('controllerName')[
        ['cpuRequest_predicted', 'memRequest_predicted', 'recommended_cpuRequest', 'recommended_memRequest']
    ].idxmax().values.flatten()]
    # Export to Excel
    # Export the Results to Excel
    base_dir_output = Path(__file__).resolve().parents[0]
    output_file_path = base_dir_output / 'output' / 'kubetune_recommended_usage.xlsx'
    df_max.to_excel(output_file_path,index=False)
    print(f"Predictions saved to {output_file_path}") 
     
if __name__ == "__main__":
    predict_usage_model()
   