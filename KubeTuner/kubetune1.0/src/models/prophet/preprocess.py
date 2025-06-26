
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib   
from pathlib import Path

def save_processed_data(df):
    print("Saving preprocessed data to output/aks01_pod_metrics_processed.json")
    # Ensure the output directory exists and save the preprocessed data
    file_path = Path(__file__).resolve().parents[0] / 'output' / 'aks01_pod_metrics_processed.json'
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
     # Save the DataFrame to a JSON file
    df.to_json(file_path, orient='records', lines=True)



def prepare_preprocess_data():
    
    # Preprocess the data by loading it from a JSON file and preparing it for training.
    
    # ========================
    # Step 1: Load JSON File
    # ========================
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'aks01_pod_metrics.json'
    
    # 1. Load the JSON file &  Remove rows with all zero values in specified columns
    df = pd.read_json(file_path)
    df = df[~(
        (df['cpuRequest'] == 0) &
        (df['cpuLimit'] == 0) &
        (df['cpuUsage'] == 0) &
        (df['memRequest'] == 0) &
        (df['memLimit'] == 0) &
        (df['memUsage'] == 0)
    )]
    df['memUsage'] = ((df['memUsage'].astype(float))/(1024 * 1024)).round(2)  # Ensure memUsage is float
    df['memRequest'] = ((df['memRequest'].astype(float))/(1024 * 1024)).round(2)  # Ensure memUsage is float
    df['memLimit'] = ((df['memLimit'].astype(float))/(1024 * 1024)).round(2)
    df['cpuRequest'] = df['cpuRequest'].astype(float).round(4)  # Ensure cpuRequest is float
    df['cpuLimit'] = df['cpuLimit'].astype(float).round(4)
    df['cpuUsage'] = df['cpuUsage'].astype(float).round(4)  # Ensure cpuUsage is float


    # Step 2: Timestamp Features
    # ========================
    df['collectionTimestamp'] = pd.to_datetime(df['collectionTimestamp'])
    df['timestamp'] = df['collectionTimestamp']
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5

    # ========================
    # Step 3: Create Utilization Features
    # ========================
     # Handle division by zero in the ratio columns only
    df['cpu_utilization_ratio'] = np.where(
        df['cpuLimit'] == 0, 0, df['cpuUsage'] / df['cpuLimit']).round(4)
    
    df['mem_utilization_ratio'] = np.where(
        df['memLimit'] == 0, 0, df['memUsage'] / df['memLimit']).round(2)
    

    # ========================
    # Step 4: Encode Categorical Features
    # ========================
    df['cat_deployment'] = LabelEncoder().fit_transform(df['deployment'])
    df['cat_namespace'] = LabelEncoder().fit_transform(df['namespace'])
    df['cat_controllerName'] = LabelEncoder().fit_transform(df['controllerName'])
    df['cat_controllerKind'] = LabelEncoder().fit_transform(df['controllerKind'])

    # ========================
    # Step 4.1: Create Rolling Features Before Training
    # ========================
    df = df.sort_values(by='collectionTimestamp')  # Ensure proper time order

    df['avg_cpu_5min'] = (df['cpuUsage'].rolling(window=5, min_periods=1).mean()).round(4)
    df['avg_mem_5min'] = (df['memUsage'].rolling(window=5, min_periods=1).mean()).round(2)
    df['max_cpu'] = (df['cpuUsage'].rolling(window=5, min_periods=1).max()).round(4)
    df['max_mem'] = (df['memUsage'].rolling(window=5, min_periods=1).max()).round(2)
   
    # ========================
    # Step 5: Select Features & Targets
    # ========================
    features = [
        'timestamp', 'controllerName','cpuRequest', 'cpuLimit', 'cpuUsage',
        'memRequest', 'memLimit', 'memUsage',
        'avg_cpu_5min', 'avg_mem_5min', 'max_cpu', 'max_mem',
        'hour', 'dayofweek', 'is_weekend',
        'cpu_utilization_ratio', 'mem_utilization_ratio',
        'cat_deployment', 'cat_namespace', 'cat_controllerName', 'cat_controllerKind'
    ]
    df_processed_data = df[features].copy()
    return df_processed_data


if __name__ == "__main__":

    print("------------------------------")
    print("Preprocessing and create feature from raw data.")
    df=prepare_preprocess_data()
    # Save the preprocessed data
    save_processed_data(df)

    print("------------------------------")
    print("Preprocessing completed and data saved.")
    print("------------------------------")


