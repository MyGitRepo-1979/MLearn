import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import evaluate as evaluate
import joblib
from utils import create_features


def train_memory_model():

    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    # Path to your Json file
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'aks01_pod_metrics.json'
    df = pd.read_json(file_path)

   
    # Remove rows with zeros to avoid log errors
    df = df[(df['memRequest'] > 0)]
   
    
    df['memUsage'] = ((df['memUsage'].astype(float))/(1014 * 1024)).round(2)  # Ensure memUsage is float
    df['memRequest'] = ((df['memRequest'].astype(float))/(1014 * 1024)).round(2)  # Ensure memUsage is float

    
    df_final = df[['collectionTimestamp', 'controllerName','memUsage', 'memRequest']].copy()
    df_final = df_final.set_index('collectionTimestamp')
    df_final.index = pd.to_datetime(df_final.index)  # Convert timestamp to datetime


    # Split by row position, not by index value
    split_index = -500
    df_train = df_final.iloc[:split_index].copy()
    df_test = df_final.iloc[split_index:].copy()

    X_train, y_train = create_features(df_train, label='memUsage')
    X_test, y_test = create_features(df_test, label='memUsage')

    # Before training, check and clean your data to remove or replace inf and NaN values:
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # -----------------------------
    # 2. Train the XGBoost Model
    model = xgb.XGBRegressor(n_estimators=1000)
    model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False) # Change verbose to True if you want to see it train

    # Save the  model
    model_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_usage.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    joblib.dump(model, model_path)

    # evaluate model
    evaluate.evaluate_model(model_path,X_test, y_test)

if __name__ == "__main__":

    print("------------------------------")
    print("Training & Evaluating xgboost  Models for Usage...")
    train_memory_model()

    # print("------------------------------")
    # print("Training & Evaluating Decision Tree Models for CPU Usage...")
    # train_cpu_model()
    # print("------------------------------")



