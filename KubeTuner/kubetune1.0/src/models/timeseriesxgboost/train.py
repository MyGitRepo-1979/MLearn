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
    # Path to your Excel file
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'AEP_Energy_consumption.xlsx'

    df = pd.read_excel(file_path, sheet_name="AEP_hourly",index_col=[0], parse_dates=[0])

    split_date = '01-Jan-2015'
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()

    X_train, y_train = create_features(df_train, label='AEP_MW')
    X_test, y_test = create_features(df_test, label='AEP_MW')

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



