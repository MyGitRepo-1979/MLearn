import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from utils import create_features

def plot_predict_usage_model(df_all):
    _ = df_all[['memUsage','memusage_Prediction']].plot(figsize=(15, 5))
    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = df_all[['memusage_Prediction','memUsage']].plot(ax=ax,
                                                style=['-','.'])
    ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
    ax.set_ylim(0, 60000)
    plot = plt.suptitle('January 2015 Forecast vs Actuals')
    plt.show()

        # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = df_all[['memusage_Prediction','memUsage']].plot(ax=ax,
                                                style=['-','.'])
    ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
    ax.set_ylim(0, 60000)
    plot = plt.suptitle('First Week of January Forecast vs Actuals')
    plt.show()

def predict_usage_model():

    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    # Path to your Json file
    base_dir_input = Path(__file__).resolve().parents[3]
    #file_path = base_dir_input / 'data' / 'aks01_pod_metrics.json'
    file_path = base_dir_input / 'data' / 'pod_metrics.json'
    df = pd.read_json(file_path)
    # Remove rows with zeros to avoid log errors
    df = df[(df['memRequest'] > 0) & (df['memLimit'] > 0) & (df['memUsage'] > 0)]

    df['memUsage'] = ((df['memUsage'].astype(float))/(1014 * 1024)).round(2)  # Ensure memUsage is float
    df['memRequest'] = ((df['memRequest'].astype(float))/(1014 * 1024)).round(2)  # Ensure memUsage is float
    df['memLimit'] = ((df['memLimit'].astype(float))/(1014 * 1024)).round(2)  # Ensure memUsage is float

    #df_final = df[['collectionTimestamp', 'controllerName','memUsage', 'memLimit','memRequest']].copy()
    df_final = df[['collectionTimestamp', 'controllerName','memRequest','memLimit','memUsage']].copy()
    df_final = df_final.set_index('collectionTimestamp')
    df_final.index = pd.to_datetime(df_final.index)  # Convert timestamp to datetime

    # Split by row position, not by index value
    split_index = -500
    df_train = df_final.iloc[:split_index].copy()
    df_test = df_final.iloc[split_index:].copy()

    X_test, y_test = create_features(df_test, label='memUsage')


    # Load the model
    model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_xgb_model_usage.pkl'
    try:
        model = joblib.load(model_file_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_file_path}")
        return

    df_test['memUsage_Prediction'] = (model.predict(X_test)).round(2)
    df_all = pd.concat([df_test], sort=False)

    # Remove timezone info from index if present
    if df_all.index.tz is not None:
        df_all.index = df_all.index.tz_localize(None)
    # Remove timezone info from datetime columns if any
    for col in df_all.select_dtypes(include=['datetimetz']).columns:
        df_all[col] = df_all[col].dt.tz_localize(None)
  
    # Export to Excel
    # Export the Results to Excel
    base_dir_output = Path(__file__).resolve().parents[0]
    output_file_path = base_dir_output / 'output' / 'kubetune_recommended_usage.xlsx'
    df_all.to_excel(output_file_path,index=False)
    print(f"Predictions saved to {output_file_path}") 

    # print(f"Predictions saved to {output_file_path}")
    # Plot the results
    #plot_predict_usage_model(df_all)
     
if __name__ == "__main__":
    predict_usage_model()
   