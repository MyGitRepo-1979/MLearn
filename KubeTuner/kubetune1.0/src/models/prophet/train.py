
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import evaluate as evaluate
from prophet import Prophet
import matplotlib.pyplot as plt


def fit_model(growth,df_train,controllerName):

    # m = Prophet(growth=growth, seasonality_mode="multiplicative", daily_seasonality=15)
    m = Prophet()
    m.fit(df_train)

    df = m.make_future_dataframe(periods=30, freq='D') # Predict next 30 days
    # Forecast
    df_prediction = m.predict(df)
    df_prediction['controllerName'] = controllerName

    return m, df_prediction[['ds', 'controllerName','yhat']]

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
    
     
    # Step 2: Initialize and train the model
    df_train=df[['ds', 'y', 'controllerName']]
    
    df_all = pd.DataFrame(columns=['ds', 'y', 'controllerName'])
    unique_controllers = df_train['controllerName'].unique()

    for controller in unique_controllers:
        print(f"Processing controller: {controller}")
        df_controller_train = df_train[df_train['controllerName'] == controller].dropna()
     
        if len(df_controller_train) < 2:
            print(f"Skipping {controller} due to insufficient data.")
            continue
        cpu_model, df_prediction = fit_model(growth='linear',df_train=df_controller_train,controllerName=controller)
        if df_all is None:
            df_all = df_prediction
        else:
            df_all = pd.concat([df_all, df_prediction], ignore_index=True)

    df_all = df_all.drop(['y'], axis=1)
    # print(df_all.head(30))
    model_dir = Path(__file__).resolve().parents[0] / 'output'
        # -----------------------------
    # 8. Export to Excel
    # -----------------------------
    output_path = model_dir / 'kubetune_recommended_usage.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        df_all.to_excel(writer, sheet_name='Recommendations', index=False)

    print(f"Predictions saved to {output_path}")

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




