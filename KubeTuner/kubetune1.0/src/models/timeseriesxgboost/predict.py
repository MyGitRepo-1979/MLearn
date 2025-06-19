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
    
    cpu_model_path = model_dir / 'kubetune_xgb_model_cpuusage.pkl'
    mem_model_path = model_dir / 'kubetune_xgb_model_memusage.pkl'

    try:
        cpu_model = joblib.load(cpu_model_path)
        mem_model = joblib.load(mem_model_path)
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return

    # -----------------------------
    # 4. Make Predictions
    # -----------------------------
    df['cpuRequest_predicted'] = cpu_model.predict(X_all).round(4)
    df['memRequest_predicted'] = mem_model.predict(X_all).round(2)

    # -----------------------------
    # 5. Generate Recommendations
    # -----------------------------
    df['recommended_cpuRequest'] = np.where(
        df['cpuRequest_predicted'] < df['cpuUsage'],
        df['cpuUsage'] * 1.2,
        df['cpuRequest_predicted'] * 1.2
    ).round(4)

    df['recommended_memRequest'] = np.where(
        df['memRequest_predicted'] < df['memUsage'],
        df['memUsage'] * 1.2,
        df['memRequest_predicted'] * 1.2
    ).round(2)

    # -----------------------------
    # 6. Add Max Recommendations per Controller
    # -----------------------------
    max_recommended = df.groupby('controllerName').agg({
        'recommended_cpuRequest': 'max',
        'recommended_memRequest': 'max'
    }).rename(columns={
        'recommended_cpuRequest': 'max_recommended_cpuRequest',
        'recommended_memRequest': 'max_recommended_memRequest'
    }).reset_index()

    df = df.merge(max_recommended, on='controllerName', how='left')

    df['recommended_request'] = (
        'cpu: ' + df['max_recommended_cpuRequest'].astype(str) +
        ', mem: ' + df['max_recommended_memRequest'].astype(str)
    )

    # -----------------------------
    # 7. Format Output
    # -----------------------------
    output_columns = [
        'controllerName',
        'cpuRequest', 'cpuLimit', 'cpuUsage',
        'memRequest', 'memLimit', 'memUsage',
        'cpuRequest_predicted', 'memRequest_predicted',
        'recommended_cpuRequest', 'recommended_memRequest',
        'recommended_request'
    ]
    df_final = df[output_columns].sort_values(by='controllerName')

    # -----------------------------
    # 8. Export to Excel (Two Sheets)
    # -----------------------------
    output_path = model_dir / 'kubetune_recommended_usage.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        # Sheet1 - full data
        df_final.to_excel(writer, sheet_name='Full Recommendations', index=False)

        # Sheet2 - one row per controller with summary recommendation
        df_summary = df_final.drop_duplicates('controllerName')[['controllerName', 'recommended_request']]
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Predictions saved to {output_path}")

    # -----------------------------
    # 9. Create Sheet2 - Aggregated Data
    # -----------------------------
    # Group by controllerName and get max for relevant columns
    sheet2_df = df.groupby('controllerName', as_index=False).agg({
        'cpuRequest': 'max',
        'cpuLimit': 'max',
        'cpuUsage': 'max',
        'memRequest': 'max',
        'memLimit': 'max',
        'memUsage': 'max',
        'cpuRequest_predicted': 'max',
        'memRequest_predicted': 'max',
        'recommended_cpuRequest': 'max',
        'recommended_memRequest': 'max'
    })

    # Create the recommended_request column as a string
    sheet2_df['recommended_request'] = (
        'cpu: ' + sheet2_df['recommended_cpuRequest'].round(4).astype(str) +
        ', mem: ' + sheet2_df['recommended_memRequest'].round(2).astype(str)
    )

    # Select and order columns for output
    output_columns = [
        'controllerName',
        'cpuRequest',
        'cpuLimit',
        'cpuUsage',
        'memRequest',
        'memLimit',
        'memUsage',
        'cpuRequest_predicted',
        'memRequest_predicted',
        'recommended_cpuRequest',
        'recommended_memRequest',
        'recommended_request'
    ]
    sheet2_df = sheet2_df[output_columns]

    # Write Sheet2 to the same Excel file
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        sheet2_df.to_excel(writer, sheet_name='Sheet2', index=False)

    # Sheet2 - one row per controller with summary recommendation
        df_summary = df_final.drop_duplicates('controllerName')[['controllerName', 'recommended_request']]
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        

    # -----------------------------
    # 10. Export Summary to JSON
    # -----------------------------
    # Select only the columns you want in the output
    json_columns = [
        'controllerName',
        'cpuRequest',
        'cpuLimit',
        'cpuUsage',
        'memRequest',
        'memLimit',
        'memUsage',
        'cpuRequest_predicted',
        'memRequest_predicted',
        'recommended_cpuRequest',
        'recommended_memRequest'
    ]

    # Make sure the DataFrame has only those columns
    json_df = sheet2_df[json_columns]

    # Write to JSON file (records format, one object per line)
    json_output_path = output_path.parent / 'controller_summary.json'
    json_df.to_json(json_output_path, orient='records', lines=True)

    print(f"Summary JSON saved to {json_output_path}")
    
if __name__ == "__main__":
    predict_usage_model()
