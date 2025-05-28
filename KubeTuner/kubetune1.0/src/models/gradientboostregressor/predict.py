import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler

def predict_model():
    # Load the data
    base_dir = Path(__file__).resolve().parents[3]
    file_path = base_dir / 'data' / 'aks_02_data_mb.xlsx'
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Remove rows with zeros to avoid log errors
    df = df[(df['memUsage'] > 0) & (df['memRequest'] > 0) & (df['cpuUsage'] > 0)]

    # -----------------------------
    # 2. Feature Engineering & New Target Definition
    # -----------------------------
    # Create a utilization feature (how much of the requested memory is actually used)
    df['memUtilization'] = df['memUsage'] / df['memRequest']

    # Define a recommended request as 20% above observed usage (in MB)
    buffer_factor = 1.2
    df['recommendedRequest'] = df['memUsage'] * buffer_factor

    # Define the optimized target as the lower of the current memRequest and the recommended value
    df['opt_memRequest'] = df[['memRequest', 'recommendedRequest']].min(axis=1)

    # Convert the optimized target from MB to bytes (since your output uses bytes)
    df['opt_memRequest_bytes'] = df['opt_memRequest'] * 1048576

    # -----------------------------
    # 3. Prepare Features and Target for Modeling
    # -----------------------------
    # Use log transformation to stabilize variance for memUsage (in MB)
    df['log_memUsage'] = np.log1p(df['memUsage'])

    # Also scale cpuUsage and the new memUtilization feature:
    scaler = StandardScaler()
    df['scaled_cpuUsage'] = scaler.fit_transform(df[['cpuUsage']])
    df['scaled_memUtilization'] = scaler.fit_transform(df[['memUtilization']])

    # Our new target is the optimized memRequest (in bytes); take its log.
    df['log_opt_memRequest'] = np.log1p(df['opt_memRequest_bytes'])

    # Define feature set – now with three features:
    # • log_memUsage
    # • scaled_cpuUsage
    # • scaled_memUtilization
    X = df[['log_memUsage', 'scaled_cpuUsage', 'scaled_memUtilization']]
    y = df['log_opt_memRequest']


   
    # Load the model
    model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'gradient_boosting_model.pkl'
    model = joblib.load(model_file_path)

    # Predict on the entire dataset (log scale), then reverse the log transform.
    df['predicted_log_opt_memRequest'] = model.predict(X)
    df['predicted_opt_memRequest_bytes'] = np.expm1(df['predicted_log_opt_memRequest'])

    # Calculate the current memRequest in bytes (for comparison)
    df['memRequest_bytes'] = df['memRequest'] * 1048576

    # Compute the reduction percent offered by the model's prediction
    df['reduction_percent'] = ((df['memRequest_bytes'] - df['predicted_opt_memRequest_bytes']) /
                               df['memRequest_bytes'] * 100).round(2)

    # Create a suggestion string (convert predicted value back to MB)
    df['suggestion'] = np.where(
        df['reduction_percent'] > 0,
        'Reduce request to ' + (df['predicted_opt_memRequest_bytes'] / 1048576).round(2).astype(str) + ' Bytes',
        'No change needed'
    )

    # Export the Results to Excel
    export_cols = ['memUsage', 'memRequest', 'memRequest_bytes', 'predicted_opt_memRequest_bytes', 'reduction_percent', 'suggestion']
    base_dir_output = Path(__file__).resolve().parents[0]
    output_file_path = base_dir_output / 'output' / 'memory_request_predictions_optimized.xlsx'
    df[export_cols].to_excel(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")

    # Plot Actual vs. Predicted Optimized Memory Request (bytes)
    plt.figure(figsize=(10,6))
    plt.scatter(df['memRequest_bytes'], df['predicted_opt_memRequest_bytes'], alpha=0.5)
    plt.plot([df['memRequest_bytes'].min(), df['memRequest_bytes'].max()],
             [df['memRequest_bytes'].min(), df['memRequest_bytes'].max()], 'r--')
    plt.xlabel('Current Memory Request (bytes)')
    plt.ylabel('Predicted Optimized Memory Request (bytes)')
    plt.title('Actual vs. Predicted Optimized Memory Request')
    plt.grid(True)
    # plt.show()
    plt.savefig(base_dir_output / 'output' / 'actual_vs_predicted_memory_request.png')

if __name__ == "__main__":
    predict_model()