import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

def plot_predict_memory_model(df_export,df):
    # 1. Bar plot: Current vs Predicted vs Final Recommended Memory Request (Sample)
    sample = df_export.sample(min(20, len(df_export)), random_state=42).reset_index(drop=True)
    x = np.arange(len(sample))
    bar_width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - bar_width, sample['memrequest'], width=bar_width, label='Current Request', color='#1976D2')
    plt.bar(x, sample['predicted_memrequest'], width=bar_width, label='Predicted Request', color='#43A047')
    plt.bar(x + bar_width, sample['recommended_memrequest'], width=bar_width, label='Final Recommended', color='#FFA000')
    plt.xticks(x, sample['podname'], rotation=45, ha='right')
    plt.xlabel('Pod')
    plt.ylabel('Memory (MB)')
    plt.title('Current vs Predicted vs Final Recommended Memory Request (Sample)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Scatter plot: Memory Usage vs Memory Request
    plt.figure(figsize=(7, 5))
    plt.scatter(df_export['memusage'], df_export['memrequest'], alpha=0.5, label='Current')
    plt.scatter(df_export['memusage'], df_export['predicted_memrequest'], alpha=0.5, label='Predicted')
    plt.scatter(df_export['memusage'], df_export['recommended_memrequest'], alpha=0.5, label='Recommended Memory Request')
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Memory Request (MB)')
    plt.title('Memory Usage vs Requests')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Cumulative savings plot
    df_export['savings_MB'] = df_export['memrequest'] - df_export['recommended_memrequest']
    df_export['savings_MB'] = df_export['savings_MB'].clip(lower=0)
    df_sorted = df_export.sort_values('savings_MB', ascending=False)
    df_sorted['cumulative_savings'] = df_sorted['savings_MB'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted['cumulative_savings'].values, color='purple')
    plt.xlabel('Pods (sorted by savings)')
    plt.ylabel('Cumulative Savings (MB)')
    plt.title('Cumulative Memory Savings if All Suggestions Applied')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def predict_memory_model():
    # Load the data
    base_dir = Path(__file__).resolve().parents[3]
    file_path = base_dir / 'data' / 'aks_02_data_mb.xlsx'
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Remove rows with zeros to avoid log errors
    df = df[(df['memUsage'] > 0) & (df['memRequest'] > 0) & (df['cpuUsage'] > 0)]

    # Feature engineering
    df['memUtilization'] = df['memUsageMB'] / df['memRequestMB']
    feature_cols = ['memRequestMB', 'memUtilization']  # Add more if available

    y = df['memUsageMB']
    X = df[feature_cols]

      
    # Load the model
    model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_rfr_model_memoryusage.pkl'
    try:
        model = joblib.load(model_file_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_file_path}")
        return


    # Predictions and suggestions
    df['predicted_memrequest'] = model.predict(X)
    
    # After prediction and rounding
    df['predicted_memrequest'] = df['predicted_memrequest'].round(3)
    df['memusage'] = df['memUsageMB'].round(3)

    # Calculate final_rec_mem_req as per your logic
    df['recommended_memrequest'] = np.where(
        df['predicted_memrequest'] < df['memusage'],
        (df['memusage'] + 0.2 * df['memusage']).round(3),
        (df['predicted_memrequest'] + 0.2 * df['predicted_memrequest']).round(3)
    )

    # Rename columns for export
    df_export = df.rename(columns={
        'pod': 'podname',
        'memRequestMB': 'memrequest'
    })

    # Select and order the columns as you want
    export_cols = [
        'podname', 'memrequest', 'memusage', 'predicted_memrequest', 'recommended_memrequest'
    ]

    # Export to Excel
       # Export the Results to Excel
    base_dir_output = Path(__file__).resolve().parents[0]
    output_file_path = base_dir_output / 'output' / 'kubetune_recommended_memrequest.xlsx'
    df_export[export_cols].to_excel(output_file_path,index=False)

    print(f"Predictions saved to {output_file_path}")
    # Plot the results
    plot_predict_memory_model(df_export, df)
 
def plot_predict_cpu_model(df_export, df):
    # 1. Bar plot: Current vs Predicted vs Final Recommended CPU Request (Sample)
    sample = df_export.sample(min(20, len(df_export)), random_state=42).reset_index(drop=True
    )
    x = np.arange(len(sample))
    bar_width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - bar_width, sample['cpurequest'], width=bar_width, label='Current Request', color='#1976D2')
    plt.bar(x, sample['predicted_cpurequest'], width=bar_width, label='Predicted Request', color='#43A047')
    plt.bar(x + bar_width, sample['recomended_cpu_request'], width=bar_width, label='Final Recommended', color='#FFA000')
    plt.xticks(x, sample['podname'], rotation=45, ha='right')
    plt.xlabel('Pod')
    plt.ylabel('CPU')
    plt.title('Current vs Predicted vs Final Recommended CPU Request (Sample)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Scatter plot: CPU Usage vs CPU Requests
    plt.figure(figsize=(7, 5))
    plt.scatter(df_export['cpuusage'], df_export['cpurequest'], alpha=0.5, label='Current')
    plt.scatter(df_export['cpuusage'], df_export['predicted_cpurequest'], alpha=0.5, label='Predicted')
    plt.scatter(df_export['cpuusage'], df_export['recomended_cpu_request'], alpha=0.5, label='Final Recommended')
    plt.xlabel('CPU Usage')
    plt.ylabel('CPU Request')
    plt.title('CPU Usage vs Requests')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Pie chart: Proportion of Pods with Increased vs Decreased Final CPU Request
    increase_count = (df_export['recomended_cpu_request'] > df_export['cpurequest']).sum()
    decrease_count = (df_export['recomended_cpu_request'] < df_export['cpurequest']).sum()
    equal_count = (df_export['recomended_cpu_request'] == df_export['cpurequest']).sum()

    labels = ['Increase', 'Decrease', 'No Change']
    sizes = [increase_count, decrease_count, equal_count]
    colors = ['#FFA000', '#43A047', '#1976D2']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.05, 0.05, 0.05))
    plt.title('Pods: Final Recommended CPU vs Current Request')
    plt.tight_layout()
    plt.show()

    # 4. Cumulative savings plot
    df_export['savings_CPU'] = df_export['cpurequest'] - df_export['recomended_cpu_request']
    df_export['savings_CPU'] = df_export['savings_CPU'].clip(lower=0)
    df_sorted = df_export.sort_values('savings_CPU', ascending=False)
    df_sorted['cumulative_savings'] = df_sorted['savings_CPU'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted['cumulative_savings'].values, color='purple')
    plt.xlabel('Pods (sorted by savings)')
    plt.ylabel('Cumulative Savings (CPU)')
    plt.title('Cumulative CPU Savings if All Suggestions Applied')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def predict_cpu_model():
    # Load the data
    base_dir = Path(__file__).resolve().parents[3]
    file_path = base_dir / 'data' / 'aks_02_data_mb.xlsx'
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Remove rows with zeros to avoid log errors
    df = df[(df['cpuUsage'] > 0) & (df['cpuRequest'] > 0)]

    # Feature engineering (must match train.py)
    df['cpuUtilization'] = df['cpuUsage'] / df['cpuRequest']
    feature_cols = ['cpuRequest', 'cpuUtilization']
    X = df[feature_cols]

    # Load the model
    model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'kubetune_rfr_model_cpuusage.pkl'
    try:
        model = joblib.load(model_file_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_file_path}")
        return

    # Predictions and suggestions
    df['predicted_cpurequest'] = model.predict(X)
    df['predicted_cpurequest'] = df['predicted_cpurequest'].round(3)
    df['cpuusage'] = df['cpuUsage'].round(3)

    # Calculate recommended cpu request
    df['recomended_cpu_request'] = np.where(
        df['predicted_cpurequest'] < df['cpuusage'],
        (df['cpuusage'] + 0.2 * df['cpuusage']).round(3),
        (df['predicted_cpurequest'] + 0.2 * df['predicted_cpurequest']).round(3)
    )

    # Rename columns for export
    df_export = df.rename(columns={
        'pod': 'podname',
        'cpuRequest': 'cpurequest'
    })

    export_cols = [
        'podname', 'cpurequest', 'cpuusage', 'predicted_cpurequest', 'recomended_cpu_request'
    ]
    base_dir_output = Path(__file__).resolve().parents[0]
    output_file_path = base_dir_output / 'output' / 'kubetune_recommended_cpurequest.xlsx'
    df_export[export_cols].to_excel(output_file_path, index=False)

    print(f"Predictions saved to {output_file_path}")
    plot_predict_cpu_model(df_export, df)

if __name__ == "__main__":
    predict_memory_model()
    predict_cpu_model()