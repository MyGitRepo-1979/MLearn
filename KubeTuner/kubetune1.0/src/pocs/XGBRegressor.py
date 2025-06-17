import os
import warnings
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from xgboost import XGBRegressor
import plotly.express as px

warnings.filterwarnings("ignore")

# -------------------- Load Data --------------------
# Set base directory and load the pod metrics JSON file into a DataFrame
base_dir = Path(__file__).resolve().parents[3]
file_path = base_dir / 'data' / 'aks01_pod_metrics.json'
df = pd.read_json(file_path)

# -------------------- Clean Data --------------------
# Remove duplicate rows to ensure data quality
df.drop_duplicates(inplace=True)
# Remove duplicates based on key columns to avoid repeated measurements
df.drop_duplicates(subset=['collectionTimestamp', 'deployment', 'controllerName', 'container'], inplace=True)
# Convert collectionTimestamp to pandas datetime for time-based feature engineering
df['collectionTimestamp'] = pd.to_datetime(df['collectionTimestamp'])

# -------------------- Feature Engineering --------------------
# Extract hour, day of week, and weekend flag from timestamp
df['hour'] = df['collectionTimestamp'].dt.hour  
df['dayofweek'] = df['collectionTimestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'] >= 5

# Replace zeros in resource columns to avoid division by zero in ratio calculations
for col in ['cpuRequest', 'cpuLimit', 'memRequest', 'memLimit']:
    df[col] = df[col].replace(0, 0.001)

# Calculate CPU and memory utilization ratios
df['cpu_util_ratio'] = df['cpuUsage'] / df['cpuLimit']
df['mem_util_ratio'] = df['memUsage'] / df['memLimit']

# Sort by timestamp for lag feature calculation
df = df.sort_values(by='collectionTimestamp')
# Create lag features for CPU and memory usage (previous timestep)
df['cpu_lag_1'] = df['cpuUsage'].shift(1)
df['mem_lag_1'] = df['memUsage'].shift(1)
# Drop rows with NaN values introduced by lagging
df.dropna(inplace=True)

# Encode deployment and controller combination as a categorical feature
df['deployment_controller'] = df['deployment'].astype(str) + '_' + df['controllerName'].astype(str)
df['deployment_controller_encoded'] = LabelEncoder().fit_transform(df['deployment_controller'])

# Flags for under-provisioning (usage exceeds request by margin)
cpu_margin = 1.2
mem_margin = 1.2
df['cpu_under'] = (df['cpuUsage'] > df['cpuRequest'] * cpu_margin).astype(int)
df['mem_under'] = (df['memUsage'] > df['memRequest'] * mem_margin).astype(int)

# Convert memory columns from bytes to MB for easier interpretation
df['memUsage_MB'] = df['memUsage'] / (1024 * 1024)
df['memRequest_MB'] = df['memRequest'] / (1024 * 1024)
df['memLimit_MB'] = df['memLimit'] / (1024 * 1024)

# -------------------- Prepare Features --------------------
# Define features for model input
features = [
    'cpuUsage', 'memUsage', 'hour', 'dayofweek', 'is_weekend',
    'cpu_util_ratio', 'mem_util_ratio', 'cpu_lag_1', 'mem_lag_1',
    'deployment_controller_encoded'
]
X = df[features]
y_cpu = df['cpuUsage']  # Target: actual CPU usage
y_mem = df['memUsage'] / (1024 * 1024)  # Target: actual memory usage in MB
y_cpu_class = df['cpu_under']           # Target: under-provisioning flag for CPU
y_mem_class = df['mem_under']           # Target: under-provisioning flag for memory

# -------------------- Train/Test Split (chronological) --------------------
# Split data chronologically to avoid data leakage (80% train, 20% test)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_cpu_train, y_cpu_test = y_cpu.iloc[:train_size], y_cpu.iloc[train_size:]
y_mem_train, y_mem_test = y_mem.iloc[:train_size], y_mem.iloc[train_size:]
y_cpu_class_train, y_cpu_class_test = y_cpu_class.iloc[:train_size], y_cpu_class.iloc[train_size:]
y_mem_class_train, y_mem_class_test = y_mem_class.iloc[:train_size], y_mem_class.iloc[train_size:]

# -------------------- Modeling --------------------
# Define hyperparameters for XGBoost regression models
param_grid = {
    'n_estimators': [100],
    'max_depth': [4],
    'learning_rate': [0.1]
}
# Train XGBoost regressor for CPU usage prediction
cpu_model = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='r2', cv=2, n_jobs=-1).fit(X_train, y_cpu_train).best_estimator_
# Train XGBoost regressor for memory usage prediction
mem_model = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='r2', cv=2, n_jobs=-1).fit(X_train, y_mem_train).best_estimator_

# Train RandomForest classifiers for under-provisioning detection
clf_cpu = RandomForestClassifier(random_state=42).fit(X_train, y_cpu_class_train)
clf_mem = RandomForestClassifier(random_state=42).fit(X_train, y_mem_class_train)

# -------------------- Predict for All Pods --------------------
MB = 1024 * 1024
X_all = df[features]

# Predict CPU and Memory usage (in cores and MB)
df['cpuRequest_predicted'] = cpu_model.predict(X_all)
df['memRequest_predicted_MB'] = mem_model.predict(X_all)
df['memRequest_predicted'] = df['memRequest_predicted_MB'] * MB

# CPU recommendation logic:
# If predicted request < actual usage, recommend usage + 20%; else, predicted + 20%
df['recommended_cpuRequest'] = np.where(
    df['cpuRequest_predicted'] < df['cpuUsage'],
    df['cpuUsage'] + 0.2 * df['cpuUsage'],
    df['cpuRequest_predicted'] + 0.2 * df['cpuRequest_predicted']
)

# Memory recommendation logic:
# If predicted request < actual usage, recommend usage + 20%; else, predicted + 20%
df['recommended_memRequest'] = np.where(
    df['memRequest_predicted'] < df['memUsage'],
    df['memUsage'] + 0.2 * df['memUsage'],
    df['memRequest_predicted'] + 0.2 * df['memRequest_predicted']
)
df['recommended_memRequest_MB'] = df['recommended_memRequest'] / MB

# -------------------- Output --------------------
output_dir = Path(r"D:\Personal\Soumalya\Internship\MAY_2025\python\week_1\Soumalya\AKS_Cost_Reduction\MLearn-main\KubeTuner\kubetune1.0\src\models\timeseriesxgboost\output")
output_dir.mkdir(parents=True, exist_ok=True)

# Ensure collectionTimestamp is timezone-unaware for Excel compatibility
if 'collectionTimestamp' in df.columns:
    df['collectionTimestamp'] = pd.to_datetime(df['collectionTimestamp']).dt.tz_localize(None)

# Define columns to output 
output_cols = [
    "key",
    "namespace", "pod", "node", "container", "deployment","controllerName", "collectionTimestamp",
    "cpuUsage", "memUsage", "cpuRequest", "memRequest", "cpuLimit", "memLimit",
    "memUsage_MB", "memRequest_MB", "memLimit_MB",
    "cpuRequest_predicted", "memRequest_predicted", "memRequest_predicted_MB",
    "recommended_cpuRequest", "recommended_memRequest", "recommended_memRequest_MB"
]
output_cols = [col for col in output_cols if col in df.columns]

# Save results to Excel
df.to_excel(output_dir / "predicted_cpu_mem_requests.xlsx", columns=output_cols, index=False)

# -------------------- Metrics & Reports --------------------
# Print regression metrics for CPU and memory usage predictions
print("CPU MAE:", mean_absolute_error(y_cpu_test, cpu_model.predict(X_test)))
print("Mem MAE (MB):", mean_absolute_error(y_mem_test, mem_model.predict(X_test)))
print("CPU R²:", r2_score(y_cpu_test, cpu_model.predict(X_test)))
print("Mem R²:", r2_score(y_mem_test, mem_model.predict(X_test)))

# Print classification reports for under-provisioning detection
print("\nCPU Under-provisioning Classification Report:")
print(classification_report(y_cpu_class_test, clf_cpu.predict(X_test)))

print("\nMemory Under-provisioning Classification Report:")
print(classification_report(y_mem_class_test, clf_mem.predict(X_test)))

# -------------------- Visualizations --------------------

# Sample data for clarity (use all data if <= 200 rows, else random sample)
sample_cpu = df.sample(n=min(200, len(df)), random_state=42) if len(df) > 200 else df
sample_mem = df.sample(n=min(200, len(df)), random_state=42) if len(df) > 200 else df
bar_sample = df.sample(n=min(40, len(df)), random_state=42).copy()
bar_sample_cpu = df.sample(n=min(40, len(df)), random_state=42).copy()

# 1. Actual vs Predicted CPU Usage (scatter with hover)
fig_cpu = px.scatter(
    sample_cpu,
    x='cpuUsage',
    y='cpuRequest_predicted',
    color='pod',
    hover_data=['pod', 'namespace', 'deployment', 'cpuUsage', 'cpuRequest_predicted'],
    labels={'cpuUsage': 'Actual CPU Usage', 'cpuRequest_predicted': 'Predicted CPU Usage'},
    title='Actual vs Predicted CPU Usage'
)
fig_cpu.add_shape(
    type="line",
    x0=sample_cpu['cpuUsage'].min(), y0=sample_cpu['cpuUsage'].min(),
    x1=sample_cpu['cpuUsage'].max(), y1=sample_cpu['cpuUsage'].max(),
    line=dict(color="red", dash="dash"),
)
fig_cpu.write_html(output_dir / "cpu_actual_vs_predicted_plot.html")

# 2. Actual vs Predicted Memory Usage (MB) (scatter with hover)
fig_mem = px.scatter(
    sample_mem,
    x='memUsage_MB',
    y='memRequest_predicted_MB',
    color='pod',
    hover_data=['pod', 'namespace', 'deployment', 'memUsage_MB', 'memRequest_predicted_MB'],
    labels={'memUsage_MB': 'Actual Memory Usage (MB)', 'memRequest_predicted_MB': 'Predicted Memory Usage (MB)'},
    title='Actual vs Predicted Memory Usage (MB)'
)
fig_mem.add_shape(
    type="line",
    x0=sample_mem['memUsage_MB'].min(), y0=sample_mem['memUsage_MB'].min(),
    x1=sample_mem['memUsage_MB'].max(), y1=sample_mem['memUsage_MB'].max(),
    line=dict(color="red", dash="dash"),
)
fig_mem.write_html(output_dir / "mem_actual_vs_predicted_plot.html")

# 3. Recommended vs Actual Requests (Memory, MB) - barplot with hover
mem_bar_data = bar_sample.melt(
    id_vars=['pod'],
    value_vars=['memRequest_MB', 'memRequest_predicted_MB', 'recommended_memRequest_MB'],
    var_name='Type',
    value_name='Memory Request (MB)'
)
fig_bar_mem = px.bar(
    mem_bar_data,
    x='pod',
    y='Memory Request (MB)',
    color='Type',
    barmode='group',
    hover_data=['pod', 'Type', 'Memory Request (MB)'],
    title='Memory Requests: Actual vs Predicted vs Recommended'
)
fig_bar_mem.write_html(output_dir / "memory_bar.html")

# 4. Recommended vs Actual Requests (CPU) - barplot with hover
cpu_bar_data = bar_sample_cpu.melt(
    id_vars=['pod'],
    value_vars=['cpuUsage', 'cpuRequest_predicted', 'recommended_cpuRequest'],
    var_name='Type',
    value_name='CPU (cores)'
)
fig_bar_cpu = px.bar(
    cpu_bar_data,
    x='pod',
    y='CPU (cores)',
    color='Type',
    barmode='group',
    hover_data=['pod', 'Type', 'CPU (cores)'],
    title='CPU: Actual vs Predicted vs Recommended'
)
fig_bar_cpu.write_html(output_dir / "cpu_bar.html")
