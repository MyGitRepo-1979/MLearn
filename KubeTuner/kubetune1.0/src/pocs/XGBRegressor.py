import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ========================
# Step 1: Load JSON File
# ========================
base_dir = Path(__file__).resolve().parents[2]
file_path = base_dir / 'data' / 'aks01_pod_metrics.json'

df = pd.read_json(file_path)

# ========================
# Step 2: Timestamp Features
# ========================
df['cpuUsageTimestamp'] = pd.to_datetime(df['cpuUsageTimestamp'])
df['timestamp'] = df['cpuUsageTimestamp']

df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'] >= 5

# ========================
# Step 3: Clean and Compute Ratios
# ========================
for col in ['cpuRequest', 'cpuLimit', 'memRequest', 'memLimit']:
    df[col] = df[col].replace(0, 0.001)

# Do NOT replace zeros in the original columns
# Instead, handle division by zero in the ratio columns only
df['cpu_utilization_ratio'] = np.where(
    df['cpuLimit'] == 0, 0, df['cpuUsage'] / df['cpuLimit']
)
df['mem_utilization_ratio'] = np.where(
    df['memLimit'] == 0, 0, df['memUsage'] / df['memLimit']
)

# ========================
# Step 4: Encode Categorical Features
# ========================
cat_feature = 'deployment'  # You can switch to 'namespace' if desired
df[cat_feature] = LabelEncoder().fit_transform(df[cat_feature])

# ========================
# Step 4.1: Create Rolling Features Before Training
# ========================
df = df.sort_values(by='cpuUsageTimestamp')  # Ensure proper time order

df['avg_cpu_5min'] = df['cpuUsage'].rolling(window=5, min_periods=1).mean()
df['avg_mem_5min'] = df['memUsage'].rolling(window=5, min_periods=1).mean()
df['max_cpu'] = df['cpuUsage'].rolling(window=5, min_periods=1).max()
df['max_mem'] = df['memUsage'].rolling(window=5, min_periods=1).max()


# ========================
# Step 5: Select Features & Targets
# ========================
features = [
    'cpuUsage', 'memUsage',
    'avg_cpu_5min', 'avg_mem_5min', 'max_cpu', 'max_mem',
    'hour', 'dayofweek', 'is_weekend',
    'cpu_utilization_ratio', 'mem_utilization_ratio',
    cat_feature
]


X = df[features]
y_cpu = df['cpuRequest']
y_mem = df['memRequest']

# ========================
# Step 6: Train/Test Split
# ========================
X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(X, y_cpu, test_size=0.2, random_state=42)
_, _, y_mem_train, y_mem_test = train_test_split(X, y_mem, test_size=0.2, random_state=42)

# ========================
# Step 7: Hyperparameter Tuning with GridSearchCV
# ========================
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2]
}

# CPU Model Grid Search
tscv = TimeSeriesSplit(n_splits=3)
cpu_grid = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
cpu_grid.fit(X_train, y_cpu_train)
cpu_model = cpu_grid.best_estimator_
print("Best CPU params:", cpu_grid.best_params_)

# Memory Model Grid Search
mem_grid = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
mem_grid.fit(X_train, y_mem_train)
mem_model = mem_grid.best_estimator_
print("Best Memory params:", mem_grid.best_params_)

# ========================
# Step 8: Evaluate
# ========================
cpu_preds = cpu_model.predict(X_test)
mem_preds = mem_model.predict(X_test)

print("CPU Request MAE:", mean_absolute_error(y_cpu_test, cpu_preds))
print("Memory Request MAE:", mean_absolute_error(y_mem_test, mem_preds))
print("CPU R²:", r2_score(y_cpu_test, cpu_preds))
print("Memory R²:", r2_score(y_mem_test, mem_preds))

# ========================
# Step 9: Plot Prediction vs Actual
# ========================
plt.figure(figsize=(10, 5))
plt.plot(cpu_preds[:100], label="Predicted CPU Request")
plt.plot(y_cpu_test.values[:100], label="Actual CPU Request")
plt.title("CPU Request Prediction vs Actual")
plt.legend()
plt.show()

# Memory plot (add here)
plt.figure(figsize=(10, 5))
plt.plot(mem_preds[:100], label="Predicted Memory Request")
plt.plot(y_mem_test.values[:100], label="Actual Memory Request")
plt.title("Memory Request Prediction vs Actual")
plt.legend()
plt.show()

# ========================
# Step 10: Save Output
# ========================
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

df_results = X_test.copy()
df_results['actual_cpu_request'] = y_cpu_test.values
df_results['predicted_cpu_request'] = cpu_preds
df_results['actual_mem_request'] = y_mem_test.values
df_results['predicted_mem_request'] = mem_preds

df_results.to_excel(output_dir / "predicted_cpu_mem_requests.xlsx", index=False)


