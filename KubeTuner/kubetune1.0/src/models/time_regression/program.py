import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

# --- Load Data ---
base_dir = Path(__file__).resolve().parents[3]
file_path = base_dir / 'data' / 'AKS02-Data.xlsx'
df = pd.read_excel(file_path, sheet_name="Sheet1")

# --- Feature Engineering: Use both timestamps ---
for col in ['cpuUsageTimestamp', 'memUsageTimestamp']:
    df[col] = pd.to_datetime(df[col])

df['cpu_time_num'] = (df['cpuUsageTimestamp'] - df['cpuUsageTimestamp'].min()).dt.total_seconds()
df['mem_time_num'] = (df['memUsageTimestamp'] - df['memUsageTimestamp'].min()).dt.total_seconds()
df['cpu_hour'] = df['cpuUsageTimestamp'].dt.hour
df['cpu_day_of_week'] = df['cpuUsageTimestamp'].dt.dayofweek
df['mem_hour'] = df['memUsageTimestamp'].dt.hour
df['mem_day_of_week'] = df['memUsageTimestamp'].dt.dayofweek

# --- Prepare features and targets for modeling ---
feature_cols = [
    'cpuUsage', 'memUsage',
    'cpu_time_num', 'mem_time_num',
    'cpu_hour', 'cpu_day_of_week',
    'mem_hour', 'mem_day_of_week'
]
target_mem = 'memRequest'
target_cpu = 'cpuRequest'

fit_mask = df[feature_cols + [target_mem, target_cpu]].notnull().all(axis=1)
fit_df = df[fit_mask]

# --- Train/Test Split (time-based, no shuffle) ---
X = fit_df[feature_cols]
y_mem = fit_df[target_mem]
y_cpu = fit_df[target_cpu]
X_train, X_test, y_mem_train, y_mem_test, y_cpu_train, y_cpu_test = train_test_split(
    X, y_mem, y_cpu, test_size=0.2, shuffle=False
)

# --- Hyperparameter grid for Random Forest ---
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# --- Grid Search for Memory Request ---
mem_rf = RandomForestRegressor(random_state=42)
mem_grid = GridSearchCV(mem_rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
mem_grid.fit(X_train, y_mem_train)
print(f"Best Memory RF Params: {mem_grid.best_params_}")

# --- Predict on test set ---
mem_predictions = mem_grid.predict(X_test)

# --- Grid Search for CPU Request ---
cpu_rf = RandomForestRegressor(random_state=42)
cpu_grid = GridSearchCV(cpu_rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
cpu_grid.fit(X_train, y_cpu_train)
print(f"Best CPU RF Params: {cpu_grid.best_params_}")

# --- Predict on test set ---
cpu_predictions = cpu_grid.predict(X_test)

# --- Evaluate Performance ---
mem_r2 = r2_score(y_mem_test, mem_predictions)
mem_mae = mean_absolute_error(y_mem_test, mem_predictions)
cpu_r2 = r2_score(y_cpu_test, cpu_predictions)
cpu_mae = mean_absolute_error(y_cpu_test, cpu_predictions)
print(f"Memory Request Prediction R²: {mem_r2:.4f}")
print(f"Memory Request Prediction MAE: {mem_mae:.4f}")
print(f"CPU Request Prediction R²: {cpu_r2:.4f}")
print(f"CPU Request Prediction MAE: {cpu_mae:.4f}")

# --- Export results for review ---
df['collectionTimestamp'] = pd.to_datetime(df['collectionTimestamp'], errors='coerce')
if pd.api.types.is_datetime64tz_dtype(df['collectionTimestamp']):
    df['collectionTimestamp'] = df['collectionTimestamp'].dt.tz_localize(None)

results = pd.DataFrame({
    'collectionTimestamp': df['collectionTimestamp'],
    'pod': df['pod'],
    'deployment': df['deployment'],
    'cpuUsage': df['cpuUsage'],
    'cpuRequest': df['cpuRequest'],
    'memUsage': df['memUsage'],
    'memRequest': df['memRequest'],
})

# Predict for all rows (where features are not null)
predict_mask = df[feature_cols].notnull().all(axis=1)
results.loc[predict_mask, 'cpuRequest_predicted'] = cpu_grid.predict(df.loc[predict_mask, feature_cols])
results.loc[predict_mask, 'memRequest_predicted'] = mem_grid.predict(df.loc[predict_mask, feature_cols])

# Convert only memory columns to MB
mem_cols = ['memUsage', 'memRequest', 'memRequest_predicted']
for col in mem_cols:
    if col in results.columns:
        results[col] = results[col] / (1024 * 1024)

# CPU recommendation logic using your formula
results['recommended_cpuRequest'] = np.where(
    results['cpuRequest_predicted'] < results['cpuUsage'],
    results['cpuUsage'] + 0.2 * results['cpuUsage'],
    results['cpuRequest_predicted'] + 0.2 * results['cpuRequest_predicted']
)

# Memory recommendation logic using your formula
results['recommended_memRequest'] = np.where(
    results['memRequest_predicted'] < results['memUsage'],
    results['memUsage'] + 0.2 * results['memUsage'],
    results['memRequest_predicted'] + 0.2 * results['memRequest_predicted']
)

# Remove mem_recommendation and cpu_recommendation columns
results = results.drop(columns=['mem_recommendation', 'cpu_recommendation'], errors='ignore')

# Save to Excel
results.to_excel('rf_mem_recommendations.xlsx', index=False)
print("\nPredictions and recommendations exported to rf_mem_recommendations.xlsx")

# Group by deployment and aggregate pod info and recommendations
grouped = results.groupby('deployment').apply(
    lambda g: pd.DataFrame({
        'pod': g['pod'],
        'collectionTimestamp': g['collectionTimestamp'],
        'cpuUsage': g['cpuUsage'],
        'cpuRequest': g['cpuRequest'],
        'cpuRequest_predicted': g['cpuRequest_predicted'],
        'recommended_cpuRequest': g['recommended_cpuRequest'],
        'memUsage': g['memUsage'],
        'memRequest': g['memRequest'],
        'memRequest_predicted': g['memRequest_predicted'],
        'recommended_memRequest': g['recommended_memRequest']
    })
).reset_index(level=0)

# Save to Excel
grouped.to_excel('rf_mem_recommendations.xlsx', index=False)
print("\nGrouped predictions and recommendations exported to rf_mem_recommendations.xlsx")