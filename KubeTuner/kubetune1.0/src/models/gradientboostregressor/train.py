import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import evaluate as evaluate
import joblib

# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------
# Read the Excel file (adjust the path as needed)

# Path to your Excel file
base_dir_input = Path(__file__).resolve().parents[3]
file_path = base_dir_input / 'data' / 'aks_02_data_mb.xlsx'

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

# -----------------------------
# 4. Split the Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Hyperparameter Tuning with GridSearchCV (Using Gradient Boosting)
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7]
}

gbm = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Save the  model
model_path = Path(__file__).resolve().parents[0] / 'output' / 'gradient_boosting_model.pkl'
model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
joblib.dump(model, model_path)

# evaluate model
evaluate.evaluate_model(X_test, y_test)



