import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

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
best_model = grid_search.best_estimator_

# -----------------------------
# 6. Evaluate the Model
# -----------------------------
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# -----------------------------
# 7. Get Predictions and Reverse Transformation
# -----------------------------
# Predict on the entire dataset (log scale), then reverse the log transform.
df['predicted_log_opt_memRequest'] = best_model.predict(X)
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

# -----------------------------
# 8. Export the Results to Excel
# -----------------------------
# Define columns for export
export_cols = ['memUsage', 'memRequest', 'recommendedRequest', 'opt_memRequest', 
               'memRequest_bytes', 'predicted_opt_memRequest_bytes', 'reduction_percent', 'suggestion']


# output_file_path = base_dir / 'output' / 'output' / 'memory_request_predictions_optimized.xlsx'
# print (f"Saving predictions to {output_file_path}")

base_dir_output = Path(__file__).resolve().parents[0]
output_file_path = base_dir_output / 'output' / 'memory_request_predictions_optimized.xlsx'


df[export_cols].to_excel(output_file_path, index=False)
print(f"Predictions saved to {output_file_path}")

# -----------------------------
# 9. Plot Actual vs. Predicted Optimized Memory Request (bytes)
# -----------------------------
plt.figure(figsize=(10,6))
plt.scatter(df['memRequest_bytes'], df['predicted_opt_memRequest_bytes'], alpha=0.5)
plt.plot([df['memRequest_bytes'].min(), df['memRequest_bytes'].max()],
         [df['memRequest_bytes'].min(), df['memRequest_bytes'].max()], 'r--')
plt.xlabel('Current Memory Request (bytes)')
plt.ylabel('Predicted Optimized Memory Request (bytes)')
plt.title('Actual vs. Predicted Optimized Memory Request')
plt.grid(True)
plt.show()
