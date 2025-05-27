import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Path to your Excel file
base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / 'data' / 'aks_02_dataset-V.03.xlsx'

# Read data
df = pd.read_excel(file_path, sheet_name="Sheet1")

df["memRequest"] =  df.memRequest.map(lambda x: x / (1024 * 1024))
df["memUsage"] = df.memRequest.map(lambda x: x / (1024 * 1024))

# Select features and targets
X = df[["cpuRequest", "memRequest"]]
y = df[["cpuUsage", "memUsage"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial features (optional — GBMs often don't need them)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Pipeline using Gradient Boosting
pipeline = Pipeline(steps=[
    ('poly', poly),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Fit the model
# NOTE: GradientBoostingRegressor does not support multi-output directly
# Train one model per target variable
y_pred_all = []
models = []

for i, col in enumerate(y.columns):
    model = pipeline.fit(X_train, y_train[col])
    y_pred = model.predict(X_test)
    y_pred_all.append(y_pred)
    models.append(model)

# Convert predictions to DataFrame
import numpy as np
y_pred_df = pd.DataFrame(np.array(y_pred_all).T, columns=y.columns)

# Evaluate
mse = mean_squared_error(y_test, y_pred_df)
r2 = r2_score(y_test, y_pred_df)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")
