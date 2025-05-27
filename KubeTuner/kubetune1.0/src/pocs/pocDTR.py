import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Polynomial features (can be removed for tree-based models)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Pipeline: Poly features + Regression
pipeline = Pipeline(steps=[
    ('poly', poly),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Fit
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")