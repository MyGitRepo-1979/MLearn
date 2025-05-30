import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'aks_02_dataset-V.04 1.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Define features and targets
X = data[['cpuUsage', 'memUsage']]
y_cpu = data['cpuRequest']
y_mem = data['memRequest']

# Split the data
X_train, X_test, y_cpu_train, y_cpu_test, y_mem_train, y_mem_test = train_test_split(
    X, y_cpu, y_mem, test_size=0.2, random_state=42
)

# Feature engineering: Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize models
dt_cpu = DecisionTreeRegressor(random_state=42)
gb_cpu = GradientBoostingRegressor(random_state=42)
dt_mem = DecisionTreeRegressor(random_state=42)
gb_mem = GradientBoostingRegressor(random_state=42)

# Train models
dt_cpu.fit(X_train_poly, y_cpu_train)
gb_cpu.fit(X_train_poly, y_cpu_train)
dt_mem.fit(X_train_poly, y_mem_train)
gb_mem.fit(X_train_poly, y_mem_train)

# Predict
y_cpu_pred_dt = dt_cpu.predict(X_test_poly)
y_cpu_pred_gb = gb_cpu.predict(X_test_poly)
y_mem_pred_dt = dt_mem.predict(X_test_poly)
y_mem_pred_gb = gb_mem.predict(X_test_poly)

# Evaluate
print("Decision Tree - CPU Request:", r2_score(y_cpu_test, y_cpu_pred_dt))
print("Gradient Boosting - CPU Request:", r2_score(y_cpu_test, y_cpu_pred_gb))
print("Decision Tree - Memory Request:", r2_score(y_mem_test, y_mem_pred_dt))
print("Gradient Boosting - Memory Request:", r2_score(y_mem_test, y_mem_pred_gb))
