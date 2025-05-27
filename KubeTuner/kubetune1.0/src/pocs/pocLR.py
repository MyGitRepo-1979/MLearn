import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt


# Path to your Excel file
base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / 'data' / 'aks_02_dataset-V.03.xlsx'

# Read data
df = pd.read_excel(file_path, sheet_name="Sheet1")

df["memRequest"] =  df.memRequest.map(lambda x: x / (1024 * 1024))
df["memUsage"] = df.memRequest.map(lambda x: x / (1024 * 1024))

usable_data = df[["cpuRequest", "memRequest","cpuUsage", "memUsage"]]

# Select features and targets
X = usable_data[["cpuRequest", "memRequest"]]
y = usable_data[["cpuUsage", "memUsage"]]



# sns.pairplot(df[["cpuUsage", "memUsage", "cpuRequest", "memRequest"]])
# plt.show()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline: preprocessing + model
scaler  = PolynomialFeatures(degree=4)
scaler.fit_transform(usable_data)

pipeline = Pipeline(steps=[
    ('scaler', scaler),      # ✅ Transformer goes here
    ('model', LinearRegression())      # ✅ Estimator goes here
])

# Fit model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"mean_squared_error is : {mse}")
print(f"r2 is : {r2}")



# residuals = y_test - y_pred
# plt.scatter(y_pred, residuals)
# plt.axhline(0, color='r', linestyle='--')
# plt.xlabel("Predicted")
# plt.ylabel("Residual")
# plt.title("Residual Plot")
# plt.grid(True)
# plt.show()