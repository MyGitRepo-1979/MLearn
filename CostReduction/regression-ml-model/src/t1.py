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

usable_data = df[["cpuRequest", "memRequest","cpuUsage", "memUsage"]]

print(usable_data.head())