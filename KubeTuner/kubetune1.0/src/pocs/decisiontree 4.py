import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
df = pd.read_excel(
    r'D:\Personal\Soumalya\Internship\MAY_2025\python\week_1\Soumalya\AKS_Cost_Reduction\my-python-project\data\aks_02_data_mb.xlsx',
    engine='openpyxl'
)
df = df[(df['memUsageMB'] > 0) & (df['memRequestMB'] > 0)]
# Prepare features and target 
y = df['memUsageMB']
X = df[['memRequestMB']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Evaluate model
y_pred = dt.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")

# Predictions and suggestions
df['recommendedmemrequest'] = dt.predict(X)

df['reduction_percent'] = ((df['memRequestMB'] - df['recommendedmemrequest']) / df['memRequestMB'] * 100).round(2)

# Correct suggestion logic
df['suggestion'] = np.where(
    np.isclose(df['recommendedmemrequest'], df['memRequestMB'], atol=1e-6),
    'No change needed',
    np.where(
        df['recommendedmemrequest'] > df['memRequestMB'],
        'Increase memory request to ' + df['recommendedmemrequest'].round(2).astype(str) + ' MB',
        'Reduce memory request to ' + df['recommendedmemrequest'].round(2).astype(str) + ' MB'
    )
)

# After prediction and rounding
df['predicted_memrequest'] = df['recommendedmemrequest'].round(3)
df['memusage'] = df['memUsageMB'].round(3)

# Calculate final_rec_mem_req as per your logic
df['final_rec_mem_req'] = np.where(
    df['predicted_memrequest'] < df['memusage'],
    (df['memusage'] + 0.2 * df['memusage']).round(3),
    (df['predicted_memrequest'] + 0.2 * df['predicted_memrequest']).round(3)
)

# Rename columns for export
df_export = df.rename(columns={
    'pod': 'podname',
    'memRequestMB': 'memrequest'
})

# Select and order the columns as you want
export_cols = [
    'podname', 'memrequest', 'memusage', 'predicted_memrequest', 'final_rec_mem_req'
]

# Export to Excel
df_export[export_cols].to_excel(
    r'D:\Personal\Soumalya\Internship\MAY_2025\python\week_1\Soumalya\AKS_Cost_Reduction\my-python-project\output\memory_request_predictions_optimized_decisiontree.xlsx',
    index=False
)

# 1. Bar plot: Current vs Predicted vs Final Recommended Memory Request (Sample)
sample = df_export.sample(min(20, len(df_export)), random_state=42).reset_index(drop=True)
x = np.arange(len(sample))
bar_width = 0.25

plt.figure(figsize=(14, 6))
plt.bar(x - bar_width, sample['memrequest'], width=bar_width, label='Current Request', color='#1976D2')
plt.bar(x, sample['predicted_memrequest'], width=bar_width, label='Predicted Request', color='#43A047')
plt.bar(x + bar_width, sample['final_rec_mem_req'], width=bar_width, label='Final Recommended', color='#FFA000')
plt.xticks(x, sample['podname'], rotation=45, ha='right')
plt.xlabel('Pod')
plt.ylabel('Memory (MB)')
plt.title('Current vs Predicted vs Final Recommended Memory Request (Sample)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Scatter plot: Memory Usage vs Memory Request
plt.figure(figsize=(7, 5))
plt.scatter(df_export['memusage'], df_export['memrequest'], alpha=0.5, label='Current')
plt.scatter(df_export['memusage'], df_export['predicted_memrequest'], alpha=0.5, label='Predicted')
plt.scatter(df_export['memusage'], df_export['final_rec_mem_req'], alpha=0.5, label='Final Recommended')
plt.xlabel('Memory Usage (MB)')
plt.ylabel('Memory Request (MB)')
plt.title('Memory Usage vs Requests')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of Reduction Percent
plt.figure(figsize=(10, 5))
reduction_data = df['reduction_percent']
reduction_data = reduction_data[reduction_data > 0]
plt.hist(reduction_data, bins=20, color='#4CAF50', edgecolor='black', alpha=0.8)
plt.xlabel('Reduction Percent (%)')
plt.ylabel('Number of Pods')
plt.title('Distribution of Suggested Memory Reductions')
plt.grid(axis='y', linestyle='--', alpha=0.7)
if not reduction_data.empty:
    plt.axvline(reduction_data.mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {reduction_data.mean():.1f}%")
    plt.legend()
plt.tight_layout()
plt.show()

# 4. Cumulative savings plot
df_export['savings_MB'] = df_export['memrequest'] - df_export['final_rec_mem_req']
df_export['savings_MB'] = df_export['savings_MB'].clip(lower=0)
df_sorted = df_export.sort_values('savings_MB', ascending=False)
df_sorted['cumulative_savings'] = df_sorted['savings_MB'].cumsum()
plt.figure(figsize=(10, 5))
plt.plot(df_sorted['cumulative_savings'].values, color='purple')
plt.xlabel('Pods (sorted by savings)')
plt.ylabel('Cumulative Savings (MB)')
plt.title('Cumulative Memory Savings if All Suggestions Applied')
plt.grid(True)
plt.tight_layout()
plt.show()

