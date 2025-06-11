import pandas as pd
from pathlib import Path

# Set up file paths
base_dir = Path(__file__).resolve().parents[3]
file_path = base_dir / 'data' / 'AKS02-Data.xlsx'

# Read the entire Excel sheet
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Save the total number of pods before any filtering/conversion
if 'pod' in df.columns:
    total_pods_raw = df['pod'].nunique()
    print(f"Total number of pods in raw Excel: {total_pods_raw}")
else:
    print("Column 'pod' not found in the data.")

# Convert relevant columns from bytes to MB (1 MB = 1024 * 1024 bytes)
cols_to_check = ['memUsage', 'memRequest', 'cpuRequest', 'cpuUsage']
for col in cols_to_check:
    df[col + '_MB'] = df[col] / (1024 * 1024)

# Save the cleaned and converted DataFrame to a pretty-printed JSON array in the output directory
output_dir = Path(__file__).resolve().parents[0] / 'output'
output_dir.mkdir(exist_ok=True)
json_path = output_dir / 'AKS02-Data-cleaned.json'
df.to_json(json_path, orient='records', indent=2)

print(f"\nAll data from Excel has been saved to JSON: {json_path}")

# Print the total number of unique pods after conversion (should match raw if no filtering)
if 'pod' in df.columns:
    total_pods = df['pod']
    print(f"Total number of pods in JSON: {total_pods}")
else:
    print("Column 'pod' not found in the data.")

