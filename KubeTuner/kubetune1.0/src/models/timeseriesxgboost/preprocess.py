
import pandas as pd
from pathlib import Path

def preprocess_data():
    """
    Preprocess the data by loading it from a JSON file and preparing it for further analysis.
    """
    # Path to your Json file
    base_dir_input = Path(__file__).resolve().parents[3]
    file_path = base_dir_input / 'data' / 'aks01_pod_metrics.json'
    
    # Load the JSON file
    df = pd.read_json(file_path)
    
    # Display the first few rows
    print(df.head())

    # Check the shape of the dataset
    print(f"Dataset Shape: {df.shape}")

    
    # Check the data types of each column
    print(df.info())

    # Check for missing values
    print(df.isnull().sum())

    # Get basic statistics for numerical columns
    print(df.describe())

    # Check for unique values in categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        print(f"Unique values in {col}: {df[col].nunique()}")



if __name__ == "__main__":

    print("------------------------------")
    print("Preprocessing and feature raw data.")
    preprocess_data()


