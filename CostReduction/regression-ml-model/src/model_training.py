import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    # Load the dataset
    data = load_data('../data/train.csv')
    print(data.columns)
    
    # Split the data into features and target variable
    X = data.drop('target', axis=1)  # Replace 'target' with the actual target column name
    y = data['target']  # Replace 'target' with the actual target column name
    
    # Train the model
    model = train_model(X, y)
    
    # Save the trained model
    save_model(model, 'regression_model.pkl')

if __name__ == "__main__":
    main()