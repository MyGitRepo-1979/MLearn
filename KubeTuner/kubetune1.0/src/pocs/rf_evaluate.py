import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model_path, X_test, y_test):
    print(f"\nEvaluating model from: {model_path.name}")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}\n")
