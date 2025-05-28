def evaluate_model(X_test, y_test):
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    from pathlib import Path

    model_file_path = Path(__file__).resolve().parents[0] / 'output' / 'gradient_boosting_model.pkl'
    model= joblib.load(model_file_path)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")