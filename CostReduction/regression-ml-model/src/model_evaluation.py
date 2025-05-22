def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error, r2_score

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        'Mean Squared Error': mse,
        'R-squared': r2
    }