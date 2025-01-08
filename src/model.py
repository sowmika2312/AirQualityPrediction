import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train):
    # Train your model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Ensure the model directory exists
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the trained model
    joblib.dump(model, os.path.join(model_dir, 'rf_model.pkl'))
    print("Model saved successfully!")
    
    return model

def evaluate_model(model, X_test, y_test):
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    
    return mse, r2
