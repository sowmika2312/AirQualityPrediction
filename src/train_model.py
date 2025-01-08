import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Update the path to the correct location of the dataset
data_path = '../data/raw/air_quality.csv'  # Corrected path to dataset

# Read the dataset
data = pd.read_csv(data_path, sep=";")

# Print the dataset columns to ensure it's read correctly
print(f"Dataset columns: {data.columns}")

# Preprocess data (example, you may need to modify this part based on your dataset structure)
# You can drop or clean columns that you do not need for modeling
data = data.dropna(axis=1)  # Drop columns with missing values

# Example of selecting features (adjust as per your dataset)
features = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH', 'O3(GT)']
target = 'CO(GT)'  # Replace with the appropriate target variable

# Check if the required columns are present in the data
missing_columns = [col for col in features if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

# Split the data into features and target
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
joblib.dump(model, '../model/rf_model.pkl')  # Save model in the model directory

print("Model training complete and saved as 'rf_model.pkl'")
