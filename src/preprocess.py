import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load the dataset using the correct delimiter and decimal handling
    data = pd.read_csv(file_path, delimiter=';', decimal=',')  # Handling comma as decimal separator
    
    # Strip any leading/trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Print the dataset columns to verify
    print("Dataset columns:", data.columns)

    # Drop unnecessary unnamed columns (if any)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Check if necessary columns are present
    required_columns = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'O3(GT)', 'T', 'RH']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing columns: {', '.join(missing_columns)}")
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Rename columns to more understandable names
    data.rename(columns={
        'CO(GT)': 'CO',
        'NOx(GT)': 'NOx',
        'NO2(GT)': 'NO2',
        'O3(GT)': 'O3',
        'T': 'Temperature',
        'RH': 'Humidity'
    }, inplace=True)

    # Convert the relevant columns to numeric values, coercing any errors into NaN
    for col in ['CO', 'NOx', 'NO2', 'O3', 'Temperature', 'Humidity']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            print(f"Column {col} not found.")

    # Drop rows with any missing values after conversion
    data.dropna(inplace=True)

    # Define the features (X) and target variable (y)
    X = data[['CO', 'NOx', 'NO2', 'O3', 'Temperature', 'Humidity']]
    y = data['CO']  # Change this to your desired target variable if needed

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
