from src.preprocess import preprocess_data
from src.model import train_model
from src.model import evaluate_model

def main():
    # Define the path to your raw data
    raw_data_path = 'data/raw/air_quality.csv'

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(raw_data_path)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
