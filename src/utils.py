import joblib

def load_model(file_path):
    # Load the trained model from the file
    return joblib.load(file_path)
