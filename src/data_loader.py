import pandas as pd

def load_data(file_path):
    """Load raw data from a CSV file."""
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Save processed data to a CSV file."""
    data.to_csv(file_path, index=False)
