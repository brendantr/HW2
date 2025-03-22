# filepath: /Users/brendanrodriguez/Library/CloudStorage/OneDrive-UniversityofCentralFlorida/spring 2025/EEL4810/HW2/prepare_dataset.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_path):
    all_data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Extract input features (2nd, 3rd, 4th columns) and labels (5th column)
    X = combined_data.iloc[:, 1:4].values  # Input features
    y = combined_data.iloc[:, 4].values    # Labels

    # Split the data into 90% training and 10% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    dataset_path = "eel4810-dataset"  # Update this path if needed
    X_train, X_test, y_train, y_test = prepare_dataset(dataset_path)

    # Print a few samples to verify
    print("Sample training data (features):", X_train[:5])
    print("Sample training data (labels):", y_train[:5])