import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data Loading and Preprocessing Module for the project
# ---------------------------
# 1. Data Preprocessing
# ---------------------------


def load_and_combine_data(data_dir):
    """
    Walk through the subdirectories, read CSVs, and combine them into one DataFrame.
    """
    all_data = []
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print("Found subdirectories:", subdirs)
    
    for sub in subdirs:
        path = os.path.join(data_dir, sub)
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        print(f"Directory: {sub} contains {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            # Read CSV without headers and assign column names
            df = pd.read_csv(csv_file, header=None)
            df.columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']
            # Use columns 2-5 (indices 1-4)
            df = df[['Col2', 'Col3', 'Col4', 'Col5']]
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def preprocess_data(df):
    """
    Preprocess the DataFrame:
      - Map the label '#' to a numeric label
      - Normalize the features using StandardScaler.
    """
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # Clean the numeric columns
    print("Cleaning feature columns...")
    for col in ['Col2', 'Col3', 'Col4']:
        # Convert to numeric, setting errors to NaN
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Report and drop rows with NaN values
    invalid_rows = df_clean.isna().any(axis=1).sum()
    print(f"Found {invalid_rows} rows with invalid numeric values")
    
    # Drop rows with NaN values
    original_len = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"Dropped {original_len - len(df_clean)} rows with invalid values")
    print(f"Remaining data shape: {df_clean.shape}")
    
    # Extract features and labels
    X = df_clean[['Col2', 'Col3', 'Col4']].values
    
    # Map labels
    label_map = {'0': 0, '1': 1, '2': 2, '#': 3}
    y = df_clean['Col5'].map(label_map).values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler

def split_data(X, y, test_size=0.1, random_state=42):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)