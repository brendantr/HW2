import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Preprocess data
def preprocess_data(dataset_path):
    all_data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)

    # Normalize input features (columns 2, 3, 4)
    scaler = StandardScaler()
    normalized_inputs = scaler.fit_transform(combined_data.iloc[:, 1:4])

    # Map labels (column 5) to numeric values
    label_mapping = {label: idx for idx, label in enumerate(combined_data.iloc[:, 4].unique())}
    print("Label Mapping:", label_mapping)
    
    numeric_labels = combined_data.iloc[:, 4].map(label_mapping).values

    return torch.tensor(normalized_inputs, dtype=torch.float32), torch.tensor(numeric_labels, dtype=torch.long)

# Define neural network
class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(label_mapping))  # Output layer size matches number of classes
        )

    def forward(self, x):
        return self.layers(x)

# Train model
def train_model(train_loader, test_loader):
    model = DeepNeuralNetwork()
    
    criterion = nn.CrossEntropyLoss()
    
   
