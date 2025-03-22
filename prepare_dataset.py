import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
      - Map the label '#' to a string (or a numeric label if desired)
      - Optionally, convert the labels to numbers: {0:0, 1:1, 2:2, '#':3}
      - Normalize the features using StandardScaler.
    """
    # Convert label column to string (if not already) to handle '#'
    df['Col5'] = df['Col5'].astype(str)

    # Map labels to numeric values.
    label_mapping = {'0': 0, '1': 1, '2': 2, '#': 3}
    df['Label'] = df['Col5'].map(label_mapping)
    
    # Separate features and labels.
    X = df[['Col2', 'Col3', 'Col4']].values.astype(float)
    y = df['Label'].values.astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# ---------------------------
# 2. Train/Test Split
# ---------------------------

def split_data(X, y, test_size=0.1, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

# ---------------------------
# 3. Define the Neural Network
# ---------------------------

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(DeepNN, self).__init__()
        # Create a 5-layer network (4 hidden layers + output)
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], num_classes)
        self.relu = nn.ReLU()
        
        # Optional: Weight initialization can be applied here
        self._initialize_weights()

    def _initialize_weights(self):
        # Example using Xavier/Glorot initialization for all layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# ---------------------------
# 4. Training and Validation Loop
# ---------------------------

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50, device='cpu'):
    train_losses = []
    val_losses = []
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f} - Validation Loss: {epoch_val_loss:.4f}")
    
    return train_losses, val_losses

# ---------------------------
# 5. Helper: Create DataLoader
# ---------------------------

from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    # Convert to torch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ---------------------------
# 6. Main Script: Preprocess Data, Train Model, and Save Results
# ---------------------------

def main():
    data_directory = "eel4810-dataset"  # Adjust this path if necessary
    df = load_and_combine_data(data_directory)
    print("Combined data shape:", df.shape)
    
    X, y, scaler = preprocess_data(df)
    print("Features shape:", X.shape, "Labels shape:", y.shape)
    
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.1)
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Define model parameters.
    input_size = X.shape[1]  # Should be 3 features (Col2, Col3, Col4)
    hidden_sizes = [64, 32, 16, 8]  # You can adjust these
    num_classes = 4  # Classes: 0, 1, 2, and '#' mapped to 3.
    
    model = DeepNN(input_size, hidden_sizes, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device)
    
    # Save the trained model.
    torch.save(model.state_dict(), "trained_model.pt")
    print("Model saved as trained_model.pt")
    
    # Plot training and validation losses.
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
