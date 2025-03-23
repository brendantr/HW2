import torch
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------
# Helper: Create DataLoader
# ---------------------------

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create PyTorch DataLoaders for training and validation sets.
    """
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