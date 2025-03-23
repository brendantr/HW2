import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Import the training functions to test
from train_model import train_model, plot_training_results

class SimpleModel(nn.Module):
    """
    Extremely simple model for testing training functions
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 3)  # Simple linear layer
        
    def forward(self, x):
        return self.fc(x)

def create_simple_dataset():
    """
    Create a very simple dataset that is guaranteed to be learnable
    """
    # Create a simple classification problem:
    # Points in the positive quadrant (x>0, y>0) are class 0
    # Points in the negative x quadrant (x<0) are class 1
    # Points in the negative y, positive x quadrant (x>0, y<0) are class 2
    np.random.seed(42)  # For reproducibility
    
    n_samples = 90
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.zeros(n_samples, dtype=np.int64)
    
    # Assign labels based on quadrants
    y[(X[:, 0] < 0)] = 1  # Negative x quadrant = class 1
    y[(X[:, 0] > 0) & (X[:, 1] < 0)] = 2  # Positive x, negative y = class 2
    
    # Create a validation set
    n_val = 10
    X_val = np.random.uniform(-1, 1, (n_val, 2))
    y_val = np.zeros(n_val, dtype=np.int64)
    y_val[(X_val[:, 0] < 0)] = 1
    y_val[(X_val[:, 0] > 0) & (X_val[:, 1] < 0)] = 2
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_tensor, y_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    return train_loader, val_loader

def test_training_function():
    """Test that the training function runs correctly"""
    print("Testing training function...")
    
    # Create a simple model and dataset
    model = SimpleModel()
    train_loader, val_loader = create_simple_dataset()
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Train for just a few epochs to test
    num_epochs = 5
    train_losses, val_losses = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs
    )
    
    # Check output shapes
    assert len(train_losses) == num_epochs, f"Expected {num_epochs} training losses, got {len(train_losses)}"
    assert len(val_losses) == num_epochs, f"Expected {num_epochs} validation losses, got {len(val_losses)}"
    print("✓ Training function recorded correct number of loss values")
    
    # Check that loss is decreasing
    assert train_losses[0] > train_losses[-1], "Training loss did not decrease"
    print("✓ Training loss decreased over epochs (from {:.4f} to {:.4f})".format(
        train_losses[0], train_losses[-1]))
    
    return train_losses, val_losses

def test_visualization_function(train_losses, val_losses):
    """Test that the visualization function works correctly"""
    print("\nTesting visualization function...")
    
    # Save the current matplotlib backend to restore it later
    original_backend = plt.get_backend()
    
    try:
        # Use the 'Agg' backend which doesn't require a GUI
        plt.switch_backend('Agg')
        
        # Call the function
        plot_training_results(train_losses, val_losses)
        
        # If we got here without errors, it worked
        print("✓ Plot function executed without errors")
        
        # Check that the output file exists
        assert os.path.exists("training_results.png"), "Plot file was not created"
        print("✓ Plot file was created successfully")
        
        # Remove the file
        os.remove("training_results.png")
        
    finally:
        # Restore the original backend
        plt.switch_backend(original_backend)
    
    return True

def test_model_learning():
    """Test that the model actually learns the simple pattern"""
    print("\nTesting that model can learn the pattern...")
    
    # Create a simple model and dataset
    model = SimpleModel()
    train_loader, val_loader = create_simple_dataset()
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Train for more epochs to ensure learning
    train_losses, _ = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=20
    )
    
    # Check that final loss is significantly lower than initial loss
    assert train_losses[0] > train_losses[-1] * 1.5, "Model didn't learn sufficiently"
    print(f"✓ Model learned successfully (loss reduced from {train_losses[0]:.4f} to {train_losses[-1]:.4f})")
    
    # Test accuracy on training data
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"✓ Final model accuracy: {accuracy:.2%}")
    assert accuracy > 0.6, "Model accuracy is too low"
    
    return True

if __name__ == "__main__":
    train_losses, val_losses = test_training_function()
    test_visualization_function(train_losses, val_losses)
    test_model_learning()
    print("\nAll training function tests passed!")