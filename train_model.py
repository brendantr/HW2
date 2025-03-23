import torch
import matplotlib.pyplot as plt

# ---------------------------
# 3. Model Training
# ---------------------------

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50, device='cpu'):
    """
    Train the model and return training/validation losses for each epoch.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        train_losses, val_losses: Lists of losses for each epoch
    """
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

def plot_training_results(train_losses, val_losses):
    """
    Plot training and validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig("training_results.png")
    plt.show()
    print("Plot saved as training_results.png")