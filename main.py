import torch
import torch.nn as nn
import torch.optim as optim

from data_preprocessing import load_and_combine_data, preprocess_data, split_data
from model import DeepNN
from utils import create_dataloaders
from train_model import train_model, plot_training_results

def main():
    # 1. Load and preprocess data
    data_directory = "eel4810-dataset"  # Adjust this path if necessary
    df = load_and_combine_data(data_directory)
    print("Combined data shape:", df.shape)
    
    X, y, scaler = preprocess_data(df)
    print("Features shape:", X.shape, "Labels shape:", y.shape)
    
    # 2. Split data
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.1)
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # 3. Define model
    input_size = X.shape[1]  # Should be 3 features (Col2, Col3, Col4)
    hidden_sizes = [64, 32, 16, 8]  # You can adjust these
    num_classes = 4  # Classes: 0, 1, 2, and '#' mapped to 3.
    
    model = DeepNN(input_size, hidden_sizes, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train the model
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    train_losses, val_losses = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs, device
    )
    
    # 5. Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'scaler': scaler,
        'hidden_sizes': hidden_sizes,
        'num_classes': num_classes
    }, "model_checkpoint.pt")
    print("Model saved as model_checkpoint.pt")
    
    # 6. Plot results
    plot_training_results(train_losses, val_losses)

if __name__ == "__main__":
    main()