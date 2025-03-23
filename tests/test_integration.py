import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

# Import all components
from data_preprocessing import load_and_combine_data, preprocess_data, split_data
from model import DeepNN
from utils import create_dataloaders
from train_model import train_model
from evaluate_model import evaluate_model

def test_full_pipeline():
    """Test the entire pipeline end-to-end with a small data sample"""
    print("Testing full integration pipeline...")
    
    # 1. Load a small subset of data (just process one subdirectory)
    print("1. Loading data subset...")
    data_dir = "eel4810-dataset"
    # Find first subdirectory
    import os
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subdirs:
        print("No subdirectories found!")
        return False
    
    # Just load files from first subdirectory to keep it small
    import glob
    import pandas as pd
    
    subdir_path = os.path.join(data_dir, subdirs[0])
    csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))[:5]  # Just use first 5 files
    
    if not csv_files:
        print("No CSV files found!")
        return False
    
    print(f"Using {len(csv_files)} files from {subdirs[0]}")
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)
        df.columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']
        df = df[['Col2', 'Col3', 'Col4', 'Col5']]
        all_data.append(df)
    
    sample_df = pd.concat(all_data, ignore_index=True)
    print(f"Sample data shape: {sample_df.shape}")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    X, y, scaler = preprocess_data(sample_df)
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # 3. Split data
    print("\n3. Splitting data...")
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)  # Use 80/20 for small sample
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}")
    
    # 4. Create dataloaders
    print("\n4. Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=8)
    
    # 5. Create model
    print("\n5. Creating model...")
    input_size = X.shape[1]  # Should be 3
    hidden_sizes = [64, 32, 16, 8]
    num_classes = 4  # Always use 4 classes regardless of what's in the sample
    
    model = DeepNN(input_size, hidden_sizes, num_classes)
    print(f"Model created with {input_size} inputs and {num_classes} outputs")
    
    # 6. Train for just a few epochs
    print("\n6. Training model (short)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = train_model(model, criterion, optimizer, 
                                          train_loader, val_loader, num_epochs=3)
    
    # 7. Evaluate model
    print("\n7. Evaluating model...")
    metrics = evaluate_model(model, val_loader)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 8. Save and load model test
    print("\n8. Testing model saving and loading...")
    
    # Save model
    checkpoint_path = "test_model_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    # Load model
    checkpoint = torch.load(checkpoint_path)
    new_model = DeepNN(input_size, hidden_sizes, num_classes)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Clean up
    os.remove(checkpoint_path)
    
    print("✓ Model saved and loaded successfully")
    
    return True

if __name__ == "__main__":
    test_full_pipeline()
    print("\nIntegration test completed successfully!")