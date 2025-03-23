import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from time import time

from data_preprocessing import load_and_combine_data, preprocess_data, split_data
from model import DeepNN
from utils import create_dataloaders
from train_model import train_model
from evaluate_model import evaluate_model

# Create directory for results
os.makedirs("experiment_results", exist_ok=True)

def run_experiment(config):
    """Run a single experiment with the given configuration"""
    experiment_name = config['name']
    print(f"\n{'='*50}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*50}")
    
    start_time = time()
    
    # 1. Load and preprocess data
    print("Loading data...")
    df = load_and_combine_data("eel4810-dataset")
    X, y, scaler = preprocess_data(df)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.1)
    
    # Apply normalization if specified
    if not config['normalize']:
        print("Skipping normalization...")
        # If not normalizing, use raw data by inverting the scaling
        X_train = scaler.inverse_transform(X_train)
        X_val = scaler.inverse_transform(X_val)
    
    # 2. Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=config['batch_size']
    )
    
    # 3. Create model with specified initialization
    input_size = X.shape[1]
    num_classes = 4
    
    # Set initialization based on config
    if config['initialization'] == 'xavier':
        model = DeepNN(input_size, [64, 32, 16, 8], num_classes)
    else:
        # Random initialization
        model = DeepNN(input_size, [64, 32, 16, 8], num_classes, init_weights='random')
    
    # 4. Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], 
                                 weight_decay=config['weight_decay'])
    else:  # momentum
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                             momentum=0.9, weight_decay=config['weight_decay'])
    
    # 5. Train model (with fewer epochs for experiments)
    num_epochs = 20  # Reduced for faster experimentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training for {num_epochs} epochs on {device}...")
    train_losses, val_losses = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs, device
    )
    
    # 6. Evaluate final model
    metrics = evaluate_model(model, val_loader, device)
    
    # 7. Save results
    results_dir = f"experiment_results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics
    }, f"{results_dir}/model.pt")
    
    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Learning Curves: {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{results_dir}/learning_curve.png")
    plt.close()
    
    end_time = time()
    runtime = end_time - start_time
    
    # Return metrics for comparison
    return {
        'name': experiment_name,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'runtime': runtime
    }

# Define experiment configurations
configs = [
    # 1. Normalization experiments
    {
        'name': 'with_normalization',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    {
        'name': 'without_normalization',
        'normalize': False,
        'batch_size': 32,
        'learning_rate': 0.001, 
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    
    # 2. Batch size experiments
    {
        'name': 'batch_size_16',
        'normalize': True,
        'batch_size': 16,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    {
        'name': 'batch_size_64',
        'normalize': True,
        'batch_size': 64,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    
    # 3. Learning rate experiments
    {
        'name': 'learning_rate_0.0001',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    {
        'name': 'learning_rate_0.01',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    
    # 4. Optimizer experiments
    {
        'name': 'optimizer_sgd',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'sgd',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    {
        'name': 'optimizer_adam',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    {
        'name': 'optimizer_rmsprop',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'rmsprop',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    {
        'name': 'optimizer_momentum',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'momentum',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    
    # 5. Initialization experiments
    {
        'name': 'init_xavier',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    },
    {
        'name': 'init_random',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'random',
        'weight_decay': 0
    },
    
    # 6. Regularization experiments
    {
        'name': 'with_l2_regularization',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0.001
    },
    {
        'name': 'without_l2_regularization',
        'normalize': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'initialization': 'xavier',
        'weight_decay': 0
    }
]

def main():
    # Run all experiments
    results = []
    for config in configs:
        result = run_experiment(config)
        results.append(result)
    
    # Generate comparative plots
    create_comparison_visualizations(results)
    
    # Save summary to file
    with open("experiment_results/summary.txt", "w") as f:
        f.write("Experimental Analysis Summary\n")
        f.write("===========================\n\n")
        
        for result in results:
            f.write(f"Experiment: {result['name']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  F1 Score: {result['f1']:.4f}\n")
            f.write(f"  Final Training Loss: {result['final_train_loss']:.4f}\n")
            f.write(f"  Final Validation Loss: {result['final_val_loss']:.4f}\n")
            f.write(f"  Runtime: {result['runtime']:.2f} seconds\n\n")
    
    print("\nAll experiments completed! Results saved to experiment_results/")

def create_comparison_visualizations(results):
    """Create comparative visualizations for all experiments"""
    
    # Group results by experiment type
    experiment_types = {
        'normalization': ['with_normalization', 'without_normalization'],
        'batch_size': ['batch_size_16', 'batch_size_64'],
        'learning_rate': ['learning_rate_0.0001', 'learning_rate_0.01'],
        'optimizer': ['optimizer_sgd', 'optimizer_adam', 'optimizer_rmsprop', 'optimizer_momentum'],
        'initialization': ['init_xavier', 'init_random'],
        'regularization': ['with_l2_regularization', 'without_l2_regularization']
    }
    
    # For each experiment type, create comparison bar charts
    for exp_type, exp_names in experiment_types.items():
        # Filter results for this experiment type
        exp_results = [r for r in results if r['name'] in exp_names]
        
        if not exp_results:
            continue
            
        # Create accuracy comparison
        plt.figure(figsize=(10, 6))
        names = [r['name'] for r in exp_results]
        accuracies = [r['accuracy'] for r in exp_results]
        
        plt.bar(names, accuracies)
        plt.title(f"Accuracy Comparison: {exp_type}")
        plt.xlabel("Configuration")
        plt.ylabel("Accuracy")
        plt.ylim(0.8, 1.0)  # Assuming accuracy is high
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"experiment_results/{exp_type}_accuracy.png")
        plt.close()
        
        # Create loss comparison
        plt.figure(figsize=(10, 6))
        names = [r['name'] for r in exp_results]
        losses = [r['final_val_loss'] for r in exp_results]
        
        plt.bar(names, losses)
        plt.title(f"Validation Loss Comparison: {exp_type}")
        plt.xlabel("Configuration")
        plt.ylabel("Validation Loss")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"experiment_results/{exp_type}_loss.png")
        plt.close()

if __name__ == "__main__":
    main()