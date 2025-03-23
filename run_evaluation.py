import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Import components from our project
from data_preprocessing import load_and_combine_data, preprocess_data, split_data
from model import DeepNN
from utils import create_dataloaders
from evaluate_model import evaluate_model, plot_confusion_matrix

def main():
    """Load trained model and evaluate its performance"""
    print("Loading trained model...")
    
    # Load the checkpoint
    checkpoint = torch.load("model_checkpoint.pt", weights_only=False)
    
    # Recreate model with same architecture
    input_size = 3  # 3 features (Col2, Col3, Col4)
    hidden_sizes = checkpoint['hidden_sizes']
    num_classes = checkpoint['num_classes']
    
    model = DeepNN(input_size, hidden_sizes, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset for evaluation (same process as in main.py)
    print("\nLoading and preprocessing dataset...")
    data_directory = "eel4810-dataset"
    df = load_and_combine_data(data_directory)
    X, y, _ = preprocess_data(df)  # We don't need scaler for evaluation
    
    # Use the same split as in training
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.1)
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader)
    
    # Display results
    print("\n===== Model Performance =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Plot confusion matrix
    class_names = ['0', '1', '2', '#']
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # Print detailed classification report
    print("\n===== Classification Report =====")
    # Determine actual labels present in the data
    unique_labels = sorted(set(metrics['true_labels']))
    used_class_names = [class_names[i] for i in unique_labels]

    class_report = classification_report(
        metrics['true_labels'], 
        metrics['predictions'],
        target_names=used_class_names
    )
    print(class_report)
    
    # Save report to file
    with open("classification_report.txt", "w") as f:
        f.write("===== Model Performance =====\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n\n")
        f.write("===== Classification Report =====\n")
        f.write(class_report)
    
    print("\nEvaluation complete! Results saved to:")
    print("- confusion_matrix.png")
    print("- classification_report.txt")

if __name__ == "__main__":
    main()