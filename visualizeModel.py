import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def create_neural_network_visualization(input_size=3, hidden_sizes=[64, 32, 16, 8], output_size=4, 
                                       figsize=(12, 9), neuron_colors=['#FFC947', '#0096FF', '#9D65C9', '#FF5757']):
    """
    Create a visualization of a neural network with the specified architecture.
    
    Args:
        input_size: Number of input neurons
        hidden_sizes: List of sizes for hidden layers
        output_size: Number of output neurons
        figsize: Figure size
        neuron_colors: Colors for different types of layers
    """
    # Setup layer sizes
    all_layers = [input_size] + hidden_sizes + [output_size]
    n_layers = len(all_layers)
    
    # Setup figure with increased height
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n_layers)
    max_neurons = max(all_layers)
    # Increase the bottom padding to make room for ReLU annotations
    ax.set_ylim(-max_neurons/2 - 2.0, max_neurons/2 + 0.8)
    
    # Turn off axis
    ax.axis('off')
    
    # Layer labels
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_sizes))] + ['Output']
    features = ['Col2', 'Col3', 'Col4']
    classes = ['0', '1', '2', '#']
    
    # Draw layers
    for i, (layer_size, layer_name) in enumerate(zip(all_layers, layer_names)):
        # Calculate y-coordinates for neurons in this layer
        y_positions = np.linspace(-layer_size/2 + 0.5, layer_size/2 - 0.5, layer_size)
        
        # Add layer annotation with more space at the top
        ax.text(i, max_neurons/2 + 0.5, layer_name, ha='center', va='bottom', fontsize=14)
        
        # Draw each neuron
        for j, y in enumerate(y_positions):
            # Choose color based on layer
            if i == 0:  # Input layer
                color = neuron_colors[0]
            elif i == n_layers - 1:  # Output layer
                color = neuron_colors[3]
            else:  # Hidden layers
                color = neuron_colors[1] if i % 2 == 0 else neuron_colors[2]
            
            # Draw neuron
            circle = plt.Circle((i, y), radius=0.2, fill=True, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            # Add feature/class labels to input/output layers
            if i == 0:  # Input layer
                ax.text(i - 0.3, y, features[j], ha='right', va='center', fontsize=12)
            elif i == n_layers - 1:  # Output layer
                ax.text(i + 0.3, y, classes[j], ha='left', va='center', fontsize=12)
            
            # Add size annotations to larger hidden layers
            if i > 0 and i < n_layers - 1 and layer_size > 10:
                if j == 0:
                    ax.text(i, y_positions[0] - 0.5, f"{layer_size} neurons", 
                           ha='center', va='top', fontsize=10)
    
    # Draw connections between layers
    for i in range(n_layers - 1):
        # Get positions of neurons in current and next layer
        layer_y = np.linspace(-all_layers[i]/2 + 0.5, all_layers[i]/2 - 0.5, all_layers[i])
        next_layer_y = np.linspace(-all_layers[i+1]/2 + 0.5, all_layers[i+1]/2 - 0.5, all_layers[i+1])
        
        # Skip drawing all connections for large layers (just draw a sample)
        if all_layers[i] * all_layers[i+1] > 100:  # Too many connections
            for j in range(all_layers[i]):
                # Draw a few sample connections per neuron
                samples = np.random.choice(all_layers[i+1], size=min(3, all_layers[i+1]), replace=False)
                for k in samples:
                    ax.add_artist(Line2D([i, i+1], [layer_y[j], next_layer_y[k]], color='gray', alpha=0.3, zorder=1))
        else:
            # Draw all connections for smaller layers
            for j in range(all_layers[i]):
                for k in range(all_layers[i+1]):
                    ax.add_artist(Line2D([i, i+1], [layer_y[j], next_layer_y[k]], color='gray', alpha=0.3, zorder=1))
    
    # Add ReLU activation annotations between hidden layers - moved lower
    for i in range(1, n_layers-1):
        # Move the ReLU text much lower to avoid overlap
        ax.text(i, -max_neurons/2 - 1.5, "ReLU", ha='center', va='top', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Add architecture summary as title
    plt.title(f"5-Layer Neural Network Architecture\nInput: {input_size} → Hidden: {hidden_sizes} → Output: {output_size}", 
              fontsize=16, pad=20)
    
    # Add Xavier/Glorot initialization note
    plt.figtext(0.5, 0.01, "All weights initialized using Xavier/Glorot uniform initialization", 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig("neural_network_architecture.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualization saved as 'neural_network_architecture.png'")

if __name__ == "__main__":
    # Create visualization with the same parameters as the model in prepare_dataset.py
    create_neural_network_visualization(
        input_size=3,           # 3 input features (Col2, Col3, Col4)
        hidden_sizes=[64, 32, 16, 8],  # 4 hidden layers with these sizes
        output_size=4           # 4 output classes (0, 1, 2, and # mapped to 3)
    )