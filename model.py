import torch
import torch.nn as nn

# Neural network definition
# -------------------------
# Define a 5-layer neural network with ReLU activation
# and Xavier/Glorot weight initialization.
# -------------------------

class DeepNN(nn.Module):
    """
    Deep Neural Network with 5 layers (4 hidden + output)
    """
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