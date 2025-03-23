import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, init_weights='xavier'):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of sizes for hidden layers
            num_classes: Number of output classes
            init_weights: Weight initialization method ('xavier' or 'random')
        """
        super(DeepNN, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], num_classes)
        
        # Initialize weights
        if init_weights == 'xavier':
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.xavier_uniform_(self.fc4.weight)
            nn.init.xavier_uniform_(self.fc5.weight)
        else:  # random initialization
            # Random normal initialization
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.fc3.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.fc4.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.fc5.weight, mean=0.0, std=0.1)
        
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
        nn.init.zeros_(self.fc5.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # No activation on the output layer
        return x