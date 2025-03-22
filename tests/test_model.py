import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from model import DeepNN

def test_model_structure():
    """Test that the model has the correct number of layers"""
    print("Testing model structure...")
    
    # Create model with the specifications from HW2
    input_size = 3  # Three features: Col2, Col3, Col4
    hidden_sizes = [64, 32, 16, 8]  # Four hidden layers
    num_classes = 4  # Four classes: 0, 1, 2, # (mapped to 3)
    
    model = DeepNN(input_size, hidden_sizes, num_classes)
    
    # Print model architecture for visual inspection
    print(model)
    
    # Check the number of layers using the model's attributes directly
    layers = [model.fc1, model.fc2, model.fc3, model.fc4, model.fc5]
    print(f"Number of linear layers: {len(layers)}")
    assert len(layers) == 5, f"Expected 5 linear layers, got {len(layers)}"
    print("✓ Model has correct number of layers")
    
    # Check dimensions directly using each layer's attributes
    expected_dimensions = [
        (3, 64),    # input → first hidden
        (64, 32),   # first hidden → second hidden
        (32, 16),   # second hidden → third hidden
        (16, 8),    # third hidden → fourth hidden
        (8, 4)      # fourth hidden → output
    ]
    
    for i, layer in enumerate(layers):
        in_features = layer.in_features
        out_features = layer.out_features
        expected = expected_dimensions[i]
        print(f"Layer {i+1}: {in_features} → {out_features}")
        assert in_features == expected[0] and out_features == expected[1], \
            f"Layer {i+1} has shape ({in_features}, {out_features}), expected {expected}"
    
    print("✓ All layer dimensions are correct")
    return True

def test_forward_pass():
    """Test that forward pass works correctly"""
    print("\nTesting forward pass...")
    
    # Create a model
    model = DeepNN(3, [64, 32, 16, 8], 4)
    
    # Create a batch of sample inputs
    batch_size = 5
    sample_input = torch.rand(batch_size, 3)
    
    # Run forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    # Check output shape
    expected_shape = (batch_size, 4)
    print(f"Output shape: {output.shape}")
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, got {output.shape}"
    print("✓ Forward pass produces correct output shape")
    
    # Check that output contains valid logits (no NaNs or infinities)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    print("✓ Output contains valid numerical values")
    
    return True

def test_weight_initialization():
    """Test that weight initialization works correctly"""
    print("\nTesting weight initialization...")
    
    # Create two models with the same architecture
    model1 = DeepNN(3, [64, 32, 16, 8], 4)
    model2 = DeepNN(3, [64, 32, 16, 8], 4)
    
    # Check that weights are initialized to different values
    # (Xavier initialization should produce different random values each time)
    different_weights = False
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(param1, param2):
            different_weights = True
            break
    
    assert different_weights, "Models have identical weights, initialization may not be working"
    print("✓ Weight initialization produces different values each time")
    
    # Check that weights follow expected distribution for Xavier initialization
    # Xavier uniform should have values distributed between -a and a where a = sqrt(6 / (fan_in + fan_out))
    for name, param in model1.named_parameters():
        if 'weight' in name:
            fan_in = param.size(1)
            fan_out = param.size(0)
            a = np.sqrt(6.0 / (fan_in + fan_out))
            
            # Check that all weights are within the expected range
            assert param.min() >= -a, f"Minimum weight {param.min()} is less than lower bound {-a}"
            assert param.max() <= a, f"Maximum weight {param.max()} is greater than upper bound {a}"
    
    print("✓ Weight values are within expected range for Xavier initialization")
    return True

if __name__ == "__main__":
    test_model_structure()
    test_forward_pass()
    test_weight_initialization()
    print("\nAll model tests passed!")