"""
Unit tests for model architecture
"""
import sys
sys.path.append('../src')

import torch
from models.mobilenet_classifier import create_model


def test_model_creation():
    """Test model can be created"""
    model = create_model(num_classes=7, pretrained=False, device='cpu')
    assert model is not None
    print("✓ Model creation test passed")


def test_forward_pass():
    """Test forward pass works"""
    model = create_model(num_classes=7, pretrained=False, device='cpu')
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 1, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (2, 7), f"Expected (2, 7), got {output.shape}"
    print("✓ Forward pass test passed")


def test_model_parameters():
    """Test model has reasonable number of parameters"""
    model = create_model(num_classes=7, pretrained=False, device='cpu')
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # MobileNetV3-Small should have ~2-3M parameters
    assert 1_000_000 < total_params < 5_000_000, \
        f"Unexpected parameter count: {total_params:,}"
    
    print(f"✓ Model parameters test passed ({total_params:,} params)")


def test_grayscale_input():
    """Test model accepts grayscale input"""
    model = create_model(num_classes=7, pretrained=False, device='cpu')
    model.eval()
    
    # Grayscale input (1 channel)
    x = torch.randn(1, 1, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (1, 7)
    print("✓ Grayscale input test passed")


if __name__ == "__main__":
    print("Running model tests...\n")
    
    test_model_creation()
    test_forward_pass()
    test_model_parameters()
    test_grayscale_input()
    
    print("\n✓ All tests passed!")
