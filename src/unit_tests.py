import torch
import numpy
from utils.globals import *

def test_forward_pass(model: torch.nn.Module):
    dummy_input = torch.randn(1, 1, 28, 28)
    output, _ = model(dummy_input)
    assert output.shape == (1, NUM_CLASSES), "Forward pass output shape mismatch."


def test_noise_injection(model: torch.nn.Module):
    noise_dict = {
        'fc1': torch.randn(1, HS1),
        'fc2': torch.randn(1, HS2),
        'fc3': torch.randn(1, NUM_CLASSES)
    }
    dummy_input = torch.randn(1, 1, 28, 28)
    output, _ = model(dummy_input, noise_dict)
    assert output.shape == (1, NUM_CLASSES), "Noisy forward pass output shape mismatch."


def test_edge_cases(model: torch.nn.Module):
    # Test with empty input
    try:
        dummy_input = torch.empty(0, 1, 28, 28)
        model(dummy_input)
    except Exception as e:
        print("Passed empty input test with exception:", e)

    # Test with extreme noise values
    noise_dict = {
        'fc1': torch.full((1, HS1), 1e6),
        'fc2': torch.full((1, HS2), 1e6),
        'fc3': torch.full((1, NUM_CLASSES), 1e6)
    }
    dummy_input = torch.randn(1, 1, 28, 28)
    try:
        output, _ = model(dummy_input, noise_dict)
        print("Extreme noise test passed.")
    except Exception as e:
        print("Failed extreme noise test:", e)