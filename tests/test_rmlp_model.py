import pytest
import torch
import config
from src.models.rmlp_snn import RecurrentMLPSNN

def test_rmlp_snn_forward_pass():
    model = RecurrentMLPSNN()
    model.eval()
    
    # Mock input batch matching (Batch_Size, Time_Steps, Input_Dim)
    batch_size = 2
    mock_input = torch.randn(batch_size, config.TIME_STEPS, config.INPUT_DIM)
    
    with torch.no_grad():
        logits, mean_spikes = model(mock_input)
        
    # Check output shape matches class count
    assert logits.shape == (batch_size, config.NUM_CLASSES), f"Expected shape ({batch_size}, {config.NUM_CLASSES}), got {logits.shape}"
    
    # Check firing rate metric falls in a valid percentage range [0.0, 1.0]
    assert 0.0 <= mean_spikes.item() <= 1.0, "Mean spike rate must be between 0 and 1"