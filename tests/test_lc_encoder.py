import pytest
import torch
from src.preprocessing.lc_encoder import level_crossing_encode

def test_lc_encoder_shape_and_binary_output():
    # Generate dummy ECG signal batch (batch_size=4, sequence_len=180)
    dummy_signal = torch.sin(torch.linspace(0, 3.14 * 4, 180)).unsqueeze(0).repeat(4, 1)
    
    threshold = 0.1
    spikes = level_crossing_encode(dummy_signal, threshold=threshold)
    
    # Assert tensor shapes match expected time-step framing
    assert spikes.dim() == 3, "Output should be a 3D tensor (Batch, Time, Channels)"
    assert spikes.shape[0] == 4, "Batch dimension mismatch"
    
    # Assert spiking output is binary (0 or 1)
    unique_vals = torch.unique(spikes)
    assert all(val in [0.0, 1.0] for val in unique_vals), "Spikes must be strictly binary (0 or 1)"