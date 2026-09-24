import pytest
import torch
import snntorch as snn
from snntorch import surrogate

def test_lif_neuron_spiking_threshold():
    beta = 0.8
    spike_grad = surrogate.fast_sigmoid(slope=25)
    lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
    
    mem = lif.init_leaky()
    
    # Below threshold input (should NOT spike)
    sub_thresh_input = torch.tensor([[0.5]])
    spk, mem = lif(sub_thresh_input, mem)
    assert spk.item() == 0.0, "Neuron should not spike below threshold"
    
    # Strong input above threshold (MUST spike)
    supra_thresh_input = torch.tensor([[2.0]])
    spk, mem = lif(supra_thresh_input, mem)
    assert spk.item() == 1.0, "Neuron must emit a spike when membrane potential exceeds threshold"