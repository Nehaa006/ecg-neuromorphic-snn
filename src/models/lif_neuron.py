import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model.
    Accumulates input charge and decays membrane potential over time.
    """
    def __init__(self, beta: float = 0.85, threshold: float = 1.0):
        super(LIFNeuron, self).__init__()
        self.beta = beta            # Membrane potential decay rate
        self.threshold = threshold  # Firing voltage threshold

    def forward(self, current_input: torch.Tensor, membrane_potential: torch.Tensor):
        # Update membrane voltage: Decay previous state + add incoming synaptic current
        updated_membrane = self.beta * membrane_potential + current_input
        
        # Generate binary spike (1.0 if threshold exceeded, else 0.0)
        spike = (updated_membrane >= self.threshold).float()
        
        # Reset membrane potential on firing
        reset_membrane = updated_membrane * (1.0 - spike)
        
        return spike, reset_membrane