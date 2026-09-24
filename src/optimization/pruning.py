import torch
import torch.nn as nn


def prune_synaptic_weights(model: nn.Module, threshold: float = 1e-2):
    """
    Prunes dead weights below threshold, reducing active synaptic connections.
    """
    total_weights = 0
    pruned_weights = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = torch.abs(param.data) >= threshold
            param.data *= mask.float()
            
            total_weights += param.numel()
            pruned_weights += torch.sum(~mask).item()

    pruning_ratio = (pruned_weights / total_weights) * 100 if total_weights > 0 else 0
    print(f"Synaptic Pruning: Removed {pruned_weights}/{total_weights} weights ({pruning_ratio:.2f}% reduction)")
    return model