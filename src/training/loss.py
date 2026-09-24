import torch
import torch.nn as nn
import config

class NeuromorphicLoss(nn.Module):
    def __init__(self):
        super(NeuromorphicLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, spike_record, model):
        # 1. Classification Loss (Cross Entropy)
        clf_loss = self.ce_loss(logits, targets)

        # 2. Sparsity Loss (Target Firing Rate Regularization)
        if spike_record is not None:
            if isinstance(spike_record, torch.Tensor):
                avg_firing_rate = spike_record.mean()
            elif isinstance(spike_record, (list, tuple)) and len(spike_record) > 0:
                avg_firing_rate = torch.stack(spike_record).mean()
            else:
                avg_firing_rate = torch.tensor(0.0, device=logits.device)
            
            sparsity_loss = torch.square(avg_firing_rate - config.TARGET_FIRING_RATE)
        else:
            sparsity_loss = torch.tensor(0.0, device=logits.device)

        # 3. L1 Weight Decay Loss (Synaptic Pruning Encouragement)
        l1_loss = sum(torch.norm(p, 1) for p in model.parameters() if p.requires_grad)

        # Total Triple-Objective Loss
        total_loss = clf_loss + (config.LAMBDA_SPARSITY * sparsity_loss) + (config.LAMBDA_WEIGHT_DECAY * l1_loss)
        return total_loss