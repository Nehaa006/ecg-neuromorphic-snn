import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import config

class RecurrentMLPSNN(nn.Module):
    def __init__(self):
        super(RecurrentMLPSNN, self).__init__()
        
        # Steep surrogate gradient to kickstart backpropagation
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.in_proj = nn.Linear(config.INPUT_DIM, config.NEURONS_IN_PROJ)
        self.rec_layer = nn.Linear(config.NEURONS_IN_PROJ + config.NEURONS_RECURRENT, config.NEURONS_RECURRENT)
        self.hidden_layer = nn.Linear(config.NEURONS_RECURRENT, config.NEURONS_HIDDEN)
        self.output_layer = nn.Linear(config.NEURONS_HIDDEN, config.NUM_CLASSES)
        
        self.lif_rec = snn.Leaky(beta=config.LIF_BETA, spike_grad=spike_grad, init_hidden=False)
        self.lif_hidden = snn.Leaky(beta=config.LIF_BETA, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x):
        batch_size = x.size(0)
        
        mem_rec = self.lif_rec.init_leaky()
        mem_hidden = self.lif_hidden.init_leaky()
        
        spk_rec = torch.zeros(batch_size, config.NEURONS_RECURRENT, device=x.device)
        logits_acc = torch.zeros(batch_size, config.NUM_CLASSES, device=x.device)
        spike_counts = []
        
        for t in range(config.TIME_STEPS):
            x_t = x[:, t, :]
            in_proj_out = self.in_proj(x_t)
            
            # Recurrent LIF dynamics
            rec_input = torch.cat([in_proj_out, spk_rec], dim=1)
            rec_h = self.rec_layer(rec_input)
            spk_rec, mem_rec = self.lif_rec(rec_h, mem_rec)
            
            # Hidden Layer LIF dynamics
            h_out = self.hidden_layer(spk_rec)
            spk_h, mem_hidden = self.lif_hidden(h_out, mem_hidden)
            
            # Direct evidence accumulation across time steps
            out_logits = self.output_layer(mem_hidden)
            logits_acc += out_logits
            
            spike_counts.append(spk_rec.mean())

        mean_spikes = torch.stack(spike_counts).mean()
        return logits_acc, mean_spikes