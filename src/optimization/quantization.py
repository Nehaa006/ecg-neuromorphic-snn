import copy
import torch
import config

def quantize_weights_6bit(
    model, 
    qmin=getattr(config, 'QUANT_MIN', -32), 
    qmax=getattr(config, 'QUANT_MAX', 31)
):
    """
    Applies uniform 6-bit symmetric quantization to all linear layer weights in the model.
    Scales weights into the range [-32, 31].
    """
    model_q = copy.deepcopy(model)
    scales = {}

    with torch.no_grad():
        for name, param in model_q.named_parameters():
            if "weight" in name:
                max_val = torch.max(torch.abs(param.data))
                scale = max_val / qmax if max_val > 0 else 1.0
                
                # Uniform quantization & clamping to 6-bit integer bounds
                q_weights = torch.clamp(torch.round(param.data / scale), qmin, qmax)
                
                # De-quantize weights back for evaluation simulation
                param.copy_(q_weights * scale)
                scales[name] = scale

    return model_q, scales