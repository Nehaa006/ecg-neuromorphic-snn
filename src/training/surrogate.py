import torch


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient function for backward pass through non-differentiable spiking thresholds.
    """
    @staticmethod
    def forward(ctx, input_tensor, threshold=1.0, slope=25.0):
        ctx.save_for_backward(input_tensor)
        ctx.threshold = threshold
        ctx.slope = slope
        return (input_tensor >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # Derivative of Fast Sigmoid: 1 / (1 + slope * |v - thr|)^2
        sg_grad = 1.0 / (1.0 + ctx.slope * torch.abs(input_tensor - ctx.threshold)) ** 2
        return grad_input * sg_grad, None, None


def spike_function(v, threshold=1.0, slope=25.0):
    return FastSigmoidSurrogate.apply(v, threshold, slope)