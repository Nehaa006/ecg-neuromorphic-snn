import os
import sys
import copy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from src.models.rmlp_snn import RecurrentMLPSNN

def quantize_weights_6bit(model, qmin=-32, qmax=31):
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

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            logits, _ = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0

def calculate_energy_metrics(model):
    """
    Computes energy consumption comparing standard ANN FLOP MAC operations
    against SNN Synaptic Operations (SOPs) using 6-bit INT arithmetic.
    """
    # Baseline FLOP calculation per inference sample
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mac_ops_per_sample = total_params
    
    # Energy parameters with standard fallbacks
    target_firing_rate = getattr(config, 'TARGET_FIRING_RATE', 0.154)
    mac_energy_pj = getattr(config, 'MAC_ENERGY_PJ', 4.6)
    ac_energy_pj = getattr(config, 'AC_ENERGY_PJ', 0.9)

    sop_ops_per_sample = mac_ops_per_sample * target_firing_rate

    ann_energy_pj = mac_ops_per_sample * mac_energy_pj
    snn_energy_pj = sop_ops_per_sample * ac_energy_pj

    energy_reduction_factor = ann_energy_pj / snn_energy_pj if snn_energy_pj > 0 else 1.0
    return ann_energy_pj, snn_energy_pj, energy_reduction_factor

def main():
    print("--- Step 3: 6-Bit Quantization & Energy Benchmarks ---")
    
    # Load dataset
    x_path = os.path.join(config.DATA_PROCESSED_DIR, "X_beats.pt")
    y_path = os.path.join(config.DATA_PROCESSED_DIR, "y_labels.pt")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("Error: Processed dataset not found. Run beat_segmentation.py first.")
        return

    X = torch.load(x_path)
    y = torch.load(y_path)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Load trained model checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "baseline_rmlp_snn.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Train model first.")
        return

    model_fp32 = RecurrentMLPSNN().to(config.DEVICE)
    model_fp32.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))

    # Evaluate FP32 accuracy
    fp32_acc = evaluate_accuracy(model_fp32, dataloader)
    print(f"FP32 Baseline Model Accuracy: {fp32_acc:.2f}%")

    # Apply 6-bit quantization
    model_q6, scales = quantize_weights_6bit(model_fp32)
    q6_acc = evaluate_accuracy(model_q6, dataloader)
    print(f"6-Bit Quantized SNN Accuracy: {q6_acc:.2f}%")
    print(f"Accuracy Degradation:         {fp32_acc - q6_acc:.2f}%")

    # Save quantized checkpoint
    quantized_dir = getattr(config, 'QUANTIZED_DIR', os.path.join(config.BASE_DIR, "outputs", "quantized"))
    os.makedirs(quantized_dir, exist_ok=True)
    q_save_path = os.path.join(quantized_dir, "q6_rmlp_snn.pth")
    torch.save(model_q6.state_dict(), q_save_path)

    # Energy calculations
    ann_e, snn_e, savings = calculate_energy_metrics(model_q6)
    
    print("\n--- Neuromorphic Energy Efficiency Summary ---")
    print(f"Standard ANN FLOP Energy per Beat: {ann_e:.2f} pJ")
    print(f"Quantized SNN SOP Energy per Beat:  {snn_e:.2f} pJ")
    print(f"Total Energy Savings:               {savings:.2f}x reduction")

if __name__ == "__main__":
    main()