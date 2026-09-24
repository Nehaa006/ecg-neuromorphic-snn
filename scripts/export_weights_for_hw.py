import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.models.rmlp_snn import RecurrentMLPSNN


def generate_weight_distribution_plots(save_path="outputs/plots/weight_distribution.png"):
    # 1. Load Model Weights
    model = RecurrentMLPSNN().to(config.DEVICE)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "baseline_rmlp_snn.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[+] Loaded model checkpoint from: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint not found at {checkpoint_path}. Plotting current weights.")

    # 2. Extract Floating-Point Weights (FP32)
    fp32_weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            fp32_weights.append(param.detach().cpu().numpy().flatten())

    fp32_weights = np.concatenate(fp32_weights)

    # 3. Compute 6-Bit Quantization Parameters & Integer Grid
    qmin, qmax = -32, 31
    max_abs_val = np.max(np.abs(fp32_weights))
    scale = max_abs_val / qmax if max_abs_val > 0 else 1.0

    # Quantize to discrete 6-bit integers
    int6_weights = np.clip(np.round(fp32_weights / scale), qmin, qmax).astype(np.int32)
    # Dequantized continuous positions for plotting alignment
    dequant_weights = int6_weights * scale

    # 4. Plot Distributions Side-by-Side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    # Focus x-limits around the dense central weight region
    xlim_bounds = [-0.25, 0.25]

    # FP32 Baseline (Continuous density histogram)
    ax1.hist(fp32_weights, bins=100, color='#4B4BB2', edgecolor='none', alpha=0.85)
    ax1.set_title("FP32 Baseline Weight Distribution\n(Continuous Floating Point)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("Weight Value", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Parameter Count", fontsize=10, fontweight='bold')
    ax1.set_xlim(xlim_bounds)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # 6-Bit Quantized (Discrete integer binning)
    # Match bins directly to 6-bit dequantized step positions
    unique_levels = np.sort(np.unique(dequant_weights))
    step_size = scale
    bin_edges = np.append(unique_levels - step_size / 2.0, unique_levels[-1] + step_size / 2.0)

    ax2.hist(dequant_weights, bins=bin_edges, color='#E64C65', edgecolor='#801b2a', lw=0.5, alpha=0.85)
    ax2.set_title("6-Bit Quantized Weight Discretization\n(64 Discrete Integer Bins)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("Quantized Weight Value", fontsize=10, fontweight='bold')
    ax2.set_xlim(xlim_bounds)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Add Text Annotations
    ax1.text(0.03, 0.92, f"Total Weights: {len(fp32_weights):,}\nPrec: 32-bit Float", 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.text(0.03, 0.92, f"Scale (Δ): {scale:.5f}\nPrec: 6-bit Int [-32, 31]", 
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. Save Artifact
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved High-Resolution Weight Distribution Plot to: {save_path}")


if __name__ == "__main__":
    generate_weight_distribution_plots()