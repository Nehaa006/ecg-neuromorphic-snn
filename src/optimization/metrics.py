import torch


def calculate_flops_vs_sops(model: torch.nn.Module, avg_firing_rate: float = 0.154, 
                             num_timesteps: int = 3, mac_energy_pj: float = 4.6, ac_energy_pj: float = 0.9):
    """
    Calculates theoretical computational energy reduction:
    - Standard CNN/LSTM FLOP energy (using Multiply-Accumulate / MAC @ 4.6 pJ)
    - Neuromorphic SNN SOP energy (using Accumulate / AC @ 0.9 pJ triggered on spikes)
    """
    total_synapses = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Count active non-zero synapses
            total_synapses += torch.sum(param != 0).item()

    # Standard Artificial Neural Network (Dense MACs per beat)
    dense_mac_ops = total_synapses
    ann_energy_pj = dense_mac_ops * mac_energy_pj

    # Neuromorphic Spiking Neural Network (Sparse ACs per spike event)
    snn_sops = total_synapses * avg_firing_rate * num_timesteps
    snn_energy_pj = snn_sops * ac_energy_pj

    energy_reduction_factor = ann_energy_pj / snn_energy_pj if snn_energy_pj > 0 else 0.0

    print("=== Theoretical Energy Efficiency Benchmark ===")
    print(f"Active Synaptic Connections: {total_synapses:,}")
    print(f"Standard ANN Compute Energy per Beat: {ann_energy_pj / 1e3:.2f} nJ")
    print(f"Event-Driven SNN Compute Energy per Beat: {snn_energy_pj / 1e3:.2f} nJ")
    print(f"Estimated Software Energy Savings: {energy_reduction_factor:.1f}x Reduction")

    return {
        "active_synapses": total_synapses,
        "ann_energy_nj": ann_energy_pj / 1e3,
        "snn_energy_nj": snn_energy_pj / 1e3,
        "energy_savings_ratio": energy_reduction_factor
    }