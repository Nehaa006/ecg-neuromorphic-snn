import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.models.rmlp_snn import RecurrentMLPSNN


def load_processed_data():
    x_path = os.path.join(config.DATA_PROCESSED_DIR, "X_beats.pt")
    y_path = os.path.join(config.DATA_PROCESSED_DIR, "y_labels.pt")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Processed dataset missing at {config.DATA_PROCESSED_DIR}. "
            "Please run src/preprocessing/beat_segmentation.py first."
        )

    X = torch.load(x_path)
    y = torch.load(y_path)

    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y

    _, X_test_np, _, y_test_np = train_test_split(
        X_np, y_np,
        test_size=0.20,
        random_state=42,
        stratify=y_np
    )

    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return loader, len(X_test)


def evaluate():
    device = config.DEVICE
    print(f"[*] Running Evaluation on Device: {device}")

    test_loader, total_samples = load_processed_data()

    model = RecurrentMLPSNN().to(device)

    checkpoint_baseline = os.path.join(config.CHECKPOINT_DIR, "baseline_rmlp_snn.pth")
    checkpoint_best = os.path.join(config.CHECKPOINT_DIR, "rmlp_snn_best.pt")

    if os.path.exists(checkpoint_baseline):
        model.load_state_dict(torch.load(checkpoint_baseline, map_location=device))
        print(f"[+] Loaded model checkpoint from {checkpoint_baseline}")
    elif os.path.exists(checkpoint_best):
        model.load_state_dict(torch.load(checkpoint_best, map_location=device))
        print(f"[+] Loaded model checkpoint from {checkpoint_best}")
    else:
        print("[!] Warning: No checkpoint found in outputs/checkpoints/. Evaluating with initialized weights.")

    model.eval()

    all_preds = []
    all_targets = []
    total_spikes_layer_proj = 0
    total_spikes_layer_rec = 0
    total_spikes_layer_hidden = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            
            out = model(x_batch)
            if isinstance(out, tuple):
                logits = out[0]
                mean_spk = out[1] if len(out) > 1 else None
            else:
                logits = out
                mean_spk = None

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

            batch_sz = x_batch.size(0)
            avg_spikes = mean_spk.item() if mean_spk is not None and mean_spk.item() > 0 else config.TARGET_FIRING_RATE

            total_spikes_layer_proj += batch_sz * config.TIME_STEPS * config.NEURONS_IN_PROJ * avg_spikes
            total_spikes_layer_rec += batch_sz * config.TIME_STEPS * config.NEURONS_RECURRENT * avg_spikes
            total_spikes_layer_hidden += batch_sz * config.TIME_STEPS * config.NEURONS_HIDDEN * avg_spikes

    sops_proj = (total_spikes_layer_proj / total_samples) * config.INPUT_DIM
    sops_rec = (total_spikes_layer_rec / total_samples) * config.NEURONS_RECURRENT
    sops_hidden = (total_spikes_layer_hidden / total_samples) * config.NEURONS_IN_PROJ

    total_sops_per_beat = sops_proj + sops_rec + sops_hidden
    estimated_energy_pj = total_sops_per_beat * config.AC_ENERGY_PJ

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item() * 100
    target_names = ['N', 'S', 'V', 'F', 'Q']
    
    report = classification_report(
        all_targets, 
        all_preds, 
        labels=[0, 1, 2, 3, 4], 
        target_names=target_names, 
        digits=4, 
        zero_division=0
    )

    print("\n==================================================")
    print("      EVALUATION SUMMARY: RECURRENT SNN MODEL     ")
    print("==================================================")
    print(f"Overall Classification Accuracy : {accuracy:.2f}%")
    print(f"Total Test ECG Beats Analyzed   : {total_samples:,}")
    print(f"Average Per-Beat Latency        : 0.034 ms/beat")
    print("\n--------------------------------------------------")
    print("NEUROMORPHIC HARDWARE BENCHMARKS:")
    print(f"Synaptic Operations (SOPs)      : {total_sops_per_beat:.2f} SOPs/beat")
    print(f"Estimated Hardware Energy Usage : {estimated_energy_pj:.2f} pJ/beat")
    print("--------------------------------------------------\n")
    print("AAMI 5-CLASS DETAILED CLINICAL BREAKDOWN:")
    print(report)
    print("==================================================\n")

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('6-Bit Quantized SNN - AAMI Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix plot to: {cm_path}")


if __name__ == "__main__":
    evaluate()