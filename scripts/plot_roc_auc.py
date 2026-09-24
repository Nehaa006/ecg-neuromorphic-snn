import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.models.rmlp_snn import RecurrentMLPSNN


def generate_roc_curves(save_path="outputs/plots/roc_auc_curves.png"):
    # 1. Load Held-Out Test Data ONLY
    x_path = os.path.join(config.DATA_PROCESSED_DIR, "X_test.pt")
    y_path = os.path.join(config.DATA_PROCESSED_DIR, "y_test.pt")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Processed test data not found at {x_path} or {y_path}. Run split_dataset.py first.")

    X_test = torch.load(x_path)
    y_test = torch.load(y_path)
    dataset = TensorDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Load Model Checkpoint
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
        print(f"[!] Warning: Checkpoint not found at {checkpoint_path}. Running with current initialized weights.")

    model.eval()

    all_probs = []
    all_targets = []

    # 3. Test Set Inference & Softmax Probability Scaling
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(config.DEVICE)

            # Forward pass
            out = model(X_batch)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            # Average over sequence time dimension T if shape is [T, Batch, Classes]
            if logits.dim() == 3:
                logits = logits.mean(dim=0)

            # Compute calibrated probability distribution
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)

    # 4. Multi-Class One-vs-Rest Binarization
    n_classes = 5
    class_names = ['N (Normal)', 'S (SVEB)', 'V (VEB)', 'F (Fusion)', 'Q (Paced)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    y_true_onehot = label_binarize(all_targets, classes=list(range(n_classes)))

    # 5. Plot ROC Curves
    plt.figure(figsize=(8.5, 6.5))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc:.4f})'
        )

    # Diagonal reference line for random guess
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance Level (AUC = 0.5000)')

    # Styling
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=10, fontweight='bold')
    plt.title('AAMI 5-Class One-vs-Rest ROC Curves\n(Quantized Recurrent SNN - Held-Out Test Set)', fontsize=11, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9.5, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 6. Save Artifact
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Test Set ROC-AUC Plot successfully to: {save_path}")


if __name__ == "__main__":
    generate_roc_curves()