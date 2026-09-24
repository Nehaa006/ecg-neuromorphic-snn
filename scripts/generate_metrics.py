import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.models.rmlp_snn import RecurrentMLPSNN

os.makedirs(config.PLOTS_DIR, exist_ok=True)

# 1. Generate Confusion Matrix Plot
X = torch.load(os.path.join(config.DATA_PROCESSED_DIR, "X_beats.pt"))
y = torch.load(os.path.join(config.DATA_PROCESSED_DIR, "y_labels.pt"))

model = RecurrentMLPSNN().to(config.DEVICE)
model.load_state_dict(torch.load(os.path.join(config.QUANTIZED_DIR, "q6_rmlp_snn.pth")))
model.eval()

with torch.no_grad():
    logits, _ = model(X.to(config.DEVICE))
    preds = logits.argmax(dim=1).cpu().numpy()

class_names = ['N', 'S', 'V', 'F', 'Q']
cm = confusion_matrix(y.numpy(), preds)

plt.figure(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("6-Bit Quantized SNN - AAMI Confusion Matrix")
plt.tight_layout()
cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved: {cm_path}")

# 2. Generate Weight Distribution Plot (FP32 vs Quantized 6-Bit)
model_fp32 = RecurrentMLPSNN()
model_fp32.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "baseline_rmlp_snn.pth")))

fp32_weights = np.concatenate([p.detach().cpu().numpy().flatten() for n, p in model_fp32.named_parameters() if 'weight' in n])
q6_weights = np.concatenate([p.detach().cpu().numpy().flatten() for n, p in model.named_parameters() if 'weight' in n])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(fp32_weights, bins=50, color='navy', alpha=0.7)
ax[0].set_title("FP32 Baseline Weight Distribution")
ax[0].set_xlabel("Weight Value")

ax[1].hist(q6_weights, bins=50, color='crimson', alpha=0.7)
ax[1].set_title("6-Bit Quantized Weight Discretization")
ax[1].set_xlabel("Weight Value")

plt.tight_layout()
weights_path = os.path.join(config.PLOTS_DIR, "weight_quantization_dist.png")
plt.savefig(weights_path, dpi=300)
plt.close()
print(f"Saved: {weights_path}")