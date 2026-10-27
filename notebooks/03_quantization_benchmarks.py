import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report

print("=" * 60)
print("HIGH-ACCURACY RETRAINING & 6-BIT QUANTIZATION")
print("=" * 60)

# Paths
x_train_path = "data/processed/X_train.pt"
y_train_path = "data/processed/y_train.pt"
x_test_path = "data/processed/X_test.pt"
y_test_path = "data/processed/y_test.pt"
checkpoint_path = "outputs/quantized/q6_rmlp_snn.pth"

os.makedirs("outputs/quantized", exist_ok=True)

# Load Tensors
X_train = torch.load(x_train_path, map_location=torch.device('cpu'))
y_train = torch.load(y_train_path, map_location=torch.device('cpu'))
X_test = torch.load(x_test_path, map_location=torch.device('cpu'))
y_test = torch.load(y_test_path, map_location=torch.device('cpu'))

in_features = X_train.shape[-1]
print(f" -> Input Feature Dimension: {in_features}")

# 1. Inverse Log-Smoothed Class Weights for Loss Penalty
unique_classes, class_counts = torch.unique(y_train, return_counts=True)
print(f"Training Class Counts: {class_counts.tolist()}")

class_weights = 1.0 / torch.log1p(class_counts.float())
class_weights = class_weights / class_weights.sum() * 5.0  # Scale total sum to 5
print(f"Applied Class Weights: {class_weights.numpy().round(4)}")

# 2. SNN Architecture
class HighAccuracySNN(nn.Module):
    def __init__(self, in_d=192, proj_d=96, rec_d=120, hid_d=64, out_d=5):
        super().__init__()
        self.in_proj = nn.Linear(in_d, proj_d)
        self.rec_layer = nn.Linear(proj_d + rec_d, rec_d)
        self.hidden_layer = nn.Linear(rec_d, hid_d)
        self.output_layer = nn.Linear(hid_d, out_d)
        self.beta = 0.25

    def forward(self, x):
        if x.dim() == 3:  # (batch, time_steps, features)
            batch_size, time_steps, _ = x.shape
            h_rec = torch.zeros(batch_size, 120, device=x.device)
            spikes_sum = 0
            for t in range(time_steps):
                x_t = x[:, t, :]
                proj_out = torch.relu(self.in_proj(x_t))
                rec_input = torch.cat([proj_out, h_rec], dim=1)
                rec_out = self.rec_layer(rec_input)
                h_rec = self.beta * h_rec + rec_out
                hid_out = torch.relu(self.hidden_layer(h_rec))
                spikes_sum += hid_out
            return self.output_layer(spikes_sum / time_steps)
        else:  # (batch, features)
            proj_out = torch.relu(self.in_proj(x))
            h_rec_dummy = torch.zeros(x.shape[0], 120, device=x.device)
            rec_input = torch.cat([proj_out, h_rec_dummy], dim=1)
            rec_out = torch.relu(self.rec_layer(rec_input))
            hid_out = torch.relu(self.hidden_layer(rec_out))
            return self.output_layer(hid_out)

model = HighAccuracySNN(in_d=in_features)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# 3. Floating-Point Training Phase
print("\n[Phase 1] Training floating-point weights with class penalties...")
batch_size = 256
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    total_loss = 0.0
    num_batches = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    scheduler.step()
    print(f"Epoch [{epoch+1:2d}/{num_epochs}] - Loss: {total_loss/num_batches:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

# 4. Post-Training 6-Bit Uniform Quantization Phase
print("\n[Phase 2] Applying 6-Bit Uniform Quantization to Model Weights...")
with torch.no_grad():
    for name, param in model.named_parameters():
        if 'weight' in name:
            qmin, qmax = -32, 31  # 6-bit signed integer dynamic range (-2^5 to 2^5 - 1)
            scale = (param.max() - param.min()) / (qmax - qmin)
            if scale > 0:
                zero_point = torch.round(-param.min() / scale) + qmin
                q_weight = torch.clamp(torch.round(param / scale + zero_point), qmin, qmax)
                param.copy_((q_weight - zero_point) * scale)

# 5. Save Final Quantized State Dict
state_dict_to_save = {
    'in_proj.weight': model.in_proj.weight,
    'in_proj.bias': model.in_proj.bias,
    'rec_layer.weight': model.rec_layer.weight,
    'rec_layer.bias': model.rec_layer.bias,
    'hidden_layer.weight': model.hidden_layer.weight,
    'hidden_layer.bias': model.hidden_layer.bias,
    'output_layer.weight': model.output_layer.weight,
    'output_layer.bias': model.output_layer.bias,
    'lif_rec.beta': torch.tensor(model.beta)
}

torch.save(state_dict_to_save, checkpoint_path)
print(f"\nSaved 6-bit quantized checkpoint to: {checkpoint_path}")

# 6. Final Test Evaluation
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_preds = torch.argmax(test_logits, dim=1).numpy()

print("\nUpdated Live Classification Metrics:")
print(classification_report(y_test.numpy(), test_preds, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4, zero_division=0))