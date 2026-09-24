import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from src.models.rmlp_snn import RecurrentMLPSNN
from src.training.trainer import TrainEngine

def get_data_dir():
    """Dynamically locates the processed data directory from config or default path."""
    for attr in ["DATA_PROCESSED_DIR", "PROCESSED_DATA_DIR", "DATA_DIR"]:
        if hasattr(config, attr):
            return getattr(config, attr)
    return os.path.join(PROJECT_ROOT, "data", "processed")

def load_datasets():
    proc_dir = get_data_dir()
    x_train_path = os.path.join(proc_dir, "X_train.pt")
    y_train_path = os.path.join(proc_dir, "y_train.pt")
    x_test_path  = os.path.join(proc_dir, "X_test.pt")
    y_test_path  = os.path.join(proc_dir, "y_test.pt")

    if os.path.exists(x_train_path) and os.path.exists(y_train_path):
        print(f"Loading split dataset from {proc_dir}...")
        X_train = torch.load(x_train_path)
        y_train = torch.load(y_train_path)
        
        if os.path.exists(x_test_path) and os.path.exists(y_test_path):
            X_val = torch.load(x_test_path)
            y_val = torch.load(y_test_path)
        else:
            # Fallback inline split if test set isn't saved separately
            train_size = int(0.8 * len(X_train))
            X_val, y_val = X_train[train_size:], y_train[train_size:]
            X_train, y_train = X_train[:train_size], y_train[:train_size]

        return TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    else:
        raise FileNotFoundError(f"Could not locate X_train.pt or y_train.pt in {proc_dir}")

def main():
    print("--- Step 2: Training 271-Neuron rMLP SNN Model ---")
    device = getattr(config, "DEVICE", "cpu")
    print(f"Using Compute Device: {device}")

    train_set, val_set = load_datasets()
    
    batch_size = getattr(config, "BATCH_SIZE", 64)
    epochs = getattr(config, "EPOCHS", 20)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = RecurrentMLPSNN().to(device)
    trainer = TrainEngine(model=model, train_loader=train_loader, val_loader=val_loader)
    
    history = trainer.run(epochs=epochs)

    checkpoint_dir = getattr(config, "CHECKPOINT_DIR", os.path.join(PROJECT_ROOT, "outputs", "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "baseline_rmlp_snn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nBaseline model checkpoint saved successfully to: {save_path}")

if __name__ == "__main__":
    main()