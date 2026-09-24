import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from sklearn.model_selection import train_test_split
import config

def main():
    print("--- Running Dataset Train/Test Split ---")
    
    proc_dir = config.DATA_PROCESSED_DIR
    x_path = os.path.join(proc_dir, "X_beats.pt")
    y_path = os.path.join(proc_dir, "y_labels.pt")  # Updated filename match

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[Error] Could not find {x_path} or {y_path}.")
        return

    print(f"Loading data from {proc_dir}...")
    X = torch.load(x_path)
    y = torch.load(y_path)
    print(f"Full Dataset Shapes -> X: {X.shape}, y: {y.shape}")

    # Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), 
        y.numpy(), 
        test_size=0.20, 
        random_state=42, 
        stratify=y.numpy()
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    torch.save(X_train, os.path.join(proc_dir, "X_train.pt"))
    torch.save(y_train, os.path.join(proc_dir, "y_train.pt"))
    torch.save(X_test,  os.path.join(proc_dir, "X_test.pt"))
    torch.save(y_test,  os.path.join(proc_dir, "y_test.pt"))

    print("\n[+] Train/Test Split Complete!")
    print(f"    - Training Set : {X_train.shape[0]} samples -> X_train.pt, y_train.pt")
    print(f"    - Testing Set  : {X_test.shape[0]} samples -> X_test.pt, y_test.pt")

if __name__ == "__main__":
    main()