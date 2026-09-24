import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_multicollinearity_analysis(data_path="data/processed/X_test.pt", save_plot_path="outputs/plots/multicollinearity_matrix.png"):
    print("[*] Starting Multicollinearity Analysis...")
    
    # 1. Load Preprocessed Feature Tensors
    if not os.path.exists(data_path):
        print(f"[!] Data file not found at {data_path}. Please check your dataset path.")
        return

    X = torch.load(data_path)
    
    # If 3D tensor (Batch, Time, Features), flatten across temporal dimension for feature analysis
    if X.dim() == 3:
        batch_size, time_steps, features = X.shape
        X = X.reshape(batch_size, time_steps * features)
        
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    
    # Sample subset if features/samples are extremely large to speed up calculation
    if X_np.shape[0] > 2000:
        indices = np.random.choice(X_np.shape[0], 2000, replace=False)
        X_np = X_np[indices]

    # Convert to DataFrame
    num_cols = X_np.shape[1]
    df = pd.DataFrame(X_np, columns=[f"F_{i}" for i in range(num_cols)])

    # Remove zero-variance features (e.g., inactive level-crossing channels)
    non_constant_cols = df.columns[df.std() > 1e-6]
    df_active = df[non_constant_cols]
    
    print(f"[+] Total Features: {num_cols} | Active (Non-Constant) Features: {len(non_constant_cols)}")

    # 2. Pearson Correlation Matrix
    corr_matrix = df_active.corr().abs()
    
    # 3. Variance Inflation Factor (VIF) Calculation
    # Subsample max 30 active features for VIF stability
    vif_features = df_active.iloc[:, :min(30, df_active.shape[1])]
    
    print("[*] Computing Variance Inflation Factors (VIF)...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = vif_features.columns
    vif_data["VIF"] = [
        variance_inflation_factor(vif_features.values, i) 
        for i in range(vif_features.shape[1])
    ]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    print("\n" + "="*50)
    print("        VIF ANALYSIS RESULT (Top 10 Features)")
    print("="*50)
    print(vif_data.head(10).to_string(index=False))
    print("="*50)

    # 4. Save Correlation Heatmap
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
    plt.title("Inter-Feature Correlation Matrix Heatmap", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Encoded Feature Channels", fontsize=10, fontweight='bold')
    plt.ylabel("Encoded Feature Channels", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    
    print(f"\n[+] Correlation Matrix Heatmap saved to: {save_plot_path}")
    print("[+] Multicollinearity check completed successfully!")

if __name__ == "__main__":
    run_multicollinearity_analysis()