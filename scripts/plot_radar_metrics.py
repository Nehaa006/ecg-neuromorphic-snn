import os
import numpy as np
import matplotlib.pyplot as plt


def plot_radar_chart(precision, recall, save_path="outputs/plots/radar_metrics.png"):
    """Generates an accurate 5-class AAMI Sensitivity vs Precision radar plot."""
    class_names = ['N (Normal)', 'S (SVEB)', 'V (VEB)', 'F (Fusion)', 'Q (Paced)']

    N = len(class_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circular loop

    # Close loops for polygon plotting
    precision_closed = np.concatenate((precision, [precision[0]]))
    recall_closed = np.concatenate((recall, [recall[0]]))

    # Initialize Polar Plot
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Styling
    plt.xticks(angles[:-1], class_names, color='black', size=10, weight='bold')
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    plt.ylim(0, 1.05)

    # Precision curve
    ax.plot(angles, precision_closed, linewidth=2, linestyle='solid', label='Precision', color='#1f77b4')
    ax.fill(angles, precision_closed, color='#1f77b4', alpha=0.15)

    # Sensitivity (Recall) curve
    ax.plot(angles, recall_closed, linewidth=2, linestyle='solid', label='Sensitivity (Recall)', color='#ff7f0e')
    ax.fill(angles, recall_closed, color='#ff7f0e', alpha=0.15)

    plt.title("AAMI 5-Class Clinical Metrics: Sensitivity vs. Precision\n(6-Bit Quantized SNN - Stratified Evaluation)", 
              size=11, weight='bold', pad=25)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Successfully saved updated Radar Chart to: {save_path}")


def main():
    # Stratified Confusion Matrix values from your latest evaluation run
    cm = np.array([
        [17988,   55,   52,    4,   23],  # N
        [  190,  358,    7,    0,    1],  # S
        [  107,    7, 1316,    9,    8],  # V
        [   47,    0,   19,   94,    0],  # F
        [   30,    1,    0,    1, 1577]   # Q
    ])

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("--- Stratified Class-wise Clinical Metrics ---")
    class_names = ['N', 'S', 'V', 'F', 'Q']
    for i, name in enumerate(class_names):
        print(f"Class {name} -> Precision: {precision[i]*100:.2f}%, Sensitivity (Recall): {recall[i]*100:.2f}%")

    plot_radar_chart(precision, recall, save_path="outputs/plots/radar_metrics.png")


if __name__ == "__main__":
    main()