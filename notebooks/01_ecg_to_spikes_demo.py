import os
import sys

# 1. Force Python to import local project root modules first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

# Local project imports
import config
from src.preprocessing.lc_encoder import load_raw_record, level_crossing_encode

def main():
    print(f"Project Root Directory: {PROJECT_ROOT}")
    print(f"Loading raw ECG record from: {config.DATA_RAW_DIR}")

    # 2. Load Record 100 Signal
    rec_path = os.path.join(config.DATA_RAW_DIR, "100")
    try:
        signal, r_peaks, labels = load_raw_record(rec_path)
    except Exception as e:
        print(f"\n[Error] Could not load WFDB record '100'. Make sure MIT-BIH data exists in '{config.DATA_RAW_DIR}'.")
        print(f"Details: {e}")
        return

    # Slice a 3-second segment (360 Hz * 3 = 1080 samples)
    sample_window = 1080
    ecg_segment = signal[:sample_window, :]
    print(f"Successfully loaded segment with shape: {ecg_segment.shape}")

    # 3. Level-Crossing Spike Encoding
    spikes = level_crossing_encode(ecg_segment, delta_v=config.DELTA_V)
    time_axis = np.arange(sample_window) / config.SAMPLING_RATE

    # 4. Plot Continuous Signal vs 4-Channel Event Spikes
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Lead 1 Continuous Signal
    axs[0].plot(time_axis, ecg_segment[:, 0], color='navy', label='Lead 1 ECG (mV)')
    axs[0].set_ylabel('Amplitude (mV)')
    axs[0].set_title('Raw Continuous ECG vs 1-Bit Event Spike Encodings')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)

    # Lead 1 Spikes (+ / -)
    axs[1].eventplot(np.where(spikes[:, 0] > 0)[0] / config.SAMPLING_RATE, lineoffsets=1, colors='crimson', label='Lead 1 (+) Spike')
    axs[1].eventplot(np.where(spikes[:, 1] > 0)[0] / config.SAMPLING_RATE, lineoffsets=-1, colors='darkorange', label='Lead 1 (-) Spike')
    axs[1].set_ylabel('Lead 1 Events')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)

    # Lead 2 Spikes (+ / -)
    axs[2].eventplot(np.where(spikes[:, 2] > 0)[0] / config.SAMPLING_RATE, lineoffsets=1, colors='seagreen', label='Lead 2 (+) Spike')
    axs[2].eventplot(np.where(spikes[:, 3] > 0)[0] / config.SAMPLING_RATE, lineoffsets=-1, colors='purple', label='Lead 2 (-) Spike')
    axs[2].set_ylabel('Lead 2 Events')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save output visualization
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(config.PLOTS_DIR, "lc_spike_raster.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to: {save_path}")

    # Calculate Compression Ratio
    raw_bits = ecg_segment.size * 32
    spike_bits = max(1, np.count_nonzero(spikes))
    print(f"Data Compression Ratio: {raw_bits / spike_bits:.2f}x Reduction")

if __name__ == "__main__":
    main()