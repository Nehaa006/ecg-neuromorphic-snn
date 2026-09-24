import os
import sys
import glob
import numpy as np
import torch
import wfdb

# Ensure root import visibility
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

# AAMI standard heartbeat class mapping
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic (S)
    'V': 2, 'E': 2,                          # Ventricular ectopic (V)
    'F': 3,                                  # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                   # Unknown / Paced (Q)
}

def encode_level_crossing(signal, delta_v=config.DELTA_V):
    """Encodes continuous ECG signal into Level-Crossing (LC) binary spike streams."""
    up_spikes = np.zeros_like(signal)
    down_spikes = np.zeros_like(signal)
    
    last_v = signal[0]
    for i in range(1, len(signal)):
        diff = signal[i] - last_v
        if diff >= delta_v:
            up_spikes[i] = 1.0
            last_v = signal[i]
        elif diff <= -delta_v:
            down_spikes[i] = 1.0
            last_v = signal[i]
            
    return up_spikes, down_spikes

def process_record(record_path):
    """Reads WFDB signal and annotations, returning segmented spike beats and labels."""
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
    except Exception as e:
        print(f"Skipping record {record_path}: {e}")
        return [], []

    signal = record.p_signal[:, 0]  # Channel 0 (MLII)
    up_spikes, down_spikes = encode_level_crossing(signal)

    beats, labels = [], []
    win_len = 180  # 180 samples centered around R-peak (360 Hz = ~0.5s)
    half_win = win_len // 2

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in AAMI_MAPPING:
            continue
        
        if sample - half_win >= 0 and sample + half_win < len(signal):
            u_seg = up_spikes[sample - half_win : sample + half_win]
            d_seg = down_spikes[sample - half_win : sample + half_win]
            
            # Reshape segments into dynamic window slots (48 slots x 4 channels = 192 features)
            # Slot aggregation across time steps (T=3)
            num_slots = config.WINDOW_SLOTS
            samples_per_slot = win_len // num_slots
            
            u_slots = u_seg[:num_slots * samples_per_slot].reshape(num_slots, samples_per_slot).sum(axis=1)
            d_slots = d_seg[:num_slots * samples_per_slot].reshape(num_slots, samples_per_slot).sum(axis=1)
            
            # Create temporal input matrix (T x 192)
            feat = np.concatenate([u_slots > 0, d_slots > 0, u_slots > 1, d_slots > 1]).astype(np.float32)
            beat_tensor = np.tile(feat, (config.TIME_STEPS, 1))
            
            beats.append(beat_tensor)
            labels.append(AAMI_MAPPING[symbol])

    return beats, labels

def main():
    print("--- Starting MIT-BIH Beat Segmentation & LC Encoding ---")
    raw_dir = config.DATA_RAW_DIR
    processed_dir = config.DATA_PROCESSED_DIR
    os.makedirs(processed_dir, exist_ok=True)

    hea_files = glob.glob(os.path.join(raw_dir, "*.hea"))
    record_bases = sorted(list(set([f.replace('.hea', '') for f in hea_files])))

    if not record_bases:
        print(f"Error: No .hea files found in {raw_dir}")
        return

    print(f"Found {len(record_bases)} record files in {raw_dir}. Processing...")

    all_beats, all_labels = [], []
    for base_path in record_bases:
        rec_name = os.path.basename(base_path)
        beats, labels = process_record(base_path)
        if beats:
            all_beats.extend(beats)
            all_labels.extend(labels)
            print(f"Processed Record {rec_name}: {len(beats)} beats extracted.")

    if not all_beats:
        print("No beats were segmented. Check raw files and annotation symbols.")
        return

    X_tensor = torch.tensor(np.array(all_beats), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)

    x_save_path = os.path.join(processed_dir, "X_beats.pt")
    y_save_path = os.path.join(processed_dir, "y_labels.pt")

    torch.save(X_tensor, x_save_path)
    torch.save(y_tensor, y_save_path)

    print("\n--- Segmentation Complete ---")
    print(f"Saved X_beats.pt: {X_tensor.shape}")
    print(f"Saved y_labels.pt: {y_tensor.shape}")
    print(f"Class Distribution (0-4): {torch.bincount(y_tensor).tolist()}")

if __name__ == "__main__":
    main()