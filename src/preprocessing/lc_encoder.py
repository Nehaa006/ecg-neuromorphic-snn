import os
import numpy as np
import wfdb


def load_raw_record(record_path: str):
    """Loads raw ECG signal array and annotation sample indices using wfdb."""
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")
    
    # Extract 2-channel voltage signal and peak annotations
    signal = record.p_signal  # Shape: (N, 2)
    r_peaks = annotation.sample
    labels = annotation.symbol
    return signal, r_peaks, labels


def level_crossing_encode(signal: np.ndarray, delta_v: float = 0.1) -> np.ndarray:
    """
    Encodes continuous 2-channel ECG signal into a 4-channel binary spike matrix (+1, -1, 0).
    Channels 0 & 1: Positive/Negative crossings for Lead 1.
    Channels 2 & 3: Positive/Negative crossings for Lead 2.
    """
    num_samples, num_channels = signal.shape
    spikes = np.zeros((num_samples, num_channels * 2), dtype=np.int8)

    for ch in range(num_channels):
        v_ref = signal[0, ch]
        for t in range(1, num_samples):
            v_current = signal[t, ch]
            diff = v_current - v_ref
            
            if diff >= delta_v:
                spikes[t, ch * 2] = 1       # Positive crossing spike
                v_ref += delta_v
            elif diff <= -delta_v:
                spikes[t, ch * 2 + 1] = 1   # Negative crossing spike
                v_ref -= delta_v

    return spikes


def extract_dynamic_beat_spikes(spikes: np.ndarray, r_peaks: np.ndarray, 
                                window_slots: int = 48, time_steps: int = 3):
    """
    Extracts beat segments using dynamic R-to-R interval scaling and bins them into
    192 spatio-temporal slots (48 slots x 4 channels) formatted for 3 time windows (T=3).
    """
    num_beats = len(r_peaks) - 2
    feature_dim = window_slots * 4  # 48 slots * 4 spike channels = 192 features
    spatio_temporal_matrices = []

    for i in range(1, len(r_peaks) - 1):
        # Dynamic R-to-R peak intervals (RR1 and RR2)
        rr1 = r_peaks[i] - r_peaks[i - 1]
        rr2 = r_peaks[i + 1] - r_peaks[i]
        
        start_idx = max(0, r_peaks[i] - int(0.4 * rr1))
        end_idx = min(len(spikes), r_peaks[i] + int(0.6 * rr2))
        
        beat_slice = spikes[start_idx:end_idx, :]
        
        # Bin spike events into fixed 48 temporal slots
        split_indices = np.array_split(beat_slice, window_slots, axis=0)
        binned_beat = np.zeros((window_slots, 4), dtype=np.float32)
        
        for slot_idx, slot_data in enumerate(split_indices):
            if len(slot_data) > 0:
                binned_beat[slot_idx] = (np.sum(slot_data, axis=0) > 0).astype(np.float32)
                
        flattened_features = binned_beat.flatten() # Shape: (192,)
        
        # Replicate features across T=3 overlapping temporal windows
        temporal_windows = np.tile(flattened_features, (time_steps, 1)) # Shape: (3, 192)
        spatio_temporal_matrices.append(temporal_windows)

    return np.array(spatio_temporal_matrices)  # Shape: (Num_Beats, 3, 192)