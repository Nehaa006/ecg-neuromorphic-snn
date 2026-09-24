import os
import torch

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoints")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
QUANTIZED_DIR = os.path.join(BASE_DIR, "outputs", "quantized")

# Signal Processing & Level-Crossing Hyperparameters
SAMPLING_RATE = 360            # MIT-BIH native frequency (Hz)
DELTA_V = 0.1                 # Level-Crossing voltage threshold (mV)
WINDOW_SLOTS = 48             # Temporal slots per dynamic beat
TIME_STEPS = 3                # Overlapping temporal windows (T=3)
INPUT_DIM = WINDOW_SLOTS * 4  # 192 features (48 slots x 4 spike channels)

# SNN Model Topology (271 Neurons total)
NEURONS_IN_PROJ = 96
NEURONS_RECURRENT = 120
NEURONS_HIDDEN = 50
NUM_CLASSES = 5               # AAMI categories: N, S, V, F, Q
LIF_BETA = 0.85               # Membrane potential decay factor
LIF_THRESHOLD = 1.0           # Spiking voltage threshold

# Training & Loss Parameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
LAMBDA_SPARSITY = 1e-4        # Target firing rate loss penalty
LAMBDA_WEIGHT_DECAY = 1e-5    # Synaptic weight decay penalty
TARGET_FIRING_RATE = 0.154    # Target average firing rate (~15.4%)

# Optimization & Benchmarking
PRUNING_THRESHOLD = 1e-2      # Weight pruning cutoff (Theta_w)
QUANT_MIN = -32               # 6-bit signed integer minimum
QUANT_MAX = 31                # 6-bit signed integer maximum
MAC_ENERGY_PJ = 4.6           # Standard FLOP energy (pJ)
AC_ENERGY_PJ = 0.9            # Spike SOP energy (pJ)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")