# Ultra-Low-Power ECG Arrhythmia Classification via Recurrent Spiking Neural Networks

An event-driven neuromorphic deep learning pipeline for real-time, 5-class AAMI electrocardiogram (ECG) arrhythmia classification. This project leverages Level-Crossing (LC) sampling, Recurrent Leaky Integrate-and-Fire (rLIF) neurons, and hardware-aware optimization (Spatio-Temporal Backpropagation, synaptic weight pruning, and 6-bit fixed-point quantization) to achieve state-of-the-art diagnostic accuracy with ultra-low active synaptic operations ($\text{SOPs}$).

---

## 🌟 Key Features

* **Event-Driven Signal Processing:** Level-Crossing (LC) delta modulation converts continuous ECG signals into sparse 1-bit temporal spike trains, eliminating Nyquist redundancy.
* **Biologically Plausible Architecture:** Recurrent Leaky Integrate-and-Fire (rMLP-SNN) network capturing temporal dynamics over ECG heartbeats.
* **Surrogate Gradient Training:** Spatio-Temporal Backpropagation (STBP) using a Fast Sigmoid surrogate function to resolve non-differentiable spiking thresholds.
* **Joint Multi-Objective Optimization:** Custom loss balancing classification cross-entropy, firing-rate sparsity, and $L_1$ weight decay.
* **Extreme Model Compression:** Post-training synaptic weight pruning ($\Theta_w = 10^{-2}$) and 6-bit uniform fixed-point quantization for ultra-low memory footprint.

---

## 🏗️ System & Network Architecture

<img width="1024" height="757" alt="image" src="https://github.com/user-attachments/assets/6e3fb297-2796-40d8-9403-458020b57a00" />

### Pipeline Breakdown

1. **Level-Crossing (LC) Encoding:** Converts raw continuous ECG voltage signals into continuous positive and negative sparse 1-bit event channels ($\Delta V$).
2. **Recurrent LIF Hidden Layer:** Incorporates intra-layer feedback weights ($W_{rec}$) allowing temporal integration of past context across consecutive time steps.
3. **Membrane Accumulation Output:** Integrates membrane potential without spiking threshold reset to compute softmax logits for 5-class prediction.

---

## 📊 Dataset & Preprocessing

The model is evaluated on the standard MIT-BIH Arrhythmia Database, aligned with the AAMI Standard EC57 5-class categorization scheme:

| AAMI Class | Description |
| :--- | :--- |
| **N** | Non-ectopic beats (Normal, LBBB, RBBB, etc.) |
| **S** | Supraventricular ectopic beats (APCs, Nodal, etc.) |
| **V** | Ventricular ectopic beats (PVCs, Ventricular escape) |
| **F** | Fusion beats (Fusion of ventricular and normal) |
| **Q** | Paced beats / Unknown / Unclassifiable |

### Pipeline Workflow

* **Filtering & Normalization:** Bandpass filtering (0.5–45 Hz) and z-score normalization.
* **R-Peak Segmentation:** Heartbeat extraction using dynamic windowing around detected R-peaks.
* **Level-Crossing (LC) Encoding:** Signal-to-spike conversion emitting $1$-bit event tokens based on positive/negative amplitude delta thresholds ($\Delta V$).

---

## 🧠 Model Dynamics

The core classifier uses a Recurrent Multi-Layer Perceptron Spiking Neural Network (rMLP-SNN) with Leaky Integrate-and-Fire (LIF) dynamics:

$$\tau_m \frac{d V(t)}{d t} = -(V(t) - V_{rest}) + R \cdot I(t)$$

$$S(t) = \Theta(V(t) - V_{th}), \quad \text{where } \Theta \text{ is the Heaviside step function.}$$

---

## 📈 Performance & Benchmarks

The model was evaluated using a 6-bit post-training quantized version on $21,894$ test ECG beats:

### Classification Metrics

* **Overall Accuracy:** 97.44%
* **Average Per-Beat Inference Latency:** 0.034 ms / beat

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **N** | 0.9796 | 0.9926 | 0.9861 | 18,122 |
| **S** | 0.8504 | 0.6439 | 0.7329 | 556 |
| **V** | 0.9440 | 0.9095 | 0.9264 | 1,447 |
| **F** | 0.8704 | 0.5875 | 0.7015 | 160 |
| **Q** | 0.9801 | 0.9801 | 0.9801 | 1,609 |

### Neuromorphic Benchmarks

* **Synaptic Operations ($\text{SOPs}$):** 10,080.68 SOPs / beat
* **Estimated Energy Footprint:** 9072.61 pJ / beat (~9.07 nJ / beat)

---

## ⚡ Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Nehaa006/ecg-neuromorphic-snn.git
cd ecg-neuromorphic-snn
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Evaluation

To run inference, check hardware metrics, and generate the AAMI confusion matrix:

```bash
python scripts/evaluate_model.py
```

### 3. Visualizations

Run analysis scripts to generate performance plots:

```bash
# Check feature collinearity
python scripts/check_multicollinearity.py

# Generate radar chart of clinical metrics
python scripts/plot_radar_metrics.py
```

---

## 🎯 Conclusion

This work demonstrates an end-to-end event-driven framework for clinical-grade ECG arrhythmia classification on ultra-low-power edge hardware. By combining Level-Crossing delta modulation with a Recurrent LIF Spiking Neural Network, the system bridges the gap between biological temporal dynamics and energy-efficient edge processing.

Key takeaways from this implementation include:
* **High diagnostic fidelity** across AAMI classes, achieving 97.44% overall accuracy.
* **An efficient energy profile** requiring only ~9.07 nJ per beat due to sparse spiking activation patterns.
* **Robust model compression** via 6-bit post-training integer quantization and synaptic pruning, rendering the network highly suitable for battery-operated wearable medical devices and always-on cardiac monitoring.
