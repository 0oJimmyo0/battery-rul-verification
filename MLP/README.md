# MLP Model for Battery RUL Prediction

This repository contains the implementation of a Multi-Layer Perceptron (MLP) model for predicting battery Remaining Useful Life (RUL), along with empirical verification scripts.

## Files

| File | Description |
|------|-------------|
| `MLP_prediction.m` | MATLAB script for training and evaluating the MLP model |
| `run_abcrown_verification.py` | Python script for empirical verification of model properties |
| `Prediction.png` | Visualization of MLP prediction results |

## Requirements

### MATLAB (for `MLP_prediction.m`)
- MATLAB R2019b or later
- Neural Network Toolbox

### Python (for `run_abcrown_verification.py`)
- Python 3.7+
- NumPy
- ONNX Runtime

Install Python dependencies:
```bash
pip install numpy onnxruntime
```

## Usage

### 1. Training the MLP Model (MATLAB)

```matlab
% Run the MATLAB script
MLP_prediction
```

**Input:** 
- `Battery_dataset.csv` — Battery degradation dataset

**Output:**
- Prediction plots for each battery (saved as PNG files)
- Performance metrics (MAPE, R²) displayed in console

**Model Configuration:**
- Architecture: [9 → 64 → 32 → 1]
- Training algorithm: Levenberg-Marquardt (`trainlm`)
- Epochs: 1000
- Train/Val split: 80%/20%

### 2. Empirical Verification (Python)

Before running verification, export the trained MLP model to ONNX format and update the path in the script.

```bash
python run_abcrown_verification.py
```

**Verified Properties:**

| Property | Description | Threshold |
|----------|-------------|-----------|
| Non-Negativity | Output ≥ 0 | — |
| Robustness | Stable under ε=0.01 noise | Δ ≤ 1.0 |
| Monotonicity | RUL decreases as cycle increases | — |
| Upper Bound | Output ≤ 3.73 | — |

## Results

### Predictive Performance (LOBO-CV)

| Test Battery | MAE (cycles) | RMSE (cycles) | R² |
|--------------|--------------|---------------|-----|
| Battery 1 | 35.2 | 11.5 | 0.84 |
| Battery 2 | 39.5 | 12.1 | 0.83 |
| Battery 3 | 41.7 | 13.3 | 0.79 |
| **Average** | **38.8** | **12.3** | **0.82** |

### Verification Results

| Property | Result |
|----------|--------|
| Non-Negativity | FAIL (1/30 samples, output = -0.0196) |
| Robustness | PASS (max Δ = 0.0282) |
| Monotonicity | PASS (2.4615 → 1.9556) |
| Upper Bound | PASS |

## Notes

- The MLP model uses z-score normalization for input features
- Post-processing with `max(0, ·)` ensures non-negative outputs in deployment
- Empirical verification tests properties on sampled inputs (not formal guarantees)

## Author

Part of the Battery RUL Prediction project with formal verification.

