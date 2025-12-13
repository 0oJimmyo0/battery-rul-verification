# Battery Remaining Useful Life Prediction with Monotone-by-Design Neural Networks

This repository contains the implementation of a Monotone-by-Design (MbD) neural network for predicting battery Remaining Useful Life (RUL) with formal verification using α-β-CROWN.

## Overview

The project implements:
- **MbD Model**: A neural network architecture that guarantees monotonicity constraints by construction
- **Formal Verification**: Verification of non-negativity properties using α-β-CROWN
- **Battery RUL Prediction**: Application to battery degradation dataset

## Repository Structure

```
/
├── final_report.md            # Complete project report (all models)
├── README.md                  # This file
├── LICENSE                    # License file
└── MbD/                       # Monotone-by-Design model implementation
    ├── preprocess.py          # Data preprocessing pipeline
    ├── mbd_train.py           # MbD model training script
    ├── verify_with_bounds_simple.py  # Verification setup script
    ├── investigate_verification_issues.py  # Verification diagnostics
    ├── prep_out/              # Preprocessed data
    ├── mbd_pt/                # Trained model outputs
    ├── verification_bounds/    # Verification configurations
    └── plots/                 # Visualization outputs
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas
- ONNX (for model export)
- α-β-CROWN (for verification)

## Usage

### 1. Data Preprocessing

```bash
cd MbD
python preprocess.py --input Battery_dataset.csv --outdir ./prep_out
```

### 2. Model Training

```bash
cd MbD
python mbd_train.py --data_dir ./prep_out --out_dir ./mbd_pt
```

### 3. Verification Setup

```bash
cd MbD
python verify_with_bounds_simple.py --model_dir ./mbd_pt --data_dir ./prep_out
```

### 4. Run Verification

```bash
cd MbD/alpha-beta-CROWN/complete_verifier
python abcrown.py --config /path/to/verification_bounds/config.yaml
```

## Results

- **Test MAE**: 34.31 cycles
- **Test RMSE**: 35.88 cycles
- **Verification**: Non-negativity verified for cycle range [50, 150]

## Citation

If you use this code, please cite the relevant papers (see `final_report.md` for full citations).

