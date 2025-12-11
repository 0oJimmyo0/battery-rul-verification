# Monotonicity Verification with α,β-CROWN

## Overview

This directory contains files for verifying pairwise monotonicity of the MbD model using α,β-CROWN.

**Property**: For any inputs x₁, x₂ where `cycle₂ ≥ cycle₁`, verify that `RUL(x₂) ≤ RUL(x₁)`.

## Workflow

The verification uses a **single script** (`verify_with_abcrown.py`) that:

1. **Creates wrapper ONNX model**: Takes `[x₁, x₂]` (concatenated) → outputs `f(x₁) - f(x₂)`
2. **Creates VNNLIB file**: Encodes the property as output constraint `Y₀ ≥ 0`
3. **Creates config file**: α,β-CROWN configuration pointing to wrapper model and VNNLIB

## Usage

```bash
# Run the setup script
python verify_with_abcrown.py --model_dir ./mbd_pt --data_dir ./prep_out

# This creates:
# - verification/monotonicity_wrapper.onnx
# - verification/monotonicity_pairwise.vnnlib
# - verification/abcrown_config.yaml

# Then run verification
conda activate alpha-beta-crown
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --config /path/to/verification/abcrown_config.yaml
```

## Files Created

- **`monotonicity_wrapper.onnx`**: Wrapper model for pairwise verification
- **`monotonicity_pairwise.vnnlib`**: VNNLIB specification encoding the property
- **`abcrown_config.yaml`**: α,β-CROWN configuration file

## Why This Approach?

- **Single script**: Everything in one place, easy to understand
- **VNNLIB format**: Standard format for neural network verification
- **Pairwise property**: Wrapper model allows comparing two inputs
- **Efficient**: One verification run instead of multiple

## Understanding Results

- **Verified**: Property holds for all inputs ✓
- **Falsified**: Counterexample found ✗
- **Timeout**: Verification incomplete (may need more time)

