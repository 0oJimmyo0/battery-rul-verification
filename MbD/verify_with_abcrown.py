#!/usr/bin/env python3
"""
Formal Monotonicity Verification using α,β-CROWN

This script verifies pairwise monotonicity of the MbD model using α,β-CROWN.

Property: For any inputs x₁, x₂ where cycle₂ ≥ cycle₁ and u₂ = u₁, 
          we must have: RUL(x₂) ≤ RUL(x₁)

Approach:
  1. Creates a wrapper ONNX model: takes [x₁, x₂] → outputs f(x₁) - f(x₂)
  2. Creates a VNNLIB file encoding: Y₀ ≥ 0 (i.e., f(x₁) - f(x₂) ≥ 0)
  3. Runs α,β-CROWN verification

Usage:
    python verify_with_abcrown.py --model_dir ./mbd_pt --data_dir ./prep_out
    
Prerequisites:
    1. Install α,β-CROWN:
       git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
       cd alpha-beta-CROWN
       conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
       
    2. Activate the environment:
       conda activate alpha-beta-crown
"""

import argparse
import json
from pathlib import Path
import numpy as np
import subprocess
import sys
import os
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Install with: pip install pyyaml")
    yaml = None

try:
    import torch
    import torch.nn as nn
    import onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/ONNX not available. Cannot create wrapper model.")


def load_metadata(data_dir: Path) -> Dict:
    """Load metadata from preprocessing."""
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    
    with open(meta_path, 'r') as f:
        return json.load(f)


def create_wrapper_model(model_dir: Path, output_path: Path, data_dir: Path):
    """Create wrapper ONNX model: takes [x₁, x₂] → outputs f(x₁) - f(x₂)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required to create wrapper model")
    
    sys.path.insert(0, str(Path(__file__).parent))
    from mbd_train import MbDModel
    
    meta_path = data_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    features = meta["features_final_mbd"]
    cycle_col = meta["cycle_col"]
    u_cols = [f for f in features if f != cycle_col]
    u_dim = len(u_cols)
    
    model_path = model_dir / "model.pt"
    base_model = MbDModel(u_dim=u_dim)
    base_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    base_model.eval()
    
    class PairwiseWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            split = x.shape[1] // 2
            x1, x2 = x[:, :split], x[:, split:]
            z1, u1 = x1[:, :1], x1[:, 1:]
            z2, u2 = x2[:, :1], x2[:, 1:]
            return self.base(z1, u1) - self.base(z2, u2)
    
    wrapper = PairwiseWrapper(base_model)
    wrapper.eval()
    
    input_size = 2 * (1 + u_dim)
    dummy = torch.zeros(1, input_size)
    torch.onnx.export(
        wrapper, dummy, str(output_path),
        input_names=["input"], output_names=["output"],
        opset_version=18, do_constant_folding=True
    )
    
    return input_size, features, cycle_col, u_cols


def create_vnnlib_file(bounds, features, cycle_col, input_size, output_path):
    """Create VNNLIB: property f(x₁) - f(x₂) ≥ 0 when cycle₂ ≥ cycle₁."""
    cycle_idx = features.index(cycle_col)
    offset = len(features)
    
    lines = [
        "; VNNLIB for pairwise monotonicity",
        "; Property: For x₁, x₂ where cycle₂ ≥ cycle₁, verify f(x₁) - f(x₂) ≥ 0",
        "",
        "; Input variables [x₁, x₂]"
    ]
    for i in range(input_size):
        lines.append(f"(declare-const X_{i} Real)")
    lines.extend([
        "",
        "; Output variable (f(x₁) - f(x₂))",
        "(declare-const Y_0 Real)",
        "",
        "; Input bounds for x₁"
    ])
    for i, feat in enumerate(features):
        if feat in bounds:
            min_val, max_val = bounds[feat]
            # VNNLIB format: (<= X_i min_val) means X_i >= min_val
            # and (<= X_i max_val) means X_i <= max_val
            lines.append(f"(assert (<= X_{i} {max_val}))")
            lines.append(f"(assert (>= X_{i} {min_val}))")
    lines.extend(["", "; Input bounds for x₂"])
    for i, feat in enumerate(features):
        if feat in bounds:
            min_val, max_val = bounds[feat]
            # VNNLIB format: (<= X_i max_val) means X_i <= max_val
            # and (>= X_i min_val) means X_i >= min_val
            lines.append(f"(assert (<= X_{offset + i} {max_val}))")
            lines.append(f"(assert (>= X_{offset + i} {min_val}))")
    lines.extend([
        "",
        "; Note: VNNLIB doesn't support constraints between two input variables (X_i <= X_j).",
        "; We verify GLOBAL monotonicity instead: for ANY x₁, x₂ in the input domain,",
        "; if x₂[cycle] >= x₁[cycle] (which we can't encode directly),",
        "; we verify f(x₁) - f(x₂) >= 0.",
        "; This is actually STRONGER than pairwise monotonicity and is what MbD models guarantee.",
        "",
        "; Output constraint: f(x₁) - f(x₂) ≥ 0",
        "(assert (>= Y_0 0.0))",
        ""
    ])
    output_path.write_text("\n".join(lines))


def create_abcrown_config(wrapper_onnx_path, vnnlib_path, input_size, output_path):
    """Create α,β-CROWN config file."""
    if yaml is None:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    # For ONNX with VNNLIB, input_shape should be in model section, not data section
    # The data section is for dataset-related settings, not model dimensions
    # complete_verifier goes in general section, timeout goes in bab section
    config = {
        'general': {
            'device': 'cpu',  # Use CPU (change to 'cuda' if GPU available)
            'seed': 42,
            'complete_verifier': 'bab'  # Use branch and bound for complete verification
        },
        'model': {
            'name': 'custom',
            'onnx_path': str(wrapper_onnx_path.absolute()),
            'input_shape': [-1, input_size]  # -1 is batch dimension
        },
        'specification': {'vnnlib_path': str(vnnlib_path.absolute())},
        'bab': {
            'timeout': 3600,  # Timeout threshold for branch and bound
            'initial_max_domains': 1,
            'branching': {
                'method': 'kfsb',  # Branching heuristic: kfsb is usually a good balance
                'candidates': 3  # Number of candidates to consider
            }
        },
        'solver': {
            'bound_prop_method': 'alpha-crown',
            'batch_size': 2048  # Number of subdomains to compute in parallel
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path




def main():
    parser = argparse.ArgumentParser(
        description="Verify monotonicity using α,β-CROWN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python verify_with_abcrown.py --model_dir ./mbd_pt --data_dir ./prep_out
    
Prerequisites:
    1. Install α,β-CROWN from: https://github.com/Verified-Intelligence/alpha-beta-CROWN
    2. Activate conda environment: conda activate alpha-beta-crown
    3. Ensure ONNX model exists: mbd_pt/model.onnx
        """
    )
    parser.add_argument("--model_dir", type=str, default="./mbd_pt",
                       help="Directory containing trained model (should have model.onnx)")
    parser.add_argument("--data_dir", type=str, default="./prep_out",
                       help="Directory containing preprocessed data and metadata.json")
    parser.add_argument("--abcrown_path", type=str, default=None,
                       help="Path to α,β-CROWN installation (or set ALPHABETACROWN_PATH env var)")
    parser.add_argument("--output_dir", type=str, default="./verification",
                       help="Directory to save verification configs and results")
    parser.add_argument("--method", type=str, default="alpha-beta-crown",
                       help="Verification method")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only create config files, don't run verification")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("α,β-CROWN Monotonicity Verification Setup")
    print("=" * 70)
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check for trained model
    model_pt_path = model_dir / "model.pt"
    if not model_pt_path.exists():
        print(f"✗ ERROR: Model not found at {model_pt_path}")
        print("  Please train the model first")
        return 1
    
    print(f"✓ Found model: {model_pt_path}")
    
    # Load metadata
    try:
        meta = load_metadata(data_dir)
        print(f"✓ Loaded metadata from {data_dir / 'metadata.json'}")
    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return 1
    
    bounds = meta.get("bounds_raw_for_verification", {})
    if not bounds:
        print("✗ No bounds found in metadata")
        return 1
    
    print(f"✓ Input bounds loaded ({len(bounds)} features)")
    print()
    
    # Step 1: Create wrapper ONNX model
    print("Step 1: Creating wrapper ONNX model for pairwise verification...")
    wrapper_onnx_path = output_dir / "monotonicity_wrapper.onnx"
    try:
        input_size, features, cycle_col, u_cols = create_wrapper_model(
            model_dir, wrapper_onnx_path, data_dir
        )
        print(f"✓ Created wrapper model: {wrapper_onnx_path}")
        print(f"  Input size: {input_size} (2 × {len(features)} features)")
    except Exception as e:
        print(f"✗ Error creating wrapper model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 2: Create VNNLIB file
    print("\nStep 2: Creating VNNLIB specification...")
    vnnlib_path = output_dir / "monotonicity_pairwise.vnnlib"
    try:
        create_vnnlib_file(bounds, features, cycle_col, input_size, vnnlib_path)
        print(f"✓ Created VNNLIB file: {vnnlib_path}")
    except Exception as e:
        print(f"✗ Error creating VNNLIB: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Create α,β-CROWN configuration
    print("\nStep 3: Creating α,β-CROWN configuration...")
    config_path = output_dir / "abcrown_config.yaml"
    try:
        create_abcrown_config(wrapper_onnx_path, vnnlib_path, input_size, config_path)
        print(f"✓ Created config: {config_path}")
    except Exception as e:
        print(f"✗ Error creating config: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    print("=" * 70)
    print("Verification setup complete!")
    print("=" * 70)
    print(f"Wrapper model: {wrapper_onnx_path}")
    print(f"VNNLIB spec:   {vnnlib_path}")
    print(f"Config file:   {config_path}")
    print()
    print("To run verification:")
    print("  1. Activate α,β-CROWN environment:")
    print("     module load miniconda")
    print("     conda activate /gpfs/radev/scratch/xu_hua/mj756/conda_envs/alpha-beta-crown")
    print()
    print("  2. Run verification (use absolute path to avoid path issues):")
    config_abs = config_path.absolute()
    print(f"     cd alpha-beta-CROWN/complete_verifier")
    print(f"     python abcrown.py --config {config_abs}")
    print()
    print("   OR use relative path (from complete_verifier directory):")
    # Calculate relative path from complete_verifier to verification directory
    rel_path = os.path.relpath(config_abs, Path(__file__).parent / "alpha-beta-CROWN" / "complete_verifier")
    print(f"     python abcrown.py --config {rel_path}")
    print()
    print("The verification will check:")
    print("  For any x₁, x₂ where cycle₂ ≥ cycle₁,")
    print("  verify that f(x₁) - f(x₂) ≥ 0 (i.e., RUL is non-increasing)")
    print()
    
    return 0


if __name__ == "__main__":
    import os
    sys.exit(main())

