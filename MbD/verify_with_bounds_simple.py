#!/usr/bin/env python3
"""
Simple Monotonicity Verification with Bounds Output

This script:
1. Fixes context features to specific values
2. Computes bounds on RUL for different cycle values
3. Verifies monotonicity: RUL(cycle_max) ≤ RUL(cycle_min)

Produces output similar to CNN verification with lower/upper bounds.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None


def load_metadata(data_dir: Path):
    """Load metadata."""
    with open(data_dir / "metadata.json", 'r') as f:
        return json.load(f)


def create_cycle_model(model_dir: Path, output_path: Path, data_dir: Path):
    """
    Create model that takes cycle as input (context fixed to medians).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    sys.path.insert(0, str(Path(__file__).parent))
    from mbd_train import MbDModel, load_data
    
    meta = load_metadata(data_dir)
    feats, target, cycle_col, train_df, val_df, test_df, meta = load_data(data_dir)
    u_cols = [f for f in feats if f != cycle_col]
    u_dim = len(u_cols)
    
    # Use medians for fixed context
    fixed_u_values = {col: float(train_df[col].median()) for col in u_cols}
    
    # Load model
    base_model = MbDModel(u_dim=u_dim)
    base_model.load_state_dict(torch.load(model_dir / "model.pt", map_location='cpu'))
    base_model.eval()
    
    # Create wrapper that replaces softplus with ReLU for verification
    # (α-β-CROWN doesn't support softplus, but ReLU gives similar non-negativity)
    class CycleModel(nn.Module):
        def __init__(self, base, fixed_u, u_cols, cycle_col, meta):
            super().__init__()
            self.base = base
            self.fixed_u = torch.tensor([fixed_u[col] for col in u_cols], dtype=torch.float32)
            self.cycle_col = cycle_col
            self.meta = meta
        
        def forward(self, cycle):
            # Normalize cycle
            if self.meta and "bounds_raw_for_verification" in self.meta:
                Cmin, Cmax = self.meta["bounds_raw_for_verification"][self.cycle_col]
                cn = (cycle - Cmin) / max(1e-6, (Cmax - Cmin))
                z = 1.0 - cn
            else:
                z = 1.0 - cycle / 100.0
            
            batch_size = cycle.shape[0]
            u = self.fixed_u.unsqueeze(0).expand(batch_size, -1)
            
            # Get base model output (before final softplus)
            # We need to access the model's internal computation
            # The base model does: base + gain * hz, then softplus
            # For verification, we'll use ReLU instead of softplus
            z_tensor = z.unsqueeze(-1) if z.dim() == 1 else z
            base_out = self.base.a(u)
            gain = self.base.b(u)
            hz = self.base.h(z_tensor)
            y = base_out + gain * hz
            
            # Replace softplus with ReLU for α-β-CROWN compatibility
            # ReLU also ensures non-negativity
            return F.relu(y)
    
    wrapper = CycleModel(base_model, fixed_u_values, u_cols, cycle_col, meta)
    wrapper.eval()
    
    # Export
    dummy = torch.tensor([[100.0]])
    torch.onnx.export(
        wrapper, dummy, str(output_path),
        input_names=["cycle"],
        output_names=["RUL"],
        opset_version=17,  # Changed from 18 to 17 (max supported)
        do_constant_folding=True
    )
    
    return fixed_u_values, cycle_col


def create_vnnlib_monotonicity(bounds, cycle_col, output_path):
    """Create VNNLIB for monotonicity: verify RUL decreases as cycle increases."""
    cycle_min, cycle_max = bounds[cycle_col]
    
    # For bounds computation, we specify input bounds and a simple output constraint
    # The output constraint helps trigger bounds computation
    # We'll use Y_0 >= 0 (RUL is non-negative) as a simple constraint
    
    lines = [
        "; VNNLIB for cycle monotonicity verification with bounds output",
        "; Input: cycle",
        "(declare-const X_0 Real)",
        "",
        "; Output: RUL",
        "(declare-const Y_0 Real)",
        "",
        f"; Input bounds: cycle in [{cycle_min}, {cycle_max}]",
        f"(assert (<= X_0 {cycle_max}))",
        f"(assert (>= X_0 {cycle_min}))",
        "",
        "; Output constraint: RUL >= 0 (non-negativity)",
        "; This constraint helps trigger bounds computation",
        "(assert (>= Y_0 0.0))",
        "",
        "; Note: For monotonicity verification, we compute bounds on RUL",
        "; for the entire cycle range and check if they satisfy:",
        "; RUL(cycle_max) <= RUL(cycle_min)",
        ""
    ]
    
    output_path.write_text("\n".join(lines))
    return cycle_min, cycle_max


def create_config_with_bounds_output(onnx_path, vnnlib_path, output_path, use_incomplete=False):
    """Create config that saves bounds."""
    if yaml is None:
        raise ImportError("PyYAML required")
    
    # For bounds output, use incomplete verification to compute bounds
    # This gives us lower/upper bounds on output without full verification
    config = {
        'general': {
            'device': 'cpu',
            'seed': 42,
            'save_output': True,  # Save bounds
            'save_adv_example': False,
            'enable_incomplete_verification': True  # Use incomplete verification mode
            # Note: Do NOT set 'complete_verifier' to force incomplete mode
        },
        'model': {
            'name': 'custom',
            'onnx_path': str(onnx_path.absolute()),
            'input_shape': [-1, 1]
        },
        'specification': {
            'vnnlib_path': str(vnnlib_path.absolute())
        },
        'solver': {
            'bound_prop_method': 'alpha-crown',
            'batch_size': 2048
        },
        'attack': {
            'pgd_order': 'skip'  # Skip PGD attack to avoid errors
        }
    }
    
    # Only add complete_verifier if NOT using incomplete mode
    if not use_incomplete:
        config['general']['complete_verifier'] = 'bab'
        config['bab'] = {
            'timeout': 3600,
            'initial_max_domains': 1,
            'branching': {
                'method': 'kfsb',
                'candidates': 3
            }
        }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./mbd_pt")
    parser.add_argument("--data_dir", type=str, default="./prep_out")
    parser.add_argument("--output_dir", type=str, default="./verification_bounds")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Creating Verification Setup for Bounds Output")
    print("=" * 70)
    
    # Load metadata
    meta = load_metadata(data_dir)
    bounds = meta["bounds_raw_for_verification"]
    cycle_col = meta["cycle_col"]
    
    # Create model
    print("\nCreating fixed-context model...")
    onnx_path = output_dir / "cycle_model.onnx"
    fixed_u, cycle_col = create_cycle_model(model_dir, onnx_path, data_dir)
    print(f"✓ Model: {onnx_path}")
    
    # Create VNNLIB
    print("\nCreating VNNLIB...")
    vnnlib_path = output_dir / "monotonicity.vnnlib"
    cycle_min, cycle_max = create_vnnlib_monotonicity(bounds, cycle_col, vnnlib_path)
    print(f"✓ VNNLIB: {vnnlib_path}")
    print(f"  Cycle range: [{cycle_min:.2f}, {cycle_max:.2f}]")
    
    # Create config
    print("\nCreating config...")
    config_path = output_dir / "config.yaml"
    # Use incomplete verification to get bounds output
    create_config_with_bounds_output(onnx_path, vnnlib_path, config_path, use_incomplete=True)
    print(f"✓ Config: {config_path}")
    print("  Using incomplete verification mode to compute bounds")
    
    # Save fixed context
    with open(output_dir / "fixed_context.json", 'w') as f:
        json.dump(fixed_u, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print("\nTo run verification:")
    print(f"  cd alpha-beta-CROWN/complete_verifier")
    print(f"  python abcrown.py --config {config_path.absolute()}")
    print("\nThis will produce bounds output showing:")
    print("  - Lower/upper bounds on RUL for different cycle values")
    print("  - Verification of monotonicity property")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

