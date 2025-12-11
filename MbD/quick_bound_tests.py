#!/usr/bin/env python3
"""
Quick Tests for Bound Improvement

This script tests a few quick configurations that might improve bounds:
1. Different bound propagation methods
2. Different batch sizes
3. Smaller input ranges (quick test with model evaluation)
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_smaller_input_ranges(model, meta, cycle_col, u_cols, train_df):
    """Test what bounds would be with smaller input ranges (using model evaluation)."""
    print("\n" + "="*70)
    print("QUICK TEST: Smaller Input Ranges")
    print("="*70)
    
    bounds = meta["bounds_raw_for_verification"]
    cycle_min, cycle_max = bounds[cycle_col]
    full_range = cycle_max - cycle_min
    
    # Test ranges
    test_ranges = [
        ("Full range", cycle_min, cycle_max),
        ("Middle range [50, 150]", 50.0, 150.0),
        ("Small range [100, 150]", 100.0, 150.0),
        ("Very small [120, 140]", 120.0, 140.0),
    ]
    
    fixed_u = {col: float(train_df[col].median()) for col in u_cols}
    u_tensor = torch.tensor([[fixed_u[col] for col in u_cols]], dtype=torch.float32)
    
    results = []
    
    for name, c_min, c_max in test_ranges:
        # Sample inputs in this range
        cycles = np.linspace(c_min, c_max, 1000)
        outputs = []
        
        with torch.no_grad():
            for cycle in cycles:
                # Normalize using full range bounds
                cn = (cycle - cycle_min) / max(1e-6, (cycle_max - cycle_min))
                z = torch.tensor([[1.0 - cn]], dtype=torch.float32)
                
                base = model.a(u_tensor)
                gain = model.b(u_tensor)
                hz = model.h(z)
                y_pre = base + gain * hz
                y = F.softplus(y_pre)
                
                outputs.append(y.item())
        
        min_out = min(outputs)
        max_out = max(outputs)
        range_size = c_max - c_min
        
        results.append({
            'name': name,
            'cycle_range': (c_min, c_max),
            'range_size': range_size,
            'min_output': min_out,
            'max_output': max_out,
            'output_range': max_out - min_out
        })
        
        print(f"\n{name}:")
        print(f"  Cycle: [{c_min:.2f}, {c_max:.2f}] (size: {range_size:.2f})")
        print(f"  Output: [{min_out:.4f}, {max_out:.4f}] (size: {max_out - min_out:.4f})")
        print(f"  → Bound would be at least: {min_out:.4f} (vs. -206.7 for full range)")
    
    return results


def create_config_variants(base_config_path, output_dir):
    """Create config variants for different bound propagation methods."""
    print("\n" + "="*70)
    print("Creating Config Variants for Different Methods")
    print("="*70)
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    methods = ['alpha-crown', 'crown', 'crown-optimized', 'ibp']
    batch_sizes = [512, 1024, 2048, 4096]
    
    configs_created = []
    
    # Create configs for different methods
    for method in methods:
        config = base_config.copy()
        config['solver']['bound_prop_method'] = method
        config_path = output_dir / f"config_{method}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        configs_created.append(str(config_path))
        print(f"  ✓ Created: {config_path.name} (method: {method})")
    
    # Create configs for different batch sizes (with alpha-crown)
    for batch_size in batch_sizes:
        if batch_size == 2048:  # Skip default
            continue
        config = base_config.copy()
        config['solver']['batch_size'] = batch_size
        config_path = output_dir / f"config_batch_{batch_size}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        configs_created.append(str(config_path))
        print(f"  ✓ Created: {config_path.name} (batch_size: {batch_size})")
    
    return configs_created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./mbd_pt")
    parser.add_argument("--data_dir", type=str, default="./prep_out")
    parser.add_argument("--config_template", type=str, default="./verification_bounds/config.yaml")
    parser.add_argument("--output_dir", type=str, default="./verification_bounds/improvement_tests")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    config_template = Path(args.config_template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Quick Bound Improvement Tests")
    print("="*70)
    
    results = {}
    
    # Test 1: Smaller input ranges (quick model evaluation)
    if TORCH_AVAILABLE:
        print("\nLoading model for input range tests...")
        sys.path.insert(0, str(Path(__file__).parent))
        from mbd_train import MbDModel, load_data
        
        feats, target, cycle_col, train_df, val_df, test_df, meta = load_data(data_dir)
        u_cols = [f for f in feats if f != cycle_col]
        u_dim = len(u_cols)
        
        model = MbDModel(u_dim=u_dim)
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location='cpu'))
        model.eval()
        
        results['input_ranges'] = test_smaller_input_ranges(
            model, meta, cycle_col, u_cols, train_df
        )
    else:
        print("\n⚠ PyTorch not available, skipping input range tests")
    
    # Test 2: Create config variants
    if config_template.exists():
        configs = create_config_variants(config_template, output_dir)
        results['configs_created'] = configs
        print(f"\n✓ Created {len(configs)} config variants")
        print(f"\nTo test these configs, run:")
        print(f"  cd alpha-beta-CROWN/complete_verifier")
        for config in configs[:4]:  # Show first 4
            config_name = Path(config).name
            print(f"  python abcrown.py --config {Path(config).absolute()}")
    else:
        print(f"\n⚠ Config template not found: {config_template}")
    
    # Save results
    results_file = output_dir / "quick_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if 'input_ranges' in results:
        print("\nInput Range Analysis:")
        for r in results['input_ranges']:
            if r['name'] != 'Full range':
                improvement = r['min_output'] - (-206.7)
                print(f"  {r['name']}: Lower bound would be {r['min_output']:.2f} "
                      f"(improvement: {improvement:+.2f} from -206.7)")
    
    print("\nNext Steps:")
    print("  1. Review input range results above")
    print("  2. Run verification with different configs (see commands above)")
    print("  3. Compare results to see if any method improves bounds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

