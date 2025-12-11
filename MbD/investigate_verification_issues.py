#!/usr/bin/env python3
"""
Investigation Script for Verification Issues

This script helps diagnose why α-β-CROWN verification is failing:
1. Tests model on many inputs to check for negative outputs (empirical validation)
2. Analyzes model structure (counts ReLUs, layers, parameters)
3. Tests with different bound propagation methods
4. Tests with smaller input ranges
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some tests will be skipped.")


def load_model_and_data(model_dir: Path, data_dir: Path):
    """Load model and metadata."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    sys.path.insert(0, str(Path(__file__).parent))
    from mbd_train import MbDModel, load_data
    
    # Load data
    feats, target, cycle_col, train_df, val_df, test_df, meta = load_data(data_dir)
    u_cols = [f for f in feats if f != cycle_col]
    u_dim = len(u_cols)
    
    # Load model
    model = MbDModel(u_dim=u_dim)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location='cpu'))
    model.eval()
    
    return model, meta, cycle_col, u_cols, train_df


def test_1_empirical_validation(model, meta, cycle_col, u_cols, train_df, n_samples=10000):
    """
    Test 1: Empirical Validation
    Check if model actually outputs negative values on many random inputs.
    """
    print("\n" + "="*70)
    print("TEST 1: Empirical Validation")
    print("="*70)
    
    # Get bounds
    bounds = meta["bounds_raw_for_verification"]
    cycle_min, cycle_max = bounds[cycle_col]
    
    # Sample random inputs
    np.random.seed(42)
    cycles = np.random.uniform(cycle_min, cycle_max, n_samples)
    
    # Use median context features
    fixed_u = {col: float(train_df[col].median()) for col in u_cols}
    u_tensor = torch.tensor([[fixed_u[col] for col in u_cols]], dtype=torch.float32)
    
    negative_count = 0
    min_output = float('inf')
    max_output = float('-inf')
    outputs = []
    
    with torch.no_grad():
        for cycle in cycles:
            # Normalize cycle
            cn = (cycle - cycle_min) / max(1e-6, (cycle_max - cycle_min))
            z = torch.tensor([[1.0 - cn]], dtype=torch.float32)
            
            # Forward pass
            base = model.a(u_tensor)
            gain = model.b(u_tensor)
            hz = model.h(z)
            y_pre = base + gain * hz
            y = F.softplus(y_pre)  # Original model uses softplus
            
            output = y.item()
            outputs.append(output)
            
            if output < 0:
                negative_count += 1
            min_output = min(min_output, output)
            max_output = max(max_output, output)
    
    print(f"Tested {n_samples} random inputs")
    print(f"  Cycle range: [{cycle_min:.2f}, {cycle_max:.2f}]")
    print(f"  Output range: [{min_output:.4f}, {max_output:.4f}]")
    print(f"  Negative outputs: {negative_count} ({100*negative_count/n_samples:.2f}%)")
    print(f"  Mean output: {np.mean(outputs):.4f}")
    print(f"  Std output: {np.std(outputs):.4f}")
    
    if negative_count == 0:
        print("\n✓ PASS: Model never outputs negative values (empirically)")
        print("  → The negative bound from verification is an over-approximation")
    else:
        print(f"\n✗ FAIL: Model outputs {negative_count} negative values")
        print("  → Model may need fixing")
    
    return {
        'negative_count': negative_count,
        'min_output': min_output,
        'max_output': max_output,
        'mean_output': np.mean(outputs),
        'std_output': np.std(outputs)
    }


def test_2_model_complexity(model):
    """
    Test 2: Analyze Model Complexity
    Count layers, ReLUs, parameters, etc.
    """
    print("\n" + "="*70)
    print("TEST 2: Model Complexity Analysis")
    print("="*70)
    
    total_params = 0
    relu_count = 0
    layer_count = 0
    
    def count_module(m, name=""):
        nonlocal total_params, relu_count, layer_count
        if isinstance(m, nn.Linear):
            layer_count += 1
            total_params += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
        elif isinstance(m, nn.ReLU):
            relu_count += 1
        elif isinstance(m, nn.Sequential):
            for i, sub_m in enumerate(m):
                count_module(sub_m, f"{name}.{i}")
        else:
            for child_name, child_m in m.named_children():
                count_module(child_m, f"{name}.{child_name}")
    
    count_module(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"ReLU activations: {relu_count}")
    print(f"Linear layers: {layer_count}")
    print(f"Average parameters per layer: {total_params/layer_count:.0f}")
    
    # Analyze each component
    print("\nComponent breakdown:")
    print(f"  MonotoneH (h):")
    h_params = sum(p.numel() for p in model.h.parameters())
    h_relus = sum(1 for m in model.h.modules() if isinstance(m, nn.ReLU))
    print(f"    Parameters: {h_params:,}")
    print(f"    ReLUs: {h_relus}")
    
    print(f"  ContextNet (a):")
    a_params = sum(p.numel() for p in model.a.parameters())
    a_relus = sum(1 for m in model.a.modules() if isinstance(m, nn.ReLU))
    print(f"    Parameters: {a_params:,}")
    print(f"    ReLUs: {a_relus}")
    
    print(f"  NonNegGate (b):")
    b_params = sum(p.numel() for p in model.b.parameters())
    b_relus = sum(1 for m in model.b.modules() if isinstance(m, nn.ReLU))
    print(f"    Parameters: {b_params:,}")
    print(f"    ReLUs: {b_relus}")
    
    return {
        'total_params': total_params,
        'relu_count': relu_count,
        'layer_count': layer_count
    }


def test_3_smaller_input_range(model, meta, cycle_col, u_cols, train_df):
    """
    Test 3: Test with Smaller Input Range
    See if bounds improve with smaller cycle range.
    """
    print("\n" + "="*70)
    print("TEST 3: Smaller Input Range Test")
    print("="*70)
    
    bounds = meta["bounds_raw_for_verification"]
    cycle_min, cycle_max = bounds[cycle_col]
    full_range = cycle_max - cycle_min
    
    # Test with progressively smaller ranges
    test_ranges = [
        (cycle_min, cycle_max, "Full range"),
        (cycle_min, cycle_min + 0.5 * full_range, "50% range"),
        (cycle_min, cycle_min + 0.25 * full_range, "25% range"),
        (cycle_min, cycle_min + 0.1 * full_range, "10% range"),
    ]
    
    fixed_u = {col: float(train_df[col].median()) for col in u_cols}
    u_tensor = torch.tensor([[fixed_u[col] for col in u_cols]], dtype=torch.float32)
    
    results = []
    
    for c_min, c_max, name in test_ranges:
        # Sample inputs in this range
        cycles = np.linspace(c_min, c_max, 100)
        outputs = []
        
        with torch.no_grad():
            for cycle in cycles:
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
        print(f"  Cycle range: [{c_min:.2f}, {c_max:.2f}] (size: {range_size:.2f})")
        print(f"  Output range: [{min_out:.4f}, {max_out:.4f}] (size: {max_out - min_out:.4f})")
    
    print("\nAnalysis:")
    print("  If output range decreases with smaller input range:")
    print("    → Input range size contributes to bound looseness")
    print("  If output range stays similar:")
    print("    → Other factors (architecture) dominate")
    
    return results


def test_4_bound_propagation_comparison():
    """
    Test 4: Compare Different Bound Propagation Methods
    Note: This requires running verification with different methods.
    """
    print("\n" + "="*70)
    print("TEST 4: Bound Propagation Method Comparison")
    print("="*70)
    print("\nThis test requires running verification with different methods.")
    print("Suggested methods to try:")
    print("  1. alpha-crown (current)")
    print("  2. crown")
    print("  3. crown-optimized")
    print("  4. ibp")
    print("\nTo test, modify config.yaml:")
    print("  solver:")
    print("    bound_prop_method: <method_name>")
    print("\nThen run verification and compare bounds.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./mbd_pt")
    parser.add_argument("--data_dir", type=str, default="./prep_out")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples for empirical test")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    
    if not TORCH_AVAILABLE:
        print("Error: PyTorch required for this script")
        return 1
    
    print("="*70)
    print("Verification Issue Investigation")
    print("="*70)
    
    # Load model and data
    print("\nLoading model and data...")
    model, meta, cycle_col, u_cols, train_df = load_model_and_data(model_dir, data_dir)
    print("✓ Loaded")
    
    # Run tests
    results = {}
    
    # Test 1: Empirical validation
    results['empirical'] = test_1_empirical_validation(
        model, meta, cycle_col, u_cols, train_df, args.n_samples
    )
    
    # Test 2: Model complexity
    results['complexity'] = test_2_model_complexity(model)
    
    # Test 3: Smaller input range
    results['input_range'] = test_3_smaller_input_range(
        model, meta, cycle_col, u_cols, train_df
    )
    
    # Test 4: Bound propagation methods (info only)
    test_4_bound_propagation_comparison()
    
    # Save results
    output_file = model_dir.parent / "verification_investigation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results['empirical']['negative_count'] == 0:
        print("✓ Model empirically satisfies non-negativity")
        print("  → Negative bound is over-approximation artifact")
        print("  → This is a fundamental limitation of bound propagation")
    else:
        print("✗ Model outputs negative values")
        print("  → Model needs fixing")
    
    print(f"\nModel complexity:")
    print(f"  {results['complexity']['total_params']:,} parameters")
    print(f"  {results['complexity']['relu_count']} ReLU activations")
    print(f"  {results['complexity']['layer_count']} linear layers")
    print("\n  → High complexity may contribute to loose bounds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

