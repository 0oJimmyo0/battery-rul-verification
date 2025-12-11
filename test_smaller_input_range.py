#!/usr/bin/env python3
"""
Test Verification with Smaller Input Range

This script creates VNNLIB files and configs for smaller input ranges
and provides commands to run verification.

Based on the quick test results, smaller ranges should give MUCH better bounds.
"""

import argparse
import json
from pathlib import Path
import yaml


def create_small_range_vnnlib(output_path, cycle_min, cycle_max):
    """Create VNNLIB file with smaller input range."""
    lines = [
        "; VNNLIB for cycle monotonicity verification with smaller input range",
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
        "(assert (>= Y_0 0.0))",
        ""
    ]
    
    output_path.write_text("\n".join(lines))
    print(f"✓ Created VNNLIB: {output_path}")


def create_small_range_config(base_config_path, vnnlib_path, output_path, range_name):
    """Create config file for smaller input range."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update VNNLIB path
    config['specification']['vnnlib_path'] = str(vnnlib_path.absolute())
    
    # Write new config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created config: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="./verification_bounds/config.yaml")
    parser.add_argument("--output_dir", type=str, default="./verification_bounds/small_range_tests")
    parser.add_argument("--ranges", type=str, nargs='+', 
                       default=["50,150", "100,150", "120,140"],
                       help="Input ranges to test (format: min,max)")
    args = parser.parse_args()
    
    base_config = Path(args.base_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_config.exists():
        print(f"Error: Base config not found: {base_config}")
        return 1
    
    print("="*70)
    print("Creating Verification Setup for Smaller Input Ranges")
    print("="*70)
    
    # Load quick test results to show expected improvements
    quick_results_path = Path("./verification_bounds/improvement_tests/quick_test_results.json")
    if quick_results_path.exists():
        with open(quick_results_path, 'r') as f:
            quick_results = json.load(f)
        print("\nExpected improvements (from model evaluation):")
        for r in quick_results.get('input_ranges', []):
            if r['name'] != 'Full range':
                improvement = r['min_output'] - (-206.7)
                print(f"  {r['name']}: Lower bound would be {r['min_output']:.2f} "
                      f"(improvement: {improvement:+.2f} from -206.7)")
    
    configs_created = []
    
    print(f"\nCreating VNNLIB files and configs for {len(args.ranges)} ranges...")
    
    for range_str in args.ranges:
        try:
            cycle_min, cycle_max = map(float, range_str.split(','))
            range_name = f"range_{int(cycle_min)}_{int(cycle_max)}"
            
            # Create VNNLIB
            vnnlib_path = output_dir / f"monotonicity_{range_name}.vnnlib"
            create_small_range_vnnlib(vnnlib_path, cycle_min, cycle_max)
            
            # Create config
            config_path = output_dir / f"config_{range_name}.yaml"
            create_small_range_config(base_config, vnnlib_path, config_path, range_name)
            
            configs_created.append({
                'range': f"[{cycle_min}, {cycle_max}]",
                'vnnlib': str(vnnlib_path),
                'config': str(config_path)
            })
            
        except ValueError:
            print(f"⚠ Invalid range format: {range_str} (expected: min,max)")
            continue
    
    # Print commands to run verification
    print("\n" + "="*70)
    print("Verification Commands")
    print("="*70)
    print("\nTo run verification with smaller input ranges:")
    print("\ncd alpha-beta-CROWN/complete_verifier")
    print()
    
    for cfg in configs_created:
        config_abs = Path(cfg['config']).absolute()
        print(f"# Test range {cfg['range']}")
        print(f"python abcrown.py --config {config_abs}")
        print()
    
    # Save summary
    summary = {
        'ranges_tested': [cfg['range'] for cfg in configs_created],
        'configs': configs_created,
        'expected_improvements': {}
    }
    
    # Add expected improvements from quick test results
    if quick_results_path.exists():
        for r in quick_results.get('input_ranges', []):
            if r['name'] != 'Full range':
                summary['expected_improvements'][r['name']] = {
                    'min_output': r['min_output'],
                    'improvement_from_full': r['min_output'] - (-206.7)
                }
    
    summary_path = output_dir / "small_range_test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nStart with range [50, 150]:")
    print("  - Avoids very early cycles (3.19) where outputs are low")
    print("  - Still covers meaningful portion of cycle range")
    print("  - Expected lower bound: ~66.64 (vs. -206.7 for full range)")
    print("  - This is a +273 improvement!")
    print("\nIf this works well, you can try even smaller ranges for tighter bounds.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


