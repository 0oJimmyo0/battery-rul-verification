#!/usr/bin/env python3
"""
Test Different Bound Propagation Methods

This script tests various bound propagation methods and configurations
to see if we can improve the bounds from the current -206.7.

Methods to test:
1. alpha-crown (current)
2. crown
3. crown-optimized
4. ibp
5. Different solver configurations
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import yaml

def create_config_with_method(config_template_path, output_path, method, additional_opts=None):
    """Create a config file with a specific bound propagation method."""
    with open(config_template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set the bound propagation method
    config['solver']['bound_prop_method'] = method
    
    # Add any additional options
    if additional_opts:
        for key, value in additional_opts.items():
            if key not in config:
                config[key] = {}
            config[key].update(value)
    
    # Write the new config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path


def run_verification(config_path, timeout=600):
    """Run verification with a given config and extract results."""
    print(f"\nRunning verification with config: {config_path.name}")
    
    # Change to the alpha-beta-CROWN directory
    abcrown_dir = Path(__file__).parent / "alpha-beta-CROWN" / "complete_verifier"
    
    # Run the verification
    start_time = time.time()
    try:
        result = subprocess.run(
            ["python", "abcrown.py", "--config", str(config_path.absolute())],
            cwd=str(abcrown_dir),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        # Parse output
        output_lines = result.stdout.split('\n')
        stderr_lines = result.stderr.split('\n')
        
        # Try to extract bounds from output
        lb_rhs = None
        status = None
        domains = None
        
        for line in output_lines + stderr_lines:
            if 'lb-rhs' in line.lower() or 'lower bound' in line.lower():
                # Try to extract number
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'lb-rhs' in part.lower() or 'lower' in part.lower():
                            if i + 1 < len(parts):
                                lb_rhs = float(parts[i + 1])
                                break
                except:
                    pass
            if 'result' in line.lower() or 'status' in line.lower():
                if 'timeout' in line.lower():
                    status = 'timeout'
                elif 'sat' in line.lower():
                    status = 'sat'
                elif 'unsat' in line.lower():
                    status = 'unsat'
            if 'domains' in line.lower():
                try:
                    parts = line.split()
                    for part in parts:
                        if part.replace(',', '').isdigit():
                            domains = int(part.replace(',', ''))
                            break
                except:
                    pass
        
        # Try to load from pickle file
        try:
            import pickle
            out_pkl = abcrown_dir / "out.pkl"
            if out_pkl.exists():
                with open(out_pkl, 'rb') as f:
                    data = pickle.load(f)
                    if 'results' in data:
                        status = data['results']
                    if 'init_alpha_crown' in data:
                        lb_rhs = float(data['init_alpha_crown'][0].item()) if hasattr(data['init_alpha_crown'][0], 'item') else float(data['init_alpha_crown'][0])
        except Exception as e:
            print(f"  Could not load pickle file: {e}")
        
        return {
            'method': config_path.stem.replace('config_', ''),
            'status': status or 'unknown',
            'lb_rhs': lb_rhs,
            'domains': domains,
            'time': elapsed,
            'timeout': elapsed >= timeout,
            'stdout': result.stdout[-500:] if result.stdout else '',  # Last 500 chars
            'stderr': result.stderr[-500:] if result.stderr else ''
        }
        
    except subprocess.TimeoutExpired:
        return {
            'method': config_path.stem.replace('config_', ''),
            'status': 'timeout',
            'lb_rhs': None,
            'domains': None,
            'time': timeout,
            'timeout': True,
            'stdout': '',
            'stderr': 'Timeout after {} seconds'.format(timeout)
        }
    except Exception as e:
        return {
            'method': config_path.stem.replace('config_', ''),
            'status': 'error',
            'lb_rhs': None,
            'domains': None,
            'time': time.time() - start_time,
            'timeout': False,
            'stdout': '',
            'stderr': str(e)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_template", type=str, default="./verification_bounds/config.yaml")
    parser.add_argument("--output_dir", type=str, default="./verification_bounds/method_tests")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per method (seconds)")
    parser.add_argument("--methods", type=str, nargs='+', 
                       default=['alpha-crown', 'crown', 'crown-optimized', 'ibp'],
                       help="Methods to test")
    args = parser.parse_args()
    
    config_template = Path(args.config_template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not config_template.exists():
        print(f"Error: Config template not found: {config_template}")
        return 1
    
    print("="*70)
    print("Testing Different Bound Propagation Methods")
    print("="*70)
    print(f"\nTemplate config: {config_template}")
    print(f"Output directory: {output_dir}")
    print(f"Timeout per method: {args.timeout} seconds")
    print(f"Methods to test: {args.methods}")
    
    results = []
    
    # Test each method
    for method in args.methods:
        print(f"\n{'='*70}")
        print(f"Testing method: {method}")
        print(f"{'='*70}")
        
        # Create config for this method
        config_path = output_dir / f"config_{method}.yaml"
        create_config_with_method(config_template, config_path, method)
        
        # Run verification
        result = run_verification(config_path, timeout=args.timeout)
        results.append(result)
        
        # Print summary
        print(f"\n  Status: {result['status']}")
        if result['lb_rhs'] is not None:
            print(f"  Lower bound (lb-rhs): {result['lb_rhs']:.4f}")
        if result['domains'] is not None:
            print(f"  Domains visited: {result['domains']:,}")
        print(f"  Time: {result['time']:.2f} seconds")
        if result['timeout']:
            print(f"  ⚠ Timeout occurred")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print(f"\n{'Method':<20} {'Status':<12} {'Lower Bound':<15} {'Time (s)':<12} {'Domains':<15}")
    print("-" * 70)
    
    for r in results:
        lb_str = f"{r['lb_rhs']:.4f}" if r['lb_rhs'] is not None else "N/A"
        domains_str = f"{r['domains']:,}" if r['domains'] is not None else "N/A"
        print(f"{r['method']:<20} {r['status']:<12} {lb_str:<15} {r['time']:<12.2f} {domains_str:<15}")
    
    # Find best method
    valid_results = [r for r in results if r['lb_rhs'] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x['lb_rhs'])
        print(f"\n✓ Best lower bound: {best['lb_rhs']:.4f} (method: {best['method']})")
        
        if best['lb_rhs'] < 0:
            print(f"  → Still negative, but {'better' if best['lb_rhs'] > -206.7 else 'worse'} than original (-206.7)")
        else:
            print(f"  → Positive! Property verified!")
    
    # Save results
    results_file = output_dir / "method_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


