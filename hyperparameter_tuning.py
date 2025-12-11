#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for MbD Model

This script performs grid search or random search over hyperparameters to find
the best configuration for the MbD model.

Usage:
    python hyperparameter_tuning.py --search_type grid --trials 10
    python hyperparameter_tuning.py --search_type random --trials 20
"""

import argparse
import json
import itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import subprocess
import sys

# Import model and training functions
sys.path.insert(0, str(Path(__file__).parent))
from mbd_train import MbDModel, load_data, to_tensors, evaluate, batch_iter


def train_model_with_params(
    ztr, utr, ytr, zva, uva, yva, zte, ute, yte,
    u_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    weight_decay: float = 1e-5,
    grad_clip: float = 10.0,
    patience: int = 20,
    device: torch.device = None,
    verbose: bool = False
) -> Dict:
    """Train model with given hyperparameters and return metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with custom architecture
    model = MbDModel(u_dim=u_dim).to(device)
    # Modify hidden sizes if needed (this requires modifying the model class)
    # For now, use default architecture
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    best_val = float("inf")
    best_state = None
    bad = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for bz, bu, by in batch_iter(ztr, utr, ytr, batch_size):
            opt.zero_grad()
            pred = model(bz, bu)
            loss = F.mse_loss(pred, by)
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
        
        val_metrics = evaluate(model, zva, uva, yva)
        scheduler.step(val_metrics["mse"])
        
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        
        if bad >= patience:
            break
        
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch}: val MAE={val_metrics['mae']:.4f}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    train_metrics = evaluate(model, ztr, utr, ytr)
    val_metrics = evaluate(model, zva, uva, yva)
    test_metrics = evaluate(model, zte, ute, yte)
    
    return {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_val_mae": best_val
    }


def grid_search(
    ztr, utr, ytr, zva, uva, yva, zte, ute, yte,
    u_dim: int,
    param_grid: Dict,
    device: torch.device,
    epochs: int
) -> List[Dict]:
    """Perform grid search over hyperparameters."""
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Grid search: {len(combinations)} combinations to try")
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        # Convert numpy types to native Python types
        params = convert_to_native_types(params)
        print(f"\n[{i}/{len(combinations)}] Testing: {params}")
        
        try:
            metrics = train_model_with_params(
                ztr, utr, ytr, zva, uva, yva, zte, ute, yte,
                u_dim=u_dim,
                lr=params.get("lr", 0.001),
                batch_size=params.get("batch_size", 128),
                epochs=epochs,
                weight_decay=params.get("weight_decay", 1e-5),
                grad_clip=params.get("grad_clip", 10.0),
                patience=20,
                device=device,
                verbose=False
            )
            
            # Convert metrics to native Python types for JSON serialization
            metrics_native = convert_to_native_types(metrics)
            
            result = {
                "params": params,
                "metrics": metrics_native,
                "val_mae": float(metrics["val"]["mae"])
            }
            results.append(result)
            print(f"  Result: val MAE={metrics['val']['mae']:.4f}, test MAE={metrics['test']['mae']:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return results


def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


def random_search(
    ztr, utr, ytr, zva, uva, yva, zte, ute, yte,
    u_dim: int,
    param_distributions: Dict,
    n_trials: int,
    device: torch.device,
    epochs: int,
    seed: int = 42
) -> List[Dict]:
    """Perform random search over hyperparameters."""
    results = []
    rng = np.random.default_rng(seed)
    
    print(f"Random search: {n_trials} trials")
    
    for i in range(1, n_trials + 1):
        # Sample random parameters
        params = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, list):
                params[key] = rng.choice(dist)
            elif isinstance(dist, tuple) and len(dist) == 2:
                # Continuous range
                if isinstance(dist[0], int):
                    params[key] = int(rng.integers(dist[0], dist[1] + 1))  # Convert to native int
                else:
                    params[key] = float(rng.uniform(dist[0], dist[1]))  # Convert to native float
            else:
                params[key] = dist
        
        # Convert numpy types to native Python types
        params = convert_to_native_types(params)
        
        print(f"\n[{i}/{n_trials}] Testing: {params}")
        
        try:
            metrics = train_model_with_params(
                ztr, utr, ytr, zva, uva, yva, zte, ute, yte,
                u_dim=u_dim,
                lr=params.get("lr", 0.001),
                batch_size=params.get("batch_size", 128),
                epochs=epochs,
                weight_decay=params.get("weight_decay", 1e-5),
                grad_clip=params.get("grad_clip", 10.0),
                patience=20,
                device=device,
                verbose=False
            )
            
            # Convert metrics to native Python types for JSON serialization
            metrics_native = convert_to_native_types(metrics)
            
            result = {
                "params": params,
                "metrics": metrics_native,
                "val_mae": float(metrics["val"]["mae"])
            }
            results.append(result)
            print(f"  Result: val MAE={metrics['val']['mae']:.4f}, test MAE={metrics['test']['mae']:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for MbD model")
    parser.add_argument("--data_dir", type=str, default="./prep_out",
                       help="Directory containing preprocessed data")
    parser.add_argument("--search_type", type=str, default="random",
                       choices=["grid", "random"],
                       help="Search type: grid or random")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of trials (for random search) or combinations (for grid)")
    parser.add_argument("--output", type=str, default="./tuning_results.json",
                       help="Output file for results")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs per trial (reduced for tuning)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    
    print("=" * 70)
    print("Hyperparameter Tuning for MbD Model")
    print("=" * 70)
    print(f"Search type: {args.search_type}")
    print(f"Trials: {args.trials}")
    print(f"Device: {device}")
    print()
    
    # Load data
    print("Loading data...")
    feats, target, cycle_col, train_df, val_df, test_df, meta = load_data(data_dir)
    ztr, utr, ytr, u_cols = to_tensors(train_df, feats, target, cycle_col, meta=meta)
    zva, uva, yva, _ = to_tensors(val_df, feats, target, cycle_col, meta=meta)
    zte, ute, yte, _ = to_tensors(test_df, feats, target, cycle_col, meta=meta)
    
    ztr, utr, ytr = ztr.to(device), utr.to(device), ytr.to(device)
    zva, uva, yva = zva.to(device), uva.to(device), yva.to(device)
    zte, ute, yte = zte.to(device), ute.to(device), yte.to(device)
    
    u_dim = utr.shape[1]
    print(f"✓ Data loaded: u_dim={u_dim}")
    print()
    
    # Define parameter space
    if args.search_type == "grid":
        param_grid = {
            "lr": [0.0005, 0.001, 0.002, 0.005],
            "batch_size": [64, 128, 256],
            "weight_decay": [0, 1e-5, 1e-4],
            "grad_clip": [1.0, 5.0, 10.0]
        }
        results = grid_search(
            ztr, utr, ytr, zva, uva, yva, zte, ute, yte,
            u_dim, param_grid, device, args.epochs
        )
    else:  # random search
        param_distributions = {
            "lr": (0.0001, 0.01),  # Log-uniform would be better, but uniform is simpler
            "batch_size": [64, 128, 256],
            "weight_decay": (0, 1e-3),
            "grad_clip": (1.0, 20.0)
        }
        results = random_search(
            ztr, utr, ytr, zva, uva, yva, zte, ute, yte,
            u_dim, param_distributions, args.trials, device, args.epochs
        )
    
    # Sort by validation MAE
    results.sort(key=lambda x: x["val_mae"])
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Tuning Results Summary")
    print("=" * 70)
    print(f"Total trials: {len(results)}")
    print(f"\nTop 5 configurations:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Val MAE: {result['val_mae']:.4f}, Test MAE: {result['metrics']['test']['mae']:.4f}")
        print(f"   Params: {result['params']}")
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

