#!/usr/bin/env python3
"""
Visualize MbD Model Performance

Creates plots showing:
1. True vs Predicted RUL scatter plot
2. Residual plots
3. Error distribution
4. Performance metrics visualization
5. Prediction trends over cycles

Usage:
    python visualize_performance.py --model_dir ./mbd_pt --data_dir ./prep_out --output_dir ./plots
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mbd_train import MbDModel, load_data, to_tensors

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_model_and_data(model_dir: Path, data_dir: Path, device: torch.device):
    """Load trained model and data."""
    # Load metadata
    meta_path = model_dir / "metrics.json"
    with open(meta_path, 'r') as f:
        metrics = json.load(f)
    
    # Load data
    feats, target, cycle_col, train_df, val_df, test_df, meta = load_data(data_dir)
    
    # Create model
    u_dim = len(metrics['u_cols'])
    model = MbDModel(u_dim=u_dim).to(device)
    
    # Load state dict
    state_dict_path = model_dir / "model.pt"
    if state_dict_path.exists():
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        print(f"✓ Loaded model from {state_dict_path}")
    else:
        raise FileNotFoundError(f"Model state dict not found at {state_dict_path}")
    
    model.eval()
    
    return model, feats, target, cycle_col, train_df, val_df, test_df, metrics, meta


def make_predictions(model, df, feats, target, cycle_col, device, meta=None):
    """Make predictions on a dataframe."""
    z, u, y_true, u_cols = to_tensors(df, feats, target, cycle_col, meta=meta)
    z, u, y_true = z.to(device), u.to(device), y_true.to(device)
    
    with torch.no_grad():
        y_pred = model(z, u).cpu().numpy().flatten()
    
    y_true = y_true.cpu().numpy().flatten()
    
    # Convert z back to cycle: z = 1 - cn, where cn = (cycle - Cmin)/(Cmax - Cmin)
    # So: cn = 1 - z, and cycle = cn * (Cmax - Cmin) + Cmin
    z_np = z.cpu().numpy().flatten()
    if meta is not None and "bounds_raw_for_verification" in meta:
        Cmin, Cmax = meta["bounds_raw_for_verification"][cycle_col]
        cn = 1.0 - z_np  # normalized cycle [0,1]
        cycles = cn * (Cmax - Cmin) + Cmin  # convert back to actual cycle values
    else:
        # Fallback: use original cycle values from dataframe
        cycles = df[cycle_col].to_numpy()
    
    return y_true, y_pred, cycles


def plot_true_vs_predicted(y_true, y_pred, title_suffix="", output_path=None):
    """Plot true vs predicted RUL."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    # Add metrics to plot
    textstr = f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('True RUL', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted RUL', fontsize=12, fontweight='bold')
    ax.set_title(f'True vs Predicted RUL {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_residuals(y_true, y_pred, cycles, title_suffix="", output_path=None):
    """Plot residuals vs predicted and vs cycle."""
    residuals = y_pred - y_true
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted RUL', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals (Predicted - True)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Residuals vs Predicted RUL {title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs Cycle
    ax2.scatter(cycles, residuals, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Predicted - True)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Residuals vs Cycle {title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_error_distribution(y_true, y_pred, title_suffix="", output_path=None):
    """Plot error distribution."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error distribution
    ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.2f}')
    ax1.set_xlabel('Error (Predicted - True)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Error Distribution {title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Absolute error distribution
    ax2.hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    ax2.axvline(x=np.mean(abs_errors), color='g', linestyle='--', linewidth=2,
                label=f'MAE: {np.mean(abs_errors):.2f}')
    ax2.set_xlabel('Absolute Error |Predicted - True|', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Absolute Error Distribution {title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_predictions_over_cycle(y_true, y_pred, cycles, title_suffix="", output_path=None):
    """Plot predictions over cycles (for visualization)."""
    # Sort by cycle for better visualization
    sort_idx = np.argsort(cycles)
    cycles_sorted = cycles[sort_idx]
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot true and predicted
    ax.scatter(cycles_sorted, y_true_sorted, alpha=0.5, s=30, label='True RUL', 
               color='blue', edgecolors='black', linewidths=0.3)
    ax.scatter(cycles_sorted, y_pred_sorted, alpha=0.5, s=30, label='Predicted RUL',
               color='red', marker='x', linewidths=1)
    
    ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax.set_ylabel('RUL', fontsize=12, fontweight='bold')
    ax.set_title(f'RUL Predictions vs Cycle {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_metrics_comparison(metrics, output_path=None):
    """Plot metrics comparison across train/val/test."""
    datasets = ['train', 'val', 'test']
    mae_values = [metrics[d]['mae'] for d in datasets]
    mse_values = [metrics[d]['mse'] for d in datasets]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE comparison
    bars1 = ax1.bar(datasets, mae_values, color=['skyblue', 'lightgreen', 'lightcoral'], 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax1.set_title('MAE Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE comparison
    bars2 = ax2.bar(datasets, mse_values, color=['skyblue', 'lightgreen', 'lightcoral'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('MSE Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, mse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize MbD model performance")
    parser.add_argument("--model_dir", type=str, default="./mbd_pt",
                       help="Directory containing trained model")
    parser.add_argument("--data_dir", type=str, default="./prep_out",
                       help="Directory containing preprocessed data")
    parser.add_argument("--output_dir", type=str, default="./plots",
                       help="Directory to save plots")
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=["train", "val", "test", "all"],
                       help="Which dataset to visualize")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("MbD Model Performance Visualization")
    print("=" * 70)
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()
    
    # Load model and data
    print("Loading model and data...")
    try:
        model, feats, target, cycle_col, train_df, val_df, test_df, metrics, meta = \
            load_model_and_data(model_dir, data_dir, device)
        print("✓ Model and data loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model/data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Plot metrics comparison
    print("\nGenerating metrics comparison plot...")
    plot_metrics_comparison(metrics, output_dir / "metrics_comparison.png")
    
    # Generate plots for each dataset
    datasets_to_plot = []
    if args.dataset == "all":
        datasets_to_plot = [
            ("train", train_df, "Training"),
            ("val", val_df, "Validation"),
            ("test", test_df, "Test")
        ]
    else:
        dataset_map = {
            "train": ("train", train_df, "Training"),
            "val": ("val", val_df, "Validation"),
            "test": ("test", test_df, "Test")
        }
        datasets_to_plot = [dataset_map[args.dataset]]
    
    for dataset_name, df, title_prefix in datasets_to_plot:
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Make predictions
        y_true, y_pred, cycles = make_predictions(
            model, df, feats, target, cycle_col, device, meta=meta
        )
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # Generate plots
        suffix = f"({title_prefix} Set)"
        
        print(f"  Generating plots...")
        plot_true_vs_predicted(
            y_true, y_pred, suffix, 
            output_dir / f"true_vs_predicted_{dataset_name}.png"
        )
        
        plot_residuals(
            y_true, y_pred, cycles, suffix,
            output_dir / f"residuals_{dataset_name}.png"
        )
        
        plot_error_distribution(
            y_true, y_pred, suffix,
            output_dir / f"error_distribution_{dataset_name}.png"
        )
        
        plot_predictions_over_cycle(
            y_true, y_pred, cycles, suffix,
            output_dir / f"predictions_over_cycle_{dataset_name}.png"
        )
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"All plots saved to: {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

