#!/usr/bin/env python3
"""
PyTorch Monotone-by-Design (MbD) model for RUL on the battery dataset.
Guarantee: as `cycle` increases, predicted RUL does NOT increase (global).

Inputs: CSVs produced by preprocess.py (mbd_{train,val,test}.csv + metadata.json)
Usage:
  python mbd_pt_train.py --data_dir ./prep_out --epochs 50 --batch_size 128 --out_dir ./mbd_pt
Outputs:
  - ./mbd_pt/metrics.json
  - ./mbd_pt/model.onnx  (ONNX export, ready for verifiers like Marabou / α,β-CROWN)
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="prep_out")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="mbd_pt")
    ap.add_argument("--cycle_col", type=str, default=None, help="override if needed")
    ap.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    ap.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clipping norm")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return ap.parse_args()

# --- Dataset helpers ---
def load_data(dir_path: Path):
    meta = json.loads((dir_path / "metadata.json").read_text())
    feats = meta["features_final_mbd"]
    target = meta["target"]
    cycle_col = meta["cycle_col"]
    train = pd.read_csv(dir_path / "mbd_train.csv")
    val   = pd.read_csv(dir_path / "mbd_val.csv")
    test  = pd.read_csv(dir_path / "mbd_test.csv")
    return feats, target, cycle_col, train, val, test, meta

def to_tensors(df, features, target, cycle_col, meta=None):
    # Normalize cycle to [0,1], then z = 1 - normalized_cycle
    # This ensures z is in [0,1] and is monotone increasing as cycle decreases
    if meta is not None and "bounds_raw_for_verification" in meta:
        Cmin, Cmax = meta["bounds_raw_for_verification"][cycle_col]
        cn = (df[cycle_col].to_numpy(dtype=np.float32) - Cmin) / max(1e-6, (Cmax - Cmin))
        z = (1.0 - cn).reshape(-1, 1).astype(np.float32)  # z in [0,1], increasing as cycle decreases
    else:
        # Fallback: use min/max from data
        c_min = df[cycle_col].min()
        c_max = df[cycle_col].max()
        cn = (df[cycle_col].to_numpy(dtype=np.float32) - c_min) / max(1e-6, (c_max - c_min))
        z = (1.0 - cn).reshape(-1, 1).astype(np.float32)
    
    u_cols = [c for c in features if c != cycle_col]
    U = df[u_cols].to_numpy(dtype=np.float32) if u_cols else np.zeros((len(df),0), np.float32)
    y = df[target].to_numpy(dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(z), torch.from_numpy(U), torch.from_numpy(y), u_cols

# --- Monotone blocks ---
class NonNegLinear(nn.Module):
    """
    Linear layer with weights constrained to be NON-NEGATIVE.
    w = softplus(raw_w), b is free (bias unconstrained).
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.raw_w = nn.Parameter(torch.empty(out_features, in_features))
        # Initialize raw_w - use xavier then ensure positive for softplus
        nn.init.xavier_uniform_(self.raw_w)
        # Ensure raw_w is positive so softplus gives reasonable weights
        with torch.no_grad():
            self.raw_w.data = torch.abs(self.raw_w.data) + 0.1
            # No special scaling needed if inputs are positive
        # Standard bias initialization (no special handling needed if inputs are positive)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        W = F.softplus(self.raw_w)  # >= 0
        out = x @ W.t()
        if self.bias is not None:
            out = out + self.bias
        return out

class MonotoneH(nn.Module):
    """
    h(z): scalar z -> hidden -> ... -> scalar, with NON-NEG weights + ReLU.
    Guarantees monotone INCREASING in z.
    
    Now z is in [0,1] (normalized), so no special bias needed.
    """
    def __init__(self, hidden_sizes=(32,32)):
        super().__init__()
        layers = []
        in_dim = 1
        for i, h in enumerate(hidden_sizes):
            # Standard initialization - z is now in [0,1]
            layers += [NonNegLinear(in_dim, h, bias=True), nn.ReLU()]
            in_dim = h
        # Final layer
        layers += [NonNegLinear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # shape (N,1)

class ContextNet(nn.Module):
    """Unconstrained context a(u) producing base output."""
    def __init__(self, in_dim, hidden=(64,64)):
        super().__init__()
        if in_dim == 0:
            # No context features
            self.net = nn.Identity()
            self.has_context = False
        else:
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            final_layer = nn.Linear(d, 1)
            # Initialize to produce outputs around 0 (will be scaled by output_scale)
            nn.init.constant_(final_layer.bias, 0.0)
            nn.init.xavier_uniform_(final_layer.weight, gain=0.5)  # Moderate initial weights
            layers += [final_layer]
            self.net = nn.Sequential(*layers)
            self.has_context = True

    def forward(self, u):
        return self.net(u) if self.has_context else torch.zeros((u.shape[0],1), device=u.device)

class NonNegGate(nn.Module):
    """b(u) >= 0 via softplus."""
    def __init__(self, in_dim, hidden=(32,)):
        super().__init__()
        if in_dim == 0:
            self.net = None
        else:
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

    def forward(self, u):
        if self.net is None:
            return torch.ones((u.shape[0],1), device=u.device)  # constant positive gain
        raw = self.net(u)
        return F.softplus(raw)  # >= 0

class MbDModel(nn.Module):
    """
    y = a(u) + b(u) * h(z)
    - h: monotone increasing in z (via nonneg weights + ReLU)
    - b(u) >= 0 (via softplus)
    => y is non-decreasing in z  => non-increasing in cycle (since z = 1 - normalized_cycle).
    
    Uses Softplus for final non-negativity (smoother gradients than ReLU).
    No global scale/offset - networks learn the range directly.
    """
    def __init__(self, u_dim):
        super().__init__()
        self.h = MonotoneH(hidden_sizes=(32,32))
        self.a = ContextNet(u_dim, hidden=(128,64))  # Increased capacity
        self.b = NonNegGate(u_dim, hidden=(64,))     # Increased capacity

    def forward(self, z, u):
        base = self.a(u)                # a(u)
        gain = self.b(u)                # b(u) >= 0
        hz   = self.h(z)                # increasing in z
        y    = base + gain * hz         # monotone in z
        return F.softplus(y)            # RUL >= 0 with smooth gradients

# --- Training / eval helpers ---
def batch_iter(z, u, y, batch):
    N = z.shape[0]
    for i in range(0, N, batch):
        yield z[i:i+batch], u[i:i+batch], y[i:i+batch]

def evaluate(model, z, u, y):
    model.eval()
    with torch.no_grad():
        pred = model(z, u)
        mse = torch.mean((pred - y)**2).item()
        mae = torch.mean(torch.abs(pred - y)).item()
    return {"mse": mse, "mae": mae}

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    feats, target, meta_cycle_col, train_df, val_df, test_df, meta = load_data(Path(args.data_dir))
    cycle_col = args.cycle_col or meta_cycle_col

    ztr, utr, ytr, u_cols = to_tensors(train_df, feats, target, cycle_col, meta=meta)
    zva, uva, yva, _     = to_tensors(val_df,   feats, target, cycle_col, meta=meta)
    zte, ute, yte, _     = to_tensors(test_df,  feats, target, cycle_col, meta=meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model no longer needs target stats - networks learn range directly
    model = MbDModel(u_dim=utr.shape[1]).to(device)
    # Use Adam optimizer with weight decay for regularization
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Learning rate scheduler to help escape local minima
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    ztr, utr, ytr = ztr.to(device), utr.to(device), ytr.to(device)
    zva, uva, yva = zva.to(device), uva.to(device), yva.to(device)
    zte, ute, yte = zte.to(device), ute.to(device), yte.to(device)

    # Diagnostic: Check initial predictions
    with torch.no_grad():
        initial_pred = model(zva[:10], uva[:10])
        print(f"Initial val predictions (first 10): {initial_pred.squeeze().cpu().numpy()}")
        print(f"Initial val targets (first 10): {yva[:10].squeeze().cpu().numpy()}")
        print(f"Target stats - mean: {ytr.mean().item():.2f}, std: {ytr.std().item():.2f}, min: {ytr.min().item():.2f}, max: {ytr.max().item():.2f}")

    best_val = float("inf"); patience=args.patience; bad=0; best_state=None

    for epoch in range(1, args.epochs+1):
        model.train()
        train_losses = []
        for bz, bu, by in batch_iter(ztr, utr, ytr, args.batch_size):
            opt.zero_grad()
            pred = model(bz, bu)
            loss = F.mse_loss(pred, by)
            train_losses.append(loss.item())
            loss.backward()
            # Gradient clipping - more aggressive for high learning rates to prevent instability
            # If LR > 0.005, use tighter clipping
            clip_norm = args.grad_clip if args.lr <= 0.005 else min(args.grad_clip, 5.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            opt.step()
        val_metrics = evaluate(model, zva, uva, yva)
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        
        # Update learning rate based on validation loss
        scheduler.step(val_metrics["mse"])
        
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]; bad=0; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
        
        current_lr = opt.param_groups[0]['lr']
        print(f"[{epoch}] train_loss={avg_train_loss:.4f} val MAE={val_metrics['mae']:.4f} MSE={val_metrics['mse']:.4f} lr={current_lr:.6f}")
        if bad >= patience:
            print("Early stopping.")
            break

    if best_state: 
        model.load_state_dict(best_state)
        # Save the best model state dict
        torch.save(best_state, out_dir / "model.pt")
        print(f"Saved best model state dict to: {out_dir / 'model.pt'}")
    else:
        # If no improvement, save current model anyway
        torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"Saved model state dict to: {out_dir / 'model.pt'} (no improvement during training)")

    metrics = {
        "train": evaluate(model, ztr, utr, ytr),
        "val":   evaluate(model, zva, uva, yva),
        "test":  evaluate(model, zte, ute, yte),
        "features": feats,
        "target": target,
        "cycle_col": cycle_col,
        "u_cols": u_cols
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved metrics:", out_dir / "metrics.json")

    # ONNX export (single input = concatenated [z, u])
    try:
        import onnx
        import onnxscript
        model.eval()
        dummy = (torch.zeros(1,1, device=device), torch.zeros(1, utr.shape[1], device=device))
        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m=m
            def forward(self, x):
                z = x[:, :1]
                u = x[:, 1:]
                return self.m(z, u)
        wrap = Wrapper(model).to(device)
        concat_in = torch.zeros(1, 1+utr.shape[1], device=device)
        onnx_path = str(out_dir / "model.onnx")
        # Use opset_version 18 to match PyTorch defaults and avoid conversion errors
        # Fixed shape export works well with verification tools (Marabou, α,β-CROWN)
        torch.onnx.export(
            wrap, concat_in, onnx_path,
            input_names=["input"], 
            output_names=["output"],
            opset_version=18,  # Updated from 13 to avoid version conversion errors
            do_constant_folding=True
            # Removed dynamic_axes - fixed shape works better for verification
        )
        print("Exported ONNX to:", onnx_path)
    except ImportError as e:
        print(f"Warning: ONNX export skipped - missing dependency: {e}")
        print("  Install with: pip install onnx onnxscript")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        print("  Model training completed successfully, but ONNX export was skipped.")

if __name__ == "__main__":
    main()
