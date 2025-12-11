#!/usr/bin/env python3
"""
preprocess.py — Battery dataset preprocessing for Monotone-by-Design (MbD) and MLP models.

Features:
- Loads a CSV (default: Battery_dataset.csv).
- Splits data by battery_id OR by time (cycle) into train/val/test.
- Imputes missing values with train medians.
- Winsorizes (clips) features to train-set percentiles (default: 1st–99th).
- Engineers monotone-friendly features (temperature abnormality).
- Exports TWO variants:
  (A) MbD: raw physical units after impute+clip (+ engineered features), no standardization.
  (B) MLP: standardized with train StandardScaler params (mean, std), saved in metadata.
- Saves verification bounds (raw and scaled) derived from train percentiles.
- Emits: CSVs for train/val/test, metadata.json with bounds + preprocessing params.

Usage:
  python preprocess.py \
      --input /path/to/Battery_dataset.csv \
      --outdir ./prep_out \
      --target RUL \
      --split-by battery \
      --val-frac 0.15 \
      --test-frac 0.2 \
      --winsor-low 0.01 \
      --winsor-high 0.99 \
      --tref 25.0 \
      --drop-battery-id

Notes:
- Monotone spec you likely want to verify later: cycle ↑ ⇒ predicted RUL ↓.
- Keep MbD inputs in raw units (post-impute/clip). The PWL calibrators/lattices handle scaling.
- For the MLP baseline, standardized features & scaled bounds are provided.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

DEFAULT_FEATURES_ORDER = [
    "cycle",
    "chI", "chV", "chT",
    "disI", "disV", "disT",
    "BCt"
]

ENGINEERED_FEATURES = [
    "abn_chT", "abn_disT"
]

METADATA_FILENAME = "metadata.json"


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess battery CSV for MbD and MLP pipelines.")
    p.add_argument("--input", type=str, default="Battery_dataset.csv", help="Path to input CSV.")
    p.add_argument("--outdir", type=str, default="prep_out", help="Output directory.")
    p.add_argument("--target", type=str, default="RUL", choices=["RUL", "SOH"], help="Prediction target.")
    p.add_argument("--split-by", type=str, default="battery", choices=["battery", "time", "random"],
                   help="How to split: by battery_id groups, time within each battery, or random rows.")
    p.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction.")
    p.add_argument("--test-frac", type=float, default=0.20, help="Test fraction.")
    p.add_argument("--winsor-low", type=float, default=0.01, help="Lower percentile for clipping (0..1).")
    p.add_argument("--winsor-high", type=float, default=0.99, help="Upper percentile for clipping (0..1).")
    p.add_argument("--tref", type=float, default=25.0, help="Reference temperature (°C) for abnormality features.")
    p.add_argument("--drop-battery-id", action="store_true", help="Drop battery_id column from outputs.")
    p.add_argument("--id-col", type=str, default="battery_id", help="Name of the battery ID column (if present).")
    p.add_argument("--cycle-col", type=str, default="cycle", help="Name of the cycle column.")
    p.add_argument("--feature-cols", type=str, default="",
                   help="Comma-separated feature columns to use (overrides default order).")
    return p.parse_args()


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def read_csv(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def choose_features(df: pd.DataFrame, feature_cols_arg: str, id_col: str, cycle_col: str, target: str) -> List[str]:
    if feature_cols_arg:
        feats = [c.strip() for c in feature_cols_arg.split(",") if c.strip()]
    else:
        feats = [c for c in DEFAULT_FEATURES_ORDER if c in df.columns]

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Requested feature columns missing from CSV: {missing}")

    if id_col in df.columns:
        feats = [c for c in feats if c != id_col]
    if target in df.columns:
        feats = [c for c in feats if c != target]
    if cycle_col in df.columns and cycle_col not in feats:
        feats = [cycle_col] + feats
    return feats


def split_by_battery(df: pd.DataFrame, id_col: str, val_frac: float, test_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found for split-by battery.")
    batteries = df[id_col].unique().tolist()
    rng.shuffle(batteries)
    n = len(batteries)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_ids = set(batteries[:n_test])
    val_ids = set(batteries[n_test:n_test+n_val])
    train_ids = set(batteries[n_test+n_val:])
    train = df[df[id_col].isin(train_ids)].copy()
    val = df[df[id_col].isin(val_ids)].copy()
    test = df[df[id_col].isin(test_ids)].copy()
    return train, val, test


def split_by_time(df: pd.DataFrame, id_col: str, cycle_col: str, val_frac: float, test_frac: float):
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found for split-by time.")
    parts = []
    for bid, grp in df.groupby(id_col):
        grp = grp.sort_values(cycle_col)
        n = len(grp)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test = grp.iloc[-n_test:]
        val = grp.iloc[-(n_test + n_val):-n_test] if n_test + n_val <= n else grp.iloc[:n_val]
        train = grp.drop(test.index).drop(val.index)
        parts.append(("train", train))
        parts.append(("val", val))
        parts.append(("test", test))
    train = pd.concat([p[1] for p in parts if p[0] == "train"]).sort_index()
    val   = pd.concat([p[1] for p in parts if p[0] == "val"]).sort_index()
    test  = pd.concat([p[1] for p in parts if p[0] == "test"]).sort_index()
    return train, val, test


def split_random(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int = 42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = df.iloc[:n_test]
    val  = df.iloc[n_test:n_test+n_val]
    train= df.iloc[n_test+n_val:]
    return train, val, test


def fit_imputer(train: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    med = train[cols].median(numeric_only=True).to_dict()
    for k, v in med.items():
        if pd.isna(v):
            med[k] = 0.0
    return med


def apply_imputer(df: pd.DataFrame, med: Dict[str, float]):
    for c, m in med.items():
        if c in df.columns:
            df[c] = df[c].fillna(m)
    return df


def winsorize_fit(train: pd.DataFrame, cols: List[str], low: float, high: float) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for c in cols:
        series = train[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) == 0:
            bounds[c] = (0.0, 0.0)
        else:
            lo = float(series.quantile(low))
            hi = float(series.quantile(high))
            if lo > hi:
                lo, hi = hi, lo
            bounds[c] = (lo, hi)
    return bounds


def winsorize_apply(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]):
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def engineer_features(df: pd.DataFrame, tref: float) -> pd.DataFrame:
    if "chT" in df.columns:
        df["abn_chT"] = (df["chT"] - float(tref)).abs()
    if "disT" in df.columns:
        df["abn_disT"] = (df["disT"] - float(tref)).abs()
    return df


def compute_bounds_from_train(train: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    b = {}
    for c in cols:
        if c in train.columns:
            s = train[c].astype(float)
            b[c] = (float(np.min(s)), float(np.max(s)))
    return b


def fit_standard_scaler(train: pd.DataFrame, cols: List[str]):
    mean = train[cols].mean().to_dict()
    std = train[cols].std(ddof=0).replace(0.0, 1.0).to_dict()
    return mean, std


def apply_standard_scaler(df: pd.DataFrame, cols: List[str], mean: Dict[str, float], std: Dict[str, float]):
    df_scaled = df.copy()
    for c in cols:
        if c in df_scaled.columns:
            m = float(mean[c])
            s = float(std[c]) if float(std[c]) != 0.0 else 1.0
            df_scaled[c] = (df_scaled[c] - m) / s
    return df_scaled


def transform_bounds_to_scaled(bounds: Dict[str, Tuple[float, float]], mean: Dict[str, float], std: Dict[str, float]):
    scaled = {}
    for c, (lo, hi) in bounds.items():
        m = float(mean.get(c, 0.0))
        s = float(std.get(c, 1.0)) or 1.0
        slo = (lo - m) / s
        shi = (hi - m) / s
        scaled[c] = (float(min(slo, shi)), float(max(slo, shi)))
    return scaled


def save_csvs(outdir: Path, prefix: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    outdir.mkdir(parents=True, exist_ok=True)
    train.to_csv(outdir / f"{prefix}_train.csv", index=False)
    val.to_csv(outdir / f"{prefix}_val.csv", index=False)
    test.to_csv(outdir / f"{prefix}_test.csv", index=False)


def main():
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_csv(input_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV. Available: {list(df.columns)}")

    features = choose_features(df, args.feature_cols, args.id_col, args.cycle_col, args.target)
    cols_needed = [c for c in features] + [args.target]
    if args.id_col in df.columns:
        cols_needed.append(args.id_col)
    df = df[cols_needed].copy()

    if args.split_by == "battery":
        train, val, test = split_by_battery(df, args.id_col, args.val_frac, args.test_frac)
    elif args.split_by == "time":
        train, val, test = split_by_time(df, args.id_col, args.cycle_col, args.val_frac, args.test_frac)
    else:
        train, val, test = split_random(df, args.val_frac, args.test_frac)

    imputer_medians = fit_imputer(train, features)

    train = apply_imputer(train, imputer_medians)
    val   = apply_imputer(val, imputer_medians)
    test  = apply_imputer(test, imputer_medians)

    clip_bounds = winsorize_fit(train, features, args.winsor_low, args.winsor_high)
    train = winsorize_apply(train, clip_bounds)
    val   = winsorize_apply(val, clip_bounds)
    test  = winsorize_apply(test, clip_bounds)

    train = engineer_features(train, args.tref)
    val   = engineer_features(val, args.tref)
    test  = engineer_features(test, args.tref)

    all_features_mbd = features + [f for f in ENGINEERED_FEATURES if f in train.columns]

    if args.drop_battery_id and args.id_col in train.columns:
        train = train.drop(columns=[args.id_col])
        val   = val.drop(columns=[args.id_col])
        test  = test.drop(columns=[args.id_col])

    mbd_train = train[all_features_mbd + [args.target]].copy()
    mbd_val   = val[all_features_mbd + [args.target]].copy()
    mbd_test  = test[all_features_mbd + [args.target]].copy()

    bounds_raw = compute_bounds_from_train(mbd_train, all_features_mbd)
    save_csvs(outdir, "mbd", mbd_train, mbd_val, mbd_test)

    mean, std = fit_standard_scaler(mbd_train, all_features_mbd)
    mlp_train = apply_standard_scaler(mbd_train, all_features_mbd, mean, std)
    mlp_val   = apply_standard_scaler(mbd_val,   all_features_mbd, mean, std)
    mlp_test  = apply_standard_scaler(mbd_test,  all_features_mbd, mean, std)

    save_csvs(outdir, "mlp", mlp_train, mlp_val, mlp_test)

    bounds_scaled = transform_bounds_to_scaled(bounds_raw, mean, std)

    metadata = {
        "input_csv": str(input_path.resolve()),
        "target": args.target,
        "split_by": args.split_by,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "winsor_low": args.winsor_low,
        "winsor_high": args.winsor_high,
        "tref_celsius": args.tref,
        "drop_battery_id": args.drop_battery_id,
        "id_col": args.id_col if args.id_col in df.columns else None,
        "cycle_col": args.cycle_col,
        "features_base": features,
        "features_engineered_included": [f for f in ENGINEERED_FEATURES if f in mbd_train.columns],
        "features_final_mbd": all_features_mbd,
        "bounds_raw_for_verification": bounds_raw,
        "scaler_mean": mean,
        "scaler_std": std,
        "bounds_scaled_for_verification": bounds_scaled,
        "notes": (
            "- MbD CSVs are in RAW physical units after imputation, winsorization, and feature engineering.\n"
            "- MLP CSVs are standardized using train-set mean/std recorded above.\n"
            "- Use bounds_raw_for_verification to build box constraints for MbD verification.\n"
            "- Use bounds_scaled_for_verification if verifying the standardized MLP (feed scaled inputs).\n"
            "- Suggested monotonicity for RUL: as 'cycle' increases, predicted RUL must not increase "
            "(verify f(y) <= f(x) with y_cycle >= x_cycle and y_other == x_other over these bounds)."
        )
    }

    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Wrote MbD CSVs + MLP CSVs + metadata.json to: {outdir.resolve()}")
    print(f"[OK] Final MbD features: {all_features_mbd}")

if __name__ == "__main__":
    main()
