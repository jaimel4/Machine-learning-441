"""
CLI Entrypoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .dataset import load_dataset, map_target
from .experiment import run_nested_cv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Path to breastCancer.csv")
    ap.add_argument("--out", type=Path, default=Path("results"), help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--outer", type=int, default=5, help="Outer CV folds")
    ap.add_argument("--inner", type=int, default=3, help="Inner CV folds")
    ap.add_argument("--calibrate", action="store_true", help="Enable isotonic calibration after inner CV")
    return ap.parse_args()

def main():
    args = parse_args()
    df = load_dataset(args.data)
    df = map_target(df)
    run_nested_cv(df, args.out, seed=args.seed, n_outer=args.outer, n_inner=args.inner, do_calibration=args.calibrate)

if __name__ == "__main__":
    main()
