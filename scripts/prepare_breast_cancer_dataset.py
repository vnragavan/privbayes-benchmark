#!/usr/bin/env python3
"""
Prepare a small medical tabular dataset for benchmarking.

This script exports scikit-learn's Breast Cancer Wisconsin dataset to:
  data/breast_cancer.csv

Run:
  source .venv/bin/activate
  python scripts/prepare_breast_cancer_dataset.py
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "data" / "breast_cancer.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()

    # Ensure an explicit, stable target column name.
    # `bunch.frame` typically includes "target" already, but keep it explicit.
    if "target" not in df.columns:
        df["target"] = bunch.target

    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (rows={len(df)}, cols={len(df.columns)})")


if __name__ == "__main__":
    main()

