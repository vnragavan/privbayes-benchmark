#!/usr/bin/env python3
"""
INTERNAL / LEGACY

Augment an existing comprehensive_results_*.json with downstream metrics.

Useful when older results have missing downstream metrics (e.g., DPMM label collapse),
and you want consistent plotting without re-running synthesis.

Example:
  source .venv/bin/activate
  python scripts/internal/augment_results_json_with_downstream.py \
    --real data/breast_cancer.csv \
    --results <results_dir>/comprehensive_results_*.json \
    --target-col target
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pbbench.enhanced_metrics import comprehensive_downstream_metrics


def _json_default(o: Any):
    # numpy scalars -> python scalars
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    # numpy arrays -> lists
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, required=True, help="Real CSV path")
    ap.add_argument("--results", type=Path, required=True, help="comprehensive_results_*.json path")
    ap.add_argument("--target-col", type=str, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path (default: <results>_augmented.json)")
    args = ap.parse_args()

    real = pd.read_csv(args.real)
    items = json.loads(args.results.read_text())
    if not isinstance(items, list):
        raise SystemExit("Expected JSON list.")

    updated = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        if not it.get("success", False):
            continue

        dm = it.get("downstream_metrics")
        needs = (dm is None) or (isinstance(dm, dict) and len(dm) == 0)
        syn_path = it.get("synthetic_data_path")
        if not needs or not syn_path:
            continue

        syn_csv = Path(syn_path)
        if not syn_csv.exists():
            # try relative to results json
            syn_csv = (args.results.parent / syn_path).resolve()
        if not syn_csv.exists():
            continue

        syn = pd.read_csv(syn_csv)
        it["downstream_metrics"] = comprehensive_downstream_metrics(
            real,
            syn,
            target_col=args.target_col,
            n_folds=5,
            random_seed=int(it.get("seed", 0) or 0),
            include_details=False,
            n_jobs=1,
            n_bootstrap=0,
        )
        updated += 1

    out_path = args.out or args.results.with_name(args.results.stem + "_augmented.json")
    out_path.write_text(json.dumps(items, indent=2, default=_json_default))
    print(f"Updated items: {updated}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

