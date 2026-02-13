#!/usr/bin/env python3
"""
Prepare a per-dataset "public schema" JSON from a CSV.

IMPORTANT:
- This script can *infer* bounds/domains from the CSV, which is NOT DP-safe if the CSV is private.
- Use it to generate a *candidate schema* that you review, edit, and then treat as public side information,
  or to populate known-public domains/bounds from documentation.

Output format matches scripts/comprehensive_comparison.py --schema:
{
  "dataset": "...",
  "target_col": "...",
  "label_domain": [...],
  "public_bounds": { col: [L,U], ... },
  "public_categories": { col: [cat1,cat2,...], ... },
  "provenance": { ... }
}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _infer_target_col(cols: list[str]) -> str | None:
    for c in ["target", "income", "label", "class", "outcome"]:
        if c in cols:
            return c
    return None


def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    if s.dtype == "object":
        x = pd.to_numeric(s, errors="coerce")
        return bool(x.notna().mean() >= 0.95)
    return False


def _numeric_bounds(s: pd.Series, pad_frac: float) -> list[float]:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0, 1.0]
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    span = vmax - vmin
    pad = (pad_frac * span) if span > 0 else max(abs(vmin) * pad_frac, 1.0)
    return [vmin - pad, vmax + pad]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Input CSV path")
    ap.add_argument("--out", type=Path, required=True, help="Output schema JSON path")
    ap.add_argument("--dataset-name", type=str, default=None, help="Optional dataset name tag")
    ap.add_argument("--target-col", type=str, default=None, help="Target/label column name (auto-detect if omitted)")
    ap.add_argument("--pad-frac", type=float, default=0.05, help="Padding fraction for numeric bounds (default 0.05)")
    ap.add_argument(
        "--infer-categories",
        action="store_true",
        help="Infer categorical domains for non-numeric columns (WARNING: data-derived unless domains are truly public).",
    )
    ap.add_argument(
        "--max-categories",
        type=int,
        default=200,
        help="If inferring categories, cap domain size; larger domains are omitted.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    cols = [str(c) for c in df.columns]
    target_col = args.target_col or _infer_target_col(cols)

    public_bounds: dict[str, list[float]] = {}
    public_categories: dict[str, list[str]] = {}

    for c in cols:
        if c == target_col:
            continue
        s = df[c]
        if _is_numeric_series(s):
            public_bounds[c] = _numeric_bounds(s, float(args.pad_frac))
        elif args.infer_categories:
            u = pd.Series(s, copy=False).astype("string").dropna().unique().tolist()
            u = [str(x) for x in u]
            if 0 < len(u) <= int(args.max_categories):
                public_categories[c] = sorted(u)

    # Label domain (if target exists)
    label_domain: list[str] = []
    if target_col and target_col in df.columns:
        u = pd.Series(df[target_col], copy=False).astype("string").dropna().unique().tolist()
        label_domain = sorted([str(x) for x in u])
        if label_domain:
            public_categories[target_col] = label_domain

    schema: dict[str, Any] = {
        "dataset": args.dataset_name or args.data.stem,
        "target_col": target_col,
        "label_domain": label_domain,
        "public_bounds": public_bounds,
        "public_categories": public_categories,
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_csv": str(args.data),
            "pad_frac": float(args.pad_frac),
            "inferred_categories": bool(args.infer_categories),
            "max_categories": int(args.max_categories),
            "note": "Candidate schema. If derived from private data, treat as NON-public unless DP-released.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote schema to: {args.out}")


if __name__ == "__main__":
    main()

