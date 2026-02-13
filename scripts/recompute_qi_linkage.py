#!/usr/bin/env python3
"""
Recompute QI-linkage rates for an existing results JSON using saved synthetic CSVs.

Why this exists:
  - QI linkage can be sensitive to discretization choices.
  - If the linkage definition changes, we can update stored metrics without
    rerunning expensive training/sampling/audits.

This script:
  - loads a comprehensive_results_*.json
  - for each successful run with a synthetic_data_path, reloads REAL from --data
  - recomputes qi_linkage_rate (and optionally ERMR) using the same robust method
    as scripts/comprehensive_comparison.py (bins derived from REAL, applied to SYN,
    and avoids degenerate QI columns)
  - writes a new JSON file next to the original.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _robust_qi_cols(real: pd.DataFrame, k: int = 3) -> list[str]:
    rnum = real.select_dtypes(include=[np.number])
    if rnum.shape[1] == 0:
        return []

    stats: list[tuple[float, str]] = []
    for c in rnum.columns:
        s = pd.to_numeric(rnum[c], errors="coerce").dropna()
        if s.empty:
            continue
        nunq = int(s.nunique(dropna=True))
        if nunq < 10:
            continue
        # Avoid columns where qcut collapses to too few bins (degenerate QI)
        try:
            codes = pd.qcut(s, q=5, labels=False, duplicates="drop")
            if int(pd.Series(codes).nunique(dropna=True)) < 3:
                continue
        except Exception:
            continue
        var = float(s.var())
        if not np.isfinite(var) or var <= 0:
            continue
        stats.append((var, c))
    stats.sort(reverse=True)
    return [c for _, c in stats[: int(k)]]


def _bin_from_real(real_col: pd.Series, syn_col: pd.Series, q: int = 5) -> tuple[pd.Series, pd.Series]:
    rr = pd.to_numeric(real_col, errors="coerce")
    ss = pd.to_numeric(syn_col, errors="coerce")
    ok = rr.notna()
    if ok.sum() < 2:
        return pd.Series([0] * len(real_col)), pd.Series([0] * len(syn_col))
    try:
        _, bins = pd.qcut(rr[ok], q=int(q), retbins=True, duplicates="drop")
        if len(bins) >= 2:
            ss = ss.clip(lower=float(bins[0]), upper=float(bins[-1]))
        r_codes = pd.cut(rr, bins=bins, labels=False, include_lowest=True)
        s_codes = pd.cut(ss, bins=bins, labels=False, include_lowest=True)
    except Exception:
        lo = float(np.nanmin(rr.to_numpy()))
        hi = float(np.nanmax(rr.to_numpy()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series([0] * len(real_col)), pd.Series([0] * len(syn_col))
        bins = np.linspace(lo, hi, int(q) + 1)
        ss = ss.clip(lower=float(bins[0]), upper=float(bins[-1]))
        r_codes = pd.cut(rr, bins=bins, labels=False, include_lowest=True)
        s_codes = pd.cut(ss, bins=bins, labels=False, include_lowest=True)

    r_out = pd.to_numeric(r_codes, errors="coerce").fillna(-1).astype(int)
    s_out = pd.to_numeric(s_codes, errors="coerce").fillna(-1).astype(int)
    return r_out, s_out


def qi_linkage_rate(real: pd.DataFrame, syn: pd.DataFrame) -> float | None:
    # Match the updated definition in scripts/comprehensive_comparison.py:
    # choose up to 2 categorical + remaining numeric, then compute fraction of SYN rows
    # that match a UNIQUE real QI tuple (support==1).
    if len(syn) == 0:
        return None

    # categorical candidates
    cat_candidates: list[tuple[float, str]] = []
    for c in real.columns:
        if c not in syn.columns:
            continue
        if pd.api.types.is_numeric_dtype(real[c]):
            continue
        rr = real[c].astype("string").fillna("__NA__")
        nunq = int(rr.nunique(dropna=True))
        if nunq < 2 or nunq > 50:
            continue
        top_frac = float(rr.value_counts(normalize=True, dropna=True).iloc[0])
        if top_frac > 0.90:
            continue
        p = rr.value_counts(normalize=True, dropna=True).to_numpy()
        ent = float(-(p * np.log(p + 1e-12)).sum())
        cat_candidates.append((ent, c))
    cat_candidates.sort(reverse=True)
    cat_cols = [c for _, c in cat_candidates[:2]]

    # numeric candidates
    num_candidates: list[tuple[float, str]] = []
    rnum = real.select_dtypes(include=[np.number])
    for c in rnum.columns:
        if c not in syn.columns:
            continue
        s = pd.to_numeric(rnum[c], errors="coerce").dropna()
        if s.empty or int(s.nunique(dropna=True)) < 10:
            continue
        try:
            codes = pd.qcut(s, q=5, labels=False, duplicates="drop")
            if int(pd.Series(codes).nunique(dropna=True)) < 3:
                continue
        except Exception:
            continue
        var = float(s.var())
        if np.isfinite(var) and var > 0:
            num_candidates.append((var, c))
    num_candidates.sort(reverse=True)
    num_cols = [c for _, c in num_candidates[: max(1, 3 - len(cat_cols))]]

    qi_cols = (cat_cols + num_cols)[:3]
    if not qi_cols:
        return None

    real_qi: dict[str, Any] = {}
    syn_qi: dict[str, Any] = {}
    for c in qi_cols:
        if pd.api.types.is_numeric_dtype(real[c]):
            rc, sc = _bin_from_real(real[c], syn[c], q=5)
            real_qi[c] = rc
            syn_qi[c] = sc
        else:
            real_qi[c] = real[c].astype("string").fillna("__NA__")
            syn_qi[c] = syn[c].astype("string").fillna("__NA__")

    real_qi_df = pd.DataFrame(real_qi)
    syn_qi_df = pd.DataFrame(syn_qi)
    real_tuples = list(map(tuple, real_qi_df.to_numpy()))
    vc = pd.Series(real_tuples).value_counts()
    unique_real = set(vc[vc == 1].index.tolist())
    syn_tuples = list(map(tuple, syn_qi_df.to_numpy()))
    return float(sum(1 for t in syn_tuples if t in unique_real) / len(syn_tuples))


def ermr(real: pd.DataFrame, syn: pd.DataFrame) -> float | None:
    try:
        real_hash = set(pd.util.hash_pandas_object(real, index=False).astype(str))
        syn_hash = pd.util.hash_pandas_object(syn, index=False).astype(str)
        return float(sum(1 for h in syn_hash if h in real_hash) / len(syn)) if len(syn) else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Real dataset CSV used for the run")
    ap.add_argument("--results-json", type=Path, required=True, help="Path to comprehensive_results_*.json")
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path (default: <in>_relink.json)")
    ap.add_argument("--recompute-ermr", action="store_true", help="Also recompute ERMR from saved CSVs.")
    args = ap.parse_args()

    real = pd.read_csv(args.data)
    rows = json.loads(args.results_json.read_text())
    if not isinstance(rows, list):
        raise SystemExit("results JSON must be a list")

    updated = 0
    skipped = 0
    for r in rows:
        if not isinstance(r, dict) or not r.get("success"):
            continue
        syn_path = r.get("synthetic_data_path")
        if not syn_path:
            skipped += 1
            continue
        p = Path(syn_path)
        if not p.exists():
            skipped += 1
            continue
        syn = pd.read_csv(p)
        r["qi_linkage_rate"] = qi_linkage_rate(real, syn)
        if args.recompute_ermr:
            r["exact_row_match_rate"] = ermr(real, syn)
        updated += 1

    out = args.out or args.results_json.with_name(args.results_json.stem + "_relink.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote: {out}")
    print(f"Updated rows: {updated}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()

