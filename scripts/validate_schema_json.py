#!/usr/bin/env python3
"""
Validate a per-dataset schema JSON used by scripts/comprehensive_comparison.py --schema.

This performs structural checks (required keys/types) and (optionally) consistency
checks against a dataset CSV (e.g., referenced columns exist).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _fail(msg: str) -> None:
    raise SystemExit(f"Schema validation failed: {msg}")


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(float(x))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        _fail(f"could not parse JSON: {e}")
    if not isinstance(obj, dict):
        _fail("top-level JSON must be an object")
    return obj


def _validate_schema_obj(schema: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    # Required keys (for our runner)
    for k in ["dataset", "target_col", "label_domain", "public_bounds", "public_categories"]:
        if k not in schema:
            _fail(f"missing required key '{k}'")

    if not isinstance(schema["dataset"], str) or not schema["dataset"].strip():
        _fail("'dataset' must be a non-empty string")

    if schema["target_col"] is not None and not isinstance(schema["target_col"], str):
        _fail("'target_col' must be a string or null")

    if not isinstance(schema["label_domain"], list):
        _fail("'label_domain' must be a list (possibly empty)")
    if any(not isinstance(x, (str, int, float, bool)) for x in schema["label_domain"]):
        warnings.append("label_domain contains non-primitive values; consider using strings.")

    pb = schema["public_bounds"]
    if not isinstance(pb, dict):
        _fail("'public_bounds' must be an object mapping col -> [L,U]")
    for col, b in pb.items():
        if not isinstance(col, str) or not col:
            _fail("public_bounds contains non-string column name")
        if not (isinstance(b, list) and len(b) == 2 and _is_num(b[0]) and _is_num(b[1])):
            _fail(f"public_bounds[{col}] must be [L,U] with finite numbers")
        if float(b[1]) <= float(b[0]):
            _fail(f"public_bounds[{col}] must satisfy U > L")

    pc = schema["public_categories"]
    if not isinstance(pc, dict):
        _fail("'public_categories' must be an object mapping col -> [cat1,cat2,...]")
    for col, cats in pc.items():
        if not isinstance(col, str) or not col:
            _fail("public_categories contains non-string column name")
        if not isinstance(cats, list):
            _fail(f"public_categories[{col}] must be a list")
        # allow empty list but warn
        if len(cats) == 0:
            warnings.append(f"public_categories[{col}] is empty; strict DP may fail for this column.")
        # ensure strings (recommended)
        if any(not isinstance(x, (str, int, float, bool)) for x in cats):
            warnings.append(f"public_categories[{col}] contains non-primitive values; consider using strings.")
        # duplicates
        s = [str(x) for x in cats]
        if len(set(s)) != len(s):
            warnings.append(f"public_categories[{col}] contains duplicates (after string-cast).")

    # Optional provenance
    if "provenance" in schema and not isinstance(schema["provenance"], dict):
        _fail("'provenance' must be an object if provided")

    return warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path, required=True, help="Schema JSON path")
    ap.add_argument("--data", type=Path, default=None, help="Optional dataset CSV path to cross-check columns")
    ap.add_argument("--strict", action="store_true", help="Fail on warnings (in addition to errors)")
    args = ap.parse_args()

    schema = _load_json(args.schema)
    warnings = _validate_schema_obj(schema)

    if args.data is not None:
        df = pd.read_csv(args.data, nrows=5)
        cols = set(map(str, df.columns))
        # cross-check columns referenced by schema
        pb = schema.get("public_bounds", {}) or {}
        pc = schema.get("public_categories", {}) or {}
        missing = sorted([c for c in set(pb.keys()) | set(pc.keys()) if c not in cols])
        if missing:
            _fail(f"schema references columns not found in data: {missing[:10]}{'...' if len(missing)>10 else ''}")
        t = schema.get("target_col")
        if t and t not in cols:
            _fail(f"target_col '{t}' not found in data columns")

    if warnings:
        for w in warnings:
            print("WARNING:", w)
        if args.strict:
            _fail(f"{len(warnings)} warning(s) treated as error due to --strict")

    print(f"OK: {args.schema}")


if __name__ == "__main__":
    main()

