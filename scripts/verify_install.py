"""
Minimal installation verification for privbayes-benchmark.

Run:
  source .venv/bin/activate
  python scripts/verify_install.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _fail(msg: str, *, exc: BaseException | None = None) -> "None":
    print(f"✗ {msg}")
    if exc is not None:
        print(f"  {type(exc).__name__}: {exc}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"✓ {msg}")


def _add_src_to_syspath(repo_root: Path) -> None:
    src = repo_root / "src"
    if not src.exists():
        _fail(f"Expected src/ directory at {src}")
    sys.path.insert(0, str(src))


def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001 - want full import errors
        _fail(f"Import failed: {name}", exc=e)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _add_src_to_syspath(repo_root)

    print("============================================================")
    print("Verifying privbayes-benchmark installation")
    print("============================================================")

    # Core deps
    numpy = _import("numpy")
    pandas = _import("pandas")
    sklearn = _import("sklearn")
    torch = _import("torch")
    _ok(f"Imported core deps: numpy={numpy.__version__}, pandas={pandas.__version__}")
    _ok(f"Imported ML deps: sklearn={sklearn.__version__}, torch={torch.__version__}")

    # Project imports (src/ layout)
    _import("pbbench")
    _import("pbbench.enhanced_metrics")
    _import("pbbench.variants.pb_enhanced")
    _import("pbbench.variants.pb_synthcity")
    _import("pbbench.variants.pb_datasynthesizer")
    _ok("Imported pbbench modules/adapters from src/")

    # Data file
    data_path = repo_root / "data" / "adult.csv"
    if not data_path.exists():
        _fail(f"Missing dataset file: {data_path}")
    df = pandas.read_csv(data_path)
    if df.empty:
        _fail(f"Dataset loaded but is empty: {data_path}")
    _ok(f"Loaded dataset: {data_path.name} rows={len(df)} cols={len(df.columns)}")

    # Optional check: dpmm is not installable on some Python versions (e.g. 3.12).
    try:
        importlib.import_module("dpmm")
        _ok("Optional dependency present: dpmm")
    except Exception:
        print("• Optional dependency not present: dpmm (this is OK on Python 3.12)")

    print("============================================================")
    print("✓ Verification passed")
    print("============================================================")


if __name__ == "__main__":
    main()

