#!/usr/bin/env python3
"""
INTERNAL / DEBUG

Smoke test for DPMM adapter in a real __main__ context.

This exists because dpmm may use multiprocessing, which fails when executed from
stdin (<stdin>) on macOS.
"""

import sys
from pathlib import Path

import pandas as pd


def _add_src_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_add_src_to_syspath()

from pbbench.variants.pb_datasynthesizer import DPMMPrivBayesAdapter  # noqa: E402


def main() -> None:
    df = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.15, 0.18, 0.22, 0.25, 0.27, 0.05, 0.4],
            "x2": [1.0, 0.5, 1.5, 1.1, 0.9, 0.6, 1.2, 0.7, 1.8, 0.4],
        }
    )

    m = DPMMPrivBayesAdapter(
        epsilon=1.0,
        delta=1e-5,
        degree=2,
        n_bins=8,
        seed=0,
        n_iters=200,
        n_jobs=1,
        compress=False,
        preprocess="dp",
        strict_dp=True,
    )
    m.fit(df)
    syn = m.sample(5)
    print(syn.head())
    print(m.privacy_report())


if __name__ == "__main__":
    main()

