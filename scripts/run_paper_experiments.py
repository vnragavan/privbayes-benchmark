"""
Run paper-ready experiment sweeps and generate figures.

This is a convenience wrapper around:
  - scripts/comprehensive_comparison.py
  - scripts/plot_utility_privacy_from_json.py

It runs:
  - Enhanced (default + strict-DP)
  - DPMM (default + strict-DP)
  - SynthCity (default only; skipped automatically under strict)

across a sweep of epsilons and seeds, then produces split figures
(utility/fidelity, privacy, performance) with uncertainty bands across seeds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adult", type=Path, default=Path("data/adult.csv"))
    ap.add_argument("--adult-schema", type=Path, default=Path("schemas/adult_public_schema.json"))
    ap.add_argument("--breast", type=Path, default=Path("data/breast_cancer.csv"))
    ap.add_argument("--breast-schema", type=Path, default=Path("schemas/breast_cancer_public_schema.json"))
    ap.add_argument("--lung", type=Path, default=None, help="Optional lung cancer CSV path.")
    ap.add_argument("--lung-schema", type=Path, default=None, help="Optional lung cancer public-schema JSON.")
    ap.add_argument("--lung-target-col", type=str, default=None, help="Target column name for lung dataset.")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.5, 1.0, 3.0, 5.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-root", type=Path, default=Path("paper_runs"))
    ap.add_argument("--n-bootstrap", type=int, default=0, help="Bootstrap resamples inside metrics (0 disables).")
    ap.add_argument("--audit", action="store_true", default=True, help="Enable minimum viable audit probes (default: on).")
    ap.add_argument("--no-audit", action="store_false", dest="audit", help="Disable minimum viable audit probes.")
    ap.add_argument("--audit-mia", action="store_true", default=True, help="Enable membership inference probe (default: on).")
    ap.add_argument("--no-audit-mia", action="store_false", dest="audit_mia", help="Disable membership inference probe.")
    ap.add_argument("--uncertainty", choices=["none", "se", "ci95"], default="ci95")
    ap.add_argument("--split", action="store_true", default=True, help="Write split figures (utility/privacy/performance) (default: on).")
    ap.add_argument("--no-split", action="store_false", dest="split", help="Disable split figures (write combined figure only).")
    ap.add_argument(
        "--implementations",
        type=str,
        nargs="+",
        default=["Enhanced", "SynthCity", "DPMM"],
        help="Base implementations to run.",
    )
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[str, Path, str | None, Path | None]] = [
        ("adult", args.adult, "income", args.adult_schema),
        ("breast_cancer", args.breast, "target", args.breast_schema),
    ]
    if args.lung is not None:
        datasets.append(("lung_cancer", args.lung, args.lung_target_col, args.lung_schema))

    for tag, path, target_col, schema in datasets:
        out_dir = args.out_root / f"{tag}_epsweep_seeds{len(args.seeds)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/comprehensive_comparison.py",
            "--data",
            str(path),
            "--eps",
            *[str(x) for x in args.eps],
            "--seeds",
            *[str(s) for s in args.seeds],
            "--out-dir",
            str(out_dir),
            "--implementations",
            *args.implementations,
            "--regimes",
            "default",
            "strict",
            "--dpmm-default-preprocess",
            "none",
            "--n-bootstrap",
            str(args.n_bootstrap),
        ]
        if schema is not None:
            cmd += ["--schema", str(schema)]
        if target_col:
            cmd += ["--target-col", target_col]
        if args.audit:
            cmd += ["--audit"]
        if args.audit_mia:
            cmd += ["--audit-mia"]
        _run(cmd)

        plot_cmd = [
            sys.executable,
            "scripts/plot_utility_privacy_from_json.py",
            str(out_dir),
            "--prefix",
            f"paper_{tag}",
            "--uncertainty",
            args.uncertainty,
        ]
        if args.split:
            plot_cmd += ["--split"]
        _run(plot_cmd)

    print("\nDone. Outputs under:", args.out_root, flush=True)


if __name__ == "__main__":
    main()

