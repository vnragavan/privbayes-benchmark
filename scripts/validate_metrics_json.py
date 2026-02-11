#!/usr/bin/env python3
"""
Validate/sanity-check comprehensive_results_*.json outputs.

Examples:
  source .venv/bin/activate
  python scripts/validate_metrics_json.py medical_quick_test/comprehensive_results_20260211_130740.json

  # validate all result json files in a directory
  python scripts/validate_metrics_json.py medical_quick_test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite_number(x: Any) -> bool:
    return _is_number(x) and math.isfinite(float(x))


def _get(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class Issue:
    level: str  # ERROR | WARN
    where: str
    message: str


def _add_issue(issues: list[Issue], level: str, where: str, message: str) -> None:
    issues.append(Issue(level=level, where=where, message=message))


def _check_range(
    issues: list[Issue],
    where: str,
    value: Any,
    lo: float | None,
    hi: float | None,
    *,
    allow_none: bool = True,
    allow_nan: bool = False,
) -> None:
    if value is None:
        if allow_none:
            return
        _add_issue(issues, "ERROR", where, "missing value")
        return

    if not _is_number(value):
        _add_issue(issues, "ERROR", where, f"not a number: {type(value).__name__}")
        return

    v = float(value)
    if not allow_nan and not math.isfinite(v):
        _add_issue(issues, "ERROR", where, f"not finite: {value!r}")
        return

    if allow_nan and math.isnan(v):
        return

    if lo is not None and v < lo:
        _add_issue(issues, "ERROR", where, f"out of range: {v} < {lo}")
    if hi is not None and v > hi:
        _add_issue(issues, "ERROR", where, f"out of range: {v} > {hi}")


def _iter_json_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    if not path.exists():
        raise FileNotFoundError(str(path))

    # directory: take any comprehensive_results*.json, else all json files
    cand = sorted(path.glob("comprehensive_results*.json"))
    if not cand:
        cand = sorted(path.glob("*.json"))
    for p in cand:
        yield p


def validate_result_item(item: dict[str, Any], *, file_label: str, idx: int) -> list[Issue]:
    issues: list[Issue] = []
    prefix = f"{file_label}[{idx}]"

    # basic required fields
    name = item.get("name")
    if not isinstance(name, str) or not name:
        _add_issue(issues, "ERROR", f"{prefix}.name", "missing/invalid implementation name")

    _check_range(issues, f"{prefix}.epsilon", item.get("epsilon"), 0.0, None, allow_none=False)
    seed = item.get("seed")
    if not isinstance(seed, int):
        _add_issue(issues, "ERROR", f"{prefix}.seed", "seed must be int")

    success = item.get("success")
    if success not in (True, False):
        _add_issue(issues, "ERROR", f"{prefix}.success", "success must be boolean")

    # perf metrics (non-negative when present)
    for k in ("fit_time_sec", "sample_time_sec", "vocab_align_time_sec", "total_time_sec", "peak_memory_mb"):
        _check_range(issues, f"{prefix}.{k}", item.get(k), 0.0, None)

    # basic utility metrics
    _check_range(issues, f"{prefix}.jaccard", item.get("jaccard"), 0.0, 1.0)
    _check_range(issues, f"{prefix}.weighted_jaccard", item.get("weighted_jaccard"), 0.0, 1.0)
    # Some implementations may legitimately skip/fail certain metrics and emit NaN.
    _check_range(issues, f"{prefix}.marginal_error", item.get("marginal_error"), 0.0, None, allow_nan=True)

    # privacy attacks (probabilities)
    _check_range(issues, f"{prefix}.exact_row_match_rate", item.get("exact_row_match_rate"), 0.0, 1.0)
    _check_range(issues, f"{prefix}.qi_linkage_rate", item.get("qi_linkage_rate"), 0.0, 1.0)

    # downstream ML
    dm = item.get("downstream_metrics") or {}
    if isinstance(dm, dict):
        _check_range(issues, f"{prefix}.downstream_metrics.syn2real_lr_auc", dm.get("syn2real_lr_auc"), 0.0, 1.0, allow_nan=True)
        _check_range(issues, f"{prefix}.downstream_metrics.syn2real_rf_auc", dm.get("syn2real_rf_auc"), 0.0, 1.0, allow_nan=True)
        _check_range(issues, f"{prefix}.downstream_metrics.symmetry_gap", dm.get("symmetry_gap"), 0.0, None, allow_nan=True)
    elif dm is not None:
        _add_issue(issues, "ERROR", f"{prefix}.downstream_metrics", "must be an object/dict")

    # MI metrics (correlations in [-1, 1], ratios >=0)
    mi = item.get("mi_metrics") or {}
    if isinstance(mi, dict):
        _check_range(issues, f"{prefix}.mi_metrics.mi_preservation_ratio_mean", mi.get("mi_preservation_ratio_mean"), 0.0, None)
        _check_range(issues, f"{prefix}.mi_metrics.nmi_spearman", mi.get("nmi_spearman"), -1.0, 1.0)
    elif mi is not None:
        _add_issue(issues, "ERROR", f"{prefix}.mi_metrics", "must be an object/dict")

    # correlation metrics (in [-1, 1])
    cm = item.get("correlation_metrics") or {}
    if isinstance(cm, dict):
        _check_range(issues, f"{prefix}.correlation_metrics.pearson_spearman", cm.get("pearson_spearman"), -1.0, 1.0)
        _check_range(issues, f"{prefix}.correlation_metrics.spearman_spearman", cm.get("spearman_spearman"), -1.0, 1.0)
    elif cm is not None:
        _add_issue(issues, "ERROR", f"{prefix}.correlation_metrics", "must be an object/dict")

    # coverage metrics
    cov = item.get("coverage_metrics") or {}
    if isinstance(cov, dict):
        _check_range(issues, f"{prefix}.coverage_metrics.kl_divergence_mean", cov.get("kl_divergence_mean"), 0.0, None)
        _check_range(issues, f"{prefix}.coverage_metrics.jaccard_coverage_mean", cov.get("jaccard_coverage_mean"), 0.0, 1.0)
    elif cov is not None:
        _add_issue(issues, "ERROR", f"{prefix}.coverage_metrics", "must be an object/dict")

    # TVD summary values if present
    tvd = item.get("tvd_metrics") or {}
    if isinstance(tvd, dict):
        s = tvd.get("summary") or {}
        if isinstance(s, dict):
            # tvd_2d/3d are true TVD => [0,1]. 1d may be EMD-normalized => >=0 (not capped).
            _check_range(issues, f"{prefix}.tvd_metrics.summary.tvd_1d_mean", s.get("tvd_1d_mean"), 0.0, None)
            _check_range(issues, f"{prefix}.tvd_metrics.summary.tvd_2d_mean", s.get("tvd_2d_mean"), 0.0, 1.0)
            _check_range(issues, f"{prefix}.tvd_metrics.summary.tvd_3d_mean", s.get("tvd_3d_mean"), 0.0, 1.0)
            _check_range(issues, f"{prefix}.tvd_metrics.summary.emd_mean", s.get("emd_mean"), 0.0, None)
        elif s is not None:
            _add_issue(issues, "ERROR", f"{prefix}.tvd_metrics.summary", "must be an object/dict")
    elif tvd is not None:
        _add_issue(issues, "ERROR", f"{prefix}.tvd_metrics", "must be an object/dict")

    # consistency checks
    if success is True and item.get("error") not in (None, ""):
        _add_issue(issues, "WARN", f"{prefix}.error", "success=true but error is set")
    if success is False and not item.get("error"):
        _add_issue(issues, "WARN", f"{prefix}.error", "success=false but error is empty")

    # privacy report epsilon sanity (if present)
    eps_total = _get(item, "privacy_report.epsilon_total")
    eps_requested = item.get("epsilon")
    if _is_finite_number(eps_total) and _is_finite_number(eps_requested):
        if float(eps_total) > float(eps_requested) + 1e-9:
            _add_issue(
                issues,
                "WARN",
                f"{prefix}.privacy_report.epsilon_total",
                f"epsilon_total={eps_total} exceeds requested epsilon={eps_requested}",
            )

    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="Path to comprehensive_results_*.json or a directory containing them")
    args = ap.parse_args()

    all_issues: list[Issue] = []
    checked_files = 0
    checked_items = 0

    for p in _iter_json_paths(args.path):
        checked_files += 1
        label = p.name
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            _add_issue(all_issues, "ERROR", label, f"failed to parse json: {e}")
            continue

        if not isinstance(data, list):
            _add_issue(all_issues, "ERROR", label, "expected top-level JSON array (list of results)")
            continue

        for i, item in enumerate(data):
            checked_items += 1
            if not isinstance(item, dict):
                _add_issue(all_issues, "ERROR", f"{label}[{i}]", "each item must be an object/dict")
                continue
            all_issues.extend(validate_result_item(item, file_label=label, idx=i))

    # print report
    errors = [x for x in all_issues if x.level == "ERROR"]
    warns = [x for x in all_issues if x.level == "WARN"]

    print(f"Checked files: {checked_files}")
    print(f"Checked items: {checked_items}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warns)}")

    if all_issues:
        print("\nIssues:")
        for it in all_issues:
            print(f"- {it.level}: {it.where}: {it.message}")

    if errors:
        raise SystemExit(1)
    print("\n✓ Validation passed")


if __name__ == "__main__":
    main()

