#!/usr/bin/env python3
"""
One-stop plotting script: create the full 9-panel utility/privacy figure
directly from comprehensive_results_*.json (no benchmark rerun required).

This is intended to be the *single* plotting entrypoint.

Features (latest):
- QI linkage plotted (ERMR omitted if constant; note added)
- TVD/EMD panel uses log-scale and includes 1D/2D/3D
- Downstream ML LR/RF curves use tiny x-offsets so overlapping points are visible

Examples:
  source .venv/bin/activate

  # plot from one JSON file
  python scripts/plot_utility_privacy_from_json.py \
    medical_breast_cancer_all3_torch241/comprehensive_results_20260211_140819_augmented.json \
    --out-dir medical_breast_cancer_all3_torch241 \
    --prefix utility_privacy_plots_latest

  # if a directory is passed, uses the most recent comprehensive_results_*.json
  python scripts/plot_utility_privacy_from_json.py medical_breast_cancer_all3_torch241 --prefix utility_privacy_plots_latest
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _get(d: dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _pick_results_json(path: Path) -> Path:
    if path.is_file():
        return path
    cand = sorted(path.glob("comprehensive_results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise SystemExit(f"No comprehensive_results_*.json found in {path}")
    return cand[0]


def _load_rows(json_path: Path) -> pd.DataFrame:
    items = json.loads(json_path.read_text())
    if not isinstance(items, list):
        raise SystemExit("Expected top-level JSON list.")

    rows: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if not it.get("success", False):
            continue

        tvd_sum = _get(it, "tvd_metrics.summary", {}) or {}
        mi_sum = _get(it, "mi_metrics.summary", {}) or {}
        mi_mc = _get(it, "mi_metrics.matrix_comparison", {}) or {}
        corr_mc = _get(it, "correlation_metrics.matrix_comparison", {}) or {}
        cov_sum = _get(it, "coverage_metrics.summary", {}) or {}

        syn2real_lr_auc = _get(it, "downstream_metrics.syn_to_real.logistic_regression.roc_auc")
        syn2real_rf_auc = _get(it, "downstream_metrics.syn_to_real.random_forest.roc_auc")

        # Note: some files store NaN as bare NaN (non-standard JSON). json.loads handles it in CPython.
        rows.append(
            {
                "implementation": it.get("name"),
                "epsilon": it.get("epsilon"),
                "seed": it.get("seed"),
                # performance breakdown (may be missing for some implementations)
                "fit_time_sec": it.get("fit_time_sec"),
                "sample_time_sec": it.get("sample_time_sec"),
                # utility
                "weighted_jaccard": it.get("weighted_jaccard"),
                "jaccard": it.get("jaccard"),
                "marginal_error": it.get("marginal_error"),
                # performance
                "total_time_sec": it.get("total_time_sec"),
                "memory_mb": it.get("peak_memory_mb"),
                # privacy
                "ermr": it.get("exact_row_match_rate"),
                "qi_linkage": it.get("qi_linkage_rate"),
                # tvd/emd
                "tvd_1d_mean": tvd_sum.get("tvd1_mean"),
                "tvd_2d_mean": tvd_sum.get("tvd2_mean"),
                "tvd_3d_mean": tvd_sum.get("tvd3_mean"),
                "emd_mean": tvd_sum.get("emd_mean"),
                # MI/corr/coverage
                "mi_preservation": mi_sum.get("preservation_ratio_mean"),
                "nmi_spearman": mi_mc.get("nmi_matrix_correlation"),
                "pearson_spearman": corr_mc.get("pearson_matrix_spearman_corr"),
                "spearman_spearman": corr_mc.get("spearman_matrix_spearman_corr"),
                "kl_divergence": cov_sum.get("kl_mean"),
                "jaccard_coverage": cov_sum.get("jaccard_mean"),
                # downstream
                "syn2real_lr_auc": syn2real_lr_auc,
                "syn2real_rf_auc": syn2real_rf_auc,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No successful result items found.")

    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    return df


def _mean_and_uncertainty(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    uncertainty: str,
) -> pd.DataFrame:
    """
    Aggregate a metric across runs (e.g., seeds) per group.

    Returns a frame with columns:
      - group_cols...
      - mean
      - err_low
      - err_high
      - n
    where error bands represent either:
      - none: err_* are NaN
      - se:   +/- standard error
      - ci95: +/- 1.96 * standard error (normal approx)
    """
    if value_col not in df.columns:
        out = df[group_cols].drop_duplicates().copy()
        out["mean"] = pd.NA
        out["err_low"] = pd.NA
        out["err_high"] = pd.NA
        out["n"] = 0
        return out

    s = pd.to_numeric(df[value_col], errors="coerce")
    tmp = df[group_cols].copy()
    tmp["_v"] = s

    g = tmp.groupby(group_cols, dropna=True)["_v"]
    mean = g.mean()
    std = g.std(ddof=1)
    n = g.count()

    se = std / (n**0.5)
    if uncertainty == "se":
        half = se
    elif uncertainty == "ci95":
        half = 1.96 * se
    else:
        half = pd.Series(pd.NA, index=mean.index)

    out = mean.reset_index().rename(columns={"_v": "mean"})
    out["n"] = n.reset_index(drop=True)
    out["err_low"] = half.reset_index(drop=True)
    out["err_high"] = half.reset_index(drop=True)
    return out


def _plot_line(
    ax: Any,
    d: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    label: str,
    color: str | None,
    marker: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    uncertainty: str = "none",
    yerr_low_col: str = "err_low",
    yerr_high_col: str = "err_high",
    capsize: float = 3.0,
) -> None:
    if d.empty or y_col not in d.columns:
        return

    x = pd.to_numeric(d[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(d[y_col], errors="coerce").to_numpy()
    ok = pd.notna(x) & pd.notna(y)
    if not ok.any():
        return

    x = x[ok]
    y = y[ok]

    if uncertainty != "none" and yerr_low_col in d.columns and yerr_high_col in d.columns:
        lo = pd.to_numeric(d[yerr_low_col], errors="coerce").to_numpy()[ok]
        hi = pd.to_numeric(d[yerr_high_col], errors="coerce").to_numpy()[ok]
        # errorbar expects symmetric or (2, N). Use asymmetric so we can later clamp if needed.
        yerr = [lo, hi]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            color=color,
            capsize=capsize,
        )
    else:
        ax.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            color=color,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="Path to results JSON or directory containing it")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", type=str, default="utility_privacy_plots_latest")
    ap.add_argument(
        "--uncertainty",
        type=str,
        default="none",
        choices=["none", "se", "ci95"],
        help="Add uncertainty bands across seeds/runs per (implementation, epsilon): none|se|ci95.",
    )
    args = ap.parse_args()

    json_path = _pick_results_json(args.path)
    out_dir = args.out_dir or (json_path.parent if json_path.is_file() else args.path)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_rows(json_path)
    impls = (
        df[["implementation"]]
        .dropna()
        .drop_duplicates()
        .sort_values("implementation")["implementation"]
        .tolist()
    )

    colors = {"SynthCity": "#2ca02c", "DPMM": "#d62728", "Enhanced": "#9467bd"}

    group_cols = ["implementation", "epsilon"]

    def agg(metric: str) -> pd.DataFrame:
        out = _mean_and_uncertainty(df, group_cols, metric, args.uncertainty)
        out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
        return out.sort_values(["implementation", "epsilon"]).reset_index(drop=True)

    # Pre-aggregate metrics used in multiple panels
    a_weighted_jaccard = agg("weighted_jaccard")
    a_qi = agg("qi_linkage")
    a_ermr = agg("ermr")
    a_total_time = agg("total_time_sec")
    a_fit_time = agg("fit_time_sec")
    a_sample_time = agg("sample_time_sec")
    a_memory = agg("memory_mb")
    a_mi = agg("mi_preservation")
    a_tvd1 = agg("tvd_1d_mean")
    a_tvd2 = agg("tvd_2d_mean")
    a_tvd3 = agg("tvd_3d_mean")
    a_pear = agg("pearson_spearman")
    a_spear = agg("spearman_spearman")
    a_lr = agg("syn2real_lr_auc")
    a_rf = agg("syn2real_rf_auc")

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    # Intentionally no figure-level title (suptitle). Subplots carry their own titles.

    # 1: Utility vs epsilon
    ax = axes[0, 0]
    for impl in impls:
        d = a_weighted_jaccard[a_weighted_jaccard["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="o",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Weighted Jaccard (Utility)")
    ax.set_title("Utility vs Privacy Budget (Higher is Better)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2: Efficiency
    # Compute per-run efficiency first, then aggregate (uncertainty on the ratio).
    df_eff = df.copy()
    df_eff["efficiency"] = pd.to_numeric(df_eff["weighted_jaccard"], errors="coerce") / pd.to_numeric(
        df_eff["epsilon"], errors="coerce"
    )
    a_eff = _mean_and_uncertainty(df_eff, group_cols, "efficiency", args.uncertainty)
    a_eff["epsilon"] = pd.to_numeric(a_eff["epsilon"], errors="coerce")
    a_eff = a_eff.sort_values(["implementation", "epsilon"]).reset_index(drop=True)

    ax = axes[0, 1]
    for impl in impls:
        d = a_eff[a_eff["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="s",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Utility per ε")
    ax.set_title("Privacy Budget Efficiency (Higher is Better)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3: QI linkage only (ERMR note if constant)
    ax = axes[0, 2]
    ermr_note = None
    if "mean" in a_ermr.columns and a_ermr["mean"].notna().any():
        er = pd.to_numeric(a_ermr["mean"], errors="coerce").dropna()
        if not er.empty and (er.max() - er.min()) < 1e-12:
            ermr_note = float(er.iloc[0])

    for impl in impls:
        d = a_qi[a_qi["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="o",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("QI linkage rate (↓)")
    ax.set_title("QI Linkage Rate vs Budget (Lower is Better)", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if ermr_note is not None:
        ax.text(
            0.02,
            0.02,
            f"Note: ERMR is constant at {ermr_note:.4f} for all mechanisms",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    # 4: Time vs epsilon (log)
    ax = axes[1, 0]
    for impl in impls:
        d = a_total_time[a_total_time["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="d",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Total time (s)")
    ax.set_title("Performance vs Privacy Budget (Lower is Better)", fontweight="bold")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5: Utility-privacy tradeoff (epsilon vs utility)
    ax = axes[1, 1]
    for impl in impls:
        d = a_weighted_jaccard[a_weighted_jaccard["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="o",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Weighted Jaccard (Utility)")
    ax.set_title("Utility–Privacy Tradeoff (Higher is Better)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6: MI preservation
    ax = axes[1, 2]
    for impl in impls:
        d = a_mi[a_mi["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="p",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("MI preservation (↑)")
    ax.set_title("Information Preservation vs Budget (Higher is Better)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7: TVD/EMD (log)
    ax = axes[2, 0]
    for impl in impls:
        d1 = a_tvd1[a_tvd1["implementation"] == impl].sort_values("epsilon")
        d2 = a_tvd2[a_tvd2["implementation"] == impl].sort_values("epsilon")
        d3 = a_tvd3[a_tvd3["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d1,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (1D)",
            color=colors.get(impl),
            marker="o",
            linewidth=1.5,
            linestyle="-",
            uncertainty=args.uncertainty,
        )
        _plot_line(
            ax,
            d2,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (2D)",
            color=colors.get(impl),
            marker="s",
            linewidth=1.5,
            linestyle="--",
            uncertainty=args.uncertainty,
        )
        _plot_line(
            ax,
            d3,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (3D)",
            color=colors.get(impl),
            marker="^",
            linewidth=1.5,
            linestyle=":",
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Distance (log scale)")
    ax.set_title("TVD/EMD Metrics (Lower is Better)", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 8: Correlation preservation
    ax = axes[2, 1]
    for impl in impls:
        dpe = a_pear[a_pear["implementation"] == impl].sort_values("epsilon")
        dsp = a_spear[a_spear["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            dpe,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (Pearson)",
            color=colors.get(impl),
            marker="o",
            linewidth=1.5,
            linestyle="-",
            uncertainty=args.uncertainty,
        )
        _plot_line(
            ax,
            dsp,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (Spearman)",
            color=colors.get(impl),
            marker="s",
            linewidth=1.5,
            linestyle="--",
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Correlation preservation (↑)")
    ax.set_title("Correlation Metrics (Higher is Better)", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 9: Downstream ML (LR/RF AUC) with small x-offsets
    ax = axes[2, 2]
    try:
        eps_span = float(pd.to_numeric(df["epsilon"], errors="coerce").max() - pd.to_numeric(df["epsilon"], errors="coerce").min())
    except Exception:
        eps_span = 0.0
    dx = 0.01 * (eps_span if eps_span > 0 else 1.0)

    for impl in impls:
        dlr = a_lr[a_lr["implementation"] == impl].sort_values("epsilon").copy()
        drf = a_rf[a_rf["implementation"] == impl].sort_values("epsilon").copy()
        dlr["epsilon"] = pd.to_numeric(dlr["epsilon"], errors="coerce") - dx
        drf["epsilon"] = pd.to_numeric(drf["epsilon"], errors="coerce") + dx

        _plot_line(
            ax,
            dlr,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (LR)",
            color=colors.get(impl),
            marker="o",
            linewidth=1.5,
            linestyle="-",
            uncertainty=args.uncertainty,
        )
        _plot_line(
            ax,
            drf,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (RF)",
            color=colors.get(impl),
            marker="s",
            linewidth=1.5,
            linestyle="--",
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("AUC (Syn→Real) (↑)")
    ax.set_title("Downstream ML Performance (Higher is Better)", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = out_dir / f"{args.prefix}.png"
    out_pdf = out_dir / f"{args.prefix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # Separate performance plot(s)
    # -----------------------------
    # Use aggregated-with-uncertainty frames
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Intentionally no figure-level title (suptitle). Subplots carry their own titles.

    # Time (total) - log
    ax = axes[0, 0]
    for impl in impls:
        d = a_total_time[a_total_time["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="o",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_title("Total time vs ε (Lower is Better, log scale)", fontweight="bold")
    ax.set_xlabel("ε")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Fit time - log (if present)
    ax = axes[0, 1]
    drew = False
    for impl in impls:
        d = a_fit_time[a_fit_time["implementation"] == impl].sort_values("epsilon")
        if d["mean"].notna().any():
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="s",
                linewidth=2,
                uncertainty=args.uncertainty,
            )
            drew = True
    ax.set_title("Fit time vs ε (Lower is Better, log scale)", fontweight="bold")
    ax.set_xlabel("ε")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if drew:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no fit_time_sec in JSON", ha="center", va="center", transform=ax.transAxes)

    # Sample time - log (if present)
    ax = axes[1, 0]
    drew = False
    for impl in impls:
        d = a_sample_time[a_sample_time["implementation"] == impl].sort_values("epsilon")
        if d["mean"].notna().any():
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="^",
                linewidth=2,
                uncertainty=args.uncertainty,
            )
            drew = True
    ax.set_title("Sample time vs ε (Lower is Better, log scale)", fontweight="bold")
    ax.set_xlabel("ε")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if drew:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no sample_time_sec in JSON", ha="center", va="center", transform=ax.transAxes)

    # Peak memory (MB)
    ax = axes[1, 1]
    for impl in impls:
        d = a_memory[a_memory["implementation"] == impl].sort_values("epsilon")
        _plot_line(
            ax,
            d,
            x_col="epsilon",
            y_col="mean",
            label=impl,
            color=colors.get(impl),
            marker="d",
            linewidth=2,
            uncertainty=args.uncertainty,
        )
    ax.set_title("Peak memory vs ε (Lower is Better)", fontweight="bold")
    ax.set_xlabel("ε")
    ax.set_ylabel("MB")
    # Memory can be extremely skewed across implementations (e.g., SynthCity vs others).
    # Use log scale so low-memory methods remain visible.
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    perf_png = out_dir / f"{args.prefix}_performance.png"
    perf_pdf = out_dir / f"{args.prefix}_performance.pdf"
    fig.savefig(perf_png, dpi=300, bbox_inches="tight")
    fig.savefig(perf_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Input JSON: {json_path}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {perf_png}")


if __name__ == "__main__":
    main()

