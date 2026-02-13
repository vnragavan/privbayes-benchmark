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
        audit = it.get("audit_metrics") or {}
        nn_mem = (audit.get("nn_memorization") or {}) if isinstance(audit, dict) else {}
        rare = (audit.get("rare_combination") or {}) if isinstance(audit, dict) else {}
        mia = (audit.get("membership_inference") or {}) if isinstance(audit, dict) else {}

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
                # privacy audit probes (optional)
                "audit_nn_emr": nn_mem.get("emr"),
                "audit_unique_pattern_leakage": audit.get("unique_pattern_leakage") if isinstance(audit, dict) else None,
                "audit_rare_rmr": rare.get("rmr"),
                "audit_rare_mae": rare.get("mae"),
                "audit_cond_disc_l1": audit.get("conditional_disclosure_l1") if isinstance(audit, dict) else None,
                "audit_mia_auc": mia.get("auc"),
                "audit_mia_advantage": mia.get("advantage"),
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
                # numeric fidelity (optional; computed by backfill scripts or future runners)
                "numeric_ks_mean": it.get("numeric_ks_mean"),
                "numeric_wasserstein_mean": it.get("numeric_wasserstein_mean"),
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
    capsize: float = 6.0,
    elinewidth: float = 2.2,
    capthick: float = 2.2,
    markersize: float = 6.5,
    markeredgewidth: float = 1.2,
    band_alpha: float = 0.12,
    zorder: float = 3.0,
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
        # errorbar expects symmetric or (2, N). Use asymmetric (2, N).
        yerr = [lo, hi]

        # Light uncertainty band behind the line (helps visibility, especially in insets).
        if pd.notna(lo).any() or pd.notna(hi).any():
            y0 = y - pd.to_numeric(pd.Series(lo)).fillna(0).to_numpy()
            y1 = y + pd.to_numeric(pd.Series(hi)).fillna(0).to_numpy()
            ax.fill_between(x, y0, y1, color=color, alpha=band_alpha, linewidth=0, zorder=zorder - 1)

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
            elinewidth=elinewidth,
            capthick=capthick,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            zorder=zorder,
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
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            zorder=zorder,
        )


def _prefix_titles_with_panel_labels(axes: list[Any], start_char: str = "a") -> None:
    """
    Prefix each subplot title with a panel label: (a), (b), ...

    This keeps the label next to the title (not inside the plotting area).
    """
    start = ord(start_char)
    for i, ax in enumerate(axes):
        t = ax.get_title() or ""
        ax.set_title(f"({chr(start + i)}) {t}".strip(), fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="Path to results JSON or directory containing it")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", type=str, default="utility_privacy_plots_latest")
    ap.add_argument(
        "--split",
        action="store_true",
        help="Write 3 separate figures: utility/fidelity, privacy, performance (instead of the combined 9-panel).",
    )
    ap.add_argument(
        "--uncertainty",
        type=str,
        default="none",
        choices=["none", "se", "ci95"],
        help="Add uncertainty bands across seeds/runs per (implementation, epsilon): none|se|ci95.",
    )
    ap.add_argument(
        "--numeric-first",
        action="store_true",
        help="For mostly-numeric datasets, replace overlap-style panels with numeric fidelity panels "
        "(mean normalized Wasserstein distance and mean KS statistic). Requires numeric_* metrics in JSON.",
    )
    args = ap.parse_args()

    json_path = _pick_results_json(args.path)
    out_dir = args.out_dir or (json_path.parent if json_path.is_file() else args.path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Publication-friendly font sizes (helps readability when embedded in LaTeX).
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    df = _load_rows(json_path)
    impls = (
        df[["implementation"]]
        .dropna()
        .drop_duplicates()
        .sort_values("implementation")["implementation"]
        .tolist()
    )

    def _base_impl(name: str) -> str:
        # e.g. "Enhanced (strict-DP)" -> "Enhanced"
        if not isinstance(name, str):
            return str(name)
        return name.split(" (", 1)[0].strip()

    def _is_strict(name: str) -> bool:
        s = str(name).lower()
        return "(strict" in s or "strict-dp" in s

    def _ls(name: str) -> str:
        # Differentiate regimes visually: default=solid, strict=dashed
        return "--" if _is_strict(name) else "-"

    def _annotate_zero_lines(
        ax: Any,
        a_metric: pd.DataFrame,
        impls_in_order: list[str],
        *,
        value_col: str = "mean",
        tol: float = 1e-12,
        loc: tuple[float, float] = (0.02, 0.02),
        fontsize: int = 9,
    ) -> None:
        """
        If any mechanism line is effectively all-zero across ε, write it on the plot.
        Helps interpret panels where curves overlap the x-axis.
        """
        if a_metric.empty or value_col not in a_metric.columns:
            return

        zero_impls: list[str] = []
        any_nonzero = False
        any_seen = False
        for impl in impls_in_order:
            d = a_metric[a_metric["implementation"] == impl]
            v = pd.to_numeric(d[value_col], errors="coerce").dropna()
            if v.empty:
                continue
            any_seen = True
            if (v.abs() > tol).any():
                any_nonzero = True
            else:
                zero_impls.append(impl)

        if not any_seen:
            return

        if not any_nonzero:
            msg = "Note: value is 0.0 for all mechanisms (all ε)"
        elif zero_impls:
            # Keep it short; list a few and summarize if many.
            show = zero_impls[:3]
            more = len(zero_impls) - len(show)
            suffix = f" (+{more} more)" if more > 0 else ""
            msg = "Note: 0.0 across ε for: " + ", ".join(show) + suffix
        else:
            return

        ax.text(
            loc[0],
            loc[1],
            msg,
            transform=ax.transAxes,
            fontsize=fontsize,
            ha="left",
            va="bottom",
        )

    base_colors = {"SynthCity": "#2ca02c", "DPMM": "#d62728", "Enhanced": "#9467bd"}
    colors = {impl: (base_colors.get(_base_impl(impl)) or "#333333") for impl in impls}

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
    a_emd = agg("emd_mean")
    a_pear = agg("pearson_spearman")
    a_spear = agg("spearman_spearman")
    a_nmi = agg("nmi_spearman")
    a_lr = agg("syn2real_lr_auc")
    a_rf = agg("syn2real_rf_auc")
    a_ks = agg("numeric_ks_mean")
    a_wass = agg("numeric_wasserstein_mean")
    a_marginal_error = agg("marginal_error")
    a_kl = agg("kl_divergence")
    a_jaccard_cov = agg("jaccard_coverage")
    a_audit_emr = agg("audit_nn_emr")
    a_audit_upl = agg("audit_unique_pattern_leakage")
    a_audit_rare_rmr = agg("audit_rare_rmr")
    a_audit_rare_mae = agg("audit_rare_mae")
    a_audit_cond = agg("audit_cond_disc_l1")
    a_audit_mia_auc = agg("audit_mia_auc")
    a_audit_mia_adv = agg("audit_mia_advantage")

    # Efficiency: compute per-run ratio first, then aggregate (uncertainty on ratio).
    df_eff = df.copy()
    df_eff["efficiency"] = pd.to_numeric(df_eff["weighted_jaccard"], errors="coerce") / pd.to_numeric(
        df_eff["epsilon"], errors="coerce"
    )
    a_eff = _mean_and_uncertainty(df_eff, group_cols, "efficiency", args.uncertainty)
    a_eff["epsilon"] = pd.to_numeric(a_eff["epsilon"], errors="coerce")
    a_eff = a_eff.sort_values(["implementation", "epsilon"]).reset_index(drop=True)

    # If split mode requested, write three focused figures.
    if args.split:
        # -----------------------------
        # Utility / fidelity figure
        # -----------------------------
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # (a) Weighted Jaccard, or (numeric-first) normalized Wasserstein/EMD
        ax = axes[0, 0]
        if args.numeric_first and (a_wass["mean"].notna().any() or a_emd["mean"].notna().any()):
            use = a_wass if a_wass["mean"].notna().any() else a_emd
            for impl in impls:
                d = use[use["implementation"] == impl].sort_values("epsilon")
                _plot_line(
                    ax,
                    d,
                    x_col="epsilon",
                    y_col="mean",
                    label=impl,
                    color=colors.get(impl),
                    marker="o",
                    linewidth=2,
                    linestyle=_ls(impl),
                    uncertainty=args.uncertainty,
                )
            ax.set_xlabel("Privacy Budget (ε)")
            ax.set_ylabel("Norm. Wasserstein/EMD (↓)")
            ax.set_title("Numeric Fidelity (Wasserstein/EMD) vs ε (Lower is Better)", fontweight="bold")
        else:
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
                    linestyle=_ls(impl),
                    uncertainty=args.uncertainty,
                )
            ax.set_xlabel("Privacy Budget (ε)")
            ax.set_ylabel("Weighted Jaccard (↑)")
            ax.set_title("Weighted Jaccard vs ε (Higher is Better)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (b) Marginal error
        ax = axes[0, 1]
        for impl in impls:
            d = a_marginal_error[a_marginal_error["implementation"] == impl].sort_values("epsilon")
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="s",
                linewidth=2,
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("Marginal error (↓)")
        ax.set_title("Marginal Error vs ε (Lower is Better)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (c) MI preservation
        ax = axes[0, 2]
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
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("MI preservation (↑)")
        ax.set_title("MI Preservation vs ε (Higher is Better)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (d) TVD (log; 1D/2D/3D)
        ax = axes[1, 0]
        for impl in impls:
            d1 = a_tvd1[a_tvd1["implementation"] == impl].sort_values("epsilon")
            d2 = a_tvd2[a_tvd2["implementation"] == impl].sort_values("epsilon")
            d3 = a_tvd3[a_tvd3["implementation"] == impl].sort_values("epsilon")
            ls = _ls(impl)
            _plot_line(
                ax,
                d1,
                x_col="epsilon",
                y_col="mean",
                label=f"{impl} (1D)",
                color=colors.get(impl),
                marker="o",
                linewidth=1.5,
                linestyle=ls,
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
                linestyle=ls,
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
                linestyle=ls,
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("TVD (log)")
        ax.set_title("TVD vs ε (Lower is Better, log scale)", fontweight="bold")
        ax.set_yscale("log")
        ax.legend(ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)

        # (e) Correlation preservation
        ax = axes[1, 1]
        for impl in impls:
            dpe = a_pear[a_pear["implementation"] == impl].sort_values("epsilon")
            dsp = a_spear[a_spear["implementation"] == impl].sort_values("epsilon")
            ls = _ls(impl)
            _plot_line(
                ax,
                dpe,
                x_col="epsilon",
                y_col="mean",
                label=f"{impl} (Pearson)",
                color=colors.get(impl),
                marker="o",
                linewidth=1.5,
                linestyle=ls,
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
                linestyle=ls,
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("Correlation preservation (↑)")
        ax.set_title("Correlation Preservation vs ε (Higher is Better)", fontweight="bold")
        ax.legend(ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)

        # (f) Coverage: KL divergence (↓)
        ax = axes[1, 2]
        for impl in impls:
            d = a_kl[a_kl["implementation"] == impl].sort_values("epsilon")
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="d",
                linewidth=2,
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("KL divergence (↓)")
        ax.set_title("Coverage KL vs ε (Lower is Better)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (g) Coverage: Jaccard coverage (↑), or (numeric-first) KS (↓), or MI-matrix rank correlation (↑)
        ax = axes[2, 0]
        if args.numeric_first and a_ks["mean"].notna().any():
            for impl in impls:
                d = a_ks[a_ks["implementation"] == impl].sort_values("epsilon")
                _plot_line(
                    ax,
                    d,
                    x_col="epsilon",
                    y_col="mean",
                    label=impl,
                    color=colors.get(impl),
                    marker="o",
                    linewidth=2,
                    linestyle=_ls(impl),
                    uncertainty=args.uncertainty,
                )
            ax.set_xlabel("Privacy Budget (ε)")
            ax.set_ylabel("KS statistic (↓)")
            ax.set_title("Numeric Fidelity (KS) vs ε (Lower is Better)", fontweight="bold")
            ax.set_ylim(0, 1)
        elif args.numeric_first and a_nmi["mean"].notna().any():
            for impl in impls:
                d = a_nmi[a_nmi["implementation"] == impl].sort_values("epsilon")
                _plot_line(
                    ax,
                    d,
                    x_col="epsilon",
                    y_col="mean",
                    label=impl,
                    color=colors.get(impl),
                    marker="o",
                    linewidth=2,
                    linestyle=_ls(impl),
                    uncertainty=args.uncertainty,
                )
            ax.set_xlabel("Privacy Budget (ε)")
            ax.set_ylabel("Spearman ρ (↑)")
            ax.set_title("MI Matrix Similarity vs ε (Higher is Better)", fontweight="bold")
            ax.set_ylim(-1, 1)
        else:
            for impl in impls:
                d = a_jaccard_cov[a_jaccard_cov["implementation"] == impl].sort_values("epsilon")
                _plot_line(
                    ax,
                    d,
                    x_col="epsilon",
                    y_col="mean",
                    label=impl,
                    color=colors.get(impl),
                    marker="o",
                    linewidth=2,
                    linestyle=_ls(impl),
                    uncertainty=args.uncertainty,
                )
            ax.set_xlabel("Privacy Budget (ε)")
            ax.set_ylabel("Jaccard coverage (↑)")
            ax.set_title("Coverage Jaccard vs ε (Higher is Better)", fontweight="bold")
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (h) Downstream ML AUC
        ax = axes[2, 1]
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
            ls = _ls(impl)
            _plot_line(
                ax,
                dlr,
                x_col="epsilon",
                y_col="mean",
                label=f"{impl} (LR)",
                color=colors.get(impl),
                marker="o",
                linewidth=1.5,
                linestyle=ls,
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
                linestyle=ls,
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("AUC (Syn→Real) (↑)")
        ax.set_title("Downstream ML vs ε (Higher is Better)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)

        # (i) Efficiency
        ax = axes[2, 2]
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
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("Utility per ε (↑)")
        ax.set_title("Utility/ε vs ε (Higher is Better)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        _prefix_titles_with_panel_labels(list(axes.ravel()), start_char="a")
        plt.tight_layout()
        util_png = out_dir / f"{args.prefix}_utility.png"
        util_pdf = out_dir / f"{args.prefix}_utility.pdf"
        fig.savefig(util_png, dpi=300, bbox_inches="tight")
        fig.savefig(util_pdf, bbox_inches="tight")
        plt.close(fig)

        # -----------------------------
        # Privacy figure
        # -----------------------------
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # (a) QI linkage
        ax = axes[0, 0]
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
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("QI linkage rate (↓)")
        ax.set_title("QI Linkage vs ε (Lower is Better)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if ermr_note is not None:
            ax.text(0.02, 0.02, f"Note: ERMR is constant at {ermr_note:.4f}", transform=ax.transAxes, fontsize=9, ha="left", va="bottom")

        # (b) NN memorization EMR
        ax = axes[0, 1]
        for impl in impls:
            d = a_audit_emr[a_audit_emr["implementation"] == impl].sort_values("epsilon")
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="s",
                linewidth=2,
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("EMR (↓)")
        ax.set_title("NN Memorization (EMR) vs ε (Lower is Better)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (c) Unique pattern leakage
        ax = axes[0, 2]
        for impl in impls:
            d = a_audit_upl[a_audit_upl["implementation"] == impl].sort_values("epsilon")
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="d",
                linewidth=2,
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        _annotate_zero_lines(ax, a_audit_upl, impls)
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("UPL (↓)")
        ax.set_title("Unique Pattern Leakage vs ε (Lower is Better)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (d) Rare combination leakage (RMR)
        ax = axes[1, 0]
        for impl in impls:
            d = a_audit_rare_rmr[a_audit_rare_rmr["implementation"] == impl].sort_values("epsilon")
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="o",
                linewidth=2,
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        _annotate_zero_lines(ax, a_audit_rare_rmr, impls)
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("RMR (↓)")
        ax.set_title("Rare Combo Leakage (RMR@τ=3) vs ε (Lower is Better)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (e) Conditional disclosure (L1)
        ax = axes[1, 1]
        for impl in impls:
            d = a_audit_cond[a_audit_cond["implementation"] == impl].sort_values("epsilon")
            _plot_line(
                ax,
                d,
                x_col="epsilon",
                y_col="mean",
                label=impl,
                color=colors.get(impl),
                marker="s",
                linewidth=2,
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("L1 distance (↓)")
        ax.set_title("Conditional Disclosure (L1) vs ε (Lower is Better)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (f) Membership inference (if present)
        ax = axes[1, 2]
        drew = False
        for impl in impls:
            d = a_audit_mia_auc[a_audit_mia_auc["implementation"] == impl].sort_values("epsilon")
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
                    linestyle=_ls(impl),
                    uncertainty=args.uncertainty,
                )
                drew = True
        if drew:
            _annotate_zero_lines(ax, a_audit_mia_auc, impls, loc=(0.02, 0.08))
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("MIA AUC (↓)")
        ax.set_title("Membership Inference (AUC) vs ε (Lower is Better)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if drew:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "MIA not in JSON (run benchmark with --audit-mia)", ha="center", va="center", transform=ax.transAxes)

        _prefix_titles_with_panel_labels(list(axes.ravel()), start_char="a")
        plt.tight_layout()
        priv_png = out_dir / f"{args.prefix}_privacy.png"
        priv_pdf = out_dir / f"{args.prefix}_privacy.pdf"
        fig.savefig(priv_png, dpi=300, bbox_inches="tight")
        fig.savefig(priv_pdf, bbox_inches="tight")
        plt.close(fig)

        # -----------------------------
        # Performance figure (existing 2x2)
        # -----------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
                linestyle=_ls(impl),
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
                    linestyle=_ls(impl),
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
                    linestyle=_ls(impl),
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

        # Peak memory (MB) - log
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
                linestyle=_ls(impl),
                uncertainty=args.uncertainty,
            )
        ax.set_title("Peak memory vs ε (Lower is Better)", fontweight="bold")
        ax.set_xlabel("ε")
        ax.set_ylabel("MB")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        _prefix_titles_with_panel_labels(list(axes.ravel()), start_char="a")
        plt.tight_layout()
        perf_png = out_dir / f"{args.prefix}_performance.png"
        perf_pdf = out_dir / f"{args.prefix}_performance.pdf"
        fig.savefig(perf_png, dpi=300, bbox_inches="tight")
        fig.savefig(perf_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Input JSON: {json_path}")
        print(f"Wrote: {util_png}")
        print(f"Wrote: {priv_png}")
        print(f"Wrote: {perf_png}")
        return

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
            linestyle=_ls(impl),
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
            linestyle=_ls(impl),
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
            linestyle=_ls(impl),
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("QI linkage rate (↓)")
    ax.set_title("QI Linkage Rate vs Budget (Lower is Better)", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
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
            linestyle=_ls(impl),
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Total time (s)")
    ax.set_title("Performance vs Privacy Budget (Lower is Better)", fontweight="bold")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5: Utility–privacy tradeoff (utility vs QI linkage)
    ax = axes[1, 1]
    a_trade = a_weighted_jaccard.merge(
        a_qi,
        on=["implementation", "epsilon"],
        how="inner",
        suffixes=("_util", "_qi"),
    )
    for impl in impls:
        d = a_trade[a_trade["implementation"] == impl].sort_values("epsilon")
        if d.empty:
            continue

        x = pd.to_numeric(d["mean_qi"], errors="coerce").to_numpy()
        y = pd.to_numeric(d["mean_util"], errors="coerce").to_numpy()
        ok = pd.notna(x) & pd.notna(y)
        if not ok.any():
            continue
        x = x[ok]
        y = y[ok]

        # Line (connect eps in order) + pointwise error bars.
        ax.plot(
            x,
            y,
            linestyle=_ls(impl),
            linewidth=2.0,
            marker="o",
            markersize=6.5,
            markeredgewidth=1.2,
            color=colors.get(impl),
            label=impl,
            zorder=3.0,
        )

        if args.uncertainty != "none":
            xlo = pd.to_numeric(d["err_low_qi"], errors="coerce").to_numpy()[ok]
            xhi = pd.to_numeric(d["err_high_qi"], errors="coerce").to_numpy()[ok]
            ylo = pd.to_numeric(d["err_low_util"], errors="coerce").to_numpy()[ok]
            yhi = pd.to_numeric(d["err_high_util"], errors="coerce").to_numpy()[ok]
            ax.errorbar(
                x,
                y,
                xerr=[xlo, xhi],
                yerr=[ylo, yhi],
                linestyle="none",
                color=colors.get(impl),
                capsize=6.0,
                elinewidth=2.2,
                capthick=2.2,
                zorder=4.0,
            )

    ax.set_xlabel("QI linkage rate (Lower is Better)")
    ax.set_ylabel("Weighted Jaccard (Higher is Better)")
    ax.set_title("Utility–Privacy Tradeoff (↑ Utility, ↓ Linkage)", fontweight="bold")
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
            linestyle=_ls(impl),
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
        ls = _ls(impl)
        _plot_line(
            ax,
            d1,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (1D)",
            color=colors.get(impl),
            marker="o",
            linewidth=1.5,
            linestyle=ls,
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
            linestyle=ls,
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
            linestyle=ls,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Distance (log scale)")
    ax.set_title("TVD/EMD Metrics (Lower is Better)", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    # 8: Correlation preservation
    ax = axes[2, 1]
    for impl in impls:
        dpe = a_pear[a_pear["implementation"] == impl].sort_values("epsilon")
        dsp = a_spear[a_spear["implementation"] == impl].sort_values("epsilon")
        ls = _ls(impl)
        _plot_line(
            ax,
            dpe,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (Pearson)",
            color=colors.get(impl),
            marker="o",
            linewidth=1.5,
            linestyle=ls,
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
            linestyle=ls,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Correlation preservation (↑)")
    ax.set_title("Correlation Metrics (Higher is Better)", fontweight="bold")
    ax.legend(ncol=2)
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
        ls = _ls(impl)

        _plot_line(
            ax,
            dlr,
            x_col="epsilon",
            y_col="mean",
            label=f"{impl} (LR)",
            color=colors.get(impl),
            marker="o",
            linewidth=1.5,
            linestyle=ls,
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
            linestyle=ls,
            uncertainty=args.uncertainty,
        )
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("AUC (Syn→Real) (↑)")
    ax.set_title("Downstream ML Performance (Higher is Better)", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    _prefix_titles_with_panel_labels(list(axes.ravel()), start_char="a")
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
            linestyle=_ls(impl),
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
                linestyle=_ls(impl),
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
                linestyle=_ls(impl),
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
            linestyle=_ls(impl),
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

    _prefix_titles_with_panel_labels(list(axes.ravel()), start_char="a")
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

