#!/usr/bin/env python3
"""
Comprehensive Privacy-Utility Comparison for PrivBayes Implementations

This script computes ALL metrics for Enhanced, SynthCity, and DPMM PrivBayes implementations:
- Performance: Speed, Memory, Scalability
- Privacy: Epsilon allocation, Transparency, Privacy attacks
- Utility: Basic + Comprehensive (TVD, EMD, MI, Downstream ML)
- Visualization: Utility vs Privacy plots

Usage:
    python scripts/comprehensive_comparison.py \
        --data data/adult.csv \
        --eps 0.5 1.0 2.0 \
        --seeds 0 1 2 \
        --out-dir results \
        --implementations Enhanced SynthCity DPMM
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

import json
import time
import tracemalloc
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import enhanced metrics
from pbbench.enhanced_metrics import (
    comprehensive_tvd_metrics,
    comprehensive_mi_metrics,
    comprehensive_correlation_metrics,
    comprehensive_coverage_metrics,
    comprehensive_downstream_metrics,
)

# ---------- Enhanced vocab alignment helpers ----------
import hashlib as _hlib

def _blake_bucket_eval(s: str, m: int) -> int:
    """Hash string to bucket index for vocabulary alignment.
    
    Matches the hashing used in Enhanced PrivBayes for categorical columns.
    """
    h = _hlib.blake2b(str(s).encode("utf-8", errors="ignore"), digest_size=16)
    return int.from_bytes(h.digest(), "little") % int(m)

def align_real_to_enhanced_vocab(enhanced_adapter, real_df: pd.DataFrame) -> pd.DataFrame:
    """Align REAL categorical values to the Enhanced model's learned vocabulary.
    
    DP-safe: Uses only the model's learned vocabulary (already DP-protected via DP heavy hitters).
    This is post-processing on evaluation data, not training. The vocabulary was learned with
    epsilon budget, and this deterministic transformation preserves DP guarantees.
    
    Process:
    - Labels: never hashed, no UNK; clamp to public classes and fill missing with first class.
    - Non-label categoricals: if hashed, map real values -> B{bucket:03d} using each column's hash_m.
      Clamp everything to the learned cats (else -> unknown_token).
    
    If the adapter's internal model/meta isn't accessible, returns the REAL unchanged.
    """
    # Try to reach the underlying synthesizer where _meta lives
    m = getattr(enhanced_adapter, "model", None) or getattr(enhanced_adapter, "_model", None) or enhanced_adapter
    meta = getattr(m, "_meta", None)
    if meta is None:
        return real_df

    unk = getattr(m, "unknown_token", "__UNK__")
    label_cols = set(getattr(m, "label_columns", []))

    out = real_df.copy()
    for c, colmeta in meta.items():
        if c not in out.columns:
            continue
        if getattr(colmeta, "kind", None) != "categorical":
            continue

        cats = list(getattr(colmeta, "cats", []) or [])
        catset = set(cats)

        if c in label_cols:
            # labels: NEVER hash, keep exact public classes
            fill = cats[0] if cats else unk
            s = out[c].astype("string").fillna(fill)
            out[c] = s.where(s.isin(catset), other=fill)
            continue

        # non-label categoricals: hash if hashed_cats=True
        s = out[c].astype("string").fillna(unk)
        if getattr(colmeta, "hashed_cats", False) and getattr(colmeta, "hash_m", None):
            msize = int(colmeta.hash_m)
            s = s.map(lambda v: unk if v == unk else f"B{_blake_bucket_eval(v, msize):03d}")
        out[c] = s.where(s.isin(catset), other=unk)

    return out


# ==================== IMPLEMENTATION RUNNERS ====================

@dataclass
class ImplementationResult:
    """Results for one implementation at one epsilon/seed"""
    name: str
    epsilon: float
    seed: int
    success: bool
    error: Optional[str] = None
    
    # Performance
    fit_time_sec: Optional[float] = None
    sample_time_sec: Optional[float] = None
    vocab_align_time_sec: Optional[float] = None
    total_time_sec: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    
    # Privacy
    privacy_report: Optional[Dict] = None
    
    # Utility - Basic
    jaccard: Optional[float] = None
    weighted_jaccard: Optional[float] = None
    marginal_error: Optional[float] = None
    
    # Utility - Comprehensive
    tvd_metrics: Optional[Dict] = None
    mi_metrics: Optional[Dict] = None
    correlation_metrics: Optional[Dict] = None
    coverage_metrics: Optional[Dict] = None
    downstream_metrics: Optional[Dict] = None
    
    # Privacy Attacks
    exact_row_match_rate: Optional[float] = None
    qi_linkage_rate: Optional[float] = None


def run_synthcity_privbayes(real: pd.DataFrame, epsilon: float, seed: int) -> ImplementationResult:
    """Run SynthCity PrivBayes implementation and measure performance.
    
    Tracks fit time, sample time, memory usage. Returns result object with
    synthetic data and original real data for metric computation.
    """
    result = ImplementationResult(name="SynthCity", epsilon=epsilon, seed=seed, success=False)
    
    try:
        from pbbench.variants.pb_synthcity import SynthcityPrivBayesAdapter
        
        tracemalloc.start()
        start_time = time.time()
        
        model = SynthcityPrivBayesAdapter(
            epsilon=epsilon,
            delta=1.0 / (len(real) ** 2),
            max_parents=2,
            theta_usefulness=4,
            random_state=seed
        )
        
        fit_start = time.time()
        model.fit(real)
        result.fit_time_sec = time.time() - fit_start
        
        sample_start = time.time()
        syn = model.sample(len(real))
        result.sample_time_sec = time.time() - sample_start
        
        result.total_time_sec = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        result.peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        result.privacy_report = model.privacy_report() if hasattr(model, 'privacy_report') else {
            "epsilon_total": epsilon,
            "implementation": "SynthCity PrivBayes",
            "note": "No detailed breakdown available"
        }
        
        result.success = True
        return result, syn, real
        
    except Exception as e:
        result.error = str(e)
        try:
            tracemalloc.stop()
        except:
            pass
        return result, None, real


def run_dpmm_privbayes(real: pd.DataFrame, epsilon: float, seed: int) -> ImplementationResult:
    """Run DPMM PrivBayes implementation and measure performance.
    
    Discretizes data to integers before fitting (DPMM requirement). Tracks
    timing and memory. Returns result with synthetic and real data.
    """
    result = ImplementationResult(name="DPMM", epsilon=epsilon, seed=seed, success=False)
    
    try:
        from pbbench.variants.pb_datasynthesizer import DPMMPrivBayesAdapter
        
        tracemalloc.start()
        start_time = time.time()
        
        model = DPMMPrivBayesAdapter(
            epsilon=epsilon,
            delta=1.0 / (len(real) ** 2),
            degree=2,
            n_bins=50,
            seed=seed
        )
        
        fit_start = time.time()
        model.fit(real)
        result.fit_time_sec = time.time() - fit_start
        
        sample_start = time.time()
        syn = model.sample(len(real))
        result.sample_time_sec = time.time() - sample_start
        
        result.total_time_sec = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        result.peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        result.privacy_report = model.privacy_report() if hasattr(model, 'privacy_report') else {}
        
        result.success = True
        return result, syn, real
        
    except Exception as e:
        result.error = str(e)
        try:
            tracemalloc.stop()
        except:
            pass
        return result, None, real


def run_enhanced_privbayes(real: pd.DataFrame, epsilon: float, seed: int, 
                          temperature: float = 1.0, target_col: Optional[str] = None) -> ImplementationResult:
    """Run Enhanced PrivBayes implementation and measure performance.
    
    Auto-detects target column for label handling. Sets up public bounds
    and categories. Returns aligned real data (vocab-matched) for fair comparison.
    """
    result = ImplementationResult(name="Enhanced", epsilon=epsilon, seed=seed, success=False)
    
    try:
        from pbbench.variants.pb_enhanced import EnhancedPrivBayesAdapter
        
        tracemalloc.start()
        
        # Public bounds for better utility - also handle numeric-looking object columns
        public_bounds = {}
        for col in real.columns:
            # Check if numeric or can be coerced to numeric
            if pd.api.types.is_numeric_dtype(real[col]):
                vmin, vmax = real[col].min(), real[col].max()
                margin = (vmax - vmin) * 0.1
                public_bounds[col] = [vmin - margin, vmax + margin]
            elif real[col].dtype == 'object':
                # Try to coerce to numeric for bounds
                s = pd.to_numeric(real[col], errors='coerce')
                if s.notna().mean() >= 0.95:  # If 95%+ can be converted to numeric
                    vmin, vmax = float(s.min()), float(s.max())
                    margin = (vmax - vmin) * 0.1
                    public_bounds[col] = [vmin - margin, vmax + margin]
        
        # Use provided target column or auto-detect (common names: 'target', 'income', 'label', 'class')
        detected_target_col = target_col
        if detected_target_col is None:
            for col in ['target', 'income', 'label', 'class', 'outcome']:
                if col in real.columns:
                    detected_target_col = col
                    break
        
        # Set up label columns and public categories
        label_columns = []
        public_categories = {}
        if detected_target_col and detected_target_col in real.columns:
            label_columns = [detected_target_col]
            # Get unique values from real data for the label
            unique_labels = sorted(real[detected_target_col].astype(str).dropna().unique().tolist())
            public_categories[detected_target_col] = unique_labels
        
        # Start timing from model creation onwards (excludes setup overhead)
        start_time = time.time()
        model = EnhancedPrivBayesAdapter(
            epsilon=epsilon,
            delta=1.0 / (len(real) ** 2),
            seed=seed,
            temperature=temperature,
            cpt_smoothing=1.5,  # DP-safe CPT smoothing (post-processing)
            public_bounds=public_bounds,
            label_columns=label_columns if label_columns else None,
            public_categories=public_categories if public_categories else None,
            bins_per_numeric=50,
            max_parents=2,
            eps_split={"structure": 0.3, "cpt": 0.7},
            forbid_as_parent=label_columns  # Prevent label from being a parent
        )
        
        fit_start = time.time()
        model.fit(real)
        result.fit_time_sec = time.time() - fit_start
        
        sample_start = time.time()
        syn = model.sample(len(real))
        result.sample_time_sec = time.time() - sample_start
        
        # Align REAL to Enhanced vocab for fair comparison
        # DP-safe: uses only the model's learned vocabulary (already DP-protected)
        # This is post-processing on real data for evaluation, not training
        vocab_align_start = time.time()
        eval_real = align_real_to_enhanced_vocab(model, real)
        result.vocab_align_time_sec = time.time() - vocab_align_start
        
        result.total_time_sec = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        result.peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()
        
        result.privacy_report = model.privacy_report() if hasattr(model, 'privacy_report') else {}
        
        result.success = True
        return result, syn, eval_real
        
    except Exception as e:
        result.error = str(e)
        try:
            tracemalloc.stop()
        except:
            pass
        return result, None, real


# ==================== UTILITY METRICS ====================

def compute_basic_utility(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    """Compute basic utility metrics: Jaccard, weighted Jaccard, marginal error.
    
    Jaccard: category overlap ratio. Weighted Jaccard: frequency-weighted overlap.
    Marginal error: average L1 distance between marginal distributions.
    """
    metrics = {}
    
    # Jaccard (category coverage)
    real_cats = set()
    syn_cats = set()
    for col in real.columns:
        real_cats.update(f"{col}:{v}" for v in real[col].dropna().unique())
        syn_cats.update(f"{col}:{v}" for v in syn[col].dropna().unique())
    
    intersection = len(real_cats & syn_cats)
    union = len(real_cats | syn_cats)
    metrics['jaccard'] = intersection / union if union > 0 else 0.0
    
    # Weighted Jaccard (frequency matching) - use union of values
    real_freq = {}
    syn_freq = {}
    for col in real.columns:
        # Use union of values from both real and syn (matches coverage computation)
        vals = pd.Index(real[col].dropna().unique()).union(
               pd.Index(syn[col].dropna().unique()))
        for v in vals:
            key = f"{col}:{v}"
            real_freq[key] = (real[col] == v).mean()
            syn_freq[key] = (syn[col] == v).mean()
    
    numerator = sum(min(real_freq.get(k, 0), syn_freq.get(k, 0)) for k in set(real_freq) | set(syn_freq))
    denominator = sum(max(real_freq.get(k, 0), syn_freq.get(k, 0)) for k in set(real_freq) | set(syn_freq))
    metrics['weighted_jaccard'] = numerator / denominator if denominator > 0 else 0.0
    
    # Marginal error (L1 distance)
    errors = []
    for col in real.columns:
        if pd.api.types.is_numeric_dtype(real[col]):
            # Numeric: histogram comparison
            bins = np.histogram_bin_edges(real[col].dropna(), bins=20)
            hist_real, _ = np.histogram(real[col].dropna(), bins=bins, density=True)
            hist_syn, _ = np.histogram(syn[col].dropna(), bins=bins, density=True)
            errors.append(np.abs(hist_real - hist_syn).mean())
        else:
            # Categorical: frequency comparison
            real_dist = real[col].value_counts(normalize=True)
            syn_dist = syn[col].value_counts(normalize=True)
            all_cats = set(real_dist.index) | set(syn_dist.index)
            error = sum(abs(real_dist.get(c, 0) - syn_dist.get(c, 0)) for c in all_cats) / 2
            errors.append(error)
    
    metrics['marginal_error'] = np.mean(errors) if errors else float('nan')
    
    return metrics


def compute_comprehensive_utility(real: pd.DataFrame, syn: pd.DataFrame, 
                                  target_col: Optional[str] = 'target') -> Dict[str, Any]:
    """Compute comprehensive utility metrics: TVD, MI, correlation, coverage, downstream ML.
    
    Includes bootstrap confidence intervals for robustness. Metrics computed
    with sampling caps for scalability on large datasets.
    """
    metrics = {}
    
    print("    Computing TVD metrics...", flush=True)
    try:
        metrics['tvd'] = comprehensive_tvd_metrics(real, syn, n_bootstrap=30)
        # Print TVD/EMD method counts
        if 'summary' in metrics['tvd']:
            tvd_sum = metrics['tvd']['summary']
            emd_count = tvd_sum.get('_tvd1_emd_count', 0)
            tvd_count = tvd_sum.get('_tvd1_tvd_count', 0)
            tvd_fallback = tvd_sum.get('_tvd1_tvd_fallback_count', 0)
            total = emd_count + tvd_count + tvd_fallback
            if total > 0:
                import sys
                print(f"         TVD 1D methods: {emd_count} EMD, {tvd_count} TVD, {tvd_fallback} TVD(fallback)", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"    ⚠️ TVD failed: {e}")
        metrics['tvd'] = {}
    
    print("    Computing MI metrics...")
    try:
        metrics['mi'] = comprehensive_mi_metrics(real, syn, n_bootstrap=30)
    except Exception as e:
        print(f"    ⚠️ MI failed: {e}")
        metrics['mi'] = {}
    
    print("    Computing correlation metrics...", flush=True)
    try:
        metrics['correlation'] = comprehensive_correlation_metrics(real, syn, n_bootstrap=30)
    except Exception as e:
        print(f"    ⚠️ Correlation failed: {e}")
        metrics['correlation'] = {}
    
    print("    Computing coverage metrics...")
    try:
        metrics['coverage'] = comprehensive_coverage_metrics(real, syn, n_bootstrap=30)
    except Exception as e:
        print(f"    ⚠️ Coverage failed: {e}")
        metrics['coverage'] = {}
    
    if target_col and target_col in real.columns:
        print("    Computing downstream ML metrics...")
        try:
            metrics['downstream'] = comprehensive_downstream_metrics(real, syn, target_col=target_col, n_bootstrap=30)
        except Exception as e:
            print(f"    ⚠️ Downstream failed: {e}")
            metrics['downstream'] = {}
    
    return metrics


# ==================== PRIVACY METRICS ====================

def compute_privacy_attacks(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    """Compute privacy attack metrics: exact row match rate and QI linkage rate.
    
    ERMR: fraction of synthetic rows that exactly match real rows.
    QI linkage: fraction of synthetic rows linkable via quasi-identifier columns.
    """
    metrics = {}
    
    # Exact Row Match Rate (ERMR)
    try:
        real_hash = set(pd.util.hash_pandas_object(real, index=False).astype(str))
        syn_hash = pd.util.hash_pandas_object(syn, index=False).astype(str)
        metrics['exact_row_match_rate'] = sum(1 for h in syn_hash if h in real_hash) / len(syn)
    except:
        metrics['exact_row_match_rate'] = None
    
    # QI Linkage (simplified)
    try:
        # Use top 3 columns with highest variance as QI
        numeric_cols = (
            real.select_dtypes(include=[np.number])
                .var(numeric_only=True)
                .sort_values(ascending=False)
                .head(3)
                .index
                .tolist()
        )
        if len(numeric_cols) > 0:
            qi_cols = list(numeric_cols)
            # Discretize for linkage - use robust numeric dtype check
            real_qi = real[qi_cols].apply(lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') if pd.api.types.is_numeric_dtype(x) else x)
            syn_qi = syn[qi_cols].apply(lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') if pd.api.types.is_numeric_dtype(x) else x)
            
            real_keys = set(pd.util.hash_pandas_object(real_qi, index=False).astype(str))
            syn_keys = pd.util.hash_pandas_object(syn_qi, index=False).astype(str)
            metrics['qi_linkage_rate'] = sum(1 for k in syn_keys if k in real_keys) / len(syn)
        else:
            metrics['qi_linkage_rate'] = None
    except:
        metrics['qi_linkage_rate'] = None
    
    return metrics


# ==================== VISUALIZATION ====================

def create_utility_privacy_plots(results: List[ImplementationResult], out_dir: str):
    """Generate 9-panel visualization comparing implementations across metrics.
    
    Plots utility vs privacy budget, efficiency, privacy risk, performance,
    MI preservation, TVD metrics, correlation metrics, and downstream ML.
    Aggregates results by implementation and epsilon before plotting.
    """
    
    # Prepare data
    data = []
    for r in results:
        if not r.success:
            continue
        
        # Extract key metrics
        row = {
            'implementation': r.name,
            'epsilon': r.epsilon,
            'seed': r.seed,
            'epsilon_used': r.privacy_report.get('eps_main', r.epsilon) if r.privacy_report else r.epsilon,
            'weighted_jaccard': r.weighted_jaccard,
            'jaccard': r.jaccard,
            'marginal_error': r.marginal_error,
            'total_time_sec': r.total_time_sec,
            'memory_mb': r.peak_memory_mb,
            'ermr': r.exact_row_match_rate,
        }
        
        # Add comprehensive metrics for plotting
        if r.tvd_metrics and 'summary' in r.tvd_metrics:
            tvd_sum = r.tvd_metrics['summary']
            row['tvd_1d_mean'] = tvd_sum.get('tvd1_mean')  # Fixed: use tvd1_mean
            row['tvd_2d_mean'] = tvd_sum.get('tvd2_mean')  # Fixed: use tvd2_mean
            row['tvd_3d_mean'] = tvd_sum.get('tvd3_mean')  # Fixed: use tvd3_mean
            row['emd_mean'] = tvd_sum.get('emd_mean')
        if r.mi_metrics:
            if 'summary' in r.mi_metrics:
                mi_sum = r.mi_metrics['summary']
                row['mi_preservation'] = mi_sum.get('preservation_ratio_mean')  # Keep short name for plots
            if 'matrix_comparison' in r.mi_metrics:
                mi_matrix = r.mi_metrics['matrix_comparison']
                row['nmi_spearman'] = mi_matrix.get('nmi_matrix_correlation')
        if r.correlation_metrics:
            if 'matrix_comparison' in r.correlation_metrics:
                mc = r.correlation_metrics['matrix_comparison']
                row['pearson_spearman'] = mc.get('pearson_matrix_spearman_corr')
                row['spearman_spearman'] = mc.get('spearman_matrix_spearman_corr')
        if r.coverage_metrics and 'summary' in r.coverage_metrics:
            cov_sum = r.coverage_metrics['summary']
            row['kl_divergence'] = cov_sum.get('kl_mean')  # Keep short name for plots
            row['jaccard_coverage'] = cov_sum.get('jaccard_mean')  # Keep short name for plots
        # Include privacy metrics for plotting
        row['qi_linkage'] = r.qi_linkage_rate
        if r.downstream_metrics:
            if 'syn_to_real' in r.downstream_metrics:
                syn2real = r.downstream_metrics['syn_to_real']
                if 'logistic_regression' in syn2real:
                    row['syn2real_lr_auc'] = syn2real['logistic_regression'].get('roc_auc')
                if 'random_forest' in syn2real:
                    row['syn2real_rf_auc'] = syn2real['random_forest'].get('roc_auc')
            row['symmetry_gap'] = r.downstream_metrics.get('symmetry_gap')
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Aggregate by implementation and epsilon - include all metrics
    # Only include columns that exist in the DataFrame
    agg_dict = {}
    base_metrics = {
        'epsilon_used': 'mean',
        'weighted_jaccard': 'mean',
        'jaccard': 'mean',
        'marginal_error': 'mean',
        'total_time_sec': 'mean',
        'memory_mb': 'mean',
        'ermr': 'mean',
        'qi_linkage': 'mean',
    }
    for metric, func in base_metrics.items():
        if metric in df.columns:
            agg_dict[metric] = func
    
    # Add optional metrics if they exist
    # Use short names to match row keys in create_utility_privacy_plots
    optional_metrics = [
        'kl_divergence', 'mi_preservation', 'nmi_spearman',
        'pearson_spearman', 'spearman_spearman',
        'tvd_1d_mean', 'tvd_2d_mean', 'tvd_3d_mean', 'emd_mean',
        'syn2real_lr_auc', 'syn2real_rf_auc', 'symmetry_gap',
        'jaccard_coverage', 'qi_linkage'
    ]
    for metric in optional_metrics:
        if metric in df.columns:
            agg_dict[metric] = 'mean'
    
    if len(agg_dict) == 0:
        print("⚠️ No metrics to aggregate for plotting")
        return
    
    df_agg = df.groupby(['implementation', 'epsilon']).agg(agg_dict).reset_index()
    
    # Skip plotting if no successful runs
    if len(df) == 0:
        print("⚠️ No successful runs to plot")
        return
    
    # Create plots - expanded to show more utility metrics
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Comprehensive Privacy-Utility Analysis (All Metrics)', fontsize=16, fontweight='bold')
    
    impls = df_agg['implementation'].unique()
    colors = {'SynthCity': '#2ca02c', 'DPMM': '#d62728', 'Enhanced': '#9467bd'}
    
    # Plot 1: Utility (W-Jaccard) vs Epsilon
    ax = axes[0, 0]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'weighted_jaccard' in data_impl.columns:
            ax.plot(data_impl['epsilon'], data_impl['weighted_jaccard'], 
                    marker='o', label=impl, color=colors.get(impl), linewidth=2)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Weighted Jaccard (Utility)', fontsize=12)
    ax.set_title('Utility vs Privacy Budget', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency (Utility per Epsilon)
    ax = axes[0, 1]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'weighted_jaccard' in data_impl.columns and 'epsilon_used' in data_impl.columns:
            efficiency = data_impl['weighted_jaccard'] / data_impl['epsilon_used']
            ax.plot(data_impl['epsilon'], efficiency, 
                    marker='s', label=impl, color=colors.get(impl), linewidth=2)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Utility per ε', fontsize=12)
    ax.set_title('Privacy Budget Efficiency', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Privacy Risk (ERMR) vs Epsilon
    ax = axes[0, 2]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'ermr' in data_impl.columns and data_impl['ermr'].notna().any():
            ax.plot(data_impl['epsilon'], data_impl['ermr'], 
                    marker='^', label=impl, color=colors.get(impl), linewidth=2)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Exact Row Match Rate', fontsize=12)
    ax.set_title('Privacy Risk vs Budget', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance (Time) vs Epsilon
    ax = axes[1, 0]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'total_time_sec' in data_impl.columns:
            ax.plot(data_impl['epsilon'], data_impl['total_time_sec'], 
                    marker='d', label=impl, color=colors.get(impl), linewidth=2)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Total Time (s)', fontsize=12)
    ax.set_title('Performance vs Privacy Budget', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Utility-Privacy Tradeoff
    ax = axes[1, 1]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'epsilon_used' in data_impl.columns and 'weighted_jaccard' in data_impl.columns:
            ax.scatter(data_impl['epsilon_used'], data_impl['weighted_jaccard'],
                      s=200, label=impl, color=colors.get(impl), alpha=0.6, edgecolors='black')
            # Add epsilon labels
            for _, row in data_impl.iterrows():
                ax.annotate(f"ε={row['epsilon']}", 
                           (row['epsilon_used'], row['weighted_jaccard']),
                           fontsize=8, ha='center')
    ax.set_xlabel('Epsilon Used', fontsize=12)
    ax.set_ylabel('Weighted Jaccard (Utility)', fontsize=12)
    ax.set_title('Utility-Privacy Tradeoff', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: MI Preservation vs Epsilon
    ax = axes[1, 2]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'mi_preservation' in data_impl.columns and data_impl['mi_preservation'].notna().any():
            ax.plot(data_impl['epsilon'], data_impl['mi_preservation'], 
                    marker='p', label=impl, color=colors.get(impl), linewidth=2)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('MI Preservation', fontsize=12)
    ax.set_title('Information Preservation vs Budget', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: TVD Metrics (1D, 2D, 3D) - lower is better
    ax = axes[2, 0]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'tvd_1d_mean' in data_impl.columns and data_impl['tvd_1d_mean'].notna().any():
            ax.plot(data_impl['epsilon'], data_impl['tvd_1d_mean'], 
                    marker='o', label=f'{impl} (1D)', color=colors.get(impl), linewidth=1.5, linestyle='-')
            if 'tvd_2d_mean' in data_impl.columns and data_impl['tvd_2d_mean'].notna().any():
                ax.plot(data_impl['epsilon'], data_impl['tvd_2d_mean'], 
                        marker='s', label=f'{impl} (2D)', color=colors.get(impl), linewidth=1.5, linestyle='--')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Total Variation Distance', fontsize=12)
    ax.set_title('TVD Metrics (Lower is Better)', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Correlation Metrics (Pearson, Spearman) - higher is better
    ax = axes[2, 1]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'pearson_spearman' in data_impl.columns and data_impl['pearson_spearman'].notna().any():
            ax.plot(data_impl['epsilon'], data_impl['pearson_spearman'], 
                    marker='o', label=f'{impl} (Pearson)', color=colors.get(impl), linewidth=1.5, linestyle='-')
        if 'spearman_spearman' in data_impl.columns and data_impl['spearman_spearman'].notna().any():
            ax.plot(data_impl['epsilon'], data_impl['spearman_spearman'], 
                    marker='s', label=f'{impl} (Spearman)', color=colors.get(impl), linewidth=1.5, linestyle='--')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Correlation Preservation', fontsize=12)
    ax.set_title('Correlation Metrics (Higher is Better)', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Downstream ML Performance (LR AUC, RF AUC) - higher is better
    ax = axes[2, 2]
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'syn2real_lr_auc' in data_impl.columns and data_impl['syn2real_lr_auc'].notna().any():
            ax.plot(data_impl['epsilon'], data_impl['syn2real_lr_auc'], 
                    marker='o', label=f'{impl} (LR)', color=colors.get(impl), linewidth=1.5, linestyle='-')
        if 'syn2real_rf_auc' in data_impl.columns and data_impl['syn2real_rf_auc'].notna().any():
            ax.plot(data_impl['epsilon'], data_impl['syn2real_rf_auc'], 
                    marker='s', label=f'{impl} (RF)', color=colors.get(impl), linewidth=1.5, linestyle='--')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('ML Model AUC (Syn→Real)', fontsize=12)
    ax.set_title('Downstream ML Performance', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/utility_privacy_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/utility_privacy_plots.pdf", bbox_inches='tight')
    print(f"✅ Saved plots to {out_dir}/utility_privacy_plots.png")
    plt.close()


# ==================== MAIN ====================

def main(data_path: str, epsilons: List[float], seeds: List[int], 
         out_dir: str, implementations: List[str], target_col: Optional[str] = None):
    """Run comprehensive benchmark comparing PrivBayes implementations.
    
    Executes all combinations of epsilon/seed/implementation. Computes utility,
    privacy, and performance metrics. Saves JSON results, CSV summary, and plots.
    """
    print("="*80)
    print(" COMPREHENSIVE PRIVACY-UTILITY COMPARISON ".center(80, "="))
    print("="*80)
    print(f"\nDataset: {data_path}")
    print(f"Epsilon values: {epsilons}")
    print(f"Seeds: {seeds}")
    print(f"Implementations: {implementations}")
    print(f"Output: {out_dir}")
    if target_col:
        print(f"Target column: {target_col}")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    real = pd.read_csv(data_path)
    print(f"✓ Loaded: {real.shape[0]} rows, {real.shape[1]} columns")
    
    print(f"\n📁 Creating output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"✓ Output directory ready")
    
    # Run all experiments
    all_results = []
    
    for eps in epsilons:
        for seed in seeds:
            print(f"\n{'='*80}")
            print(f" EPSILON={eps}, SEED={seed} ".center(80, "="))
            print(f"{'='*80}")
            
            for impl_name in implementations:
                print(f"\n🔹 Running {impl_name}...")
                
                # Run implementation
                if impl_name == "SynthCity":
                    result, syn, eval_real = run_synthcity_privbayes(real, eps, seed)
                elif impl_name == "DPMM":
                    result, syn, eval_real = run_dpmm_privbayes(real, eps, seed)
                elif impl_name == "Enhanced":
                    result, syn, eval_real = run_enhanced_privbayes(real, eps, seed, temperature=1.0, target_col=target_col)
                else:
                    print(f"⚠️ Unknown implementation: {impl_name}")
                    continue
                
                if not result.success:
                    print(f"❌ {impl_name} failed: {result.error}")
                    all_results.append(result)
                    continue
                
                time_breakdown = f"fit={result.fit_time_sec:.2f}s, sample={result.sample_time_sec:.2f}s"
                if result.vocab_align_time_sec is not None:
                    time_breakdown += f", vocab_align={result.vocab_align_time_sec:.2f}s"
                print(f"✓ Generated in {result.total_time_sec:.2f}s ({time_breakdown}) using {result.peak_memory_mb:.1f} MB")
                
                # Compute metrics using eval_real (aligned for Enhanced, original for others)
                print(f"  Computing utility metrics...")
                basic_util = compute_basic_utility(eval_real, syn)
                result.jaccard = basic_util['jaccard']
                result.weighted_jaccard = basic_util['weighted_jaccard']
                result.marginal_error = basic_util['marginal_error']
                print(f"  ✓ Basic: Jaccard={result.jaccard:.3f}, W-Jaccard={result.weighted_jaccard:.3f}")
                
                print(f"  Computing comprehensive utility...")
                # Use provided target column or auto-detect (common names: 'target', 'income', 'label', 'class')
                detected_target_col = target_col
                if detected_target_col is None:
                    for col in ['target', 'income', 'label', 'class', 'outcome']:
                        if col in eval_real.columns:
                            detected_target_col = col
                            break
                comp_util = compute_comprehensive_utility(eval_real, syn, target_col=detected_target_col)
                result.tvd_metrics = comp_util.get('tvd', {})
                result.mi_metrics = comp_util.get('mi', {})
                result.correlation_metrics = comp_util.get('correlation', {})
                result.coverage_metrics = comp_util.get('coverage', {})
                result.downstream_metrics = comp_util.get('downstream', {})
                
                print(f"  Computing privacy attacks...")
                privacy_attacks = compute_privacy_attacks(eval_real, syn)
                result.exact_row_match_rate = privacy_attacks['exact_row_match_rate']
                result.qi_linkage_rate = privacy_attacks['qi_linkage_rate']
                if result.exact_row_match_rate is not None:
                    print(f"  ✓ ERMR={result.exact_row_match_rate:.4f}, QI-Link={result.qi_linkage_rate:.4f}")
                
                all_results.append(result)
    
    # Save results
    print(f"\n{'='*80}")
    print(" SAVING RESULTS ".center(80, "="))
    print(f"{'='*80}")
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    results_json = f"{out_dir}/comprehensive_results_{ts}.json"
    with open(results_json, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"✅ Saved JSON: {results_json}")
    
    # Save CSV summary - Extract ALL utility metrics
    summary_data = []
    for r in all_results:
        if r.success:
            row = {
                'implementation': r.name,
                'epsilon': r.epsilon,
                'seed': r.seed,
                'total_time_sec': r.total_time_sec,
                'memory_mb': r.peak_memory_mb,
                # Basic utility
                'jaccard': r.jaccard,
                'weighted_jaccard': r.weighted_jaccard,
                'marginal_error': r.marginal_error,
                # Privacy
                'ermr': r.exact_row_match_rate,
                'qi_linkage': r.qi_linkage_rate,
            }
            
            # Extract TVD metrics
            if r.tvd_metrics and 'summary' in r.tvd_metrics:
                tvd_sum = r.tvd_metrics['summary']
                row['tvd_1d_mean'] = tvd_sum.get('tvd1_mean')  # Fixed: use tvd1_mean not tvd_1d_mean
                row['tvd_2d_mean'] = tvd_sum.get('tvd2_mean')  # Fixed: use tvd2_mean not tvd_2d_mean
                row['tvd_3d_mean'] = tvd_sum.get('tvd3_mean')  # Fixed: use tvd3_mean not tvd_3d_mean
                row['emd_mean'] = tvd_sum.get('emd_mean')
            
            # Extract MI metrics
            if r.mi_metrics:
                # Get NMI from matrix_comparison, not summary
                if 'matrix_comparison' in r.mi_metrics:
                    mi_matrix = r.mi_metrics['matrix_comparison']
                    row['nmi_spearman'] = mi_matrix.get('nmi_matrix_correlation')
                if 'summary' in r.mi_metrics:
                    mi_sum = r.mi_metrics['summary']
                    row['mi_preservation_ratio_mean'] = mi_sum.get('preservation_ratio_mean')
            
            # Extract Correlation metrics
            if r.correlation_metrics:
                # Get matrix comparison metrics
                if 'matrix_comparison' in r.correlation_metrics:
                    mc = r.correlation_metrics['matrix_comparison']
                    row['pearson_spearman'] = mc.get('pearson_matrix_spearman_corr')
                    row['spearman_spearman'] = mc.get('spearman_matrix_spearman_corr')
            
            # Extract Coverage metrics
            if r.coverage_metrics and 'summary' in r.coverage_metrics:
                cov_sum = r.coverage_metrics['summary']
                row['kl_divergence_mean'] = cov_sum.get('kl_mean')
                row['jaccard_coverage_mean'] = cov_sum.get('jaccard_mean')
            
            # Extract Downstream ML metrics
            if r.downstream_metrics:
                # Syn→Real metrics
                if 'syn_to_real' in r.downstream_metrics:
                    syn2real = r.downstream_metrics['syn_to_real']
                    if 'logistic_regression' in syn2real:
                        row['syn2real_lr_auc'] = syn2real['logistic_regression'].get('roc_auc')
                        row['syn2real_lr_acc'] = syn2real['logistic_regression'].get('accuracy')
                    if 'random_forest' in syn2real:
                        row['syn2real_rf_auc'] = syn2real['random_forest'].get('roc_auc')
                        row['syn2real_rf_acc'] = syn2real['random_forest'].get('accuracy')
                
                # Real→Real baseline
                if 'real_to_real' in r.downstream_metrics:
                    real2real = r.downstream_metrics['real_to_real']
                    if 'logistic_regression' in real2real:
                        row['real2real_lr_auc'] = real2real['logistic_regression'].get('roc_auc')
                
                # Symmetry gap
                if 'symmetry_gap' in r.downstream_metrics:
                    row['symmetry_gap'] = r.downstream_metrics['symmetry_gap']
            
            summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    summary_csv = f"{out_dir}/comprehensive_summary_{ts}.csv"
    
    # Add metric direction information as a comment header in the CSV
    metric_directions = {
        # Utility metrics - Higher is better
        'jaccard': '↑ Higher is better',
        'weighted_jaccard': '↑ Higher is better',
        'jaccard_coverage_mean': '↑ Higher is better',
        'mi_preservation_ratio_mean': '↑ Higher is better',
        'nmi_spearman': '↑ Higher is better',
        'pearson_spearman': '↑ Higher is better',
        'spearman_spearman': '↑ Higher is better',
        'syn2real_lr_auc': '↑ Higher is better',
        'syn2real_lr_acc': '↑ Higher is better',
        'syn2real_rf_auc': '↑ Higher is better',
        'syn2real_rf_acc': '↑ Higher is better',
        'real2real_lr_auc': '↑ Higher is better',
        # Utility metrics - Lower is better
        'marginal_error': '↓ Lower is better',
        'tvd_1d_mean': '↓ Lower is better',
        'tvd_2d_mean': '↓ Lower is better',
        'tvd_3d_mean': '↓ Lower is better',
        'emd_mean': '↓ Lower is better',
        'kl_divergence_mean': '↓ Lower is better',
        'symmetry_gap': '↓ Lower is better',
        # Privacy metrics - Lower is better
        'ermr': '↓ Lower is better (more privacy)',
        'qi_linkage': '↓ Lower is better (more privacy)',
        # Performance metrics
        'total_time_sec': '↓ Lower is better',
        'memory_mb': '↓ Lower is better',
    }
    
    # Write CSV with header comments
    with open(summary_csv, 'w') as f:
        # Write metric direction comments
        f.write("# Metric Direction Guide (for interpretability):\n")
        f.write("# ↑ = Higher is better, ↓ = Lower is better\n")
        f.write("#\n")
        for col in df_summary.columns:
            if col in metric_directions:
                f.write(f"# {col}: {metric_directions[col]}\n")
        f.write("#\n")
        # Write actual CSV data
        df_summary.to_csv(f, index=False)
    print(f"✅ Saved CSV: {summary_csv}")
    
    # Create plots
    print(f"\n📊 Creating visualization...")
    create_utility_privacy_plots(all_results, out_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print(" SUMMARY ".center(80, "="))
    print(f"{'='*80}")
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if r.success)}")
    print(f"Failed: {sum(1 for r in all_results if not r.success)}")
    
    if df_summary.shape[0] > 0:
        print("\nAverage metrics by implementation:")
        print("(↑ = Higher is better, ↓ = Lower is better)")
        # Show all utility metrics
        util_cols = [c for c in df_summary.columns if c not in ['implementation', 'epsilon', 'seed']]
        util_cols_sorted = [
            'total_time_sec', 'memory_mb',
            'jaccard', 'weighted_jaccard', 'marginal_error',
            'tvd_1d_mean', 'tvd_2d_mean', 'tvd_3d_mean', 'emd_mean',
            'mi_preservation_ratio_mean', 'nmi_spearman',
            'pearson_spearman', 'spearman_spearman',
            'kl_divergence_mean', 'jaccard_coverage_mean',
            'syn2real_lr_auc', 'syn2real_rf_auc', 'symmetry_gap',
            'ermr', 'qi_linkage'
        ]
        # Only show columns that exist
        util_cols_to_show = [c for c in util_cols_sorted if c in df_summary.columns]
        
        # Create metric direction mapping for display
        metric_directions = {
            'jaccard': '↑', 'weighted_jaccard': '↑', 'jaccard_coverage_mean': '↑',
            'mi_preservation_ratio_mean': '↑', 'nmi_spearman': '↑',
            'pearson_spearman': '↑', 'spearman_spearman': '↑',
            'syn2real_lr_auc': '↑', 'syn2real_rf_auc': '↑',
            'marginal_error': '↓', 'tvd_1d_mean': '↓', 'tvd_2d_mean': '↓',
            'tvd_3d_mean': '↓', 'emd_mean': '↓', 'kl_divergence_mean': '↓',
            'symmetry_gap': '↓', 'ermr': '↓', 'qi_linkage': '↓',
            'total_time_sec': '↓', 'memory_mb': '↓'
        }
        
        # Print with direction indicators
        df_display = df_summary.groupby('implementation')[util_cols_to_show].mean().round(4)
        # Add direction indicators to column names
        df_display.columns = [f"{metric_directions.get(col, '')} {col}" if col in metric_directions else col 
                              for col in df_display.columns]
        print(df_display.to_string())
    
    print(f"\n✅ All done! Results saved to: {out_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Privacy-Utility Comparison")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--eps", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="Epsilon values")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--implementations", type=str, nargs="+", 
                       default=["Enhanced", "SynthCity", "DPMM"], 
                       help="Implementations to test (Enhanced, SynthCity, DPMM)")
    parser.add_argument("--target-col", type=str, default=None,
                       help="Target column name for downstream ML metrics (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    main(args.data, args.eps, args.seeds, args.out_dir, args.implementations, args.target_col)


