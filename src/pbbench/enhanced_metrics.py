"""
Comprehensive metrics module addressing scalability, comparability, robustness, and coverage depth issues.

This module provides comprehensive versions of all privacy and utility metrics with:
- Scalable computation (sampling caps, deterministic subsampling)
- Confidence intervals via bootstrap
- Normalized and comparable metrics
- Bidirectional evaluation
- Statistical testing and aggregation (bootstrap CIs and matrix-level Spearman tests)
"""

from __future__ import annotations

__version__ = "0.1.2"

# NOTE: requires pandas >= 1.1 for groupby(dropna=False) and nullable Int64 dtypes

from typing import Dict, Tuple, List, Optional, Union, Any
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

# SciPy version guard for warnings
ConstantInputWarning = getattr(stats, "ConstantInputWarning", Warning)
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
from sklearn.exceptions import ConvergenceWarning
import hashlib
# Scope warnings filter to specific sklearn warnings only
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')

def _stable_seed(*parts, base=42):
    """Generate a stable seed from parts for per-metric bootstrap seeding."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return (int(h[:8], 16) ^ base) & 0xFFFFFFFF

def _as_prob_array(s: pd.Series, eps: float = 1e-12) -> np.ndarray:
    """
    Convert pandas Series to numpy probability array for scipy functions.
    Ensures float array, strictly positive, normalized.
    
    Args:
        s: pandas Series with probability values
        eps: small epsilon for numerical stability
        
    Returns:
        Normalized numpy array of probabilities (float, strictly positive)
    """
    # ensure float array, strictly positive, normalized
    a = np.asarray(pd.Series(s).astype(float).values, dtype=float)
    a = a + eps
    a = a / a.sum()
    return a

def _percentile_ci_from_list(values, confidence_level=0.95):
    """Percentile CI for a list/array of scalars (NaNs ignored)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    alpha = (1 - confidence_level) / 2 * 100
    lo = np.percentile(arr, alpha)
    hi = np.percentile(arr, 100 - alpha)
    return (float(lo), float(hi))

# Helper functions for consistent formatting
def _round_metric(value: float, decimals: int = 3) -> float:
    """Round metric values consistently."""
    if pd.isna(value) or np.isinf(value):
        return value
    return round(float(value), decimals)

def _round_ci(interval: Tuple[float, float], decimals: int = 2) -> Tuple[float, float]:
    """Round confidence intervals consistently."""
    if interval is None or any(pd.isna(x) or np.isinf(x) for x in interval):
        return interval
    return (round(float(interval[0]), decimals), round(float(interval[1]), decimals))

def _binary_auc(y_true, proba, classes_, pos_label=None):
    """Robust AUC for binary classification, handling 1D/2D probs and arbitrary labels."""
    proba = np.asarray(proba)
    classes_ = np.asarray(classes_)

    # 1D probability vector (unknown which class it corresponds to)
    if proba.ndim == 1:
        # Guard against >2 unique y_true values for 1D probabilities
        y_unique = np.unique(y_true)
        if len(y_unique) > 2:
            return np.nan
        
        # Use deterministic default if pos_label not specified
        if pos_label is None:
            # deterministic default: prefer 1 if present, else max label (string/numeric)
            pos_label = 1 if 1 in classes_ else np.sort(classes_)[-1]
        try:
            return roc_auc_score((np.asarray(y_true) == pos_label).astype(int), proba)
        except Exception:
            return np.nan

    # Use deterministic default if pos_label not specified
    if pos_label is None:
        # deterministic default: prefer 1 if present, else max label (string/numeric)
        pos_label = 1 if 1 in classes_ else np.sort(classes_)[-1]
    
    # If pos_label is present in classes, use it
    if pos_label in classes_:
        pos_idx = np.where(classes_ == pos_label)[0][0]
        return roc_auc_score((np.asarray(y_true) == pos_label).astype(int), proba[:, pos_idx])
    
    return np.nan

def _coerce_datetime_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime and timedelta columns to numeric for evaluation."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].view('int64')  # ns since epoch
        elif pd.api.types.is_timedelta64_dtype(out[c]):
            out[c] = out[c].view('int64')
    return out

def _fit_safely(estimator, X, y):
    """Fit estimator with scoped warnings to avoid hiding useful messages in caller's code."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        return estimator.fit(X, y)

def _split_cols(real: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical, ensuring no overlap and preserving order."""
    numeric = []
    for c in real.select_dtypes(include=[np.number]).columns.tolist():
        if not pd.api.types.is_bool_dtype(real[c]):
            numeric.append(c)
    categorical = real.select_dtypes(include=['object', 'category', 'bool', 'boolean']).columns.tolist()
    # preserve order, remove dups
    numeric = list(dict.fromkeys(numeric))
    categorical = [c for c in categorical if c not in numeric]
    return numeric, categorical

def _rows_signature(df: pd.DataFrame) -> np.ndarray:
    """Hash per-row, ignoring index, robust to dtypes."""
    return np.sort(pd.util.hash_pandas_object(df, index=False).to_numpy())

def _is_regression_target(y: pd.Series) -> bool:
    """Heuristic: small, integer-like label sets → classification; else regression."""
    y_non = pd.Series(y).dropna()
    n_unique = y_non.nunique()
    
    # Non-numeric data is always classification
    if not pd.api.types.is_numeric_dtype(y_non):
        return False
    
    # If label set is small and (near-)integer valued, assume classification
    if n_unique <= 20:
        uniq = y_non.unique()
        if np.all(np.isfinite(uniq)) and np.all(np.isclose(uniq, np.round(uniq))):
            return False
    # Otherwise treat as regression if float dtype or very high cardinality
    return pd.api.types.is_float_dtype(y) or n_unique > max(20, int(len(y) * 0.1))

# Public API
__all__ = [
    "comprehensive_tvd_metrics",
    "comprehensive_mi_metrics", 
    "comprehensive_correlation_metrics",
    "comprehensive_coverage_metrics",
    "comprehensive_downstream_metrics",
    "run_sanity_checks",
    "validate_metric_consistency"
]

# ============================================================================
# TVD METRICS WITH SCALABILITY AND ROBUSTNESS
# ============================================================================

def comprehensive_tvd_metrics(
    real: pd.DataFrame, 
    syn: pd.DataFrame, 
    max_pairs: int = 50,
    max_triples: int = 20,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    use_emd: bool = True,
    random_seed: int = 42,
    int_cardinality_as_categorical: int = 20,
    tvd_quantiles: int = 10,
    emd_sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    TVD computation with sampling, confidence intervals, and EMD for continuous variables.
    
    Args:
        real: Real dataset
        syn: Synthetic dataset
        max_pairs: Maximum number of pairs to sample for 2D TVD
        max_triples: Maximum number of triples to sample for 3D TVD
        n_bootstrap: Number of bootstrap samples for confidence intervals
        confidence_level: Confidence level for intervals
        use_emd: Whether to use Earth Mover's Distance for continuous variables
        random_seed: Random seed for deterministic sampling
        int_cardinality_as_categorical: Integer columns with <= this many unique values are treated as categorical
        tvd_quantiles: Number of quantiles for TVD discretization
        emd_sample_size: Optional cap on sample size for EMD computation (for very long columns)
    
    Returns:
        Dictionary with TVD metrics including confidence intervals.
        EMD is normalized by real IQR (fixed for CI stability).
        
    Notes:
        - Boolean columns are treated as categorical (TVD) and won't be EMD-binned
        - EMD is only applied to numeric (non-boolean) columns
        - Integer columns with <= int_cardinality_as_categorical unique values are treated as categorical
    """
    import random
    random.seed(random_seed)  # Set Python's random seed for any future use
    rng = np.random.default_rng(random_seed)
    results = {}
    
    # Convert datetime columns to numeric for evaluation
    real = _coerce_datetime_to_numeric(real)
    syn = _coerce_datetime_to_numeric(syn)
    
    # Get numerical and categorical columns (disjoint sets)
    numerical_cols, categorical_cols = _split_cols(real)
    all_cols = list(dict.fromkeys([*numerical_cols, *categorical_cols]))
    
    # Pre-fit bin edges on full real data for numeric (non-bool) cols that aren't small-cardinality integers
    num_for_bins = []
    for c in numerical_cols:
        if pd.api.types.is_bool_dtype(real[c]):
            continue
        if pd.api.types.is_integer_dtype(real[c]) and real[c].dropna().nunique() <= int_cardinality_as_categorical:
            continue  # treat as categorical (no edges => no discretization)
        num_for_bins.append(c)
    
    tvd_edges = {c: _fit_quantile_bins(real[c], q=tvd_quantiles) for c in num_for_bins}
    
    # Pre-bin once for TVD 2D/3D to avoid repeated discretization
    binned_real = {}
    binned_syn = {}
    for c in all_cols:
        if c in tvd_edges:  # numeric-with-edges
            binned_real[c] = _discretize_with_edges(pd.to_numeric(real[c], errors='coerce'), tvd_edges[c])
            binned_syn[c] = _discretize_with_edges(pd.to_numeric(syn[c], errors='coerce'), tvd_edges[c])
        else:
            # Harmonize small-cardinality ints to Int64 once here
            if pd.api.types.is_integer_dtype(real[c]) and real[c].dropna().nunique() <= int_cardinality_as_categorical:
                binned_real[c] = real[c].astype('Int64')
                binned_syn[c] = pd.to_numeric(syn[c], errors='coerce').round().astype('Int64')
            else:
                binned_real[c] = real[c]
                binned_syn[c] = syn[c]
    
    def _tvd2_fixed_factory(c1, c2):
        e1 = tvd_edges.get(c1, None); e2 = tvd_edges.get(c2, None)
        def _f(d1, d2):
            A = d1[[c1, c2]].copy()
            B = d2[[c1, c2]].copy()

            # NEW: harmonize dtypes before PMF (covers small-cardinality ints)
            for c in (c1, c2):
                r = d1[c]; s = d2[c]
                if pd.api.types.is_integer_dtype(r):
                    A[c] = r.astype('Int64')
                    B[c] = pd.to_numeric(s, errors='coerce').round().astype('Int64')
                elif (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r)
                      and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
                    A[c] = pd.to_numeric(r, errors='coerce')
                    B[c] = pd.to_numeric(s, errors='coerce')

            if e1 is not None:
                A[c1] = _discretize_with_edges(pd.to_numeric(A[c1], errors="coerce"), e1)
                B[c1] = _discretize_with_edges(pd.to_numeric(B[c1], errors="coerce"), e1)
            if e2 is not None:
                A[c2] = _discretize_with_edges(pd.to_numeric(A[c2], errors="coerce"), e2)
                B[c2] = _discretize_with_edges(pd.to_numeric(B[c2], errors="coerce"), e2)
            rp = _pmf(A, [c1, c2]); sp = _pmf(B, [c1, c2])
            idx = rp.index.union(sp.index)
            return 0.5 * np.abs(rp.reindex(idx, fill_value=0).values - sp.reindex(idx, fill_value=0).values).sum()
        return _f

    def _tvd3_fixed_factory(c1, c2, c3):
        e = {c: tvd_edges.get(c, None) for c in (c1, c2, c3)}
        def _f(d1, d2):
            A = d1[[c1, c2, c3]].copy()
            B = d2[[c1, c2, c3]].copy()

            # NEW: harmonize dtypes before PMF
            for c in (c1, c2, c3):
                r = d1[c]; s = d2[c]
                if pd.api.types.is_integer_dtype(r):
                    A[c] = r.astype('Int64')
                    B[c] = pd.to_numeric(s, errors='coerce').round().astype('Int64')
                elif (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r)
                      and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
                    A[c] = pd.to_numeric(r, errors='coerce')
                    B[c] = pd.to_numeric(s, errors='coerce')

                if e[c] is not None:
                    A[c] = _discretize_with_edges(pd.to_numeric(A[c], errors="coerce"), e[c])
                    B[c] = _discretize_with_edges(pd.to_numeric(B[c], errors="coerce"), e[c])
            rp = _pmf(A, [c1, c2, c3]); sp = _pmf(B, [c1, c2, c3])
            idx = rp.index.union(sp.index)
            return 0.5 * np.abs(rp.reindex(idx, fill_value=0).values - sp.reindex(idx, fill_value=0).values).sum()
        return _f
    
    # 1D TVD with confidence intervals
    tvd1_results = {}
    for col in all_cols:
        if col in syn.columns:
            is_numeric = col in numerical_cols
            is_small_int = (
                is_numeric and pd.api.types.is_integer_dtype(real[col]) and
                real[col].dropna().nunique() <= int_cardinality_as_categorical
            )
            is_constant_numeric = is_numeric and real[col].dropna().nunique() <= 1
            if is_numeric and use_emd and not is_small_int and not is_constant_numeric:
                # Use Earth Mover's Distance for continuous variables (IQR-normalized)
                real_vals_full = _clean_numeric(real[col]).dropna().to_numpy()
                syn_vals_full = _clean_numeric(syn[col]).dropna().to_numpy()
                if len(real_vals_full) > 0 and len(syn_vals_full) > 0:
                    # Compute scale on full data for CI consistency
                    q25, q75 = np.nanpercentile(real_vals_full, [25, 75])
                    scale = (q75 - q25) if (q75 > q25) else (real_vals_full.max() - real_vals_full.min())
                    scale0 = scale  # Use same scale for CI
                    
                    # Optional sampling for very long columns (without replacement)
                    real_vals = real_vals_full
                    syn_vals = syn_vals_full
                    if emd_sample_size:
                        m1 = min(emd_sample_size, real_vals_full.size)
                        m2 = min(emd_sample_size, syn_vals_full.size)
                        idx1 = rng.choice(real_vals_full.size, size=m1, replace=False)
                        idx2 = rng.choice(syn_vals_full.size, size=m2, replace=False)
                        real_vals = real_vals_full[idx1]
                        syn_vals = syn_vals_full[idx2]
                    
                    emd_raw = wasserstein_distance(real_vals, syn_vals)
                    emd_norm = emd_raw / (scale + 1e-8)
                    
                    # Use fixed scale for CI to avoid mixing drift with changing scale
                    
                    ci = bootstrap_confidence_interval(
                        real_vals, syn_vals,
                        lambda a, b, s=scale0: wasserstein_distance(a, b) / (s + 1e-8),
                        n_bootstrap, confidence_level, _stable_seed('tvd1', col, base=random_seed)
                    )
                    
                    tvd1_results[col] = {
                        'value': _round_metric(emd_norm),
                        'raw': _round_metric(emd_raw),
                        'scale': _round_metric(scale),
                        'method': 'EMD(IQR-normalized)',
                        'confidence_interval': _round_ci(ci)
                    }
                else:
                    # Fallback: compute TVD when EMD arrays are empty (e.g., syn is non-numeric tokens)
                    tvd_val = _compute_tvd_1d(real, syn, col)
                    ci = bootstrap_confidence_interval(
                        real[[col]], syn[[col]], _tvd_wrapper, n_bootstrap, confidence_level,
                        _stable_seed('tvd1_cat', col, base=random_seed)
                    )
                    tvd1_results[col] = {
                        'value': _round_metric(tvd_val),
                        'method': 'TVD(fallback from EMD)',
                        'confidence_interval': _round_ci(ci)
                    }
            else:
                # Use TVD for categorical variables
                tvd_val = _compute_tvd_1d(real, syn, col)
                ci = bootstrap_confidence_interval(
                    real[[col]], syn[[col]], _tvd_wrapper, n_bootstrap, confidence_level, _stable_seed('tvd1_cat', col, base=random_seed)
                )
                tvd1_results[col] = {
                    'value': _round_metric(tvd_val),
                    'method': 'TVD',
                    'confidence_interval': _round_ci(ci)
                }
    
    results['tvd1'] = tvd1_results
    
    # 2D TVD with sampling
    if len(all_cols) >= 2:
        pairs = list(itertools.combinations(all_cols, 2))
        if len(pairs) > max_pairs:
            # Deterministic sampling
            selected_pairs = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in selected_pairs]
        
        tvd2_results = {}
        for col1, col2 in pairs:
            if col1 in syn.columns and col2 in syn.columns:
                tvd_val = _compute_tvd_2d(real, syn, col1, col2, tvd_edges, tvd_quantiles, binned_real, binned_syn)
                ci = bootstrap_confidence_interval(
                    real[[col1, col2]], syn[[col1, col2]], _tvd2_fixed_factory(col1, col2), n_bootstrap, confidence_level, _stable_seed('tvd2', col1, col2, base=random_seed)
                )
                tvd2_results[f"{col1}|{col2}"] = {
                    'value': _round_metric(tvd_val),
                    'confidence_interval': _round_ci(ci),
                    'method': 'TVD'
                }
        
        results['tvd2'] = tvd2_results
    
    # 3D TVD with sampling
    if len(all_cols) >= 3:
        triples = list(itertools.combinations(all_cols, 3))
        if len(triples) > max_triples:
            # Deterministic sampling
            selected_triples = rng.choice(len(triples), max_triples, replace=False)
            triples = [triples[i] for i in selected_triples]
        
        tvd3_results = {}
        for col1, col2, col3 in triples:
            if all(col in syn.columns for col in [col1, col2, col3]):
                tvd_val = _compute_tvd_3d(real, syn, col1, col2, col3, tvd_edges, tvd_quantiles, binned_real, binned_syn)
                ci = bootstrap_confidence_interval(
                    real[[col1, col2, col3]], syn[[col1, col2, col3]], _tvd3_fixed_factory(col1, col2, col3), n_bootstrap, confidence_level, _stable_seed('tvd3', col1, col2, col3, base=random_seed)
                )
                tvd3_results[f"{col1}|{col2}|{col3}"] = {
                    'value': _round_metric(tvd_val),
                    'confidence_interval': _round_ci(ci),
                    'method': 'TVD'
                }
        
        results['tvd3'] = tvd3_results
    
    # Summary statistics
    results['summary'] = _compute_tvd_summary(results)
    
    # Add counts for auditability
    results['counts'] = {
        'tvd1_n': len(results.get('tvd1', {})),
        'tvd2_n': len(results.get('tvd2', {})),
        'tvd3_n': len(results.get('tvd3', {}))
    }
    
    # Expose prefit edges for auditability
    results['tvd_bins'] = {k: v.tolist() for k, v in tvd_edges.items()}
    
    # Add parameters for auditability
    results['params'] = {
        'random_seed': random_seed,
        'max_pairs': max_pairs,
        'max_triples': max_triples,
        'use_emd': use_emd,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'int_cardinality_as_categorical': int_cardinality_as_categorical,
        'tvd_quantiles': tvd_quantiles,
        'emd_sample_size': emd_sample_size,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'tvd_edges_features_count': len(tvd_edges),
        'version': __version__
    }
    
    return results

def _compute_tvd_1d(real: pd.DataFrame, syn: pd.DataFrame, col: str) -> float:
    """Compute 1D TVD for a single column with consistent NaN handling."""
    r = real[col]
    s = syn[col]
    # If real is integer-like, coerce syn to nullable Int64 (preserves NaN)
    if pd.api.types.is_integer_dtype(r):
        s = pd.to_numeric(s, errors='coerce').round().astype('Int64')
        r = r.astype('Int64')
    # If both are numeric (but not bool), align numeric types to avoid 1 vs 1.0 splits
    elif (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r) and
          pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
        r = pd.to_numeric(r, errors='coerce')
        s = pd.to_numeric(s, errors='coerce')

    real_counts = pd.Series(r).value_counts(normalize=True, dropna=False)
    syn_counts = pd.Series(s).value_counts(normalize=True, dropna=False)
    idx = real_counts.index.union(syn_counts.index)
    rp = real_counts.reindex(idx, fill_value=0).values
    sp = syn_counts.reindex(idx, fill_value=0).values
    return 0.5 * np.abs(rp - sp).sum()

def _compute_tvd_2d(real: pd.DataFrame, syn: pd.DataFrame, col1: str, col2: str, tvd_edges: Optional[Dict[str, np.ndarray]] = None, tvd_quantiles: int = 10, binned_real: Optional[Dict[str, pd.Series]] = None, binned_syn: Optional[Dict[str, pd.Series]] = None) -> float:
    """Compute 2D TVD for a pair of columns with discretization for continuous variables."""
    # Use pre-binned data if available (performance optimization)
    if binned_real is not None and binned_syn is not None and col1 in binned_real and col2 in binned_real:
        real_ = pd.DataFrame({col1: binned_real[col1], col2: binned_real[col2]})
        syn_ = pd.DataFrame({col1: binned_syn[col1], col2: binned_syn[col2]})
    else:
        # Apply consistent bin fitting for stability
        if (pd.api.types.is_numeric_dtype(real[col1]) and not pd.api.types.is_bool_dtype(real[col1])) or (pd.api.types.is_numeric_dtype(real[col2]) and not pd.api.types.is_bool_dtype(real[col2])):
            # Use prefit edges if available, otherwise fit bins on real data
            if tvd_edges is not None:
                edges1 = tvd_edges.get(col1, None)
                edges2 = tvd_edges.get(col2, None)
            else:
                edges1 = _fit_quantile_bins(real[col1], q=tvd_quantiles) if (pd.api.types.is_numeric_dtype(real[col1]) and not pd.api.types.is_bool_dtype(real[col1])) else None
                edges2 = _fit_quantile_bins(real[col2], q=tvd_quantiles) if (pd.api.types.is_numeric_dtype(real[col2]) and not pd.api.types.is_bool_dtype(real[col2])) else None
            
            real_ = real[[col1, col2]].copy()
            syn_ = syn[[col1, col2]].copy()
            
            # Align dtypes for both sides before the PMF to avoid 1 vs 1.0 splits
            for c in (col1, col2):
                r = real[c]
                s = syn[c]
                if pd.api.types.is_integer_dtype(r):
                    # keep discrete integer semantics identical on both sides
                    real_[c] = r.astype('Int64')
                    syn_[c] = pd.to_numeric(s, errors='coerce').round().astype('Int64')
                elif (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r)
                      and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
                    # avoid 1 vs 1.0 splits
                    real_[c] = pd.to_numeric(r, errors='coerce')
                    syn_[c] = pd.to_numeric(s, errors='coerce')
            
            if edges1 is not None:
                real_[col1] = _discretize_with_edges(real_[col1], edges1)
                syn_[col1] = _discretize_with_edges(syn_[col1], edges1)
            if edges2 is not None:
                real_[col2] = _discretize_with_edges(real_[col2], edges2)
                syn_[col2] = _discretize_with_edges(syn_[col2], edges2)
        else:
            real_ = real[[col1, col2]]
            syn_ = syn[[col1, col2]]
    
    real_pmf = _pmf(real_, [col1, col2])
    syn_pmf = _pmf(syn_, [col1, col2])
    
    # Align indexes
    idx = real_pmf.index.union(syn_pmf.index)
    real_probs = real_pmf.reindex(idx, fill_value=0).values
    syn_probs = syn_pmf.reindex(idx, fill_value=0).values
    
    return 0.5 * np.abs(real_probs - syn_probs).sum()

def _compute_tvd_3d(real: pd.DataFrame, syn: pd.DataFrame, col1: str, col2: str, col3: str, tvd_edges: Optional[Dict[str, np.ndarray]] = None, tvd_quantiles: int = 10, binned_real: Optional[Dict[str, pd.Series]] = None, binned_syn: Optional[Dict[str, pd.Series]] = None) -> float:
    """Compute 3D TVD for a triple of columns with discretization for continuous variables."""
    cols = [col1, col2, col3]
    
    # Use pre-binned data if available (performance optimization)
    if binned_real is not None and binned_syn is not None and all(col in binned_real for col in cols):
        real_ = pd.DataFrame({col1: binned_real[col1], col2: binned_real[col2], col3: binned_real[col3]})
        syn_ = pd.DataFrame({col1: binned_syn[col1], col2: binned_syn[col2], col3: binned_syn[col3]})
    else:
        # Apply consistent bin fitting for stability
        real_ = real[cols].copy()
        syn_ = syn[cols].copy()
    
    for col in cols:
        r = real[col]
        s = syn[col]
        if pd.api.types.is_integer_dtype(r):
            real_[col] = r.astype('Int64')
            syn_[col] = pd.to_numeric(s, errors='coerce').round().astype('Int64')
        elif (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r)
              and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
            real_[col] = pd.to_numeric(r, errors='coerce')
            syn_[col] = pd.to_numeric(s, errors='coerce')
        
        # Use prefit edges if available, otherwise fit bins on real data
        if tvd_edges is not None:
            edges = tvd_edges.get(col, None)
        else:
            edges = _fit_quantile_bins(r, q=tvd_quantiles) if (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r)) else None
        if edges is not None and not pd.api.types.is_bool_dtype(r):
            real_[col] = _discretize_with_edges(real_[col], edges)
            syn_[col] = _discretize_with_edges(syn_[col], edges)
    
    real_pmf = _pmf(real_, cols)
    syn_pmf = _pmf(syn_, cols)
    
    # Align indexes
    idx = real_pmf.index.union(syn_pmf.index)
    real_probs = real_pmf.reindex(idx, fill_value=0).values
    syn_probs = syn_pmf.reindex(idx, fill_value=0).values
    
    return 0.5 * np.abs(real_probs - syn_probs).sum()

def _clean_numeric(s: pd.Series) -> pd.Series:
    """
    Clean numeric series by converting to numeric and replacing inf values with NaN.
    
    Args:
        s: Input series to clean
        
    Returns:
        Cleaned series with inf values replaced by NaN
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def _pmf(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Compute probability mass function for given columns."""
    counts = df.groupby(cols, dropna=False).size()
    return counts / counts.sum()

def _harmonize_numeric_like(real_s: pd.Series, syn_s: pd.Series) -> pd.Series:
    """Coerce synthetic data to match real data type when real is numeric (excluding boolean)."""
    if pd.api.types.is_numeric_dtype(real_s) and not pd.api.types.is_bool_dtype(real_s):
        return _clean_numeric(syn_s)
    return syn_s.astype(real_s.dtype, errors="ignore")

def _compare_corr_matrices(real_df: pd.DataFrame, syn_df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """Compare correlation matrices between real and synthetic data.

    Coerces both sides to numeric, drops columns that are fully NA post-coercion,
    computes Pearson correlation matrices, then Spearman-correlates their upper triangles.
    This is good and stable for comparing correlation structure.
    
    Guarantees exact same ordered column list for both matrices.
    """
    if len(cols) < 2:
        # expose both new and legacy keys
        return {
            "pearson_matrix_spearman_corr": np.nan,
            "pearson_matrix_spearman_p_value": np.nan,
            "corr_matrix_spearman": np.nan,
            "corr_matrix_p_value": np.nan,
            "spearman_matrix_spearman_corr": np.nan,
            "spearman_matrix_spearman_p_value": np.nan,
            "matrix_columns_used": [],
            "matrix_columns_count": 0,
            "strong_pair_sign_agreement": np.nan,
        }

    # CRITICAL: Use exact same ordered column list for both dataframes
    # Filter to columns that exist in both, then sort for consistent ordering
    cols_ordered = sorted([c for c in cols if c in real_df.columns and c in syn_df.columns])
    if len(cols_ordered) < 2:
        return {
            "pearson_matrix_spearman_corr": np.nan,
            "pearson_matrix_spearman_p_value": np.nan,
            "corr_matrix_spearman": np.nan,
            "corr_matrix_p_value": np.nan,
            "spearman_matrix_spearman_corr": np.nan,
            "spearman_matrix_spearman_p_value": np.nan,
            "matrix_columns_used": [],
            "matrix_columns_count": 0,
            "strong_pair_sign_agreement": np.nan,
        }

    # Coerce both to numeric using EXACT SAME column order
    Rn = real_df[cols_ordered].apply(pd.to_numeric, errors="coerce")
    Sn = syn_df[cols_ordered].apply(pd.to_numeric, errors="coerce")

    # Keep only columns that are not all-NA on BOTH sides
    keep = [c for c in cols_ordered if not (Rn[c].isna().all() or Sn[c].isna().all())]
    if len(keep) < 2:
        return {
            "pearson_matrix_spearman_corr": np.nan,
            "pearson_matrix_spearman_p_value": np.nan,
            "corr_matrix_spearman": np.nan,
            "corr_matrix_p_value": np.nan,
            "spearman_matrix_spearman_corr": np.nan,
            "spearman_matrix_spearman_p_value": np.nan,
            "matrix_columns_used": [],
            "matrix_columns_count": 0,
            "strong_pair_sign_agreement": np.nan,
        }
    
    # GUARANTEE: Use exact same ordered column list for both matrices
    # keep is already sorted from cols_ordered, but ensure it's sorted
    keep = sorted(keep)
    
    # Extract with EXACT SAME column order - this is critical
    Rn_keep = Rn[keep]
    Sn_keep = Sn[keep]
    
    # Compute correlation matrices using EXACT SAME column order
    # Both will have columns in the same order: keep
    R = Rn_keep.corr(method="pearson")
    S = Sn_keep.corr(method="pearson")
    
    # Convert to numpy AFTER ensuring column alignment
    # Verify column order matches
    assert list(R.columns) == list(S.columns) == keep, "Column order mismatch!"
    assert list(R.index) == list(S.index) == keep, "Row order mismatch!"
    
    R_np = R.to_numpy()
    S_np = S.to_numpy()
    
    # Extract upper triangle (excluding diagonal) - same order guaranteed
    i = np.triu_indices_from(R_np, k=1)
    Ri, Si = R_np[i], S_np[i]
    mask = np.isfinite(Ri) & np.isfinite(Si)
    if mask.sum() < 2:
        return {
            "pearson_matrix_spearman_corr": np.nan,
            "pearson_matrix_spearman_p_value": np.nan,
            "corr_matrix_spearman": np.nan,
            "corr_matrix_p_value": np.nan,
            "spearman_matrix_spearman_corr": np.nan,
            "spearman_matrix_spearman_p_value": np.nan,
            "matrix_columns_used": keep,
            "matrix_columns_count": len(keep),
            "strong_pair_sign_agreement": np.nan,
        }
    Ri_masked = Ri[mask]
    Si_masked = Si[mask]
    
    # Check for constant arrays to avoid warnings
    Ri_std = np.std(Ri_masked)
    Si_std = np.std(Si_masked)
    
    if Ri_std == 0 or Si_std == 0:
        rho, p = np.nan, np.nan
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConstantInputWarning)
            rho, p = stats.spearmanr(Ri_masked, Si_masked)
    
    # Spearman-of-Spearman matrices - use EXACT SAME column order
    Rs = Rn_keep.corr(method="spearman")
    Ss = Sn_keep.corr(method="spearman")
    
    # Verify alignment again
    assert list(Rs.columns) == list(Ss.columns) == keep, "Spearman column order mismatch!"
    assert list(Rs.index) == list(Ss.index) == keep, "Spearman row order mismatch!"
    
    Rs_np = Rs.to_numpy()
    Ss_np = Ss.to_numpy()
    
    i2 = np.triu_indices_from(Rs_np, k=1)
    r2, s2 = Rs_np[i2], Ss_np[i2]
    mask2 = np.isfinite(r2) & np.isfinite(s2)
    if mask2.sum() >= 2:
        r2_masked = r2[mask2]
        s2_masked = s2[mask2]
        r2_std = np.std(r2_masked)
        s2_std = np.std(s2_masked)
        if r2_std == 0 or s2_std == 0:
            rho_ss, p_ss = np.nan, np.nan
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConstantInputWarning)
                rho_ss, p_ss = stats.spearmanr(r2_masked, s2_masked)
    else:
        rho_ss, p_ss = np.nan, np.nan
    
    # Check sign agreement on strong pairs (|r| >= 0.4)
    # Use the correlation DataFrames to get proper column names for reporting
    sign_agreements = []
    flipped_pairs = []  # Track which pairs flipped
    for i in range(len(keep)):
        for j in range(i+1, len(keep)):
            r_real = R_np[i, j]
            r_syn = S_np[i, j]
            if abs(r_real) >= 0.4 and np.isfinite(r_real) and np.isfinite(r_syn):
                sign_match = np.sign(r_real) == np.sign(r_syn)
                sign_agreements.append(sign_match)
                if not sign_match:
                    flipped_pairs.append((keep[i], keep[j], r_real, r_syn))
    
    strong_pair_sign_agreement = np.nan
    if len(sign_agreements) > 0:
        strong_pair_sign_agreement = sum(sign_agreements) / len(sign_agreements)
    
    return {
        "pearson_matrix_spearman_corr": float(rho),
        "pearson_matrix_spearman_p_value": float(p),
        "spearman_matrix_spearman_corr": float(rho_ss),
        "spearman_matrix_spearman_p_value": float(p_ss),
        # legacy aliases
        "corr_matrix_spearman": float(rho),
        "corr_matrix_p_value": float(p),
        # Debugging info
        "matrix_columns_used": keep,
        "matrix_columns_count": len(keep),
        "strong_pair_sign_agreement": float(strong_pair_sign_agreement) if np.isfinite(strong_pair_sign_agreement) else np.nan,
        "_flipped_pairs": flipped_pairs[:10] if flipped_pairs else [],  # Store top 10 flipped pairs
    }

# ---------- Helpers for fixed discretization (used by MI bootstrap) ----------
def _build_mi_discretizers(
    real_df: pd.DataFrame, q: int, int_cardinality_as_categorical: int
) -> Dict[str, Dict[str, Any]]:
    """
    For each column, return either {'kind':'edges','value':np.ndarray}
    or {'kind':'map','value':dict} fitted on REAL data only.
    """
    spec = {}
    for c in real_df.columns:
        r = real_df[c]
        # booleans & small-cardinality integers → discrete map
        is_bool_like = pd.api.types.is_bool_dtype(r)
        is_small_int = (pd.api.types.is_integer_dtype(r) and r.dropna().nunique() <= int_cardinality_as_categorical)
        unique_vals = set(r.dropna().unique())
        is_binary_int = (pd.api.types.is_integer_dtype(r) and unique_vals.issubset({0, 1, np.int64(0), np.int64(1)}))
        if is_bool_like or is_small_int or is_binary_int:
            cats = pd.Index(sorted(pd.unique(r.dropna())))
            spec[c] = {"kind": "map", "value": {v: i for i, v in enumerate(cats)}}
        else:
            spec[c] = {"kind": "edges", "value": _fit_quantile_bins(r, q)}
    return spec

def _apply_mi_discretizers(df: pd.DataFrame, disc_spec: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        s = df[c]
        kind = disc_spec[c]["kind"]
        val = disc_spec[c]["value"]
        if kind == "map":
            out[c] = s.map(val)
        else:
            out[c] = _discretize_with_edges(pd.to_numeric(s, errors="coerce"), val)
    return pd.DataFrame(out)

# ---------- Helpers for correlation CIs ----------
def _corr_delta_factory(c1, c2, method):
    def _f(d1: pd.DataFrame, d2: pd.DataFrame):
        r = _safe_corr_pair(d1[c1], d1[c2], method)
        s = _safe_corr_pair(d2[c1], d2[c2], method)
        return float(r - s) if (isinstance(r, (int, float)) and isinstance(s, (int, float))) else np.nan
    return _f

def _pointbiserial_delta_factory(num_col, cat_col, mapping=None):
    """Delta for point-biserial with fixed mapping from full REAL support."""
    def _f(d1: pd.DataFrame, d2: pd.DataFrame):
        if mapping is None:
            # Fallback: derive from d1 if not provided (may be unstable if category missing)
            cats = pd.Categorical(d1[cat_col])
            if cats.categories.size != 2:
                return np.nan
            levels = list(cats.categories)
            mapping_local = {levels[0]: 0, levels[1]: 1}
        else:
            mapping_local = mapping
        r_codes = d1[cat_col].map(mapping_local)
        s_codes = d2[cat_col].map(mapping_local)
        r = _safe_corr_pair(d1[num_col], r_codes, "pearson")
        s = _safe_corr_pair(d2[num_col], s_codes, "pearson")
        return float(r - s) if (isinstance(r, (int, float)) and isinstance(s, (int, float))) else np.nan
    return _f

# ---------- Helpers for coverage CIs ----------
def _build_coverage_spec(real: pd.DataFrame, bin_numerics: bool, q: int, int_cardinality_as_categorical: int):
    spec = {}
    for c in real.columns:
        r = real[c]
        if pd.api.types.is_bool_dtype(r) or (
            pd.api.types.is_integer_dtype(r) and r.dropna().nunique() <= int_cardinality_as_categorical
        ):
            cats = pd.Index(sorted(pd.unique(r.dropna())))
            spec[c] = {"kind": "map", "value": {v: i for i, v in enumerate(cats)}, "label": c}
        elif bin_numerics and pd.api.types.is_numeric_dtype(r):
            edges = _fit_quantile_bins(r, q=q)
            spec[c] = {"kind": "edges", "value": edges, "label": f"{c}__binned"}
        else:
            spec[c] = {"kind": "raw", "value": None, "label": c}
    return spec

def _apply_coverage_spec(series: pd.Series, spec_entry):
    if spec_entry["kind"] == "map":
        return pd.Series(series).map(spec_entry["value"])
    if spec_entry["kind"] == "edges":
        return _discretize_with_edges(_clean_numeric(series), spec_entry["value"])
    return series


# ============================================================================
# MUTUAL INFORMATION METRICS
# ============================================================================

def comprehensive_mi_metrics(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    topk: int = 10,
    random_seed: int = 42,
    max_pairs: Optional[int] = None,
    bins: int = 10,
    matrix_col_cap: Optional[int] = None,
    int_cardinality_as_categorical: int = 20,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Mutual information metrics with normalization and matrix comparison.
    
    Args:
        real: Real dataset
        syn: Synthetic dataset
        topk: Number of top MI pairs to report
        random_seed: Random seed for reproducibility
        max_pairs: Maximum number of pairs to evaluate (None = no cap)
        bins: Number of bins for discretization
        matrix_col_cap: Maximum number of columns for matrix computation (None = no cap)
        int_cardinality_as_categorical: Integer columns with <= this many unique values treated as categorical
    
    Returns:
        Dictionary with MI metrics
    """
    results = {}
    
    # Convert datetime columns to numeric for evaluation
    real = _coerce_datetime_to_numeric(real)
    syn = _coerce_datetime_to_numeric(syn)
    
    # Get numerical and boolean columns (disjoint sets)
    numerical_cols, categorical_cols = _split_cols(real)
    numerical_cols = [col for col in numerical_cols if col in syn.columns]
    boolean_cols = [col for col in categorical_cols if col in syn.columns and pd.api.types.is_bool_dtype(real[col])]
    
    if len(numerical_cols) < 2 and len(boolean_cols) == 0:
        return {'error': 'Insufficient numerical or boolean columns for MI computation'}
    
    # Prepare data for discretization (ensure no overlap)
    all_cols = list(dict.fromkeys([*numerical_cols, *boolean_cols]))
    
    # Guard against insufficient columns
    if len(all_cols) < 2:
        return {'error': 'Need at least two numerical/boolean columns for MI computation'}
    
    # Apply matrix column cap if specified (deterministic selection)
    matrix_cols = all_cols
    mi_dropped = []
    if matrix_col_cap is not None and len(all_cols) > matrix_col_cap:
        # Use stable hash for deterministic selection across runs
        def _stable_int(s: str) -> int:
            return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        
        sorted_cols = sorted(all_cols)
        order = np.argsort([_stable_int(c) for c in sorted_cols])
        matrix_cols = [sorted_cols[i] for i in order[:matrix_col_cap]]
        
        # Stage dropped columns for later inclusion in params
        mi_dropped = sorted(set(all_cols) - set(matrix_cols))
    
    real_data = real[matrix_cols].copy()
    syn_data = syn[matrix_cols].copy()
    
    # Convert boolean columns to 0/1 for MI computation (only those in matrix)
    bool_in_matrix = [c for c in boolean_cols if c in real_data.columns]
    for col in bool_in_matrix:
        # Preserve NaNs; use nullable Int64 to keep them
        real_data[col] = real_data[col].astype('Int64')
        syn_data[col] = syn_data[col].astype('Int64')
    
    # Discretize both datasets using bins fitted on real data for consistency
    real_disc, syn_disc = _discretize_cols_with_fitted_bins(real_data, syn_data, q=bins, int_cardinality_as_categorical=int_cardinality_as_categorical)
    
    # Store discretization edges for auditability (only for columns that were actually discretized)
    discretization_edges = {}
    for col in matrix_cols:  # Only check columns that were actually used in the matrix
        if pd.api.types.is_numeric_dtype(real_data[col]) and not pd.api.types.is_bool_dtype(real_data[col]):
            # Check if this column would be treated as discrete
            unique_vals = set(real_data[col].dropna().unique())
            is_binary_int = (pd.api.types.is_integer_dtype(real_data[col]) and unique_vals.issubset({0, 1, np.int64(0), np.int64(1)}))
            is_small_int = (pd.api.types.is_integer_dtype(real_data[col]) and real_data[col].dropna().nunique() <= int_cardinality_as_categorical)
            
            if not is_binary_int and not is_small_int:
                discretization_edges[col] = _fit_quantile_bins(real_data[col], q=bins).tolist()
    
    # Compute MI and NMI matrices on discretized data
    real_mi_matrix, real_nmi_matrix = _compute_mi_and_nmi(real_disc)
    syn_mi_matrix, syn_nmi_matrix = _compute_mi_and_nmi(syn_disc)
    
    # Fixed discretizers for pair-level bootstrap (fitted on REAL full data)
    disc_spec = _build_mi_discretizers(real_data[matrix_cols], bins, int_cardinality_as_categorical)

    def _mi_pair_delta_factory(c1, c2):
        def _f(d1: pd.DataFrame, d2: pd.DataFrame):
            R = _apply_mi_discretizers(d1[[c1, c2]], disc_spec)
            S = _apply_mi_discretizers(d2[[c1, c2]], disc_spec)
            mi_r, mi_s = _compute_mi_and_nmi(R)[0], _compute_mi_and_nmi(S)[0]
            return float(mi_r[0, 1] - mi_s[0, 1])
        return _f

    def _nmi_pair_delta_factory(c1, c2):
        def _f(d1: pd.DataFrame, d2: pd.DataFrame):
            R = _apply_mi_discretizers(d1[[c1, c2]], disc_spec)
            S = _apply_mi_discretizers(d2[[c1, c2]], disc_spec)
            nmi_r, nmi_s = _compute_mi_and_nmi(R)[1], _compute_mi_and_nmi(S)[1]
            return float(nmi_r[0, 1] - nmi_s[0, 1])
        return _f

    # Per-pair MI comparison with optional subsampling
    mi_pairs = {}
    
    # Build list of upper-triangular indices for matrix columns
    matrix_idxs = [(i, j) for i in range(len(matrix_cols)) for j in range(i+1, len(matrix_cols))]
    
    # Apply subsampling if requested
    if max_pairs and len(matrix_idxs) > max_pairs:
        rng = np.random.default_rng(random_seed)
        sel = rng.choice(len(matrix_idxs), max_pairs, replace=False)
        matrix_idxs = [matrix_idxs[k] for k in sel]
    
    for i, j in matrix_idxs:
        col1, col2 = matrix_cols[i], matrix_cols[j]
        pair_key = f"{col1}|{col2}"
        
        real_mi = real_mi_matrix[i, j]
        syn_mi = syn_mi_matrix[i, j]
        real_nmi = real_nmi_matrix[i, j]
        syn_nmi = syn_nmi_matrix[i, j]
        
        # Delta MI
        delta_mi = real_mi - syn_mi
        delta_nmi = real_nmi - syn_nmi
        
        # Compute both preservation metrics
        # Handle near-zero real MI to avoid unstable ratios
        if real_mi < 1e-6:
            preservation_ratio = None  # Unstable when real MI is too small
        else:
            preservation_ratio = syn_mi / (real_mi + 1e-8)
        preservation_bounded = min(syn_mi, real_mi) / (max(syn_mi, real_mi) + 1e-12)
        
        ci_delta_mi = ci_delta_nmi = ci_pres_ratio = None
        if n_bootstrap and n_bootstrap > 0:
            ci_delta_mi = bootstrap_confidence_interval(
                real[matrix_cols], syn[matrix_cols],
                _mi_pair_delta_factory(col1, col2),
                n_bootstrap, confidence_level,
                _stable_seed("mi_delta", col1, col2, base=random_seed),
                sample_size=0
            )
            ci_delta_nmi = bootstrap_confidence_interval(
                real[matrix_cols], syn[matrix_cols],
                _nmi_pair_delta_factory(col1, col2),
                n_bootstrap, confidence_level,
                _stable_seed("nmi_delta", col1, col2, base=random_seed),
                sample_size=0
            )
            # preservation_ratio is scale-free but bounded; bootstrap on syn/real MI ratio
            def _pres_ratio_factory(c1, c2):
                def _f(d1, d2):
                    R = _apply_mi_discretizers(d1[[c1, c2]], disc_spec)
                    S = _apply_mi_discretizers(d2[[c1, c2]], disc_spec)
                    mi_r, mi_s = _compute_mi_and_nmi(R)[0], _compute_mi_and_nmi(S)[0]
                    r, s = float(mi_r[0, 1]), float(mi_s[0, 1])
                    return s / (r + 1e-8)
                return _f
            ci_pres_ratio = bootstrap_confidence_interval(
                real[matrix_cols], syn[matrix_cols],
                _pres_ratio_factory(col1, col2),
                n_bootstrap, confidence_level,
                _stable_seed("mi_pres_ratio", col1, col2, base=random_seed),
                sample_size=0
            )

        mi_pairs[pair_key] = {
            'real_mi': _round_metric(real_mi),
            'syn_mi': _round_metric(syn_mi),
            'delta_mi': _round_metric(delta_mi),
            'real_nmi': _round_metric(real_nmi),
            'syn_nmi': _round_metric(syn_nmi),
            'delta_nmi': _round_metric(delta_nmi),
            'preservation_ratio': _round_metric(preservation_ratio),
            'preservation_bounded': _round_metric(preservation_bounded),
            'confidence_intervals': {
                'delta_mi': _round_ci(ci_delta_mi) if ci_delta_mi else None,
                'delta_nmi': _round_ci(ci_delta_nmi) if ci_delta_nmi else None,
                'preservation_ratio': _round_ci(ci_pres_ratio) if ci_pres_ratio else None
            }
        }
    
    results['mi_pairs'] = mi_pairs
    
    # Top-k MI pairs (filter to finite deltas first)
    finite_items = [(k, v) for k, v in mi_pairs.items() if np.isfinite(v.get('delta_mi', np.nan))]
    sorted_pairs = sorted(finite_items, key=lambda x: abs(x[1]['delta_mi']), reverse=True)
    results['topk_delta_mi'] = dict(sorted_pairs[:topk])
    
    # Top-k NMI pairs (filter to finite deltas first)
    finite_nmi = [(k, v) for k, v in mi_pairs.items() if np.isfinite(v.get('delta_nmi', np.nan))]
    sorted_nmi = sorted(finite_nmi, key=lambda x: abs(x[1]['delta_nmi']), reverse=True)
    results['topk_delta_nmi'] = dict(sorted_nmi[:topk])
    
    # Matrix-level comparison (+ bootstrap CI on MI/NMI matrix Spearman)
    matrix_comp = _compare_mi_matrices(real_mi_matrix, syn_mi_matrix, real_nmi_matrix, syn_nmi_matrix)

    if n_bootstrap and n_bootstrap > 0:
        # Bootstrap CI for MI matrix Spearman correlation
        def _mi_mat_spear(d1, d2):
            R = _apply_mi_discretizers(d1[matrix_cols], disc_spec)
            S = _apply_mi_discretizers(d2[matrix_cols], disc_spec)
            rm, _ = _compute_mi_and_nmi(R)
            sm, _ = _compute_mi_and_nmi(S)
            return _compare_mi_matrices(rm, sm)['matrix_correlation']
        ci_mat = bootstrap_confidence_interval(
            real[matrix_cols], syn[matrix_cols], _mi_mat_spear,
            n_bootstrap, confidence_level, _stable_seed("mi_matrix", "spearman", base=random_seed),
            sample_size=0
        )
        # Bootstrap CI for NMI matrix Spearman correlation
        def _nmi_mat_spear(d1, d2):
            R = _apply_mi_discretizers(d1[matrix_cols], disc_spec)
            S = _apply_mi_discretizers(d2[matrix_cols], disc_spec)
            _, rn = _compute_mi_and_nmi(R)
            _, sn = _compute_mi_and_nmi(S)
            return _compare_mi_matrices(np.zeros_like(rn), np.zeros_like(sn), rn, sn)['nmi_matrix_correlation']
        ci_nmi = bootstrap_confidence_interval(
            real[matrix_cols], syn[matrix_cols], _nmi_mat_spear,
            n_bootstrap, confidence_level, _stable_seed("nmi_matrix", "spearman", base=random_seed),
            sample_size=0
        )
        matrix_comp['matrix_correlation_ci'] = _round_ci(ci_mat)
        matrix_comp['nmi_matrix_correlation_ci'] = _round_ci(ci_nmi)

    results['matrix_comparison'] = {k: (_round_metric(v) if isinstance(v, (int, float)) else v)
                                   for k, v in matrix_comp.items()}
    
    # Discretization edges for auditability
    results['discretization_edges'] = discretization_edges
    
    # Summary statistics
    results['summary'] = _compute_mi_summary(mi_pairs)
    
    # Add counts for auditability
    results['counts'] = {
        'mi_pairs_n': len(results.get('mi_pairs', {}))
    }
    
    # Add parameters for auditability
    results['params'] = {
        'random_seed': random_seed,
        'max_pairs': max_pairs,
        'bins': bins,
        'topk': topk,
        'matrix_col_cap': matrix_col_cap,
        'int_cardinality_as_categorical': int_cardinality_as_categorical,
        'numerical_cols': numerical_cols,
        'boolean_cols': boolean_cols,
        'matrix_cols_count': len(matrix_cols),
        'mi_matrix_kept_cols': matrix_cols,
        'mi_matrix_dropped_cols': mi_dropped,
        'version': __version__,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }
    
    return results


def _compare_mi_matrices(real_mi: np.ndarray, syn_mi: np.ndarray, real_nmi: np.ndarray = None, syn_nmi: np.ndarray = None) -> Dict[str, Any]:
    """Compare MI matrices using various metrics."""
    def _upper(a):
        return a[np.triu_indices_from(a, k=1)]
    
    r, s = _upper(real_mi), _upper(syn_mi)
    mask = np.isfinite(r) & np.isfinite(s)

    if mask.sum() < 2:
        out = dict(
            matrix_correlation=np.nan, 
            matrix_p_value=np.nan,
            mean_absolute_error=np.nan, 
            root_mean_square_error=np.nan, 
            max_error=np.nan
        )
    else:
        r_masked = r[mask]
        s_masked = s[mask]
        
        # Check for constant arrays to avoid warnings
        r_std = np.std(r_masked)
        s_std = np.std(s_masked)
        
        if r_std == 0 or s_std == 0:
            # Constant array - correlation is not defined
            rho, p = np.nan, np.nan
        else:
            # Suppress ConstantInputWarning for spearmanr
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConstantInputWarning)
                rho, p = stats.spearmanr(r_masked, s_masked)
        
        diff = r_masked - s_masked
        out = dict(
            matrix_correlation=float(rho),
            matrix_p_value=float(p),
            mean_absolute_error=float(np.mean(np.abs(diff))),
            root_mean_square_error=float(np.sqrt(np.mean(diff**2))),
            max_error=float(np.max(np.abs(diff)))
        )

    if real_nmi is not None and syn_nmi is not None:
        rn, sn = _upper(real_nmi), _upper(syn_nmi)
        m2 = np.isfinite(rn) & np.isfinite(sn)
        if m2.sum() >= 2:
            rn_masked = rn[m2]
            sn_masked = sn[m2]
            
            # Check for constant arrays
            rn_std = np.std(rn_masked)
            sn_std = np.std(sn_masked)
            
            if rn_std == 0 or sn_std == 0:
                rho2, p2 = np.nan, np.nan
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConstantInputWarning)
                    rho2, p2 = stats.spearmanr(rn_masked, sn_masked)
            out['nmi_matrix_correlation'] = float(rho2)
            out['nmi_matrix_p_value'] = float(p2)
        else:
            out['nmi_matrix_correlation'] = np.nan
            out['nmi_matrix_p_value'] = np.nan
    
    return out

# ============================================================================
# CORRELATION METRICS
# ============================================================================

def comprehensive_correlation_metrics(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    random_seed: int = 42,
    max_pairs: Optional[int] = None,
    max_pairs_mixed: Optional[int] = None,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Correlation metrics with multiple correlation types and summary statistics.
    
    Args:
        real: Real dataset
        syn: Synthetic dataset
        random_seed: Random seed for reproducibility
        max_pairs: Maximum number of pairs to evaluate (None = no cap)
        max_pairs_mixed: Maximum number of mixed pairs to evaluate (None = no cap)
                      Note: On very wide tables, mixed pairs can dominate runtime
    
    Returns:
        Dictionary with correlation metrics
    """
    results = {}
    
    # Convert datetime columns to numeric for evaluation
    real = _coerce_datetime_to_numeric(real)
    syn = _coerce_datetime_to_numeric(syn)
    
    # Get numerical and categorical columns (disjoint sets)
    numerical_cols, categorical_cols = _split_cols(real)
    
    # CRITICAL: Build numeric column list from REAL only, then filter to those in syn
    # This ensures consistent ordering - use exact same list for both matrices
    num_cols = sorted([c for c in numerical_cols if c in syn.columns])
    cat_cols = [c for c in categorical_cols if c in syn.columns]
    
    # Coerce synthetic numeric columns to match real data types
    syn = syn.copy()
    for c in num_cols:
        syn[c] = _harmonize_numeric_like(real[c], syn[c])
    
    # Numerical-numerical correlations with optional subsampling
    num_num_correlations = {}
    if len(num_cols) >= 2:
        # Build list of upper-triangular indices
        idxs = [(i, j) for i in range(len(num_cols)) for j in range(i+1, len(num_cols))]
        
        # Apply subsampling if requested
        if max_pairs and len(idxs) > max_pairs:
            rng = np.random.default_rng(random_seed)
            sel = rng.choice(len(idxs), max_pairs, replace=False)
            idxs = [idxs[k] for k in sel]
        
        for i, j in idxs:
            col1, col2 = num_cols[i], num_cols[j]
            pair_key = f"{col1}|{col2}"
            
            # Pearson correlation
            real_pearson = _safe_corr_pair(real[col1], real[col2], "pearson")
            syn_pearson = _safe_corr_pair(syn[col1], syn[col2], "pearson")
            
            # Spearman correlation
            real_spearman = _safe_corr_pair(real[col1], real[col2], "spearman")
            syn_spearman = _safe_corr_pair(syn[col1], syn[col2], "spearman")
            
            ci_pears = ci_spear = None
            if n_bootstrap and n_bootstrap > 0:
                ci_pears = bootstrap_confidence_interval(
                    real[[col1, col2]], syn[[col1, col2]],
                    _corr_delta_factory(col1, col2, "pearson"),
                    n_bootstrap, confidence_level,
                    _stable_seed("corr_delta_pearson", col1, col2, base=random_seed),
                    sample_size=0
                )
                ci_spear = bootstrap_confidence_interval(
                    real[[col1, col2]], syn[[col1, col2]],
                    _corr_delta_factory(col1, col2, "spearman"),
                    n_bootstrap, confidence_level,
                    _stable_seed("corr_delta_spearman", col1, col2, base=random_seed),
                    sample_size=0
                )

            num_num_correlations[pair_key] = {
                'pearson': {
                    'real': _round_metric(real_pearson),
                    'syn': _round_metric(syn_pearson),
                    'error': _round_metric(abs(real_pearson - syn_pearson)),
                    'preservation_ratio': _round_metric(syn_pearson / (real_pearson + 1e-8)),
                    'confidence_interval_delta': _round_ci(ci_pears) if ci_pears else None
                },
                'spearman': {
                    'real': _round_metric(real_spearman),
                    'syn': _round_metric(syn_spearman),
                    'error': _round_metric(abs(real_spearman - syn_spearman)),
                    'preservation_ratio': _round_metric(syn_spearman / (real_spearman + 1e-8)),
                    'confidence_interval_delta': _round_ci(ci_spear) if ci_spear else None
                }
            }
    
    results['numerical_correlations'] = num_num_correlations
    
    # Categorical-categorical correlations (Cramer's V) with optional subsampling
    cat_cat_correlations = {}
    if len(cat_cols) >= 2:
        # Build list of upper-triangular indices
        idxs = [(i, j) for i in range(len(cat_cols)) for j in range(i+1, len(cat_cols))]
        
        # Apply subsampling if requested
        if max_pairs and len(idxs) > max_pairs:
            rng = np.random.default_rng(random_seed)
            sel = rng.choice(len(idxs), max_pairs, replace=False)
            idxs = [idxs[k] for k in sel]
        
        for i, j in idxs:
            col1, col2 = cat_cols[i], cat_cols[j]
            pair_key = f"{col1}|{col2}"
            
            # Cramer's V (bias-corrected)
            real_cramers = _cramers_v_corrected(real[col1], real[col2])
            syn_cramers = _cramers_v_corrected(syn[col1], syn[col2])
            
            ci = None
            if n_bootstrap and n_bootstrap > 0:
                def _cat_pair_err_factory(c1, c2):
                    def _f(d1, d2):
                        r = _cramers_v_corrected(d1[c1], d1[c2])
                        s = _cramers_v_corrected(d2[c1], d2[c2])
                        return abs(r - s) if np.isfinite(r) and np.isfinite(s) else np.nan
                    return _f
                ci = bootstrap_confidence_interval(
                    real[cat_cols], syn[cat_cols],
                    _cat_pair_err_factory(col1, col2),
                    n_bootstrap, confidence_level, _stable_seed("corr_cat", col1, col2, base=random_seed), sample_size=0
                )

            cat_cat_correlations[pair_key] = {
                'cramers_v': {
                    'real': _round_metric(real_cramers),
                    'syn': _round_metric(syn_cramers),
                    'error': _round_metric(abs(real_cramers - syn_cramers)),
                    'preservation_ratio': _round_metric(syn_cramers / (real_cramers + 1e-8)),
                    'confidence_interval_delta': _round_ci(ci) if ci else None
                }
            }
    
    results['categorical_correlations'] = cat_cat_correlations
    
    # Mixed correlations (numerical-categorical) with optional subsampling
    mixed_correlations = {}
    if len(num_cols) > 0 and len(cat_cols) > 0:
        # Build list of mixed pairs
        mixed_pairs = [(num_col, cat_col) for num_col in num_cols 
                      for cat_col in cat_cols]
        
        # Apply subsampling if requested
        if max_pairs_mixed and len(mixed_pairs) > max_pairs_mixed:
            rng = np.random.default_rng(random_seed)
            sel = rng.choice(len(mixed_pairs), max_pairs_mixed, replace=False)
            mixed_pairs = [mixed_pairs[k] for k in sel]
        
        # Build mapping (binary) once from REAL for reproducibility
        bin_maps = {}
        for cat_col in cat_cols:
            r = real[cat_col].astype('category')
            if r.cat.categories.size == 2:
                levels = list(r.cat.categories)
                pos = levels[-1]
                neg = levels[0] if levels[1] == pos else levels[1]
                bin_maps[cat_col] = {neg: 0, pos: 1}

        def _mixed_err_factory(num_c, cat_c, is_binary):
            def _f(d1, d2):
                if is_binary:
                    m = bin_maps.get(cat_c, None)
                    if m is None:
                        return np.nan
                    r_codes = d1[cat_c].map(m)
                    s_codes = d2[cat_c].map(m)
                    r = _safe_corr_pair(d1[num_c], r_codes, "pearson")
                    s = _safe_corr_pair(d2[num_c], s_codes, "pearson")
                else:
                    r = _correlation_ratio(d1[cat_c], d1[num_c])
                    s = _correlation_ratio(d2[cat_c], d2[num_c])
                return abs(r - s) if np.isfinite(r) and np.isfinite(s) else np.nan
            return _f

        for num_col, cat_col in mixed_pairs:
            pair_key = f"{num_col}|{cat_col}"
            
            # Use point-biserial only for binary categoricals, otherwise correlation ratio
            cats = real[cat_col].astype('category')
            if cats.cat.categories.size == 2:
                # Point-biserial correlation for binary categoricals with consistent coding
                r_codes, s_codes, dropped_count = _binary_codes_pair(real[cat_col], syn[cat_col])
                if r_codes is not None and s_codes is not None:
                    real_pb = _safe_corr_pair(real[num_col], r_codes, "pearson")
                    syn_pb = _safe_corr_pair(syn[num_col], s_codes, "pearson")
                else:
                    real_pb = syn_pb = np.nan
                    dropped_count = 0
                corr_type = "point_biserial"
                
                # Track unseen categories separately (exclude NaNs)
                unseen_mask = syn[cat_col].notna() & ~syn[cat_col].isin(real[cat_col].dropna().unique())
                dropped_unseen = int(unseen_mask.sum())
                dropped_total = int(dropped_count)  # includes NaNs + unseen
            else:
                # Correlation ratio (η²) for multi-level categoricals
                real_pb = _correlation_ratio(real[cat_col], real[num_col])
                syn_pb = _correlation_ratio(syn[cat_col], syn[num_col])
                corr_type = "correlation_ratio"
                dropped_total = 0
                dropped_unseen = 0
            
            ci = None
            if n_bootstrap and n_bootstrap > 0:
                is_bin = cats.cat.categories.size == 2
                ci = bootstrap_confidence_interval(
                    real[[num_col, cat_col]], syn[[num_col, cat_col]],
                    _mixed_err_factory(num_col, cat_col, is_bin),
                    n_bootstrap, confidence_level, _stable_seed("corr_mixed", num_col, cat_col, base=random_seed), sample_size=0
                )

            mixed_correlations[pair_key] = {
                corr_type: {
                    'real': _round_metric(real_pb),
                    'syn': _round_metric(syn_pb),
                    'error': _round_metric(abs(real_pb - syn_pb)),
                    'preservation_ratio': _round_metric(syn_pb / (real_pb + 1e-8)),
                    'confidence_interval_delta': _round_ci(ci) if ci else None
                },
                'dropped_rows': dropped_total,
                'dropped_unseen_categories': dropped_unseen
            }
    
    results['mixed_correlations'] = mixed_correlations
    
    # Matrix-level comparison for numerical correlations
    # CRITICAL: Use exact same column list (from real) for both matrices
    # num_cols is already sorted and filtered to columns in both real and syn
    if len(num_cols) >= 2:
        # Verify all columns exist in both (should already be true, but double-check)
        num_cols_final = [c for c in num_cols if c in real.columns and c in syn.columns]
        if len(num_cols_final) < 2:
            num_cols_final = num_cols  # Fallback
        
        # Use this exact list for matrix comparison
        matrix_comparison = _compare_corr_matrices(real, syn, num_cols_final)
        
        # Extract and log debugging info
        used_cols = matrix_comparison.get('matrix_columns_used', [])
        col_count = matrix_comparison.get('matrix_columns_count', 0)
        sign_agreement = matrix_comparison.get('strong_pair_sign_agreement', np.nan)
        
        if col_count > 0:
            import sys
            print(f"         Matrix comparison: {col_count} numeric columns used: {', '.join(used_cols[:10])}{'...' if len(used_cols) > 10 else ''}", file=sys.stderr, flush=True)
            if col_count < 5:
                print(f"         ⚠️  Warning: Only {col_count} columns - matrix score may be noisy", file=sys.stderr, flush=True)
            
            if np.isfinite(sign_agreement):
                # Count strong pairs and show flipped pairs
                try:
                    real_subset = real[used_cols].apply(pd.to_numeric, errors='coerce')
                    strong_count = 0
                    for i in range(len(used_cols)):
                        for j in range(i+1, len(used_cols)):
                            r_val = real_subset[[used_cols[i], used_cols[j]]].corr().iloc[0,1]
                            if abs(r_val) >= 0.4 and np.isfinite(r_val):
                                strong_count += 1
                    
                    flipped_pairs = matrix_comparison.get('_flipped_pairs', [])
                    if flipped_pairs:
                        print(f"         Strong pair sign agreement: {sign_agreement:.2%} ({strong_count} strong pairs, {len(flipped_pairs)} flipped)", file=sys.stderr, flush=True)
                        if len(flipped_pairs) > 0:
                            top_flip = flipped_pairs[0]
                            print(f"         Top flipped pair: {top_flip[0]}-{top_flip[1]} (real={top_flip[2]:.3f}, syn={top_flip[3]:.3f})", file=sys.stderr, flush=True)
                    else:
                        print(f"         Strong pair sign agreement: {sign_agreement:.2%} ({strong_count} strong pairs)", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"         Strong pair sign agreement: {sign_agreement:.2%} (error counting: {e})", file=sys.stderr, flush=True)
            else:
                print(f"         No strong pairs found (|r| >= 0.4) for sign agreement check", file=sys.stderr, flush=True)
        elif len(num_cols_final) >= 2:
            print(f"         ⚠️  Warning: Matrix comparison failed - columns dropped during coercion", file=sys.stderr, flush=True)
        
        if n_bootstrap and n_bootstrap > 0:
            def _corr_mat_spear(d1, d2):
                comp = _compare_corr_matrices(d1, d2, num_cols_final)
                return comp.get("pearson_matrix_spearman_corr", np.nan)
            ci_mat = bootstrap_confidence_interval(
                real[num_cols_final], syn[num_cols_final], _corr_mat_spear,
                n_bootstrap, confidence_level, _stable_seed("corr_matrix", "spearman", base=random_seed),
                sample_size=0
            )
            matrix_comparison['pearson_matrix_spearman_corr_ci'] = _round_ci(ci_mat)
        
        # Remove non-serializable items before storing
        matrix_comparison_clean = {k: v for k, v in matrix_comparison.items() 
                                   if k not in ['matrix_columns_used']}  # Keep count and sign_agreement
        results['matrix_comparison'] = {k: (_round_metric(v) if isinstance(v, (int, float)) else v)
                                       for k, v in matrix_comparison_clean.items()}
        # Store column info separately for debugging
        if used_cols:
            results['matrix_comparison']['_debug_columns'] = used_cols[:10]  # Store first 10 for reference
    
    # Summary statistics
    results['summary'] = _compute_correlation_summary(results)
    
    # Add counts for auditability
    results['counts'] = {
        'num_pairs_n': len(results.get('numerical_correlations', {})),
        'cat_pairs_n': len(results.get('categorical_correlations', {})),
        'mixed_pairs_n': len(results.get('mixed_correlations', {}))
    }
    
    # Add parameters for auditability
    results['params'] = {
        'random_seed': random_seed,
        'max_pairs': max_pairs,
        'max_pairs_mixed': max_pairs_mixed,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols_used': num_cols,
        'categorical_cols_used': cat_cols,
        'version': __version__,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }
    
    return results

def _cramers_v_corrected(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V with guards for sparse/degenerate tables."""
    confusion = pd.crosstab(x, y, dropna=True)
    n = confusion.values.sum()
    if n == 0 or confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0

    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    r, k = confusion.shape
    phi2 = chi2 / n
    # Bias correction (Bergsma 2013)
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(max(kcorr - 1, 0.0), max(rcorr - 1, 0.0))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0

def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Compute correlation ratio (η²) for categorical-numerical relationship."""
    df = pd.DataFrame({"g": categories.astype("category"), "x": values})
    df = df.dropna()
    if df.empty:
        return np.nan
    groups = [g["x"].values for _, g in df.groupby("g")]
    grand_mean = df["x"].mean()
    ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in map(pd.Series, groups))
    ss_total = ((df["x"]-grand_mean)**2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0

def _binary_codes_pair(real_cat: pd.Series, syn_cat: pd.Series, positive=None):
    """Apply consistent binary coding across real and synthetic data."""
    r = real_cat.astype('category')
    if r.cat.categories.size != 2:
        return None, None, 0
    levels = list(r.cat.categories)
    pos = positive if (positive in levels) else levels[-1]
    neg = levels[0] if levels[1] == pos else levels[1]
    mapping = {neg: 0, pos: 1}
    
    # Count dropped rows (unseen categories in syn map to NaN)
    r_codes = real_cat.map(mapping)
    s_codes = syn_cat.map(mapping)
    dropped_count = s_codes.isna().sum()
    
    return r_codes, s_codes, dropped_count

def _safe_corr_pair(x, y, kind="pearson"):
    """Compute correlation with NaN and constant column handling.
    
    Note: Requires ≥3 overlapping rows after NaN filtering for stable correlation.
    """
    s = pd.concat([x, y], axis=1)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 3 or s.nunique().min() < 2:
        return np.nan
    
    col0 = s.iloc[:,0]
    col1 = s.iloc[:,1]
    
    # Check for constant arrays
    if np.std(col0) == 0 or np.std(col1) == 0:
        return np.nan
    
    if kind == "pearson":
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConstantInputWarning)
            return stats.pearsonr(col0, col1)[0]
    elif kind == "spearman":
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConstantInputWarning)
            return stats.spearmanr(col0, col1)[0]
    else:
        return np.nan

# (REMOVED) duplicate _binary_auc — the robust version above is the single source of truth.

def _safe_eval_block(clf, X_test, y_test, pos_label=None):
    """Safely evaluate classifier handling unseen test classes and Pipelines."""
    # Get classes_ from estimator or final step in a Pipeline
    classes = getattr(clf, "classes_", None)
    if classes is None and hasattr(clf, "named_steps"):
        last = getattr(clf, "named_steps", {}).get("clf", None)
        classes = getattr(last, "classes_", None)
    if classes is None:
        # Fallback: infer from predictions if absolutely necessary
        classes = np.unique(pd.Series(y_test).dropna())
    classes = np.asarray(classes)

    mask = pd.Series(y_test).isin(classes).to_numpy()
    if not mask.any():
        return (
            {'accuracy': np.nan, 'f1_score': np.nan, 'log_loss': np.nan, 'roc_auc': np.nan, 'dropped_test_rows': int((~mask).sum()), 'kept_classes': classes.tolist()},
            np.array([], dtype=object),
            np.empty((0, len(classes)))
        )

    X_t, y_t = X_test[mask], np.asarray(y_test)[mask]
    proba = clf.predict_proba(X_t)
    pred = clf.predict(X_t)

    out = {
        'accuracy': _round_metric(accuracy_score(y_t, pred)),
        'f1_score': _round_metric(f1_score(y_t, pred, average='weighted', zero_division=0)),
        # Prefer estimator classes_ when available for correct column order
        'log_loss': _round_metric(log_loss(y_t, proba, labels=classes)) if classes is not None else _round_metric(log_loss(y_t, proba)),
        'dropped_test_rows': int((~mask).sum()),
        'kept_classes': classes.tolist()
    }
    
    # ROC-AUC: binary or multiclass (macro OvR)
    if np.unique(y_t).size == 2:
        out['roc_auc'] = _round_metric(_binary_auc(y_t, proba, classes, pos_label))
    else:
        # macro OvR for multiclass
        try:
            out['roc_auc'] = _round_metric(roc_auc_score(y_t, proba, multi_class='ovr', average='macro'))
        except Exception:
            out['roc_auc'] = None
    
    return out, pred, proba

def _safe_eval_regression(model, X_test, y_test) -> Tuple[Dict[str, float], np.ndarray]:
    """Safely evaluate regression model handling edge cases."""
    y_t = np.asarray(y_test)
    if y_t.size == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}, np.array([])
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_t, pred)))
    mae = float(mean_absolute_error(y_t, pred))
    r2 = float(r2_score(y_t, pred))
    return {'rmse': _round_metric(rmse), 'mae': _round_metric(mae), 'r2': _round_metric(r2)}, pred

def _fit_quantile_bins(s: pd.Series, q=10):
    """Fit quantile bin edges on a series with handling for degenerate/NaN series."""
    s_non = _clean_numeric(s).dropna()
    if s_non.empty:
        # Default symmetric edges for all-NaN columns
        return np.array([-1.0, 1.0], dtype=float)
    q = min(q, s_non.nunique())
    # Returns unique, sorted bin edges inclusive
    edges = np.unique(np.nanpercentile(s_non, np.linspace(0, 100, q+1)))
    if edges.size < 2:  # Constant feature
        v = float(s_non.iloc[0])
        return np.array([v-1.0, v+1.0], dtype=float)
    return edges

def _discretize_with_edges(s: pd.Series, edges):
    """Discretize series using pre-fitted bin edges."""
    # right=True to mirror qcut; include_lowest=True to capture min
    return pd.cut(s, bins=edges, labels=False, include_lowest=True, duplicates='drop')

def _discretize_cols_with_fitted_bins(real_df: pd.DataFrame, syn_df: pd.DataFrame, q=10, int_cardinality_as_categorical=20):
    """Discretize using bins fitted on real data, but keep truly discrete small-cardinality
    numeric features (incl. booleans / 0-1 ints) as categorical codes to avoid bin collapse.
    
    Note: For discrete categorical mapping, we use only real data support for stability and 
    auditability. Synthetic-only categories will map to NaN, which is filtered out during 
    MI computation. This ensures consistent evaluation across different synthetic datasets.
    """
    out_real = {}
    out_syn = {}

    for c in real_df.columns:
        r = real_df[c]
        s = syn_df[c]

        # Harmonize synthetic dtype if real is numeric
        if pd.api.types.is_numeric_dtype(r):
            s = _clean_numeric(s)

        # Treat booleans and small-cardinality integers as discrete categories
        is_bool_like = pd.api.types.is_bool_dtype(r)
        is_small_int = (pd.api.types.is_integer_dtype(r) and r.dropna().nunique() <= int_cardinality_as_categorical)
        # Also treat 0/1 integer columns as discrete (converted booleans)
        unique_vals = set(r.dropna().unique())
        is_binary_int = (pd.api.types.is_integer_dtype(r) and unique_vals.issubset({0, 1, np.int64(0), np.int64(1)}))

        if is_bool_like or is_small_int or is_binary_int:
            cats = pd.Index(sorted(pd.unique(r.dropna())))
            mapping = {v: i for i, v in enumerate(cats)}
            out_real[c] = r.map(mapping)
            out_syn[c] = s.map(mapping)
        else:
            edges = _fit_quantile_bins(r, q)
            out_real[c] = _discretize_with_edges(r, edges)
            out_syn[c] = _discretize_with_edges(s, edges)

    return pd.DataFrame(out_real), pd.DataFrame(out_syn)

def _maybe_bin_for_coverage(real_vals, syn_vals, name, bin_numerics, q=10, int_cardinality_as_categorical=20):
    """Apply binning to numerical values for coverage computation when enabled."""
    # Treat small-cardinality integers as categorical (like TVD/MI)
    if pd.api.types.is_bool_dtype(real_vals) or (
        pd.api.types.is_integer_dtype(real_vals) and pd.Series(real_vals).dropna().nunique() <= int_cardinality_as_categorical
    ):
        return real_vals, syn_vals, name  # categorical path
    
    if bin_numerics and pd.api.types.is_numeric_dtype(real_vals):
        # Clean and harmonize types to match real data
        real_vals = _clean_numeric(real_vals)
        syn_vals = _clean_numeric(syn_vals)
        edges = _fit_quantile_bins(real_vals, q=q)
        real_b = _discretize_with_edges(real_vals, edges)
        syn_b = _discretize_with_edges(syn_vals, edges)
        return real_b, syn_b, f"{name}__binned"
    return real_vals, syn_vals, name

def _compute_mi_and_nmi(df_disc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MI and NMI matrices on discretized data."""
    cols = df_disc.columns
    n = len(cols)
    mi = np.full((n, n), np.nan, dtype=float)
    nmi = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(mi, np.nan)
    np.fill_diagonal(nmi, np.nan)
    for i in range(n):
        for j in range(i+1, n):
            a, b = df_disc.iloc[:, i], df_disc.iloc[:, j]
            mask = ~(a.isna() | b.isna())
            if mask.any():
                mi[i,j] = mi[j,i] = mutual_info_score(a[mask], b[mask])
                nmi[i,j] = nmi[j,i] = normalized_mutual_info_score(a[mask], b[mask], average_method='arithmetic')
    return mi, nmi

# ============================================================================
# COVERAGE METRICS
# ============================================================================

def comprehensive_coverage_metrics(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    frequency_weighted: bool = True,
    bin_numerics: bool = True,
    int_cardinality_as_categorical: int = 20,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Coverage metrics with Jaccard index, KL divergence, and frequency weighting.
    
    Args:
        real: Real dataset
        syn: Synthetic dataset
        frequency_weighted: Whether to use frequency weighting
        bin_numerics: Whether to bin numerical values for coverage computation.
                     When True, continuous coverage uses real-fitted quantile bins.
        int_cardinality_as_categorical: Integer columns with <= this many unique values are treated as categorical
    
    Returns:
        Dictionary with coverage metrics
    """
    results = {}
    
    # Convert datetime columns to numeric for evaluation
    real = _coerce_datetime_to_numeric(real)
    syn = _coerce_datetime_to_numeric(syn)
    
    # Overall coverage metrics (per-column aggregation)
    # Note: Flattened overall coverage is removed as it mixes unrelated domains
    # Instead, we aggregate per-column coverage in the summary
    results['overall'] = {
        'note': 'Overall coverage computed as per-column aggregation (see summary)',
        'real_shape': real.shape,
        'syn_shape': syn.shape
    }
    
    # Per-column coverage
    column_coverage = {}
    
    # Pre-fit coverage spec for bootstrap stability
    cov_spec = _build_coverage_spec(real, bin_numerics, q=10, int_cardinality_as_categorical=int_cardinality_as_categorical)
    
    for col in real.columns:
        if col in syn.columns:
            real_vals = real[col].dropna()
            syn_vals = syn[col].dropna()
            
            if len(real_vals) > 0 and len(syn_vals) > 0:
                # Apply spec-based discretization for coverage computation
                spec_e = cov_spec[col]
                real_used = _apply_coverage_spec(real_vals, spec_e)
                syn_used = _apply_coverage_spec(syn_vals, spec_e)
                label = spec_e["label"]
                
                # Basic coverage on the processed values
                real_unique_col = set(pd.Series(real_used).dropna().unique())
                syn_unique_col = set(pd.Series(syn_used).dropna().unique())
                
                # Guard against zero denominator
                union_sz = len(real_unique_col.union(syn_unique_col))
                jaccard_col = float(len(real_unique_col.intersection(syn_unique_col)) / union_sz) if union_sz > 0 else 0.0
                
                # Frequency-weighted coverage on the processed values
                if frequency_weighted:
                    real_freq = pd.Series(real_used).value_counts(normalize=True)
                    syn_freq = pd.Series(syn_used).value_counts(normalize=True)
                    
                    # KL divergence (normalized on common support)
                    common_vals = real_freq.index.intersection(syn_freq.index)
                    if len(common_vals) > 0:
                        # capture original mass on the common support (pre-normalization)
                        real_mass = float(real_freq[common_vals].sum())
                        syn_mass = float(syn_freq[common_vals].sum())
                        # Get common values and convert to probability arrays
                        real_common = real_freq[common_vals]
                        syn_common = syn_freq[common_vals]
                        rc = _as_prob_array(real_common)   # float np.array
                        sc = _as_prob_array(syn_common)    # float np.array
                        kl_div = stats.entropy(rc, sc)
                        js_div = float(jensenshannon(rc, sc)**2)  # JS divergence (squared distance)
                        common_support_mass = (
                            float(real_freq[common_vals].sum()),
                            float(syn_freq[common_vals].sum()),
                        )
                    else:
                        kl_div = None
                        js_div = None
                        common_support_mass = (0.0, 0.0)
                    
                    # Proper weighted Jaccard
                    all_vals = real_freq.index.union(syn_freq.index)
                    real_aligned = real_freq.reindex(all_vals, fill_value=0)
                    syn_aligned = syn_freq.reindex(all_vals, fill_value=0)
                    num = np.minimum(real_aligned, syn_aligned).sum()
                    den = np.maximum(real_aligned, syn_aligned).sum()
                    freq_jaccard = float(num/den) if den > 0 else 0.0
                else:
                    kl_div = None
                    js_div = None
                    freq_jaccard = None
                
                # Set KL note only when frequency_weighted is True and KL was actually computed
                kl_note = None
                if frequency_weighted:
                    kl_note = "no common support" if kl_div is None else None
                
                # Add per-column bootstrap CIs
                ci_j = ci_wj = None
                if n_bootstrap and n_bootstrap > 0:
                    # Jaccard CI
                    def _jacc(d1: pd.DataFrame, d2: pd.DataFrame, e=spec_e):
                        a = _apply_coverage_spec(d1[col].dropna(), e)
                        b = _apply_coverage_spec(d2[col].dropna(), e)
                        A, B = set(pd.Series(a).dropna().unique()), set(pd.Series(b).dropna().unique())
                        den = len(A.union(B))
                        return float(len(A.intersection(B)) / den) if den > 0 else np.nan

                    ci_j = bootstrap_confidence_interval(
                        real[[col]], syn[[col]], _jacc, n_bootstrap, confidence_level,
                        _stable_seed("coverage_jaccard", col, base=42), sample_size=0
                    )

                    # Weighted Jaccard CI (only if frequency_weighted)
                    if frequency_weighted:
                        def _wj(d1: pd.DataFrame, d2: pd.DataFrame, e=spec_e):
                            a = _apply_coverage_spec(d1[col].dropna(), e)
                            b = _apply_coverage_spec(d2[col].dropna(), e)
                            rf = pd.Series(a).value_counts(normalize=True)
                            sf = pd.Series(b).value_counts(normalize=True)
                            allv = rf.index.union(sf.index)
                            ra = rf.reindex(allv, fill_value=0); sa = sf.reindex(allv, fill_value=0)
                            den = np.maximum(ra, sa).sum()
                            return float(np.minimum(ra, sa).sum()/den) if den > 0 else np.nan

                        ci_wj = bootstrap_confidence_interval(
                            real[[col]], syn[[col]], _wj, n_bootstrap, confidence_level,
                            _stable_seed("coverage_weighted_jaccard", col, base=42), sample_size=0
                        )

                column_coverage[col] = {
                    'jaccard_index': _round_metric(jaccard_col),
                    'kl_divergence': _round_metric(kl_div) if kl_div is not None else kl_div,
                    'js_divergence': _round_metric(js_div) if js_div is not None else js_div,  # Squared JS distance
                    'frequency_weighted_jaccard': _round_metric(freq_jaccard) if freq_jaccard is not None else freq_jaccard,
                    'real_unique_count': len(real_unique_col),
                    'syn_unique_count': len(syn_unique_col),
                    'coverage_ratio': _round_metric(len(real_unique_col.intersection(syn_unique_col)) / len(real_unique_col)),
                    'syn_coverage_ratio': _round_metric(len(real_unique_col.intersection(syn_unique_col)) / len(syn_unique_col) if len(syn_unique_col) > 0 else 0.0),
                    'label': label,  # Store label to indicate if column was binned
                    'common_support_mass': common_support_mass if frequency_weighted else None,
                    'kl_note': kl_note,
                    'confidence_intervals': {
                        'jaccard_index': _round_ci(ci_j) if ci_j else None,
                        'frequency_weighted_jaccard': _round_ci(ci_wj) if ci_wj else None
                    }
                }
    
    results['column_coverage'] = column_coverage
    
    # Summary statistics
    results['summary'] = _compute_coverage_summary(results)
    
    # Add parameters for auditability
    results['params'] = {
        'frequency_weighted': frequency_weighted,
        'bin_numerics': bin_numerics,
        'int_cardinality_as_categorical': int_cardinality_as_categorical,
        'all_columns': list(real.columns),
        'version': __version__,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }
    
    return results

# ============================================================================
# DOWNSTREAM PERFORMANCE METRICS
# ============================================================================

def comprehensive_downstream_metrics(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target_col: str,
    n_folds: int = 5,
    random_seed: int = 42,
    include_details: bool = False,
    n_jobs: int = -1,
    pos_label: Optional[Any] = None,
    force_task_type: Optional[str] = None,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Downstream performance with bidirectional evaluation and cross-validation.
    
    Args:
        real: Real dataset
        syn: Synthetic dataset
        target_col: Target column name
        n_folds: Number of CV folds
        random_seed: Random seed for reproducibility
        include_details: Whether to include detailed predictions/probabilities
        n_jobs: Number of parallel jobs for cross-validation and RandomForest
        pos_label: Positive class label for binary classification (if None, auto-detects)
        force_task_type: Force task type ('classification' or 'regression'), overrides auto-detection
    
    Returns:
        Dictionary with downstream metrics
    """
    results = {}
    
    if target_col not in real.columns or target_col not in syn.columns:
        return {'error': f'Target column {target_col} not found in datasets'}
    
    # Prepare data
    X_real = real.drop(columns=[target_col])
    y_real = real[target_col]
    X_syn = syn.drop(columns=[target_col])
    y_syn = syn[target_col]
    
    # Ensure same columns
    common_cols = X_real.columns.intersection(X_syn.columns)
    X_real = X_real[common_cols]
    X_syn = X_syn[common_cols]
    
    # Convert datetime columns to numeric for evaluation
    X_real = _coerce_datetime_to_numeric(X_real)
    X_syn = _coerce_datetime_to_numeric(X_syn)
    
    # Convert small-cardinality integer columns to categorical for proper one-hot encoding
    int_cardinality_as_categorical = 20  # Match other functions
    for col in X_real.columns:
        if (pd.api.types.is_integer_dtype(X_real[col]) and 
            X_real[col].dropna().nunique() <= int_cardinality_as_categorical):
            X_real[col] = X_real[col].astype('category')
            X_syn[col] = X_syn[col].astype('category')
    
    # Encode categorical variables
    X_real_encoded = pd.get_dummies(X_real)
    X_syn_encoded = pd.get_dummies(X_syn)
    
    # Align columns
    all_cols = X_real_encoded.columns.union(X_syn_encoded.columns)
    X_real_encoded = X_real_encoded.reindex(columns=all_cols, fill_value=0)
    X_syn_encoded = X_syn_encoded.reindex(columns=all_cols, fill_value=0)
    
    # Determine if this is a regression problem
    is_regression = (force_task_type == 'regression') if force_task_type else _is_regression_target(y_real)
    results['task_type'] = 'regression' if is_regression else 'classification'
    
    # 1. Train on synthetic, test on real
    syn_to_real_results = _evaluate_direction(
        X_syn_encoded, y_syn, X_real_encoded, y_real, 
        f"syn_to_real", random_seed, include_details, n_jobs, pos_label,
        is_regression=is_regression
    )
    results['syn_to_real'] = syn_to_real_results
    
    # 2. Train on real, test on synthetic
    real_to_syn_results = _evaluate_direction(
        X_real_encoded, y_real, X_syn_encoded, y_syn,
        f"real_to_syn", random_seed, include_details, n_jobs, pos_label,
        is_regression=is_regression
    )
    results['real_to_syn'] = real_to_syn_results
    
    # 3. Cross-validation on real data (baseline)
    real_cv_results = _cross_validate_models(
        X_real_encoded, y_real, "real_cv", n_folds, random_seed, n_jobs,
        is_regression=is_regression, confidence_level=confidence_level
    )
    results['real_cv'] = real_cv_results
    
    # 4. Cross-validation on synthetic data
    syn_cv_results = _cross_validate_models(
        X_syn_encoded, y_syn, "syn_cv", n_folds, random_seed, n_jobs,
        is_regression=is_regression, confidence_level=confidence_level
    )
    results['syn_cv'] = syn_cv_results
    
    # 5. Calibration analysis (only for classification)
    results['calibration'] = {} if is_regression else _analyze_calibration(
        X_real_encoded, y_real, X_syn_encoded, y_syn, random_seed, pos_label
    )
    
    # 6. Summary statistics
    results['summary'] = _compute_downstream_summary(results)
    
    # Add parameters for auditability
    results['params'] = {
        'n_folds': n_folds,
        'random_seed': random_seed,
        'include_details': include_details,
        'n_jobs': n_jobs,
        'pos_label': pos_label,
        'target_col': target_col,
        'feature_cols': list(real.columns.drop(target_col)),
        'version': __version__,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }
    
    return results

def _evaluate_direction(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    direction: str, random_seed: int,
    include_details: bool = False, n_jobs: int = -1,
    pos_label: Optional[Any] = None,
    is_regression: bool = False
) -> Dict[str, Any]:
    """Evaluate models trained on one dataset and tested on another."""
    results = {'direction': direction}
    
    if is_regression:
        # Linear Regression
        lin = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", LinearRegression())
        ])
        _fit_safely(lin, X_train, y_train)
        lin_out, lin_pred = _safe_eval_regression(lin, X_test, y_test)
        if include_details:
            lin_out['predictions'] = np.asarray(lin_pred).tolist()
        results['linear_regression'] = lin_out

        # Random Forest Regressor
        rfr = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(random_state=random_seed, n_estimators=100, n_jobs=n_jobs))
        ])
        _fit_safely(rfr, X_train, y_train)
        rfr_out, rfr_pred = _safe_eval_regression(rfr, X_test, y_test)
        if include_details:
            rfr_out['predictions'] = np.asarray(rfr_pred).tolist()
        results['random_forest'] = rfr_out
        return results
    
    # Logistic Regression with StandardScaler for stability
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False for sparse one-hots
        ("clf", LogisticRegression(random_state=random_seed, max_iter=1000, solver="lbfgs"))
    ])
    _fit_safely(lr, X_train, y_train)
    lr_out, lr_pred, lr_proba = _safe_eval_block(lr, X_test, y_test, pos_label)
    lr_block = {**lr_out}
    if include_details:
        lr_block['predictions'] = lr_pred.tolist()
        lr_block['probabilities'] = lr_proba.tolist()
    results['logistic_regression'] = lr_block
    
    # Random Forest
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(random_state=random_seed, n_estimators=100, n_jobs=n_jobs))
    ])
    _fit_safely(rf, X_train, y_train)
    rf_out, rf_pred, rf_proba = _safe_eval_block(rf, X_test, y_test, pos_label)
    rf_block = {**rf_out}
    if include_details:
        rf_block['predictions'] = rf_pred.tolist()
        rf_block['probabilities'] = rf_proba.tolist()
    results['random_forest'] = rf_block
    
    return results

def _cross_validate_models(
    X: pd.DataFrame, y: pd.Series, 
    dataset_name: str, n_folds: int, random_seed: int, n_jobs: int = -1,
    is_regression: bool = False, confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Perform cross-validation on a single dataset."""
    results = {}
    
    if is_regression:
        n = len(y)
        n_splits = min(max(2, n_folds), n)  # ensure viable
        if n_splits < 2:
            return {
                'linear_regression': {'r2_mean': np.nan, 'r2_std': np.nan, 'mae_mean': np.nan, 'mae_std': np.nan, 'rmse_mean': np.nan, 'rmse_std': np.nan, 'cv_scores': []},
                'random_forest': {'r2_mean': np.nan, 'r2_std': np.nan, 'mae_mean': np.nan, 'mae_std': np.nan, 'rmse_mean': np.nan, 'rmse_std': np.nan, 'cv_scores': []},
                'meta': {'n_splits': None, 'note': 'CV skipped: not enough samples'}
            }
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        # Linear Regression
        lin = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False)), ("reg", LinearRegression())])
        r2_lin = cross_val_score(lin, X, y, cv=cv, scoring='r2', n_jobs=n_jobs)
        mae_lin = -cross_val_score(lin, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=n_jobs)
        mse_lin = -cross_val_score(lin, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=n_jobs)
        rmse_lin = np.sqrt(mse_lin)

        # Random Forest Regressor
        rfr = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(random_state=random_seed, n_estimators=100, n_jobs=n_jobs))
        ])
        r2_rfr = cross_val_score(rfr, X, y, cv=cv, scoring='r2', n_jobs=n_jobs)
        mae_rfr = -cross_val_score(rfr, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=n_jobs)
        mse_rfr = -cross_val_score(rfr, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=n_jobs)
        rmse_rfr = np.sqrt(mse_rfr)

        # Add percentile CIs for regression CV scores
        def _pct_ci(vals, cl=0.95):
            return _round_ci(_percentile_ci_from_list(vals, cl))

        return {
            'linear_regression': {
                'r2_mean': _round_metric(r2_lin.mean()), 'r2_std': _round_metric(r2_lin.std()),
                'mae_mean': _round_metric(mae_lin.mean()), 'mae_std': _round_metric(mae_lin.std()),
                'rmse_mean': _round_metric(rmse_lin.mean()), 'rmse_std': _round_metric(rmse_lin.std()),
                'cv_scores': [_round_metric(v) for v in r2_lin.tolist()],
                'r2_ci': _pct_ci(r2_lin, confidence_level),
                'mae_ci': _pct_ci(mae_lin, confidence_level),
                'rmse_ci': _pct_ci(rmse_lin, confidence_level)
            },
            'random_forest': {
                'r2_mean': _round_metric(r2_rfr.mean()), 'r2_std': _round_metric(r2_rfr.std()),
                'mae_mean': _round_metric(mae_rfr.mean()), 'mae_std': _round_metric(mae_rfr.std()),
                'rmse_mean': _round_metric(rmse_rfr.mean()), 'rmse_std': _round_metric(rmse_rfr.std()),
                'cv_scores': [_round_metric(v) for v in r2_rfr.tolist()],
                'r2_ci': _pct_ci(r2_rfr, confidence_level),
                'mae_ci': _pct_ci(mae_rfr, confidence_level),
                'rmse_ci': _pct_ci(rmse_rfr, confidence_level)
            },
            'meta': {'n_splits': n_splits}
        }
    
    # Guard against tiny classes - check if any class has < 2 samples
    vc = pd.Series(y).value_counts()
    min_class = int(vc.min()) if not vc.empty else 0
    if min_class < 2:
        return {
            'logistic_regression': {'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'f1_mean': np.nan, 'f1_std': np.nan, 'cv_scores': []},
            'random_forest': {'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'f1_mean': np.nan, 'f1_std': np.nan, 'cv_scores': []},
            'meta': {'n_splits': None, 'note': 'CV skipped: at least one class has < 2 samples'}
        }
    
    # Cap folds by both min_class size and total dataset size (defensive for tiny datasets)
    n = len(y)
    n_splits = min(n_folds, min_class, n)  # Cap by dataset size too
    if n_splits < 2:
        return {
            'logistic_regression': {'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'f1_mean': np.nan, 'f1_std': np.nan, 'cv_scores': [], 'accuracy_ci': None, 'f1_ci': None},
            'random_forest': {'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'f1_mean': np.nan, 'f1_std': np.nan, 'cv_scores': [], 'accuracy_ci': None, 'f1_ci': None},
            'meta': {'n_splits': None, 'note': 'CV skipped: not enough samples or classes'}
        }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Logistic Regression CV with StandardScaler
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(random_state=random_seed, max_iter=1000, solver="lbfgs"))
    ])
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    lr_f1_scores = cross_val_score(lr, X, y, cv=cv, scoring='f1_weighted', n_jobs=n_jobs)
    
    # Add percentile CIs for CV scores
    def _pct_ci(vals, cl=0.95):
        return _round_ci(_percentile_ci_from_list(vals, cl))

    results['logistic_regression'] = {
        'accuracy_mean': _round_metric(lr_scores.mean()),
        'accuracy_std': _round_metric(lr_scores.std()),
        'f1_mean': _round_metric(lr_f1_scores.mean()),
        'f1_std': _round_metric(lr_f1_scores.std()),
        'cv_scores': [_round_metric(v) for v in lr_scores.tolist()],
        'accuracy_ci': _pct_ci(lr_scores, confidence_level),
        'f1_ci': _pct_ci(lr_f1_scores, confidence_level)
    }
    
    # Random Forest CV
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(random_state=random_seed, n_estimators=100, n_jobs=n_jobs))
    ])
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    rf_f1_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_weighted', n_jobs=n_jobs)
    
    results['random_forest'] = {
        'accuracy_mean': _round_metric(rf_scores.mean()),
        'accuracy_std': _round_metric(rf_scores.std()),
        'f1_mean': _round_metric(rf_f1_scores.mean()),
        'f1_std': _round_metric(rf_f1_scores.std()),
        'cv_scores': [_round_metric(v) for v in rf_scores.tolist()],
        'accuracy_ci': _pct_ci(rf_scores, confidence_level),
        'f1_ci': _pct_ci(rf_f1_scores, confidence_level)
    }
    
    # Add meta information about CV configuration
    results['meta'] = {"n_splits": n_splits}
    
    return results

def _analyze_calibration(
    X_real: pd.DataFrame, y_real: pd.Series,
    X_syn: pd.DataFrame, y_syn: pd.Series,
    random_seed: int,
    pos_label: Optional[Any] = None
) -> Dict[str, Any]:
    """Analyze calibration of models trained on synthetic data."""
    from sklearn.metrics import brier_score_loss
    
    results = {}
    
    # Train on synthetic, test on real
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(random_state=random_seed, max_iter=1000, solver="lbfgs"))
    ])
    _fit_safely(lr, X_syn, y_syn)
    classes = lr.named_steps["clf"].classes_
    
    # choose positive label: use provided or 1 if present else the lexicographically larger class
    classes = np.asarray(classes)
    if pos_label is None or pos_label not in classes:
        pos_label = 1 if 1 in classes else classes[-1]
    pos_idx = np.where(classes == pos_label)[0][0]
    proba_pos = lr.predict_proba(X_real)[:, pos_idx]
    
    # Log the pos_label used for debugging
    results['pos_label_used'] = pos_label

    if np.unique(y_real).size == 2:
        # binarize y to {0,1} w.r.t pos_label
        y_bin = (pd.Series(y_real).to_numpy() == pos_label).astype(int)
        
        # Adaptive bin count for stability
        n_bins = min(10, max(2, int(np.sqrt(len(y_bin)))))
        frac_pos, mean_pred = calibration_curve(y_bin, proba_pos, n_bins=n_bins, strategy="quantile")
        
        # Filter out empty bins
        valid_mask = ~np.isnan(frac_pos) & ~np.isnan(mean_pred)
        if valid_mask.any():
            results['calibration_curve'] = {
                'fraction_of_positives': [_round_metric(float(x)) for x in frac_pos[valid_mask]],
                'mean_predicted_value': [_round_metric(float(x)) for x in mean_pred[valid_mask]]
            }
        else:
            results['calibration_curve'] = {
                'fraction_of_positives': [],
                'mean_predicted_value': []
            }
        results['brier_score'] = _round_metric(brier_score_loss(y_bin, proba_pos))
    
    return results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def bootstrap_confidence_interval(
    data1: Union[pd.DataFrame, pd.Series, np.ndarray],
    data2: Union[pd.DataFrame, pd.Series, np.ndarray],
    metric_func: callable,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    max_sample_size: int = 50000,
    sample_size: Optional[int] = None
) -> Tuple[float, float]:
    """Two-sample bootstrap CI for an arbitrary metric.

    Samples WITH replacement independently from data1 and data2.
    Returns (lo, hi) central two-sided CI.
    
    Args:
        sample_size: If None, uses min(n1, n2, max_sample_size) (balanced).
                    If 0 or "auto-left-right", draws n1 and n2 separately (per-sample).
                    If positive int, uses that size for both samples.
    """
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_seed)
    
    # Determine sample sizes
    if sample_size is None:
        # Default balanced behavior
        n_samples1 = n_samples2 = min(n1, n2, max_sample_size)
    elif sample_size == 0 or sample_size == "auto-left-right":
        # Per-sample bootstrapping at original sizes
        n_samples1 = min(n1, max_sample_size)
        n_samples2 = min(n2, max_sample_size)
    else:
        # Fixed sample size
        n_samples1 = n_samples2 = min(sample_size, n1, n2, max_sample_size)

    scores = []
    for _ in range(n_bootstrap):
        if isinstance(data1, pd.DataFrame):
            idx1 = rng.integers(0, n1, n_samples1)
            idx2 = rng.integers(0, n2, n_samples2)
            s1 = data1.iloc[idx1]
            s2 = data2.iloc[idx2]
        else:
            a1 = np.asarray(data1)
            a2 = np.asarray(data2)
            idx1 = rng.integers(0, n1, n_samples1)
            idx2 = rng.integers(0, n2, n_samples2)
            s1 = a1[idx1]
            s2 = a2[idx2]
        try:
            val = metric_func(s1, s2)
            if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                scores.append(float(val))
        except Exception:
            # swallow metric errors on pathologies; continue
            pass

    if not scores:
        return (np.nan, np.nan)

    alpha = (1 - confidence_level) / 2 * 100
    lo = np.percentile(scores, alpha)
    hi = np.percentile(scores, 100 - alpha)
    return (float(lo), float(hi))

# Wrapper functions for bootstrap
def _tvd_wrapper(data1, data2):
    """Wrapper for 1D TVD computation that handles Series/ndarray/DataFrame."""
    # Convert to DataFrame if needed
    if isinstance(data1, pd.Series): 
        data1 = data1.rename("x").to_frame()
    if isinstance(data2, pd.Series): 
        data2 = data2.rename("x").to_frame()
    if isinstance(data1, np.ndarray): 
        data1 = pd.DataFrame({"x": data1})
    if isinstance(data2, np.ndarray): 
        data2 = pd.DataFrame({"x": data2})
    
    if isinstance(data1, pd.DataFrame) and data1.shape[1] == 1:
        col = data1.columns[0]
        return _compute_tvd_1d(data1, data2, col)
    return np.nan


# Summary computation functions
def _compute_tvd_summary(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary statistics for TVD results."""
    summary = {}
    
    for tvd_type in ['tvd1', 'tvd2', 'tvd3']:
        if tvd_type in results:
            values = [item['value'] for item in results[tvd_type].values()]
            # Filter out NaNs
            values = [v for v in values if np.isfinite(v)]
            if values:
                summary[f'{tvd_type}_mean'] = _round_metric(np.mean(values))
                summary[f'{tvd_type}_std'] = _round_metric(np.std(values))
                summary[f'{tvd_type}_max'] = _round_metric(np.max(values))
                summary[f'{tvd_type}_min'] = _round_metric(np.min(values))
    
    # Add EMD-only mean if present among tvd1 entries
    # Also count EMD vs TVD usage for diagnostics
    if 'tvd1' in results:
        emd_vals = []
        emd_count = 0
        tvd_count = 0
        tvd_fallback_count = 0
        for v in results['tvd1'].values():
            if isinstance(v, dict):
                method = v.get('method', '')
                if method.startswith('EMD'):
                    emd_count += 1
                    val = v.get('value')
                    if val is not None and np.isfinite(val):
                        emd_vals.append(val)
                elif method == 'TVD':
                    tvd_count += 1
                elif 'fallback' in method.lower():
                    tvd_fallback_count += 1
        
        if emd_vals:
            summary['emd_mean'] = _round_metric(np.mean(emd_vals))
        
        # Add diagnostic counts
        summary['_tvd1_emd_count'] = emd_count
        summary['_tvd1_tvd_count'] = tvd_count
        summary['_tvd1_tvd_fallback_count'] = tvd_fallback_count
    
    return summary

def _compute_mi_summary(mi_pairs: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary statistics for MI results."""
    if not mi_pairs:
        return {}
    
    delta_mis = []
    preservation_ratios = []
    preservation_bounded = []
    
    for v in mi_pairs.values():
        if np.isfinite(v.get('delta_mi', np.nan)):
            delta_mis.append(v['delta_mi'])
        # Handle None preservation_ratio (when real MI is too small)
        pres_ratio = v.get('preservation_ratio')
        if pres_ratio is not None and np.isfinite(pres_ratio):
            preservation_ratios.append(pres_ratio)
        if np.isfinite(v.get('preservation_bounded', np.nan)):
            preservation_bounded.append(v['preservation_bounded'])
    
    summary = {}
    if delta_mis:
        summary['delta_mi_mean'] = _round_metric(np.mean(delta_mis))
        summary['delta_mi_std'] = _round_metric(np.std(delta_mis))
    if preservation_ratios:
        summary['preservation_ratio_mean'] = _round_metric(np.mean(preservation_ratios))
        summary['preservation_ratio_std'] = _round_metric(np.std(preservation_ratios))
    if preservation_bounded:
        summary['preservation_bounded_mean'] = _round_metric(np.mean(preservation_bounded))
        summary['preservation_bounded_std'] = _round_metric(np.std(preservation_bounded))
    
    return summary

def _compute_correlation_summary(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary statistics for correlation results."""
    summary = {}
    
    for corr_type in ['numerical_correlations', 'categorical_correlations', 'mixed_correlations']:
        if corr_type in results:
            errors = []
            for pair_data in results[corr_type].values():
                for metric_data in pair_data.values():
                    if isinstance(metric_data, dict) and 'error' in metric_data:
                        errors.append(metric_data['error'])
            
            # Filter out NaNs
            errors = [v for v in errors if np.isfinite(v)]
            if errors:
                summary[f'{corr_type}_error_mean'] = _round_metric(np.mean(errors))
                summary[f'{corr_type}_error_std'] = _round_metric(np.std(errors))
    
    # Add matrix comparison metrics to summary for easier access
    if 'matrix_comparison' in results:
        mc = results['matrix_comparison']
        summary['pearson_spearman'] = mc.get('pearson_matrix_spearman_corr')
        summary['spearman_spearman'] = mc.get('spearman_matrix_spearman_corr')
    
    return summary

def _compute_coverage_summary(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary statistics for coverage results."""
    summary = {}
    
    if 'column_coverage' in results:
        jaccard_indices = [item['jaccard_index'] for item in results['column_coverage'].values()]
        coverage_ratios = [item['coverage_ratio'] for item in results['column_coverage'].values()]
        
        # Include KL divergence, weighted Jaccard, and common support mass
        # FIX: robust KL filtering
        kls = []
        for v in results['column_coverage'].values():
            kl = v.get('kl_divergence', None)
            if isinstance(kl, (int, float)) and np.isfinite(kl):
                kls.append(kl)
        
        wj = [float(v['frequency_weighted_jaccard']) for v in results['column_coverage'].values() 
              if isinstance(v.get('frequency_weighted_jaccard'), (int, float)) and np.isfinite(v['frequency_weighted_jaccard'])]
        common_masses = [v['common_support_mass'] for v in results['column_coverage'].values() 
                         if v.get('common_support_mass') is not None]
        
        # Filter out NaNs
        jaccard_indices = [v for v in jaccard_indices if np.isfinite(v)]
        coverage_ratios = [v for v in coverage_ratios if np.isfinite(v)]
        
        if jaccard_indices:
            summary['jaccard_mean'] = _round_metric(np.mean(jaccard_indices))
            summary['jaccard_std'] = _round_metric(np.std(jaccard_indices))
        if coverage_ratios:
            summary['coverage_ratio_mean'] = _round_metric(np.mean(coverage_ratios))
            summary['coverage_ratio_std'] = _round_metric(np.std(coverage_ratios))
        if kls:
            summary['kl_mean'] = _round_metric(np.mean(kls))
            summary['kl_std'] = _round_metric(np.std(kls))
        if wj:
            summary['weighted_jaccard_mean'] = _round_metric(np.mean(wj))
            summary['weighted_jaccard_std'] = _round_metric(np.std(wj))
        if common_masses:
            # Extract real and syn masses separately
            real_masses = [m[0] for m in common_masses if len(m) == 2]
            syn_masses = [m[1] for m in common_masses if len(m) == 2]
            if real_masses:
                summary['mean_common_support_mass_real'] = _round_metric(np.mean(real_masses))
                summary['mean_common_support_mass_syn'] = _round_metric(np.mean(syn_masses))
    
    return summary

def _compute_downstream_summary(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary statistics for downstream results."""
    summary = {}
    t = results.get('task_type')
    
    # Compare syn_to_real vs real_to_syn
    if 'syn_to_real' in results and 'real_to_syn' in results:
        if t == 'classification':
            for model in ['logistic_regression', 'random_forest']:
                if model in results['syn_to_real'] and model in results['real_to_syn']:
                    a = results['syn_to_real'][model].get('accuracy')
                    b = results['real_to_syn'][model].get('accuracy')
                    if a is not None and b is not None:
                        summary[f'{model}_symmetry_gap'] = _round_metric(abs(a - b))
                        summary[f'{model}_syn_to_real_acc'] = _round_metric(a)
                        summary[f'{model}_real_to_syn_acc'] = _round_metric(b)
        elif t == 'regression':
            for model in ['linear_regression', 'random_forest']:
                if model in results['syn_to_real'] and model in results['real_to_syn']:
                    a = results['syn_to_real'][model].get('rmse')
                    b = results['real_to_syn'][model].get('rmse')
                    if a is not None and b is not None:
                        summary[f'{model}_symmetry_gap_rmse'] = _round_metric(abs(a - b))
                        summary[f'{model}_syn_to_real_rmse'] = _round_metric(a)
                        summary[f'{model}_real_to_syn_rmse'] = _round_metric(b)
    
    return summary

# ============================================================================
# SANITY CHECK FUNCTIONS
# ============================================================================

def run_sanity_checks(real_data: pd.DataFrame, syn_data: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """Run quick sanity checks to validate metric computation."""
    results = {}
    
    # 1. Perfect copy test (syn = real)
    if real_data.equals(syn_data):
        results['perfect_copy'] = {
            'tvd1_mean': 0.0,
            'delta_mi_mean': 0.0,
            'correlation_error_mean': 0.0,
            'jaccard_index': 1.0,
            'syn_to_real_expected_optimal': True if target_col else None
        }
        results['notes'] = ['Perfect copy detected - all metrics should be optimal']
    
    # 2. Shuffled rows test (same distribution)
    elif real_data.shape == syn_data.shape and real_data.columns.equals(syn_data.columns):
        # Check if it's just row shuffling using stable hashing
        if np.array_equal(_rows_signature(real_data), _rows_signature(syn_data)):
            results['shuffled_rows'] = {
                'tvd1_mean': 0.0,
                'delta_mi_mean': 0.0,
                'correlation_error_mean': 0.0,
                'jaccard_index': 1.0,
                'syn_to_real_expected_optimal': True if target_col else None
            }
            results['notes'] = ['Shuffled rows detected - all metrics should be optimal']
    
    # 3. Single feature noise test
    else:
        # Check for single feature differences
        diff_cols = []
        for col in real_data.columns:
            if col in syn_data.columns:
                if not real_data[col].equals(syn_data[col]):
                    diff_cols.append(col)
        
        if len(diff_cols) == 1:
            results['single_feature_noise'] = {
                'modified_column': diff_cols[0],
                'notes': ['Single feature modified - expect increased EMD for this feature, decreased MI with neighbors']
            }
        elif len(diff_cols) > 1:
            results['multiple_differences'] = {
                'modified_columns': diff_cols,
                'notes': [f'{len(diff_cols)} columns differ between real and synthetic data']
            }
    
    return results

def _dtype_family(s):
    """Get semantic dtype family for comparison."""
    from pandas.api import types as ptypes
    if ptypes.is_bool_dtype(s): 
        return "bool"
    if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s): 
        return "numeric"
    if ptypes.is_datetime64_any_dtype(s) or ptypes.is_timedelta64_dtype(s): 
        return "datetime"
    return "categorical"

def validate_metric_consistency(real_data: pd.DataFrame, syn_data: pd.DataFrame) -> Dict[str, Any]:
    """Validate that metrics are consistent and interpretable."""
    results = {}
    notes = []
    
    # Check for constant columns
    constant_cols = []
    for col in real_data.columns:
        if col in syn_data.columns:
            if real_data[col].nunique() <= 1 or syn_data[col].nunique() <= 1:
                constant_cols.append(col)
    
    if constant_cols:
        results['constant_columns'] = constant_cols
        notes.append(f'Constant columns detected: {constant_cols} - correlations will be NaN')
    
    # Check for missing columns
    missing_in_syn = set(real_data.columns) - set(syn_data.columns)
    missing_in_real = set(syn_data.columns) - set(real_data.columns)
    
    if missing_in_syn or missing_in_real:
        results['missing_columns'] = {
            'missing_in_syn': list(missing_in_syn),
            'missing_in_real': list(missing_in_real)
        }
        notes.append('Column mismatch detected - some metrics may be incomplete')
    
    # Check data type families (semantic, not exact dtypes)
    fam_mismatches = []
    for col in real_data.columns:
        if col in syn_data.columns:
            if _dtype_family(real_data[col]) != _dtype_family(syn_data[col]):
                fam_mismatches.append(col)
    
    if fam_mismatches:
        results['type_mismatches'] = fam_mismatches
        notes.append(f'Data type family mismatches: {fam_mismatches} - may affect metric computation')
    
    if notes:
        results['notes'] = notes
    
    return results
