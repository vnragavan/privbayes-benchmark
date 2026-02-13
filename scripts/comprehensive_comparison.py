#!/usr/bin/env python3
"""
Comprehensive Privacy-Utility Comparison for PrivBayes Implementations

Computes metrics for Enhanced, SynthCity, and DPMM PrivBayes implementations:
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

# Minimum viable privacy audit probes (optional)
from pbbench.privacy_audit import (
    _select_qi_cols as _audit_select_qi_cols,
    conditional_disclosure_l1,
    membership_inference_distance_attack,
    nearest_neighbor_memorization,
    rare_combination_leakage,
    unique_pattern_leakage,
)

# ---------------- DP-safe public domains (for strict-dp mode) ----------------

_ADULT_PUBLIC_CATEGORIES: dict[str, list[str]] = {
    # UCI Adult / Census Income (public schema)
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital-status": [
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ],
    # label column
    "income": ["<=50K", ">50K"],
}


def _infer_dataset_tag(df: pd.DataFrame) -> str:
    cols = set(map(str, df.columns))
    if "income" in cols and "workclass" in cols and "education" in cols:
        return "adult"
    if "target" in cols and "mean radius" in cols:
        return "breast_cancer"
    return "unknown"


def _public_label_categories(target_col: Optional[str]) -> Optional[list[str]]:
    if target_col is None:
        return None
    if target_col == "income":
        return ["<=50K", ">50K"]
    if target_col == "target":
        return ["0", "1"]
    return None


def _load_public_schema(schema_path: Optional[str]) -> dict[str, Any]:
    if not schema_path:
        return {}
    p = schema_path
    try:
        with open(p, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("schema must be a JSON object")
        return obj
    except Exception as e:
        raise RuntimeError(f"Failed to load schema JSON from '{p}': {e}") from e


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
    
    Uses the model's DP-protected vocabulary for alignment. Post-processing
    on evaluation data, not training, so it preserves DP guarantees.
    
    Labels: never hashed, no UNK; clamp to public classes.
    Non-label categoricals: hash if needed, then clamp to learned vocabulary.
    
    Returns REAL unchanged if model metadata isn't accessible.
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
    implementation_base: str
    regime: str  # "default" | "strict"
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

    # Privacy audit probes (minimum viable set; optional)
    audit_metrics: Optional[Dict[str, Any]] = None
    
    # Synthetic data path
    synthetic_data_path: Optional[str] = None


def _label_impl(base: str, regime: str) -> str:
    if regime == "strict":
        return f"{base} (strict-DP)"
    return f"{base} (default)"

def run_synthcity_privbayes(
    real: pd.DataFrame,
    epsilon: float,
    seed: int,
    n_samples: Optional[int] = None,
    strict_dp: bool = False,
    regime: str = "default",
) -> ImplementationResult:
    """Run SynthCity PrivBayes implementation and measure performance.
    
    Tracks fit time, sample time, memory usage. Returns result object with
    synthetic data and original real data for metric computation.
    """
    base = "SynthCity"
    result = ImplementationResult(
        name=_label_impl(base, regime),
        implementation_base=base,
        regime=regime,
        epsilon=epsilon,
        seed=seed,
        success=False,
    )
    
    try:
        if strict_dp:
            raise RuntimeError("strict-dp enabled: SynthCity PrivBayes is not DP-compliant; skipping.")
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
        n = n_samples if n_samples is not None else len(real)
        syn = model.sample(n)
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


def run_dpmm_privbayes(
    real: pd.DataFrame,
    epsilon: float,
    seed: int,
    n_samples: Optional[int] = None,
    strict_dp: bool = False,
    dataset_tag: str = "unknown",
    target_col: Optional[str] = None,
    regime: str = "default",
    preprocess: str = "none",
    public_bounds: Optional[dict] = None,
    public_categories: Optional[dict] = None,
) -> ImplementationResult:
    """Run DPMM PrivBayes implementation and measure performance.
    
    Discretizes data to integers before fitting (DPMM requirement). Tracks
    timing and memory. Returns result with synthetic and real data.
    """
    base = "DPMM"
    result = ImplementationResult(
        name=_label_impl(base, regime),
        implementation_base=base,
        regime=regime,
        epsilon=epsilon,
        seed=seed,
        success=False,
    )
    
    try:
        from pbbench.variants.pb_datasynthesizer import DPMMPrivBayesAdapter
        
        tracemalloc.start()
        start_time = time.time()

        pb = dict(public_bounds or {})
        pc = dict(public_categories or {})
        if strict_dp and dataset_tag == "adult" and not pc:
            pc = dict(_ADULT_PUBLIC_CATEGORIES)

        # Ensure label domain is treated as categorical for DPMM too.
        # Without this, binary labels like breast-cancer "target" get treated as numeric,
        # decoded as continuous midpoints, and downstream ML becomes undefined.
        detected_target_col = target_col
        if detected_target_col is None:
            for col in ["target", "income", "label", "class", "outcome"]:
                if col in real.columns:
                    detected_target_col = col
                    break
        if detected_target_col and detected_target_col in real.columns:
            pub = _public_label_categories(detected_target_col)
            if pub is not None:
                pc.setdefault(detected_target_col, pub)
            elif not strict_dp:
                unique_labels = sorted(real[detected_target_col].astype(str).dropna().unique().tolist())
                pc.setdefault(detected_target_col, unique_labels)

        # dpmm multiprocessing is fragile on macOS; keep single-process by default
        model = DPMMPrivBayesAdapter(
            epsilon=epsilon,
            delta=1.0 / (len(real) ** 2),
            degree=2,
            n_bins=50,
            seed=seed,
            preprocess=("dp" if strict_dp else str(preprocess)),
            strict_dp=bool(strict_dp),
            public_bounds=pb if pb else None,
            public_categories=pc if pc else None,
            int_cardinality_as_categorical=20,
            n_jobs=1,
            compress=True,
            n_iters=2000 if dataset_tag == "adult" else 1000,
        )
        
        fit_start = time.time()
        model.fit(real)
        result.fit_time_sec = time.time() - fit_start
        
        sample_start = time.time()
        n = n_samples if n_samples is not None else len(real)
        syn = model.sample(n)
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
                          temperature: float = 1.0, target_col: Optional[str] = None,
                          n_samples: Optional[int] = None,
                          strict_dp: bool = False,
                          dataset_tag: str = "unknown",
                          regime: str = "default",
                          public_bounds_schema: Optional[dict] = None,
                          public_categories_schema: Optional[dict] = None) -> ImplementationResult:
    """Run Enhanced PrivBayes implementation and measure performance.
    
    Auto-detects target column for label handling. Sets up public bounds
    and categories. Returns aligned real data (vocab-matched) for fair comparison.
    """
    base = "Enhanced"
    result = ImplementationResult(
        name=_label_impl(base, regime),
        implementation_base=base,
        regime=regime,
        epsilon=epsilon,
        seed=seed,
        success=False,
    )
    
    try:
        from pbbench.variants.pb_enhanced import EnhancedPrivBayesAdapter
        
        tracemalloc.start()
        
        # Compute public bounds with margin for DP bounds discovery
        # WARNING: Using actual min/max from data is a privacy leak if bounds aren't public knowledge.
        # Set use_dp_bounds=True to use DP bounds discovery instead (fully DP-compliant).
        # For now, we compute bounds from data for utility, but this should be avoided for private data.
        use_dp_bounds = bool(strict_dp)  # True => DP bounds discovery via eps_disc
        public_bounds = {}
        original_data_bounds = {}
        
        if not use_dp_bounds:
            if public_bounds_schema:
                # Public-schema mode: bounds are assumed public side information.
                public_bounds = dict(public_bounds_schema)
                original_data_bounds = None
            else:
                # Non-DP bounds discovery (privacy leak if bounds aren't public knowledge)
                for col in real.columns:
                    if pd.api.types.is_numeric_dtype(real[col]):
                        vmin, vmax = float(real[col].min()), float(real[col].max())
                        margin = (vmax - vmin) * 0.1
                        public_bounds[col] = [vmin - margin, vmax + margin]
                        original_data_bounds[col] = [vmin, vmax]
                    elif real[col].dtype == 'object':
                        s = pd.to_numeric(real[col], errors='coerce')
                        if s.notna().mean() >= 0.95:
                            vmin, vmax = float(s.min()), float(s.max())
                            margin = (vmax - vmin) * 0.1
                            public_bounds[col] = [vmin - margin, vmax + margin]
                            original_data_bounds[col] = [vmin, vmax]
        else:
            # DP-compliant: Use DP bounds discovery (set public_bounds=None)
            # The model will use eps_disc to discover bounds with DP
            public_bounds = None
            original_data_bounds = None
        
        # Use provided target column or auto-detect (common names: 'target', 'income', 'label', 'class')
        detected_target_col = target_col
        if detected_target_col is None:
            for col in ['target', 'income', 'label', 'class', 'outcome']:
                if col in real.columns:
                    detected_target_col = col
                    break
        
        # Set up label columns and public categories
        label_columns = []
        public_categories = dict(public_categories_schema or {})
        if detected_target_col and detected_target_col in real.columns:
            label_columns = [detected_target_col]
            pub = _public_label_categories(detected_target_col)
            if pub is not None:
                public_categories[detected_target_col] = pub
            elif strict_dp:
                raise RuntimeError(
                    f"strict-dp enabled: unknown public label domain for target_col='{detected_target_col}'."
                )
            else:
                # Legacy (non-DP): infer label domain from data
                unique_labels = sorted(real[detected_target_col].astype(str).dropna().unique().tolist())
                public_categories[detected_target_col] = unique_labels
        
        # WARNING: Auto-discovering categories from training data is NOT DP-compliant.
        # This reveals the exact set of categories in the data, which is a privacy leak.
        # Only use this if categories are public knowledge (e.g., US states, ISO codes).
        # For private categories, set public_categories=None and use DP heavy hitters.
        # 
        # Disabled by default for DP compliance. Uncomment only if categories are public.
        # for col in real.columns:
        #     if col in public_categories:
        #         continue
        #     if real[col].dtype == 'object' or real[col].dtype.name == 'category':
        #         unique_vals = sorted(real[col].astype(str).dropna().unique().tolist())
        #         if len(unique_vals) <= 1000:
        #             public_categories[col] = unique_vals
        
        # Start timing from model creation onwards (excludes setup overhead)
        start_time = time.time()
        model = EnhancedPrivBayesAdapter(
            epsilon=epsilon,
            delta=1.0 / (len(real) ** 2),
            seed=seed,
            temperature=temperature,
            cpt_smoothing=1.5,  # CPT smoothing (post-processing, DP-safe)
            public_bounds=public_bounds if public_bounds else None,
            label_columns=label_columns if label_columns else None,
            public_categories=public_categories if public_categories else None,
            # Keeping all non-zero noisy buckets is DP-safe (post-processing) and reduces UNK.
            cat_keep_all_nonzero=True,
            bins_per_numeric=50,
            max_parents=2,
            eps_split={"structure": 0.3, "cpt": 0.7},
            forbid_as_parent=label_columns,  # Prevent label from being a parent
            original_data_bounds=original_data_bounds if original_data_bounds else None
        )
        
        fit_start = time.time()
        model.fit(real)
        result.fit_time_sec = time.time() - fit_start
        
        sample_start = time.time()
        n = n_samples if n_samples is not None else len(real)
        syn = model.sample(n)
        result.sample_time_sec = time.time() - sample_start
        
        # Align REAL to Enhanced vocab for fair comparison (post-processing, DP-safe)
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


def compute_comprehensive_utility(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target_col: Optional[str] = "target",
    n_bootstrap: int = 30,
) -> Dict[str, Any]:
    """Compute comprehensive utility metrics: TVD, MI, correlation, coverage, downstream ML.
    
    Includes bootstrap confidence intervals for robustness. Metrics computed
    with sampling caps for scalability on large datasets.
    """
    metrics = {}
    
    print("    Computing TVD metrics...", flush=True)
    try:
        metrics["tvd"] = comprehensive_tvd_metrics(real, syn, n_bootstrap=n_bootstrap)
        # Print TVD/EMD method counts
        if "summary" in metrics["tvd"]:
            tvd_sum = metrics["tvd"]["summary"]
            emd_count = tvd_sum.get('_tvd1_emd_count', 0)
            tvd_count = tvd_sum.get('_tvd1_tvd_count', 0)
            tvd_fallback = tvd_sum.get('_tvd1_tvd_fallback_count', 0)
            total = emd_count + tvd_count + tvd_fallback
            if total > 0:
                import sys
                print(f"         TVD 1D methods: {emd_count} EMD, {tvd_count} TVD, {tvd_fallback} TVD(fallback)", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"    ⚠️ TVD failed: {e}")
        metrics["tvd"] = {}
    
    print("    Computing MI metrics...", flush=True)
    try:
        metrics["mi"] = comprehensive_mi_metrics(real, syn, n_bootstrap=n_bootstrap)
    except Exception as e:
        print(f"    ⚠️ MI failed: {e}")
        metrics["mi"] = {}
    
    print("    Computing correlation metrics...", flush=True)
    try:
        metrics["correlation"] = comprehensive_correlation_metrics(real, syn, n_bootstrap=n_bootstrap)
    except Exception as e:
        print(f"    ⚠️ Correlation failed: {e}")
        metrics["correlation"] = {}
    
    print("    Computing coverage metrics...", flush=True)
    try:
        metrics["coverage"] = comprehensive_coverage_metrics(real, syn, n_bootstrap=n_bootstrap)
    except Exception as e:
        print(f"    ⚠️ Coverage failed: {e}")
        metrics["coverage"] = {}
    
    if target_col and target_col in real.columns:
        print("    Computing downstream ML metrics...", flush=True)
        try:
            metrics["downstream"] = comprehensive_downstream_metrics(
                real,
                syn,
                target_col=target_col,
                n_bootstrap=n_bootstrap,
                n_jobs=1,  # avoid joblib/multiprocessing hangs on some platforms (notably macOS)
            )
        except Exception as e:
            print(f"    ⚠️ Downstream failed: {e}")
            metrics["downstream"] = {}
    
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
    
    # QI Linkage (robust + "linkable" = matches a UNIQUE real QI pattern)
    # This avoids the degenerate behavior of simple "tuple exists in real" overlap,
    # which can saturate at 1.0 on large datasets.
    try:
        def _qcut_bin_count(s: pd.Series, q: int = 5) -> int:
            ss = pd.to_numeric(s, errors="coerce").dropna()
            if ss.size < 2:
                return 0
            try:
                codes = pd.qcut(ss, q=int(q), labels=False, duplicates="drop")
                return int(pd.Series(codes).nunique(dropna=True))
            except Exception:
                return 0

        # Candidate categorical QIs: moderate cardinality, not dominated by one value.
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
            # entropy proxy
            p = rr.value_counts(normalize=True, dropna=True).to_numpy()
            ent = float(-(p * np.log(p + 1e-12)).sum())
            cat_candidates.append((ent, c))
        cat_candidates.sort(reverse=True)
        cat_cols = [c for _, c in cat_candidates[:2]]

        # Candidate numeric QIs: variance high, and real qcut produces >=3 bins.
        num_candidates: list[tuple[float, str]] = []
        rnum = real.select_dtypes(include=[np.number])
        for c in rnum.columns:
            if c not in syn.columns:
                continue
            s = pd.to_numeric(rnum[c], errors="coerce").dropna()
            if s.empty or int(s.nunique(dropna=True)) < 10:
                continue
            if _qcut_bin_count(s, q=5) < 3:
                continue
            var = float(s.var())
            if np.isfinite(var) and var > 0:
                num_candidates.append((var, c))
        num_candidates.sort(reverse=True)
        num_cols = [c for _, c in num_candidates[: max(1, 3 - len(cat_cols))]]

        qi_cols = cat_cols + num_cols
        qi_cols = qi_cols[:3]
        if len(qi_cols) == 0:
            metrics["qi_linkage_rate"] = None
            return metrics

        def _bin_numeric_from_real(real_col: pd.Series, syn_col: pd.Series, q: int = 5) -> tuple[pd.Series, pd.Series]:
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

        # Build aligned discrete QI views
        real_qi: dict[str, Any] = {}
        syn_qi: dict[str, Any] = {}
        for c in qi_cols:
            if pd.api.types.is_numeric_dtype(real[c]):
                rc, sc = _bin_numeric_from_real(real[c], syn[c], q=5)
                real_qi[c] = rc
                syn_qi[c] = sc
            else:
                rr = real[c].astype("string").fillna("__NA__")
                ss = syn[c].astype("string").fillna("__NA__")
                # unseen synthetic categories are kept as-is; linkage is about exact matching
                real_qi[c] = rr
                syn_qi[c] = ss

        real_qi_df = pd.DataFrame(real_qi)
        syn_qi_df = pd.DataFrame(syn_qi)

        # Define "linkable" as matching a UNIQUE real QI tuple (support==1).
        real_tuples = list(map(tuple, real_qi_df.to_numpy()))
        vc = pd.Series(real_tuples).value_counts()
        unique_real = set(vc[vc == 1].index.tolist())
        if len(syn) == 0:
            metrics["qi_linkage_rate"] = None
            return metrics
        syn_tuples = list(map(tuple, syn_qi_df.to_numpy()))
        metrics["qi_linkage_rate"] = float(sum(1 for t in syn_tuples if t in unique_real) / len(syn))
    except:
        metrics['qi_linkage_rate'] = None
    
    return metrics


def compute_min_viable_privacy_audit(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Minimum viable audit set (model-agnostic, computed from REAL + SYN):
    - nearest-neighbor memorization (EMR)
    - unique pattern leakage (support=1 patterns)
    - rare combination leakage (tau=3)
    - conditional disclosure leakage (L1 distance) using QI heuristic + sensitive=target_col
    """
    out: Dict[str, Any] = {}

    qi_cols = _audit_select_qi_cols(real, k=3)
    sens = target_col if target_col and target_col in real.columns else None
    if sens is None and len(real.columns) > 0:
        sens = str(real.columns[-1])

    mem = nearest_neighbor_memorization(real, syn, n_bins=20)
    out["nn_memorization"] = {"emr": mem.emr, "mean_dsyn": mem.mean_dsyn, "mean_dreal": mem.mean_dreal}

    out["unique_pattern_leakage"] = unique_pattern_leakage(real, syn, n_bins=20)
    rare = rare_combination_leakage(real, syn, tau=3, n_bins=20)
    out["rare_combination"] = {"tau": 3, "rmr": rare.rmr, "mae": rare.mae}

    if sens is not None and sens in real.columns and sens in syn.columns:
        out["conditional_disclosure_l1"] = conditional_disclosure_l1(
            real,
            syn,
            qi_cols=qi_cols,
            sensitive_col=sens,
            n_bins=20,
        )
        out["conditional_disclosure_meta"] = {"qi_cols": qi_cols, "sensitive_col": sens}
    else:
        out["conditional_disclosure_l1"] = None
        out["conditional_disclosure_meta"] = {"qi_cols": qi_cols, "sensitive_col": sens}

    return out


# ==================== VISUALIZATION ====================

def create_utility_privacy_plots(results: List[ImplementationResult], out_dir: str, filename_prefix: str = "utility_privacy_plots"):
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
    
    # Plot 3: Privacy Risk (QI linkage) vs Epsilon
    ax = axes[0, 2]
    # If ERMR is constant (often 0.0), don't clutter the panel; annotate instead.
    ermr_note = None
    if 'ermr' in df_agg.columns and df_agg['ermr'].notna().any():
        ermr_vals = df_agg['ermr'].dropna().astype(float)
        if len(ermr_vals) > 0 and (ermr_vals.max() - ermr_vals.min()) < 1e-12:
            ermr_note = float(ermr_vals.iloc[0])

    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'qi_linkage' in data_impl.columns and data_impl['qi_linkage'].notna().any():
            ax.plot(
                data_impl['epsilon'],
                data_impl['qi_linkage'],
                marker='o',
                label=f"{impl}",
                color=colors.get(impl),
                linewidth=2,
            )
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Rate (lower is better)', fontsize=12)
    ax.set_title('QI Linkage Rate vs Budget', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, ncol=1)
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
            if 'tvd_3d_mean' in data_impl.columns and data_impl['tvd_3d_mean'].notna().any():
                ax.plot(data_impl['epsilon'], data_impl['tvd_3d_mean'],
                        marker='^', label=f'{impl} (3D)', color=colors.get(impl), linewidth=1.5, linestyle=':')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Distance (log scale)', fontsize=12)
    ax.set_title('TVD/EMD Metrics (Lower is Better)', fontweight='bold')
    # tvd_1d_mean can be EMD(IQR-normalized) for numeric columns, and can be orders of magnitude
    # larger than 2D/3D TVD; use log scale so all implementations remain visible.
    ax.set_yscale('log')
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
    # If LR and RF overlap (common when a method collapses labels), apply a small x-offset
    # so both markers are visible.
    try:
        eps_span = float(df_agg['epsilon'].max() - df_agg['epsilon'].min())
    except Exception:
        eps_span = 0.0
    dx = 0.01 * (eps_span if eps_span > 0 else 1.0)
    for impl in impls:
        data_impl = df_agg[df_agg['implementation'] == impl]
        if 'syn2real_lr_auc' in data_impl.columns and data_impl['syn2real_lr_auc'].notna().any():
            ax.plot(data_impl['epsilon'] - dx, data_impl['syn2real_lr_auc'], 
                    marker='o', label=f'{impl} (LR)', color=colors.get(impl), linewidth=1.5, linestyle='-',
                    markerfacecolor='none', markeredgewidth=1.8)
        if 'syn2real_rf_auc' in data_impl.columns and data_impl['syn2real_rf_auc'].notna().any():
            ax.plot(data_impl['epsilon'] + dx, data_impl['syn2real_rf_auc'], 
                    marker='s', label=f'{impl} (RF)', color=colors.get(impl), linewidth=1.5, linestyle='--',
                    markerfacecolor='none', markeredgewidth=1.8)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('ML Model AUC (Syn→Real)', fontsize=12)
    ax.set_title('Downstream ML Performance', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/{filename_prefix}.pdf", bbox_inches='tight')
    print(f"✅ Saved plots to {out_dir}/{filename_prefix}.png")
    plt.close()


# ==================== MAIN ====================

def main(
    data_path: str,
    epsilons: List[float],
    seeds: List[int],
    out_dir: str,
    implementations: List[str],
    target_col: Optional[str] = None,
    n_samples: Optional[int] = None,
    n_bootstrap: int = 30,
    audit: bool = False,
    audit_mia: bool = False,
    audit_mia_holdout_frac: float = 0.3,
    audit_mia_impls: Optional[List[str]] = None,
    regimes: Optional[List[str]] = None,  # ["default"] or ["default","strict"]
    dpmm_default_preprocess: str = "none",  # "none" | "public" | "dp"
    schema: Optional[str] = None,
):
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
    regimes = regimes or ["default"]
    regimes = [r.strip().lower() for r in regimes]
    regimes = [r for r in regimes if r in {"default", "strict"}]
    if not regimes:
        regimes = ["default"]
    print(f"Regimes: {regimes}")
    print(f"Output: {out_dir}")
    print(f"Bootstrap resamples (for CI): {n_bootstrap}")
    print(f"DPMM default preprocess: {dpmm_default_preprocess}")
    if audit:
        print("Privacy audit probes: enabled (min viable)")
    if audit_mia:
        print(f"Membership inference: enabled (holdout_frac={audit_mia_holdout_frac})")
    if target_col:
        print(f"Target column: {target_col}")
    if n_samples:
        print(f"Synthetic rows: {n_samples} (custom)")
    else:
        print(f"Synthetic rows: same as training data (default)")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    real = pd.read_csv(data_path)
    print(f"✓ Loaded: {real.shape[0]} rows, {real.shape[1]} columns")

    dataset_tag = _infer_dataset_tag(real)
    print(f"Dataset tag: {dataset_tag}")

    public_schema = _load_public_schema(schema)
    public_bounds_schema = public_schema.get("public_bounds") if isinstance(public_schema, dict) else None
    public_categories_schema = public_schema.get("public_categories") if isinstance(public_schema, dict) else None
    if schema:
        print(f"Public schema: {schema}")
        if isinstance(public_bounds_schema, dict):
            print(f"  - public_bounds: {len(public_bounds_schema)} entries")
        if isinstance(public_categories_schema, dict):
            print(f"  - public_categories: {len(public_categories_schema)} columns")
    
    print(f"\n📁 Creating output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"✓ Output directory ready")
    
    # Run all experiments
    all_results = []
    
    for regime in regimes:
        strict_dp = regime == "strict"
        print(f"\n{'='*80}")
        print(f" REGIME={regime} ".center(80, "="))
        print(f"{'='*80}")

        for eps in epsilons:
            for seed in seeds:
                print(f"\n{'='*80}")
                print(f" EPSILON={eps}, SEED={seed} ".center(80, "="))
                print(f"{'='*80}")
                
                for impl_name in implementations:
                    if strict_dp and impl_name == "SynthCity":
                        print("\n🔹 Running SynthCity...")
                        print("  ⏭️  Skipped in strict regime (not DP-compliant).")
                        # still record a failure row so plots/tables can show omission if desired
                        r = ImplementationResult(
                            name=_label_impl("SynthCity", regime),
                            implementation_base="SynthCity",
                            regime=regime,
                            epsilon=eps,
                            seed=seed,
                            success=False,
                            error="Skipped: strict regime; SynthCity not DP-compliant.",
                        )
                        all_results.append(r)
                        continue

                    disp = _label_impl(impl_name, regime) if impl_name in {"Enhanced", "DPMM", "SynthCity"} else impl_name
                    print(f"\n🔹 Running {disp}...")
                    
                    # Run implementation
                    if impl_name == "SynthCity":
                        result, syn, eval_real = run_synthcity_privbayes(
                            real,
                            eps,
                            seed,
                            n_samples=n_samples,
                            strict_dp=False,
                            regime=regime,
                        )
                    elif impl_name == "DPMM":
                        result, syn, eval_real = run_dpmm_privbayes(
                            real,
                            eps,
                            seed,
                            n_samples=n_samples,
                            strict_dp=strict_dp,
                            dataset_tag=dataset_tag,
                            target_col=target_col,
                            regime=regime,
                            preprocess=dpmm_default_preprocess,
                            public_bounds=(public_bounds_schema if (not strict_dp) else None),
                            public_categories=(public_categories_schema if (not strict_dp) else None),
                        )
                    elif impl_name == "Enhanced":
                        result, syn, eval_real = run_enhanced_privbayes(
                            real,
                            eps,
                            seed,
                            temperature=1.0,
                            target_col=target_col,
                            n_samples=n_samples,
                            strict_dp=strict_dp,
                            dataset_tag=dataset_tag,
                            regime=regime,
                            public_bounds_schema=(public_bounds_schema if (not strict_dp) else None),
                            # Allow strict-DP to consume PUBLIC categorical domains (same principle as DPMM):
                            # domains are public side information and do not cost epsilon.
                            public_categories_schema=public_categories_schema,
                        )
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
                    comp_util = compute_comprehensive_utility(
                        eval_real, syn, target_col=detected_target_col, n_bootstrap=n_bootstrap
                    )
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

                    if audit:
                        print("  Computing privacy audit probes (min viable)...", flush=True)
                        result.audit_metrics = compute_min_viable_privacy_audit(
                            eval_real, syn, target_col=detected_target_col
                        )

                    if audit_mia:
                        mia_impls = audit_mia_impls or implementations
                        if result.implementation_base in set(mia_impls):
                            print("  Computing membership inference (distance attack; extra fit)...", flush=True)
                            rng = np.random.default_rng(seed)
                            idx = np.arange(len(real))
                            rng.shuffle(idx)
                            frac = float(np.clip(audit_mia_holdout_frac, 0.05, 0.5))
                            n_holdout = int(frac * len(idx))
                            hold = idx[:n_holdout]
                            train = idx[n_holdout:]
                            real_train = real.iloc[train].reset_index(drop=True)
                            real_holdout = real.iloc[hold].reset_index(drop=True)

                            syn_train = None
                            try:
                                if result.implementation_base == "Enhanced":
                                    _, syn_train, _ = run_enhanced_privbayes(
                                        real_train,
                                        eps,
                                        seed,
                                        temperature=1.0,
                                        target_col=target_col,
                                        n_samples=len(real_train),
                                        strict_dp=strict_dp,
                                        dataset_tag=dataset_tag,
                                        regime=regime,
                                        public_categories_schema=public_categories_schema,
                                    )
                                elif result.implementation_base == "SynthCity":
                                    _, syn_train, _ = run_synthcity_privbayes(
                                        real_train,
                                        eps,
                                        seed,
                                        n_samples=len(real_train),
                                        strict_dp=False,
                                        regime=regime,
                                    )
                                elif result.implementation_base == "DPMM":
                                    _, syn_train, _ = run_dpmm_privbayes(
                                        real_train,
                                        eps,
                                        seed,
                                        n_samples=len(real_train),
                                        strict_dp=strict_dp,
                                        dataset_tag=dataset_tag,
                                        target_col=target_col,
                                        regime=regime,
                                        preprocess=dpmm_default_preprocess,
                                    )
                            except Exception as e:
                                if result.audit_metrics is None:
                                    result.audit_metrics = {}
                                result.audit_metrics["membership_inference_error"] = str(e)

                            if syn_train is not None:
                                mia = membership_inference_distance_attack(
                                    real_train, real_holdout, syn_train, n_bins=20
                                )
                                if result.audit_metrics is None:
                                    result.audit_metrics = {}
                                result.audit_metrics["membership_inference"] = {
                                    "auc": mia.auc,
                                    "advantage": mia.advantage,
                                }

                    # Save synthetic dataset
                    syn_filename = f"synthetic_{result.name}_eps{eps}_seed{seed}.csv"
                    syn_filename = syn_filename.replace(" ", "_").replace("/", "_")
                    syn_path = os.path.join(out_dir, syn_filename)
                    syn.to_csv(syn_path, index=False)
                    result.synthetic_data_path = syn_path
                    print(f"  💾 Saved synthetic data: {syn_filename} ({len(syn)} rows, {syn.shape[1]} columns)")

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
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of rows to generate (default: same as training data size)")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=30,
        help="Number of bootstrap resamples for confidence intervals (0 disables).",
    )

    # Minimum viable privacy audit probes
    parser.add_argument(
        "--audit",
        action="store_true",
        default=True,
        help="Compute additional privacy audit probes (NN memorization, unique/rare leakage, conditional disclosure).",
    )
    parser.add_argument(
        "--no-audit",
        action="store_false",
        dest="audit",
        help="Disable privacy audit probes (use for faster runs).",
    )
    parser.add_argument(
        "--audit-mia",
        action="store_true",
        default=True,
        help="Compute membership inference (distance-based) by training on a train split (extra cost).",
    )
    parser.add_argument(
        "--no-audit-mia",
        action="store_false",
        dest="audit_mia",
        help="Disable membership inference probe (use for faster runs).",
    )
    parser.add_argument(
        "--audit-mia-holdout-frac",
        type=float,
        default=0.3,
        help="Holdout fraction for membership inference (default 0.3).",
    )
    parser.add_argument(
        "--audit-mia-impls",
        type=str,
        nargs="+",
        default=None,
        help="Subset of implementations to run membership inference for (default: all selected impls).",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        nargs="+",
        default=["default"],
        choices=["default", "strict"],
        help="Which configuration regimes to run. Use both to compare default vs strict-DP.",
    )
    parser.add_argument(
        "--dpmm-default-preprocess",
        type=str,
        default="none",
        choices=["none", "public", "dp"],
        help="DPMM preprocessing mode when regime=default. 'none' is non-DP (data-derived bounds/domains).",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to per-dataset public-schema JSON (public bounds/categories). Consumed only in regime=default.",
    )
    parser.add_argument(
        "--strict-dp",
        action="store_true",
        help="(Deprecated) Alias for --regimes strict.",
    )
    
    args = parser.parse_args()

    regimes = list(args.regimes or ["default"])
    if args.strict_dp:
        regimes = ["strict"]
    
    main(
        args.data,
        args.eps,
        args.seeds,
        args.out_dir,
        args.implementations,
        args.target_col,
        args.n_samples,
        args.n_bootstrap,
        args.audit,
        args.audit_mia,
        args.audit_mia_holdout_frac,
        args.audit_mia_impls,
        regimes,
        args.dpmm_default_preprocess,
        args.schema,
    )


