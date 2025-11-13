from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_bool_dtype

SMOOTH: float = 1e-8  # additive smoothing for probabilities

# ---- Minimal "register" shim for compatibility ----

def register(*args, **kwargs):
    def decorator(cls_or_func):
        return cls_or_func
    return decorator

# ======================== Auto-tuning (utility ↑ with ε) ========================

@dataclass
class PBTune:
    eps_split: Dict[str, float]
    eps_disc: float
    bins_per_numeric: int
    max_parents: int
    cat_buckets: int
    cat_topk: int
    dp_bounds_mode: str  # "public" or "smooth"
    dp_quantile_alpha: float  # e.g., 0.01 => [1%, 99%] bounds

def auto_tune_for_epsilon(
    epsilon: float,
    n: int,
    d: int,
    *,
    have_public_bounds: bool,
    target_high_utility: bool = True
) -> PBTune:
    """
    Heuristic tuning schedule aimed at monotonic utility w.r.t ε.
    - More ε to CPTs as ε grows (structure still gets a meaningful slice).
    - bins_per_numeric increases slowly with ε (caps to avoid CPT blow-ups).
    - max_parents increases at higher ε (2 -> 3).
    - Small metadata budget; use "smooth" DP bounds if no public coarse bounds.
    """
    eps = float(max(epsilon, 1e-6))
    # Structure/CPT split: favor CPTs slightly as ε grows
    s_frac = 0.35 if eps < 0.5 else (0.30 if eps < 2 else 0.25)
    c_frac = 1.0 - s_frac
    # Reserve a thin slice for metadata (bounds/domains). Use less as ε grows.
    disc_frac = 0.12 if eps < 0.5 else (0.08 if eps < 2 else 0.05)
    disc_frac = min(disc_frac, 0.15)
    # Numeric discretization granularity grows slowly with ε (cap to 64)
    base_bins = 8
    extra = int(np.floor(np.log2(1 + eps * 10.0)))
    bins_per_numeric = int(np.clip(base_bins + extra, 8, 64))
    # Parent width: keep small at low ε; allow 3 at higher ε if d is large.
    max_parents = 2 if eps < 1.5 else (3 if d >= 16 else 2)
    # DP categorical via hash buckets: keep domain bounded & stable
    cat_buckets = 64 if eps < 1.0 else (96 if eps < 2.0 else 128)
    cat_topk = 24 if eps < 1.0 else (28 if eps < 2.0 else 32)
    dp_bounds_mode = "public" if have_public_bounds else "smooth"
    dp_quantile_alpha = 0.01  # [1%, 99%] clipping for smooth DP bounds
    eps_disc = float(np.clip(disc_frac * eps, 0.0, eps))
    eps_split = {"structure": s_frac, "cpt": c_frac}
    return PBTune(
        eps_split=eps_split,
        eps_disc=eps_disc,
        bins_per_numeric=bins_per_numeric,
        max_parents=max_parents,
        cat_buckets=cat_buckets,
        cat_topk=cat_topk,
        dp_bounds_mode=dp_bounds_mode,
        dp_quantile_alpha=dp_quantile_alpha,
    )

# ========================== Helpers for DP metadata ===========================

def _blake_bucket(s: str, m: int) -> int:
    """Hash string to bucket index using BLAKE2b.
    
    Deterministic mapping for DP categorical heavy hitters. Returns integer
    in range [0, m-1] for bucket assignment.
    """
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=16)
    return int.from_bytes(h.digest(), "little") % int(m)

def _quantile_indices(n: int, q: float) -> int:
    """Compute array index for quantile q in sorted array of length n.
    
    Returns 0-based index. Uses ceiling to handle edge cases consistently.
    """
    q = float(np.clip(q, 0.0, 1.0))
    return int(np.clip(int(np.ceil(q * n)) - 1, 0, max(n - 1, 0)))

def _smooth_sensitivity_quantile(
    x: np.ndarray,
    q: float,
    eps: float,
    delta: float,
    rng: np.random.Generator,
    beta_scale: float = 1.0,
) -> float:
    """
    Approximate smooth sensitivity mechanism for a quantile (Nissim–Raskhodnikova–Smith'07).
    Produces an (ε, δ)-DP noisy quantile without public bounds.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    x.sort()
    n = x.size
    i = _quantile_indices(n, q)
    delta = float(np.clip(delta, 1e-15, 1.0 - 1e-12))
    eps = float(max(eps, 1e-12))
    # NRS'07 calibration
    beta = beta_scale * (eps / (2.0 * np.log(1.0 / delta)))
    max_s = 0.0
    k_max = min(n - 1, int(np.ceil(4.0 * np.sqrt(n + 1))))
    for k in range(0, k_max + 1):
        l = max(i - k, 0)
        r = min(i + k, n - 1)
        ls = float(x[r] - x[l])
        ss = np.exp(-beta * k) * ls
        if ss > max_s:
            max_s = ss
    # Correct scale factor: 2 * S* / ε
    scale = (2.0 * max_s) / eps
    noise = rng.laplace(0.0, scale)
    y = float(x[i] + noise)
    return y

def _dp_numeric_bounds_public(
    col: pd.Series,
    eps_min: float,
    eps_max: float,
    coarse_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Pure ε-DP: with public coarse [L,U], add Laplace noise to min/max of data clipped to [L,U].
    """
    L, U = coarse_bounds
    x = pd.to_numeric(col, errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(L), float(U)
    xc = np.clip(x, L, U)
    sens = float(max(U - L, 0.0))
    lo = float(np.min(xc) + rng.laplace(0.0, sens / max(eps_min, 1e-12)))
    hi = float(np.max(xc) + rng.laplace(0.0, sens / max(eps_max, 1e-12)))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + (U - L) / 100.0
    lo = float(np.clip(lo, L, U))
    hi = float(np.clip(hi, L, U))
    return lo, hi

def _dp_numeric_bounds_smooth(
    col: pd.Series,
    eps_total: float,
    delta_total: float,
    alpha: float,
    rng: np.random.Generator,
    public_coarse: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    (ε,δ)-DP bounds via smooth-sensitivity quantiles.
    """
    x = pd.to_numeric(col, errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        if public_coarse is not None:
            return float(public_coarse[0]), float(public_coarse[1])
        return 0.0, 1.0
    eps_each = max(eps_total, 1e-12) * 0.5
    delta_each = max(delta_total, 1e-15) * 0.5
    qL = _smooth_sensitivity_quantile(x, alpha, eps_each, delta_each, rng)
    qU = _smooth_sensitivity_quantile(x, 1.0 - alpha, eps_each, delta_each, rng)
    if public_coarse is not None:
        Lc, Uc = public_coarse
        qL = float(np.clip(qL, Lc, Uc))
        qU = float(np.clip(qU, Lc, Uc))
    if not np.isfinite(qU) or qU <= qL:
        if public_coarse is not None:
            span = (public_coarse[1] - public_coarse[0]) / 100.0
            qU = float(qL + max(span, 1.0))
        else:
            qU = float(qL + (np.nanmax(x) - np.nanmin(x) + 1.0) / 100.0)
    return float(qL), float(qU)

# ============================ Model internals ============================

@dataclass
class _ColMeta:
    kind: str  # "numeric" or "categorical"
    k: int
    bins: Optional[np.ndarray] = None
    cats: Optional[List[str]] = None
    is_int: bool = False
    bounds: Optional[Tuple[float, float]] = None
    binary_numeric: bool = False
    original_dtype: Optional[np.dtype] = None
    all_nan: bool = False
    # DP hashed-categorical flags
    hashed_cats: bool = False
    hash_m: Optional[int] = None

@register("model", "privbayes")
class PrivBayesSynthesizerEnhanced:
    """Enhanced Differentially Private PrivBayes with QI-linkage reduction.
    
    Supported data types:
    - Numeric: int, float, decimal (discretized into bins)
    - Categorical: string/varchar (uses DP heavy hitters when domain unknown)
    - Boolean: binary numeric or categorical
    - Datetime/timedelta: converted to nanoseconds since epoch
      * Handles datetime64[ns] and string-formatted dates from CSV
      * Formats: '2023-01-15 10:30:00', '2023-01-15T10:30:00', etc.
    - Object columns: auto-detected as datetime (if 95%+ parseable), then numeric (if 95%+ convertible), else categorical
    
    __UNK__ tokens and how to avoid them:
    
    Without public categories, categoricals use DP heavy hitters:
    - Values hashed into buckets (B000, B001, etc.) for privacy
    - Only top-K buckets kept in vocabulary
    - Values in non-top-K buckets become __UNK__
    
    Strategies to reduce/avoid __UNK__:
    
    1. Provide public_categories (best - no UNK, no DP cost):
       For public domains (US states, ISO codes, etc.), provide all values.
       Example: public_categories={'state': ['CA', 'NY', 'TX', ...]}
    
    2. Increase cat_topk (DP-safe, uses more epsilon):
       Keeps more top-K buckets. Trade-off: more epsilon for discovery, less UNK.
       Use cat_topk_overrides for per-column control.
    
    3. Increase cat_buckets (DP-safe, may help):
       More hash buckets can capture more categories, but UNK still occurs if not in top-K.
    
    4. Use label_columns (best for target variables):
       Label columns never use hashing, never get UNK. Example: label_columns=['income']
    
    5. Allocate more epsilon to categorical discovery:
       Increase eps_disc to learn more categories. Trade-off: less epsilon for structure/CPT.
    
    6. Use cat_keep_all_nonzero=True (universal strategy):
       Keeps all observed buckets instead of just top-K. Captures all training categories,
       minimizing __UNK__ to near-zero. DP-safe, but uses more memory. Default in adapter.
    
    Additional features:
      • temperature: flatten CPTs at sampling (p -> p^(1/temperature), temperature>=1)
      • forbid_as_parent: columns never allowed as parents (e.g., QIs)
      • parent_blacklist: {child: [parents_not_allowed]} for fine-grained edge bans
      • numeric_bins_overrides: {col: k} to coarsen discretization per column
      • integer_decode_mode: 'round' | 'stochastic' | 'granular'
      • numeric_granularity: {col: step} snaps floats to bands on decode
      • cat_topk_overrides / cat_buckets_overrides: per-column tail compression
    """

    def __init__(
        self,
        *,
        epsilon: float,
        delta: float = 1e-6,
        seed: int = 0,
        # tuning / privacy split
        eps_split: Optional[Dict[str, float]] = None,
        eps_disc: Optional[float] = None,
        max_parents: int = 2,
        bins_per_numeric: int = 16,
        adjacency: str = "unbounded",
        # DP metadata strategy
        dp_bounds_mode: str = "smooth",
        dp_quantile_alpha: float = 0.01,
        public_bounds: Optional[Dict[str, List[float]]] = None,
        public_categories: Optional[Dict[str, List[str]]] = None,
        public_binary_numeric: Optional[Dict[str, bool]] = None,
        original_data_bounds: Optional[Dict[str, List[float]]] = None,  # Original data min/max for clipping
        # DP heavy hitters for categoricals when domain private
        cat_buckets: int = 64,
        cat_topk: int = 28,
        # Universal strategy: if cat_topk is None or -1, keep all non-zero buckets
        cat_keep_all_nonzero: bool = False,  # If True, keep all buckets with noisy_count > 0
        # decoding
        decode_binary_as_bool: bool = False,
        cpt_dtype: str = "float64",
        # misc
        require_public: bool = False,
        strict_dp: bool = True,
        # ===== New knobs (all optional) =====
        temperature: float = 1.0,
        cpt_smoothing: float = 1.5,  # Pseudo-counts added after DP noise (post-processing, DP-safe)
        forbid_as_parent: Optional[List[str]] = None,
        parent_blacklist: Optional[Dict[str, List[str]]] = None,
        numeric_bins_overrides: Optional[Dict[str, int]] = None,
        integer_decode_mode: str = "round",
        integer_granularity: Optional[Dict[str, int]] = None,
        numeric_granularity: Optional[Dict[str, float]] = None,
        cat_topk_overrides: Optional[Dict[str, int]] = None,
        cat_buckets_overrides: Optional[Dict[str, int]] = None,
        # categorical unknown handling
        unknown_token: str = "__UNK__",
        categorical_unknown_to_nan: bool = False,
        # label columns (no hashing, no UNK)
        label_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        if cpt_dtype not in ("float32", "float64"):
            raise ValueError("cpt_dtype must be 'float32' or 'float64'")
        self.cpt_dtype = cpt_dtype

        self.adjacency = str(adjacency).lower()
        if self.adjacency not in {"unbounded", "bounded"}:
            raise ValueError("adjacency must be 'unbounded' or 'bounded'")
        self._sens_count = 1.0 if self.adjacency == "unbounded" else 2.0

        # metadata / strategy
        self.require_public = bool(require_public)
        self.dp_bounds_mode = str(dp_bounds_mode).lower()
        if self.dp_bounds_mode not in {"public", "smooth"}:
            raise ValueError("dp_bounds_mode must be 'public' or 'smooth'")
        self.dp_quantile_alpha = float(dp_quantile_alpha)
        self.strict_dp = bool(strict_dp)

        # public hints
        self.public_bounds: Dict[str, List[float]] = dict(public_bounds or {})
        self.public_categories: Dict[str, List[str]] = {k: list(v or []) for k, v in (public_categories or {}).items()}
        self.public_binary_numeric: Dict[str, bool] = dict(public_binary_numeric or {})
        self.original_data_bounds: Dict[str, List[float]] = dict(original_data_bounds or {})

        # DP heavy hitters defaults
        self.cat_buckets = int(cat_buckets)
        self.cat_topk = int(cat_topk) if cat_topk is not None and cat_topk > 0 else None
        self.cat_keep_all_nonzero = bool(cat_keep_all_nonzero)

        # main knobs
        self.max_parents = int(max_parents)
        self.bins_per_numeric = int(bins_per_numeric)

        # Default DP metadata budget
        if eps_disc is None:
            self.eps_disc = float(min(max(0.10 * self.epsilon, 1e-6), 0.15 * self.epsilon))
        else:
            self.eps_disc = float(eps_disc)

        es = eps_split or {"structure": 0.3, "cpt": 0.7}
        s = max(0.0, float(es.get("structure", 0.3)))
        c = max(0.0, float(es.get("cpt", 0.7)))
        if s + c == 0:
            s, c = 0.3, 0.7
        z = s + c
        main_eps = max(self.epsilon - max(self.eps_disc, 0.0), 0.0)
        self._eps_struct = main_eps * (s / z)
        self._eps_cpt = main_eps * (c / z)
        self._eps_main = main_eps

        # store decode flags
        self.decode_binary_as_bool = bool(decode_binary_as_bool)

        # ===== store new knobs =====
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.cpt_smoothing = float(cpt_smoothing)
        if self.cpt_smoothing < 0:
            raise ValueError("cpt_smoothing must be >= 0")
        self.forbid_as_parent_set = set(forbid_as_parent or [])
        self.parent_blacklist = {k: set(v or []) for k, v in (parent_blacklist or {}).items()}
        self.numeric_bins_overrides: Dict[str, int] = dict(numeric_bins_overrides or {})
        self.integer_decode_mode = str(integer_decode_mode).lower()
        if self.integer_decode_mode not in {"round", "stochastic", "granular"}:
            raise ValueError("integer_decode_mode must be 'round', 'stochastic', or 'granular'")
        self.integer_granularity: Dict[str, int] = dict(integer_granularity or {})
        self.numeric_granularity: Dict[str, float] = dict(numeric_granularity or {})
        self.cat_topk_overrides: Dict[str, int] = dict(cat_topk_overrides or {})
        self.cat_buckets_overrides: Dict[str, int] = dict(cat_buckets_overrides or {})
        
        # categorical unknown handling
        self.unknown_token = str(unknown_token)
        self.categorical_unknown_to_nan = bool(categorical_unknown_to_nan)
        
        # label columns (no hashing, no UNK)
        self.label_columns = set(label_columns or [])

        # learned state
        self._meta: Dict[str, _ColMeta] = {}
        self._order: List[str] = []
        self._cpt: Dict[str, Dict[str, Any]] = {}

        # book-keeping
        self._dp_metadata_used_bounds: set[str] = set()
        self._dp_metadata_used_cats: set[str] = set()
        self._dp_metadata_delta_used: float = 0.0
        self._all_nan_columns: int = 0

    def _lap(self, eps: float, shape: Any, *, sens: Optional[float] = None) -> np.ndarray:
        """Generate Laplace noise for differential privacy.
        
        Noise scale is sensitivity/epsilon. Uses adjacency mode to determine
        default sensitivity (1.0 for unbounded, 2.0 for bounded).
        """
        base_sens = float(self._sens_count) if sens is None else float(sens)
        scale = base_sens / max(float(eps), 1e-12)
        return self._rng.laplace(0.0, scale, size=shape)

    def _build_meta(self, df: pd.DataFrame) -> None:
        """Build column metadata: discretization bins, categorical vocabularies, bounds.
        
        Handles DP metadata generation for columns without public bounds/categories.
        Uses smooth sensitivity quantiles for numeric bounds when public bounds unavailable.
        For categoricals, uses DP heavy hitters via hash buckets when domain is private.
        """
        self._dp_metadata_used_bounds.clear()
        self._dp_metadata_used_cats.clear()
        self._dp_metadata_delta_used = 0.0
        self._all_nan_columns = 0

        pb = dict(self.public_bounds or {})
        pc = {k: list(v or []) for k, v in (self.public_categories or {}).items()}
        pbn = dict(self.public_binary_numeric or {})

        m_cols_need_bounds: List[str] = []
        m_cols_need_cats: List[str] = []

        if (self.eps_disc > 0.0) and (not self.require_public):
            for c in df.columns:
                if is_numeric_dtype(df[c]) and c not in pb:
                    m_cols_need_bounds.append(c)
                elif (not is_numeric_dtype(df[c])) and not pc.get(c):
                    m_cols_need_cats.append(c)

        m_total = len(m_cols_need_bounds) + len(m_cols_need_cats)

        if not self.require_public and self.strict_dp and m_total > 0 and self.eps_disc <= 0.0:
            raise ValueError(
                "DP metadata required by default, but eps_disc=0. "
                "Either provide public bounds/categories or set a positive eps_disc."
            )

        eps_disc_per_col = (self.eps_disc / m_total) if m_total > 0 else 0.0

        smooth_cols: List[str] = []
        if not self.require_public and eps_disc_per_col > 0.0 and self.dp_bounds_mode == "smooth":
            for c in m_cols_need_bounds:
                smooth_cols.append(c)

        n_smooth = len(smooth_cols)
        delta_per_smooth_col = (self.delta / n_smooth) if n_smooth > 0 else 0.0

        meta: Dict[str, _ColMeta] = {}

        for c in df.columns:
            # Handle datetime/timedelta by converting to numeric (nanoseconds)
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].view('int64')
            elif pd.api.types.is_timedelta64_dtype(df[c]):
                df[c] = df[c].view('int64')
            # Handle string-formatted datetime columns (common in CSV exports)
            elif df[c].dtype == 'object':
                try:
                    dt_parsed = pd.to_datetime(df[c], errors='coerce')
                    if dt_parsed.notna().mean() >= 0.95:
                        df[c] = dt_parsed.astype('int64')
                except (ValueError, TypeError, OverflowError):
                    pass
            
            is_bool = is_bool_dtype(df[c])
            is_num = is_numeric_dtype(df[c])

            if self.require_public and is_bool and c not in pb:
                pb[c] = [0.0, 1.0]
                pbn[c] = True

            if is_num or is_bool or (c in pb):
                raw = pd.to_numeric(df[c], errors="coerce").to_numpy()
                all_nan_flag = False

                if self.require_public and is_num and not is_bool and c not in pb:
                    raise ValueError(f"Numeric column {c} requires public bounds when require_public=True.")

                if c in pb and isinstance(pb[c], (list, tuple)) and len(pb[c]) == 2:
                    L, U = pb[c]
                    if not (np.isfinite(L) and np.isfinite(U) and U > L):
                        if self.require_public:
                            raise ValueError(f"Invalid public bounds for {c}.")
                        x = raw[np.isfinite(raw)]
                        if x.size == 0:
                            L, U = 0.0, 1.0
                            self._all_nan_columns += 1
                            all_nan_flag = True
                        else:
                            L, U = float(np.min(x)), float(np.max(x))
                        pb[c] = [float(L), float(U)]
                else:
                    if self.require_public:
                        raise ValueError(f"Column {c} missing public bounds while require_public=True.")
                    if eps_disc_per_col > 0.0:
                        coarse = None
                        for key in ("*", "__all__", "__global__"):
                            if key in pb and isinstance(pb[key], (list, tuple)) and len(pb[key]) == 2:
                                L, U = pb[key]
                                if np.isfinite(L) and np.isfinite(U) and U > L:
                                    coarse = (float(L), float(U))
                                    break
                        if self.dp_bounds_mode == "public" and coarse is not None:
                            L, U = _dp_numeric_bounds_public(
                                pd.Series(raw),
                                eps_min=eps_disc_per_col * 0.5,
                                eps_max=eps_disc_per_col * 0.5,
                                coarse_bounds=coarse,
                                rng=self._rng
                            )
                            self._dp_metadata_used_bounds.add(c)
                        else:
                            L, U = _dp_numeric_bounds_smooth(
                                pd.Series(raw),
                                eps_total=eps_disc_per_col,
                                delta_total=max(delta_per_smooth_col, 1e-15),
                                alpha=self.dp_quantile_alpha,
                                rng=self._rng,
                                public_coarse=coarse
                            )
                            self._dp_metadata_used_bounds.add(c)
                            self._dp_metadata_delta_used += float(max(delta_per_smooth_col, 1e-15)) if n_smooth > 0 else 0.0
                        pb[c] = [float(L), float(U)]
                    else:
                        if self.strict_dp:
                            raise ValueError(
                                f"DP bounds required for column '{c}' but eps_disc_per_col=0 under strict_dp."
                            )
                        x = raw[np.isfinite(raw)]
                        if x.size == 0:
                            L, U = 0.0, 1.0
                            self._all_nan_columns += 1
                            all_nan_flag = True
                        else:
                            L, U = float(np.min(x)), float(np.max(x))
                        pb[c] = [float(L), float(U)]

                is_int = is_integer_dtype(df[c])
                if is_int:
                    try:
                        dt = df[c].to_numpy(copy=False).dtype
                    except Exception:
                        dt = np.dtype("int64")
                    original_dtype = dt
                else:
                    original_dtype = None

                if self.require_public:
                    binary_numeric = bool(self.public_binary_numeric.get(c, False))
                else:
                    vals = pd.to_numeric(df[c], errors="coerce")
                    u = pd.unique(vals.dropna())
                    try:
                        binary_numeric = len(u) <= 2 and set([0.0, 1.0]).issuperset(set(pd.Series(u).astype(float)))
                    except Exception:
                        binary_numeric = False

                if binary_numeric or is_bool:
                    k = 2
                    bins = np.array([0.0, 0.5, 1.0], dtype=float)
                else:
                    k_override = self.numeric_bins_overrides.get(c) if hasattr(self, "numeric_bins_overrides") else None
                    k = max(2, int(k_override)) if k_override is not None else max(2, int(self.bins_per_numeric))
                    bins = np.linspace(0.0, 1.0, k + 1)

                meta[c] = _ColMeta(
                    kind="numeric",
                    k=k,
                    bins=bins,
                    cats=None,
                    is_int=bool(is_int),
                    bounds=(float(pb[c][0]), float(pb[c][1])),
                    binary_numeric=bool(binary_numeric),
                    original_dtype=original_dtype,
                    all_nan=all_nan_flag,
                    hashed_cats=False,
                    hash_m=None,
                )
            else:
                # Labels: no hashing, no UNK, fixed public categories
                if c in self.label_columns:
                    pub = list(self.public_categories.get(c, []) or [])
                    if not pub:
                        if self.strict_dp:
                            raise ValueError(
                                f"Label column '{c}' requires public_categories[{c}] "
                                "so the class names stay fixed (e.g., ['<=50K','>50K'])."
                            )
                        # non-DP fallback (only if strict_dp=False)
                        vals = pd.Series(df[c], copy=False).astype("string").dropna().unique().tolist()
                        pub = sorted([str(v) for v in vals])
                    # Do not add unknown token to labels
                    cats = [x for x in pub if x != self.unknown_token]
                    if len(cats) < 2:
                        warnings.warn(f"Label column '{c}' has <2 classes after filtering; ensure sufficient class diversity.", stacklevel=1)
                    meta[c] = _ColMeta(kind="categorical", k=len(cats), cats=cats, hashed_cats=False, hash_m=None)
                    self.public_categories[c] = cats
                    continue  # skip the generic categorical logic below
                
                unk = getattr(self, "unknown_token", "__UNK__")
                pub = list(self.public_categories.get(c, []) or [])
                hashed = False
                hash_m = None
                if self.require_public:
                    cats = ([unk] if unk not in pub else []) + [x for x in pub if x != unk]
                    if not cats:
                        cats = [unk]
                else:
                    if pub:
                        # Use public_categories directly when provided (no __UNK__ needed)
                        # Only add __UNK__ if it's already in the public categories list
                        if unk in pub:
                            cats = [x for x in pub if x != unk]  # Remove UNK if present
                            cats = [unk] + cats  # Put UNK first if it was in pub
                        else:
                            # No UNK in public categories - use them as-is (no UNK needed)
                            cats = list(pub)
                    elif eps_disc_per_col > 0.0:
                        ser = pd.Series(df[c], copy=False).astype("string")
                        m_default = int(self.cat_buckets)
                        m = max(8, int(self.cat_buckets_overrides.get(c, m_default)))
                        buckets = ser.fillna(unk).map(lambda v: f"B{_blake_bucket(str(v), m):03d}")
                        counts = buckets.value_counts(dropna=False).to_dict()
                        eps_col = max(eps_disc_per_col, 1e-12)
                        noisy = {b: (float(cnt) + float(self._lap(eps_col, (), sens=1.0))) for b, cnt in counts.items()}
                        order = sorted(noisy.keys(), key=lambda t: noisy[t], reverse=True)
                        
                        # Universal strategy: keep all observed buckets
                        if self.cat_keep_all_nonzero:
                            # Keep all buckets that appeared in training data
                            # DP noise might make counts negative, but we keep all observed buckets anyway
                            topk = list(order)
                            # Keep __UNK__ for edge cases, but it's rarely used since we keep all categories
                            cats = [unk] + topk if len(topk) > 0 else [unk]
                        else:
                            # Original top-K strategy
                            k_default = int(self.cat_topk) if self.cat_topk is not None else 28
                            K = max(8, int(min(self.cat_topk_overrides.get(c, k_default), len(order))))
                            topk = order[:K] if K > 0 else []
                            cats = [unk] + topk
                        self._dp_metadata_used_cats.add(c)
                        hashed = True
                        hash_m = m
                    else:
                        if self.strict_dp:
                            raise ValueError(
                                f"DP categorical discovery required for '{c}' but eps_disc_per_col=0 under strict_dp."
                            )
                        warnings.warn(
                            "Non-DP categorical discovery used due to eps_disc=0 and strict_dp=False. "
                            "This is NOT differentially private and reveals the exact set of categories.",
                            stacklevel=1
                        )
                        vals = pd.Series(df[c], copy=False).astype("string").dropna().unique().tolist()
                        cats = ([unk] if unk not in pub else []) + [x for x in vals if x != unk]

                meta[c] = _ColMeta(
                    kind="categorical",
                    k=len(cats),
                    cats=cats,
                    hashed_cats=hashed,
                    hash_m=hash_m,
                )
                self.public_categories[c] = cats

        self._meta = meta
        self.public_bounds = pb
        self.public_categories = self.public_categories

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert dataframe to integer codes using learned metadata.
        
        Numeric columns: normalize to [0,1], bin using learned bins, map to integer codes.
        Categorical columns: map values to category indices. For hashed categoricals,
        apply hash bucket mapping first. Label columns skip hashing and unknown tokens.
        """
        out: Dict[str, np.ndarray] = {}
        for c, m in self._meta.items():
            if m.kind == "numeric":
                lo, hi = m.bounds if m.bounds is not None else (0.0, 1.0)
                x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                z = (x - lo) / max(hi - lo, 1e-12)
                z = np.where(np.isfinite(z), z, 0.5)
                z = np.clip(z, 0.0, 1.0)
                idx = np.digitize(z, m.bins, right=False) - 1
                idx = np.clip(idx, 0, m.k - 1)
                out[c] = idx.astype(int, copy=False)
            else:
                unk = getattr(self, "unknown_token", "__UNK__")
                cats = list(m.cats or [])
                
                # Non-label categoricals get unknown bucket, unless public_categories provided
                has_public_cats = c in (self.public_categories or {})
                if c not in self.label_columns and not has_public_cats:
                    if unk not in cats:
                        cats = [unk] + [x for x in cats if x != unk]
                        m.cats = cats
                        self.public_categories[c] = cats
                
                col = df[c].astype("string").fillna(unk if c not in self.label_columns else (cats[0] if cats else unk))
                
                # Never hash labels
                if getattr(m, "hashed_cats", False) and m.hash_m and (c not in self.label_columns):
                    msize = int(m.hash_m)
                    col = col.map(lambda v: unk if v == unk else f"B{_blake_bucket(str(v), msize):03d}")
                
                cat = pd.Categorical(col, categories=cats, ordered=False)
                codes = np.asarray(cat.codes, dtype=int)
                # For non-labels, unseen -> UNK (code 0). For labels, unseen (shouldn't happen) -> first class.
                codes = np.where(codes < 0, 0, codes)
                out[c] = codes
        return pd.DataFrame(out, index=df.index)

    def fit(self, df: pd.DataFrame, schema: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        """Fit PrivBayes model: learn structure and conditional probability tables.
        
        Process: (1) build metadata, (2) discretize data, (3) compute DP mutual information
        scores for all pairs, (4) select parents greedily, (5) estimate noisy CPTs with
        Laplace mechanism. Epsilon split between structure learning and CPT estimation.
        """
        cfg = config or {}
        kw = dict(cfg.get("kwargs", {}))

        if "eps_split" in kw:
            es = kw["eps_split"] or {}
            s = max(0.0, float(es.get("structure", 0.3)))
            c = max(0.0, float(es.get("cpt", 0.7)))
            if s + c == 0:
                s, c = 0.3, 0.7
            z = s + c
            self._eps_main = max(self.epsilon - max(self.eps_disc, 0.0), 0.0)
            self._eps_struct = self._eps_main * (s / z)
            self._eps_cpt = self._eps_main * (c / z)

        if "bins_per_numeric" in kw:
            self.bins_per_numeric = int(kw["bins_per_numeric"])
        if "require_public" in kw:
            self.require_public = bool(kw["require_public"])
        if "strict_dp" in kw:
            self.strict_dp = bool(kw["strict_dp"])

        self._build_meta(df)
        disc = self._discretize(df)
        cols = list(disc.columns)
        self._order = cols[:]

        parents: Dict[str, List[str]] = {c: [] for c in cols}
        dp_mi_scores: Dict[Tuple[str, str], float] = {}

        if self._eps_struct > 0:
            pair_info = []
            for j, c in enumerate(cols):
                for p in cols[:j]:
                    pair_info.append((c, p))
            n_pairs = len(pair_info)
            if n_pairs > 0:
                eps_per_pair = self._eps_struct / n_pairs
                for c, p in pair_info:
                    x = disc[c].to_numpy()
                    y = disc[p].to_numpy()
                    kx = self._meta[c].k
                    ky = self._meta[p].k
                    joint = np.zeros((kx, ky), dtype=float)
                    np.add.at(joint, (x, y), 1.0)
                    joint += self._lap(eps_per_pair, joint.shape, sens=1.0)
                    joint = np.maximum(joint, 0.0) + SMOOTH
                    pxy = joint / joint.sum()
                    px = pxy.sum(axis=1, keepdims=True)
                    py = pxy.sum(axis=0, keepdims=True)
                    denom = (px @ py)
                    ratio = np.divide(pxy, denom, out=np.ones_like(pxy), where=denom > 0)
                    mi = float(max(0.0, (pxy * np.log(ratio)).sum()))
                    dp_mi_scores[(c, p)] = mi
                    dp_mi_scores[(p, c)] = mi

            for j, c in enumerate(cols):
                cand = [p for p in cols[:j] if p not in self.forbid_as_parent_set
                        and p not in self.parent_blacklist.get(c, set())]
                if not cand:
                    continue
                scores = [(dp_mi_scores.get((c, p), 0.0), p) for p in cand]
                scores.sort(key=lambda t: (-t[0], t[1]))
                parents[c] = [p for _, p in scores[: self.max_parents]]

        self._cpt = {}
        n_vars = len(cols)
        eps_per_var = (self._eps_cpt / n_vars) if (self._eps_cpt > 0 and n_vars > 0) else 0.0

        for c in cols:
            k_child = self._meta[c].k
            pa = parents[c]
            if len(pa) == 0:
                counts = np.bincount(disc[c].to_numpy(), minlength=k_child).astype(float)
                if eps_per_var > 0:
                    counts += self._lap(eps_per_var, counts.shape, sens=1.0)
                # Apply smoothing to DP-noisy counts
                counts = np.maximum(counts, 0.0)
                counts += self.cpt_smoothing
                probs = (counts / counts.sum().clip(min=1e-12)).reshape(1, k_child).astype(self.cpt_dtype)
                self._cpt[c] = {"parents": [], "parent_card": [], "probs": probs}
            else:
                par_ks = [self._meta[p].k for p in pa]
                S = int(np.prod(par_ks, dtype=object))
                max_cells = int(2_000_000)
                while S * k_child > max_cells and len(pa) > 0:
                    pa = pa[:-1]
                    par_ks = [self._meta[p].k for p in pa]
                    S = int(np.prod(par_ks, dtype=object))
                if S * k_child > max_cells:
                    raise MemoryError(f"CPT for {c} too large after pruning.")
                if len(pa) == 0:
                    counts = np.bincount(disc[c].to_numpy(), minlength=k_child).astype(float)
                    if eps_per_var > 0:
                        counts += self._lap(eps_per_var, counts.shape, sens=1.0)
                    # Apply smoothing to DP-noisy counts
                    counts = np.maximum(counts, 0.0)
                    counts += self.cpt_smoothing
                    probs = (counts / counts.sum().clip(min=1e-12)).reshape(1, k_child).astype(self.cpt_dtype)
                    self._cpt[c] = {"parents": [], "parent_card": [], "probs": probs}
                    continue
                counts = np.zeros((S, k_child), dtype=float)
                pa_codes = np.stack([disc[p].to_numpy(dtype=np.int64, copy=False) for p in pa], axis=0)
                keys = np.ravel_multi_index(pa_codes, dims=tuple(par_ks), mode="raise")
                child = disc[c].to_numpy()
                np.add.at(counts, (keys, child), 1.0)
                if eps_per_var > 0:
                    counts += self._lap(eps_per_var, counts.shape, sens=1.0)
                row_sums = counts.sum(axis=1, keepdims=True)
                deg = (row_sums <= 1e-12).flatten()
                if np.any(deg):
                    counts[deg, :] = 1.0
                # Apply smoothing to DP-noisy counts
                counts = np.maximum(counts, 0.0)
                counts += self.cpt_smoothing
                probs = (counts / counts.sum(axis=1, keepdims=True).clip(min=1e-12)).astype(self.cpt_dtype)
                self._cpt[c] = {"parents": pa, "parent_card": par_ks, "probs": probs}

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Flatten probability distribution using temperature scaling.
        
        Temperature > 1 makes distribution more uniform, reducing linkage risk.
        Formula: p' = p^(1/T) / Z where Z is normalization constant.
        """
        T = float(self.temperature)
        if not np.isfinite(T) or T <= 0:
            T = 1.0
        if abs(T - 1.0) < 1e-12:
            return probs
        p = np.clip(probs, 1e-12, 1.0) ** (1.0 / T)
        Z = p.sum(axis=1, keepdims=True)
        out = np.divide(p, Z, out=np.full_like(p, 1.0 / p.shape[1]), where=(Z > 0))
        return out

    def _sample_categorical_rows(self, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample categorical values from probability matrix using inverse CDF.
        
        Applies temperature scaling before sampling. Uses cumulative distribution
        function with uniform random draws for each row.
        """
        n, _ = probs.shape
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        probs = self._apply_temperature(probs)
        cdf = np.cumsum(probs, axis=1)
        r = np.minimum(rng.random(n), np.nextafter(1.0, 0.0))
        return (cdf >= r[:, None]).argmax(axis=1).astype(int, copy=False)

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data by ancestral sampling from Bayesian network.
        
        Samples variables in topological order, conditioning on parent values.
        Returns decoded dataframe with original dtypes restored.
        """
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        if not self._cpt or not self._meta or not self._order:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        codes: Dict[str, np.ndarray] = {}
        for c in self._order:
            info = self._cpt[c]
            pa = info["parents"]
            probs = info["probs"]
            if len(pa) == 0:
                row_probs = np.repeat(probs, n, axis=0)
                picks = self._sample_categorical_rows(row_probs, rng)
            else:
                par_ks = info["parent_card"]
                pa_mat = np.stack([codes[p].astype(np.int64, copy=False) for p in pa], axis=0)
                keys = np.ravel_multi_index(pa_mat, dims=tuple(par_ks), mode="raise")
                row_probs = probs[keys]
                picks = self._sample_categorical_rows(row_probs, rng)
            codes[c] = picks
        return self._decode(codes, n, rng)

    def _decode(self, codes: Dict[str, np.ndarray], n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Convert integer codes back to original data types and value ranges.
        
        Numeric: uniform random within bins, scaled to original range. Integers use
        decode mode (round/stochastic/granular). Datetime columns decoded as numeric.
        Categorical: map codes to category strings. Binary numerics threshold at midpoint.
        """
        out: Dict[str, np.ndarray] = {}
        for c in self._order:
            m = self._meta[c]
            z = codes[c]
            if m.kind == "numeric":
                lo, hi = m.bounds if m.bounds is not None else (0.0, 1.0)
                left = m.bins[z]
                right = m.bins[np.minimum(z + 1, m.k)]
                u = rng.random(n)
                val01 = left + (right - left) * u
                val = lo + val01 * (hi - lo)
                # Clip to original data bounds if provided
                # WARNING: original_data_bounds reveals exact data range and is NOT DP-compliant
                # Only use if bounds are public knowledge (e.g., age is always 0-120)
                # For DP compliance, set original_data_bounds=None and let DP bounds handle it
                if c in (self.original_data_bounds or {}):
                    orig_lo, orig_hi = self.original_data_bounds[c]
                    if orig_lo is not None and orig_hi is not None:
                        val = np.clip(val, orig_lo, orig_hi)
                if m.binary_numeric:
                    if getattr(self, "decode_binary_as_bool", False):
                        val = (val >= (lo + (hi - lo) * 0.5))
                    else:
                        val = (val >= (lo + (hi - lo) * 0.5)).astype(int)
                elif m.is_int:
                    mode = self.integer_decode_mode
                    if mode == "stochastic":
                        base = np.floor(val)
                        frac = val - base
                        draw = rng.random(n)
                        val = base + (draw < frac).astype(float)
                    elif mode == "granular":
                        g = int(self.integer_granularity.get(c, 1))
                        g = max(1, g)
                        val = np.round(val / g) * g
                    else:
                        val = np.rint(val)
                    if m.original_dtype is not None:
                        info = np.iinfo(m.original_dtype) if m.original_dtype.kind in ("i", "u") else None
                        if info is not None:
                            val = np.clip(val, info.min, info.max)
                        val = val.astype(m.original_dtype)
                    else:
                        val = val.astype(int)
                else:
                    step = float(self.numeric_granularity.get(c, 0.0)) if hasattr(self, "numeric_granularity") else 0.0
                    if np.isfinite(step) and step > 0:
                        val = np.round(val / step) * step
                    val = val.astype(float)
                out[c] = val
            else:
                # Categorical: keep unknowns as token (no NaNs)
                unk = getattr(self, "unknown_token", "__UNK__")
                cats = m.cats or [unk]
                z = np.clip(z, 0, len(cats) - 1)
                vals = np.array(cats, dtype=object)[z]
                
                # Legacy behavior (off by default)
                if getattr(self, "categorical_unknown_to_nan", False):
                    vals = np.where(vals == unk, np.nan, vals)
                
                out[c] = vals
        return pd.DataFrame(out, columns=self._order)

    @property
    def parents_(self) -> Dict[str, List[str]]:
        """Return learned Bayesian network structure: mapping from child to parent columns."""
        if not self._cpt:
            raise RuntimeError("Model is not fitted.")
        return {c: list(self._cpt[c]["parents"]) for c in self._order}

    def privacy_report(self) -> Dict[str, Any]:
        """Return privacy accounting: epsilon allocation, mechanism type, metadata usage.
        
        Reports actual epsilon consumption across structure learning, CPT estimation,
        and metadata generation. Indicates whether (ε,δ)-DP was used for bounds.
        """
        eps_struct = float(getattr(self, "_eps_struct", 0.0))
        eps_cpt = float(getattr(self, "_eps_cpt", 0.0))
        eps_main = float(getattr(self, "_eps_main", 0.0))
        eps_disc_cfg = float(getattr(self, "eps_disc", 0.0))
        used_bounds = len(self._dp_metadata_used_bounds) > 0
        used_cats = len(self._dp_metadata_used_cats) > 0
        metadata_dp_used = bool(used_bounds or used_cats)
        eps_disc_used = float(eps_disc_cfg if metadata_dp_used else 0.0)
        eps_actual = eps_struct + eps_cpt + eps_disc_used
        mech = "pure"
        delta_used = float(self._dp_metadata_delta_used) if used_bounds and self.dp_bounds_mode == "smooth" else 0.0
        if used_bounds and self.dp_bounds_mode == "smooth":
            mech = "(ε,δ)-DP"
        return {
            "mechanism": mech,
            "epsilon": float(self.epsilon),
            "delta": float(self.delta),
            "eps_main": eps_main,
            "eps_struct": eps_struct,
            "eps_cpt": eps_cpt,
            "n_pairs": int((len(self._order) * (len(self._order) - 1)) // 2),
            "n_vars": int(len(self._order)),
            "eps_disc_configured": eps_disc_cfg,
            "eps_disc_used": eps_disc_used,
            "epsilon_total_configured": float(self.epsilon),
            "epsilon_total_actual": eps_actual,
            "delta_used": delta_used,
            "adjacency": self.adjacency,
            "sensitivity_count": float(self._sens_count),
            "metadata_dp": metadata_dp_used,
            "metadata_mode": ("public" if self.require_public else ("dp_bounds_" + self.dp_bounds_mode)),
            "max_parents": int(self.max_parents),
            "temperature": float(self.temperature),
        }

