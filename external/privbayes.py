from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_bool_dtype

SMOOTH: float = 1e-8  # additive smoothing for probabilities


# ---- Minimal "register" shim (replace with your project's registry if you have one) ----
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
    """Stable hash → bucket id in [0, m-1]."""
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=16)
    return int.from_bytes(h.digest(), "little") % int(m)


def _quantile_indices(n: int, q: float) -> int:
    """Order-statistic index for quantile q (0..1), 0-based."""
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

    Calibration:
      β ≈ ε / (2 ln(1/δ)) (optionally scaled by beta_scale).
      Local sensitivity proxy: LS_k ≈ x[i+k] − x[i−k]; smooth bound S* = max_k e^{-βk} LS_k.
      Laplace scale = 2 S* / ε (pure post-processing after noise).
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
    return y  # no private clamping here


def _dp_numeric_bounds_public(
    col: pd.Series,
    eps_min: float,
    eps_max: float,
    coarse_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Pure ε-DP: with public coarse [L,U], add Laplace noise to min/max of data clipped to [L,U].
    Sensitivity = (U-L) for min and for max.
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
    # Clamp returned bounds to public coarse bounds (safe)
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
    (ε,δ)-DP bounds via smooth-sensitivity quantiles:
      L = q_alpha^DP, U = q_(1-alpha)^DP
    We split eps/delta equally between the two quantiles.
    If public_coarse is provided, clamp DP quantiles to it (safe).
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
        # fall back to a tiny positive span within a safe public interval if available
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


@register("model", "privbayes")
class PrivBayesSynthesizer:
    """
    Differentially Private PrivBayes with DP metadata.

    Key features:
      • DP structure: pairwise DP counts → MI ranking (up to K parents).
      • DP CPTs: Laplace noise with per-variable ε.
      • DP metadata:
          - Numeric bounds: "public" (pure ε) or "smooth" ((ε,δ)-DP) quantile bounds.
          - Categorical domain: public list OR DP hash-bucket heavy hitters (bounded domain).
      • Unbounded adjacency ("add/remove") by default: sensitivity for count cells = 1.
      • Tunable via `auto_tune_for_epsilon(...)` or manual kwargs.

    Parameters (subset):
      epsilon: total privacy budget
      delta  : used only if dp_bounds_mode="smooth"
      eps_split: {"structure": s, "cpt": c} for the main budget
      eps_disc: budget reserved for metadata (bounds/domains)
      max_parents: K
      bins_per_numeric: discretization granularity
      dp_bounds_mode: "public" or "smooth"
      public_bounds: per-column or {"*": [L, U]} coarse bounds for "public" mode
      public_categories: dict of known categorical domains (optional)
      cat_buckets, cat_topk: DP hash-bucket heavy hitters when category domain is private
      adjacency: "unbounded" (add/remove) or "bounded" (replace-one)
    """

    def __init__(
        self,
        *,
        epsilon: float,
        delta: float = 1e-6,
        seed: int = 0,
        # tuning / privacy split
        eps_split: Optional[Dict[str, float]] = None,   # {"structure": 0.3, "cpt": 0.7}
        eps_disc: Optional[float] = None,               # DP metadata budget (default set below)
        max_parents: int = 2,
        bins_per_numeric: int = 16,
        adjacency: str = "unbounded",
        # DP metadata strategy
        dp_bounds_mode: str = "smooth",                 # default to smooth (ε,δ)-DP bounds
        dp_quantile_alpha: float = 0.01,                # for "smooth" mode bounds
        public_bounds: Optional[Dict[str, List[float]]] = None,
        public_categories: Optional[Dict[str, List[str]]] = None,
        public_binary_numeric: Optional[Dict[str, bool]] = None,
        # DP heavy hitters for categoricals when domain private
        cat_buckets: int = 64,
        cat_topk: int = 28,
        # decoding
        decode_binary_as_bool: bool = False,
        cpt_dtype: str = "float64",
        # misc
        require_public: bool = False,  # if True, metadata must be public (no DP metadata)
        strict_dp: bool = True,        # if True, disallow non-DP fallbacks
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

        # DP heavy hitters for categoricals
        self.cat_buckets = int(cat_buckets)
        self.cat_topk = int(cat_topk)

        # main knobs
        self.max_parents = int(max_parents)
        self.bins_per_numeric = int(bins_per_numeric)

        # Default DP metadata budget: 10% of epsilon (cap at 0.15)
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

        # learned state
        self._meta: Dict[str, _ColMeta] = {}
        self._order: List[str] = []
        self._cpt: Dict[str, Dict[str, Any]] = {}

        # book-keeping
        self._dp_metadata_used_bounds: set[str] = set()
        self._dp_metadata_used_cats: set[str] = set()
        self._dp_metadata_delta_used: float = 0.0  # track δ spent by smooth bounds
        self._all_nan_columns: int = 0  # internal only; not exposed

    # ------------------ primitives ------------------

    def _lap(self, eps: float, shape: Any, *, sens: Optional[float] = None) -> np.ndarray:
        base_sens = float(self._sens_count) if sens is None else float(sens)
        scale = base_sens / max(float(eps), 1e-12)
        return self._rng.laplace(0.0, scale, size=shape)

    # ------------------ metadata build ------------------

    def _build_meta(self, df: pd.DataFrame) -> None:
        self._dp_metadata_used_bounds.clear()
        self._dp_metadata_used_cats.clear()
        self._dp_metadata_delta_used = 0.0
        self._all_nan_columns = 0

        pb = dict(self.public_bounds or {})
        pc = {k: list(v or []) for k, v in (self.public_categories or {}).items()}
        pbn = dict(self.public_binary_numeric or {})

        m_cols_need_bounds: List[str] = []
        m_cols_need_cats: List[str] = []

        # Identify columns needing DP metadata if allowed
        if (self.eps_disc > 0.0) and (not self.require_public):
            for c in df.columns:
                if is_numeric_dtype(df[c]) and c not in pb:
                    m_cols_need_bounds.append(c)
                elif (not is_numeric_dtype(df[c])) and not pc.get(c):
                    m_cols_need_cats.append(c)

        m_total = len(m_cols_need_bounds) + len(m_cols_need_cats)

        # If DP metadata is required by policy but impossible (eps_disc==0), block.
        if not self.require_public and self.strict_dp and m_total > 0 and self.eps_disc <= 0.0:
            raise ValueError(
                "DP metadata required by default, but eps_disc=0. "
                "Either provide public bounds/categories or set a positive eps_disc."
            )

        eps_disc_per_col = (self.eps_disc / m_total) if m_total > 0 else 0.0

        # Determine which numeric columns will use 'smooth' bounds and split δ across them
        smooth_cols: List[str] = []
        if not self.require_public and eps_disc_per_col > 0.0 and self.dp_bounds_mode == "smooth":
            for c in m_cols_need_bounds:
                # Candidate for smooth (no public bounds provided for c)
                smooth_cols.append(c)

        n_smooth = len(smooth_cols)
        # Distribute the global delta across smooth columns (0 means no smooth usage)
        delta_per_smooth_col = (self.delta / n_smooth) if n_smooth > 0 else 0.0

        meta: Dict[str, _ColMeta] = {}

        # Resolve numeric/categorical metadata
        for c in df.columns:
            is_bool = is_bool_dtype(df[c])
            is_num = is_numeric_dtype(df[c])

            # Auto public for booleans if require_public=True
            if self.require_public and is_bool and c not in pb:
                pb[c] = [0.0, 1.0]
                pbn[c] = True

            # Numeric path
            if is_num or is_bool or (c in pb):
                raw = pd.to_numeric(df[c], errors="coerce").to_numpy()
                all_nan_flag = False

                if self.require_public and is_num and not is_bool and c not in pb:
                    raise ValueError(f"Numeric column {c} requires public bounds when require_public=True.")

                # Determine bounds
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
                    # need to derive bounds
                    if self.require_public:
                        raise ValueError(f"Column {c} missing public bounds while require_public=True.")
                    if eps_disc_per_col > 0.0:
                        # Optionally look for public coarse fallback
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
                            # smooth sensitivity (ε,δ)-DP; clamp to public coarse if available
                            L, U = _dp_numeric_bounds_smooth(
                                pd.Series(raw),
                                eps_total=eps_disc_per_col,
                                delta_total=max(delta_per_smooth_col, 1e-15),
                                alpha=self.dp_quantile_alpha,
                                rng=self._rng,
                                public_coarse=coarse
                            )
                            self._dp_metadata_used_bounds.add(c)
                            # Each column consumes exactly its per-column δ (0 if n_smooth == 0)
                            self._dp_metadata_delta_used += float(max(delta_per_smooth_col, 1e-15)) if n_smooth > 0 else 0.0
                        pb[c] = [float(L), float(U)]
                    else:
                        # Non-DP fallback only if strict_dp is False
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

                # Integer round-trip preservation
                is_int = is_integer_dtype(df[c])
                if is_int:
                    try:
                        dt = df[c].to_numpy(copy=False).dtype
                    except Exception:
                        dt = np.dtype("int64")
                    original_dtype = dt
                else:
                    original_dtype = None

                # Binary detection (public hint or data-driven if public not required)
                if self.require_public:
                    binary_numeric = bool(self.public_binary_numeric.get(c, False))
                else:
                    vals = pd.to_numeric(df[c], errors="coerce")
                    u = pd.unique(vals.dropna())
                    try:
                        binary_numeric = len(u) <= 2 and set([0.0, 1.0]).issuperset(set(pd.Series(u).astype(float)))
                    except Exception:
                        binary_numeric = False

                # Discretization bins (uniform in [0,1] after DP bounds)
                if binary_numeric or is_bool:
                    k = 2
                    bins = np.array([0.0, 0.5, 1.0], dtype=float)
                else:
                    k = max(2, int(self.bins_per_numeric))
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
                )

            # Categorical path
            else:
                pub = list(self.public_categories.get(c, []) or [])
                if self.require_public:
                    cats = (["__UNK__"] if "__UNK__" not in pub else []) + [x for x in pub if x != "__UNK__"]
                    if not cats:
                        cats = ["__UNK__"]
                else:
                    if pub:
                        cats = (["__UNK__"] if "__UNK__" not in pub else []) + [x for x in pub if x != "__UNK__"]
                    elif eps_disc_per_col > 0.0:
                        # DP hash-bucket heavy hitters (bounded m buckets)
                        ser = pd.Series(df[c], copy=False).astype("string")
                        m = max(8, int(self.cat_buckets))
                        buckets = ser.fillna("__UNK__").map(lambda v: f"B{_blake_bucket(str(v), m):03d}")
                        counts = buckets.value_counts(dropna=False).to_dict()
                        eps_col = max(eps_disc_per_col, 1e-12)
                        # Explicit sens=1.0 per bucket count under add/remove adjacency
                        noisy = {b: (float(cnt) + float(self._lap(eps_col, (), sens=1.0))) for b, cnt in counts.items()}
                        order = sorted(noisy.keys(), key=lambda t: noisy[t], reverse=True)
                        K = max(8, int(min(self.cat_topk, len(order))))
                        topk = order[:K] if K > 0 else []
                        cats = ["__UNK__"] + topk
                        self._dp_metadata_used_cats.add(c)
                    else:
                        if self.strict_dp:
                            raise ValueError(
                                f"DP categorical discovery required for '{c}' but eps_disc_per_col=0 under strict_dp."
                            )
                        warnings.warn(
                            "Non-DP categorical discovery used due to eps_disc=0 and strict_dp=False.",
                            stacklevel=1
                        )
                        vals = pd.Series(df[c], copy=False).astype("string").dropna().unique().tolist()
                        cats = (["__UNK__"] if "__UNK__" not in pub else []) + [x for x in vals if x != "__UNK__"]

                meta[c] = _ColMeta(kind="categorical", k=len(cats), cats=cats)
                self.public_categories[c] = cats  # record resolved domain

        # Note: We do not fold back eps_disc if unused; it simply remains unused.

        self._meta = meta
        self.public_bounds = pb
        self.public_categories = self.public_categories  # keep reference

    # ------------------ discretization ------------------

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
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
                cats = list(m.cats or [])
                if "__UNK__" not in cats:
                    cats = ["__UNK__"] + [x for x in cats if x != "__UNK__"]
                    m.cats = cats
                    self.public_categories[c] = cats
                col = df[c].astype("string").fillna("__UNK__")
                cat = pd.Categorical(col, categories=cats, ordered=False)
                codes = np.asarray(cat.codes, dtype=int)
                codes = np.where(codes < 0, 0, codes)  # unseen → __UNK__
                out[c] = codes
        return pd.DataFrame(out, index=df.index)

    # ------------------ fit / learn ------------------

    def fit(self, df: pd.DataFrame, schema: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        kw = dict(cfg.get("kwargs", {}))

        # runtime overrides
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

        # Build metadata (DP or public as configured)
        self._build_meta(df)
        disc = self._discretize(df)

        cols = list(disc.columns)
        self._order = cols[:]

        # DP structure via pairwise DP counts → MI cache
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
                    # sens=1 per cell under add/remove adjacency
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

            # Parent selection is pure post-processing of DP scores
            for j, c in enumerate(cols):
                cand = cols[:j]
                if not cand:
                    continue
                scores = [(dp_mi_scores.get((c, p), 0.0), p) for p in cand]
                scores.sort(key=lambda t: (-t[0], t[1]))
                parents[c] = [p for _, p in scores[: self.max_parents]]

        # DP CPTs with per-variable ε
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
                counts = np.maximum(counts, 0.0) + SMOOTH
                probs = (counts / counts.sum()).reshape(1, k_child).astype(self.cpt_dtype)
                self._cpt[c] = {"parents": [], "parent_card": [], "probs": probs}
            else:
                par_ks = [self._meta[p].k for p in pa]
                S = int(np.prod(par_ks, dtype=object))
                # guard against blow-up
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
                    counts = np.maximum(counts, 0.0) + SMOOTH
                    probs = (counts / counts.sum()).reshape(1, k_child).astype(self.cpt_dtype)
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
                counts = np.maximum(counts, 0.0) + SMOOTH
                probs = (counts / counts.sum(axis=1, keepdims=True).clip(min=1e-12)).astype(self.cpt_dtype)
                self._cpt[c] = {"parents": pa, "parent_card": par_ks, "probs": probs}

    # ------------------ sampling ------------------

    def _sample_categorical_rows(self, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, _ = probs.shape
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        r = np.minimum(rng.random(n), np.nextafter(1.0, 0.0))
        return (cdf >= r[:, None]).argmax(axis=1).astype(int, copy=False)

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
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

    # ------------------ decoding ------------------

    def _decode(self, codes: Dict[str, np.ndarray], n: int, rng: np.random.Generator) -> pd.DataFrame:
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
                if m.binary_numeric:
                    if getattr(self, "decode_binary_as_bool", False):
                        val = (val >= (lo + (hi - lo) * 0.5))
                    else:
                        val = (val >= (lo + (hi - lo) * 0.5)).astype(int)
                elif m.is_int and m.original_dtype is not None:
                    val = np.rint(val)
                    info = np.iinfo(m.original_dtype) if m.original_dtype.kind in ("i", "u") else None
                    if info is not None:
                        val = np.clip(val, info.min, info.max)
                    val = val.astype(m.original_dtype)
                elif m.is_int:
                    val = np.rint(val).astype(int)
                else:
                    val = val.astype(float)
                out[c] = val
            else:
                cats = m.cats or ["__UNK__"]
                z = np.clip(z, 0, len(cats) - 1)
                vals = np.array(cats, dtype=object)[z]
                vals = np.where(vals == "__UNK__", np.nan, vals)
                out[c] = vals
        return pd.DataFrame(out, columns=self._order)

    # ------------------ debugging / report ------------------

    @property
    def parents_(self) -> Dict[str, List[str]]:
        if not self._cpt:
            raise RuntimeError("Model is not fitted.")
        return {c: list(self._cpt[c]["parents"]) for c in self._order}

    def privacy_report(self) -> Dict[str, Any]:
        eps_struct = float(getattr(self, "_eps_struct", 0.0))
        eps_cpt = float(getattr(self, "_eps_cpt", 0.0))
        eps_main = float(getattr(self, "_eps_main", 0.0))
        eps_disc_cfg = float(getattr(self, "eps_disc", 0.0))
        used_bounds = len(self._dp_metadata_used_bounds) > 0
        used_cats = len(self._dp_metadata_used_cats) > 0
        metadata_dp_used = bool(used_bounds or used_cats)

        # Count eps_disc only if DP metadata was actually used
        eps_disc_used = float(eps_disc_cfg if metadata_dp_used else 0.0)
        eps_actual = eps_struct + eps_cpt + eps_disc_used

        # Mechanism labeling
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
            "delta_used": delta_used,   # exact δ consumed by smooth bounds (per-column split)
            "adjacency": self.adjacency,
            "sensitivity_count": float(self._sens_count),
            "metadata_dp": metadata_dp_used,
            "metadata_mode": ("public" if self.require_public else ("dp_bounds_" + self.dp_bounds_mode)),
            "max_parents": int(self.max_parents),
        }
