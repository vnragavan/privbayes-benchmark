"""DPMM PrivBayes adapter for benchmarking comparison.

Important (DP compliance):
The upstream `dpmm` package implements a DP PrivBayes-style mechanism for *discrete*
data. Any discretization / category-domain discovery performed *outside* that DP
mechanism must itself be public or differentially private, otherwise the overall
pipeline is NOT DP.

This adapter therefore supports a DP-safe preprocessing mode that:
- releases DP numeric bounds (smooth-sensitivity quantiles), then uses fixed bins
  (post-processing) to discretize numeric columns, and
- requires public categorical domains (or, if `strict_dp=False`, falls back to
  non-DP category inference with a warning).
"""

from typing import Optional
import pandas as pd
import numpy as np
import warnings
from pandas.api.types import is_numeric_dtype


class DPMMPrivBayesAdapter:
    """Adapter to make DPMM's PrivBayes compatible with our benchmark interface."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: Optional[float] = 1e-5,
        degree: int = 2,
        n_bins: int = 50,
        seed: int = 42,
        # dpmm engine knobs (performance/robustness)
        n_iters: int = 5000,
        n_jobs: int = -1,
        compress: bool = True,
        max_model_size: Optional[int] = None,
        # --- DP-safe preprocessing options ---
        preprocess: str = "dp",  # "dp" | "public" | "none"
        eps_disc: Optional[float] = None,
        dp_quantile_alpha: float = 0.01,
        public_bounds: Optional[dict] = None,  # {col: [L, U]} or {"*": [L, U]}
        public_categories: Optional[dict] = None,  # {col: [cat1, cat2, ...]}
        strict_dp: bool = True,
        int_cardinality_as_categorical: int = 20,
        **kwargs
    ):
        """
        Initialize DPMM PrivBayes.
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter (default 1e-5)
            degree: Maximum degree of the Bayesian network (max parents)
            n_bins: Number of bins for discretization (default 50)
            seed: Random seed
            preprocess: Preprocessing mode:
              - "dp": spend eps_disc to make numeric discretization DP-safe (recommended)
              - "public": require public_bounds/public_categories (no DP spend here)
              - "none": legacy behavior (non-DP); kept for backwards compatibility
            eps_disc: Epsilon to spend on DP preprocessing (numeric bounds). If None,
              defaults to min(0.10*epsilon, 0.15*epsilon) similar to Enhanced.
            dp_quantile_alpha: Quantile clipping for smooth DP bounds (default 0.01 => [1%,99%]).
            public_bounds: Optional public coarse bounds.
            public_categories: Optional public categorical domains.
            strict_dp: If True, disallow non-DP fallbacks in preprocessing.
            n_iters: dpmm inference iterations (default 5000; smaller is faster).
            n_jobs: dpmm parallelism (default -1 = many processes).
            compress: dpmm compression flag (default True).
            max_model_size: optional dpmm model size cap.
        """
        try:
            from dpmm.pipelines import PrivBayesPipeline
        except ImportError:
            raise ImportError(
                "DPMM not installed. Install with: pip install dpmm"
            )
        
        self.epsilon = epsilon
        self.delta = delta or 1e-5
        self.degree = degree
        self.n_bins = n_bins if isinstance(n_bins, int) else 50
        self.seed = seed
        self.kwargs = kwargs
        self.preprocess = str(preprocess).lower()
        if self.preprocess not in {"dp", "public", "none"}:
            raise ValueError("preprocess must be one of: dp | public | none")
        self.strict_dp = bool(strict_dp)

        self.public_bounds = dict(public_bounds or {})
        self.public_categories = {k: list(v or []) for k, v in (public_categories or {}).items()}
        self.dp_quantile_alpha = float(dp_quantile_alpha)
        self.int_cardinality_as_categorical = int(int_cardinality_as_categorical)

        # DP preprocessing budget (numeric bounds)
        if eps_disc is None:
            self.eps_disc = float(min(max(0.10 * float(self.epsilon), 1e-6), 0.15 * float(self.epsilon)))
        else:
            self.eps_disc = float(eps_disc)

        # learned preprocess state (for decode)
        self._num_bounds: dict[str, tuple[float, float]] = {}
        self._num_bins: dict[str, np.ndarray] = {}
        self._cat_maps: dict[str, list] = {}
        # For numeric columns treated as categorical, cast back on decode.
        # Values: "int" | "float"
        self._cat_decode_numeric: dict[str, str] = {}
        
        # Initialize DPMM PrivBayes pipeline.
        # DPMM requires integer data, so we disable processing and handle it ourselves.
        # If we do DP preprocessing, we pass the remaining epsilon/delta into dpmm.
        eps_for_dpmm = float(self.epsilon)
        delta_for_dpmm = float(self.delta)
        self._eps_disc_used = 0.0
        self._delta_disc_used = 0.0
        if self.preprocess == "dp":
            # Spend eps_disc on DP bounds; pass remainder to dpmm mechanism.
            self._eps_disc_used = float(min(max(self.eps_disc, 0.0), max(self.epsilon, 0.0)))
            eps_for_dpmm = float(max(self.epsilon - self._eps_disc_used, 1e-12))
            # Allocate delta only to the DP bounds release; pass remainder to dpmm.
            # (If dpmm uses pure-DP internally, this is conservative.)
            self._delta_disc_used = float(min(max(self.delta, 0.0), max(self.delta * 0.5, 1e-15)))
            delta_for_dpmm = float(max(self.delta - self._delta_disc_used, 1e-15))
        elif self.preprocess == "public":
            # No DP spend; requires public metadata for preprocessing.
            pass
        else:
            # legacy non-DP preprocessing
            pass

        self.model = PrivBayesPipeline(
            epsilon=eps_for_dpmm,
            delta=delta_for_dpmm,
            n_jobs=int(n_jobs),
            compress=bool(compress),
            max_model_size=max_model_size,
            disable_processing=True,  # We handle discretization manually
            gen_kwargs={"degree": int(degree), "n_iters": int(n_iters)},
        )
        
        self._fitted = False
        self._real_data = None
        self._discretized_data = None

    def _treat_numeric_as_discrete_category(self, s: pd.Series) -> bool:
        """
        Heuristic: treat small-cardinality / integer-like numeric columns as categorical.

        This is important for label columns like {0,1} where decoding as continuous
        midpoints breaks downstream ML and coverage mapping.
        """
        try:
            s_num = pd.to_numeric(s, errors="coerce")
            s_non = s_num[np.isfinite(s_num)]
            if s_non.empty:
                return False
            nun = int(pd.Series(s_non).nunique())
            if nun <= max(self.int_cardinality_as_categorical, 2):
                # If all values are (near-)integers, treat as discrete category.
                vals = np.asarray(pd.Series(s_non).unique(), dtype=float)
                if np.all(np.isfinite(vals)) and np.all(np.isclose(vals, np.round(vals), atol=1e-8)):
                    return True
        except Exception:
            return False
        return False

    def _numeric_discrete_kind(self, s: pd.Series) -> Optional[str]:
        """If numeric and discrete-like, return 'int' or 'float', else None."""
        try:
            s_num = pd.to_numeric(s, errors="coerce")
            s_non = s_num[np.isfinite(s_num)]
            if s_non.empty:
                return None
            nun = int(pd.Series(s_non).nunique())
            if nun > max(self.int_cardinality_as_categorical, 2):
                return None
            vals = np.asarray(pd.Series(s_non).unique(), dtype=float)
            if np.all(np.isfinite(vals)) and np.all(np.isclose(vals, np.round(vals), atol=1e-8)):
                return "int"
            return "float"
        except Exception:
            return None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit DPMM PrivBayes after discretizing data to integers.
        
        Numeric columns: DP bounds + fixed bins (when preprocess='dp').
        Categorical: requires public categories (or strict_dp=False fallback).
        DPMM requires integer input, so preprocessing happens here.
        """
        self._real_data = X
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Discretize all data to integers (DPMM requirement)
        X_disc = X.copy()
        for col in X_disc.columns:
            if is_numeric_dtype(X_disc[col]):
                # Treat small-cardinality integer-like numerics as categorical.
                # This preserves discrete labels (e.g., 0/1 target) rather than decoding
                # them back as continuous midpoints.
                kind = self._numeric_discrete_kind(X[col])
                if kind is not None:
                    self._cat_decode_numeric[col] = kind
                    X_disc[col] = self._encode_categorical_col(X[col], col)
                else:
                    X_disc[col] = self._discretize_numeric_col(X_disc[col], col)
            else:
                X_disc[col] = self._encode_categorical_col(X_disc[col], col)
        
        self._discretized_data = X_disc
        
        # Fit DPMM model on discretized data
        self.model.fit(X_disc)
        
        self._fitted = True
        return self
    
    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic dataframe (decoded back to approximate original types)."""
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if n_samples is None:
            n_samples = len(self._real_data)
        
        # Generate synthetic data (discretized integers)
        syn_disc = self.model.generate(n_samples)
        return self._decode_synthetic(syn_disc)
    
    def privacy_report(self) -> dict:
        """Return privacy parameters."""
        eps_dpmm = None
        delta_dpmm = None
        try:
            eps_dpmm = float(getattr(getattr(self.model, "gen", None), "epsilon", None))
        except Exception:
            eps_dpmm = None
        try:
            delta_dpmm = float(getattr(getattr(self.model, "gen", None), "delta", None))
        except Exception:
            delta_dpmm = None
        return {
            "epsilon_total_configured": float(self.epsilon),
            "delta_total_configured": float(self.delta),
            "preprocess": self.preprocess,
            "eps_disc_used": float(getattr(self, "_eps_disc_used", 0.0)),
            "delta_disc_used": float(getattr(self, "_delta_disc_used", 0.0)),
            "epsilon_passed_to_dpmm": eps_dpmm,
            "delta_passed_to_dpmm": delta_dpmm,
            "degree": self.degree,
            "n_bins": self.n_bins,
            "implementation": "DPMM PrivBayes",
            "note": "DP compliance depends on dpmm internals + preprocessing mode. Avoid preprocess='none' for DP claims."
        }

    # ---------------------------------------------------------------------
    # DP-safe preprocessing helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _quantile_index(n: int, q: float) -> int:
        q = float(np.clip(q, 0.0, 1.0))
        return int(np.clip(int(np.ceil(q * n)) - 1, 0, max(n - 1, 0)))

    @classmethod
    def _smooth_sensitivity_quantile(
        cls,
        x: np.ndarray,
        q: float,
        eps: float,
        delta: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Approximate smooth sensitivity quantile mechanism (NRS'07 style).
        Used only for numeric bounds when preprocess='dp'.
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        x.sort()
        n = x.size
        i = cls._quantile_index(n, q)
        delta = float(np.clip(delta, 1e-15, 1.0 - 1e-12))
        eps = float(max(eps, 1e-12))
        beta = eps / (2.0 * np.log(1.0 / delta))
        max_s = 0.0
        k_max = min(n - 1, int(np.ceil(4.0 * np.sqrt(n + 1))))
        for k in range(0, k_max + 1):
            l = max(i - k, 0)
            r = min(i + k, n - 1)
            ls = float(x[r] - x[l])
            ss = np.exp(-beta * k) * ls
            if ss > max_s:
                max_s = ss
        scale = (2.0 * max_s) / eps
        return float(x[i] + rng.laplace(0.0, scale))

    def _dp_bounds_for_numeric(self, s: pd.Series, col: str) -> tuple[float, float]:
        # Public coarse bounds can be passed either per-column or via "*" wildcard.
        if col in self.public_bounds and isinstance(self.public_bounds[col], (list, tuple)) and len(self.public_bounds[col]) == 2:
            L, U = self.public_bounds[col]
            if np.isfinite(L) and np.isfinite(U) and float(U) > float(L):
                return float(L), float(U)
        if "*" in self.public_bounds and isinstance(self.public_bounds["*"], (list, tuple)) and len(self.public_bounds["*"]) == 2:
            L, U = self.public_bounds["*"]
            if np.isfinite(L) and np.isfinite(U) and float(U) > float(L):
                return float(L), float(U)

        if self.preprocess != "dp":
            if self.strict_dp:
                raise ValueError(f"Numeric column '{col}' requires public_bounds when preprocess='{self.preprocess}' and strict_dp=True.")
            # Non-DP fallback
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return 0.0, 1.0
            return float(np.min(x)), float(np.max(x))

        # DP bounds (smooth sensitivity quantiles)
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0, 1.0
        # Split eps_disc evenly across numeric columns; delta similarly.
        num_cols = [c for c in self._real_data.columns if is_numeric_dtype(self._real_data[c])]
        m = max(len(num_cols), 1)
        eps_per_col = max(self._eps_disc_used / m, 1e-12)
        delta_per_col = max(self._delta_disc_used / m, 1e-15)
        rng = np.random.default_rng(self.seed)
        eps_each = eps_per_col * 0.5
        delta_each = delta_per_col * 0.5
        qL = self._smooth_sensitivity_quantile(x, self.dp_quantile_alpha, eps_each, delta_each, rng)
        qU = self._smooth_sensitivity_quantile(x, 1.0 - self.dp_quantile_alpha, eps_each, delta_each, rng)
        if not np.isfinite(qU) or qU <= qL:
            span = (np.nanmax(x) - np.nanmin(x) + 1.0) / 100.0
            qU = float(qL + max(span, 1.0))
        return float(qL), float(qU)

    def _discretize_numeric_col(self, s: pd.Series, col: str) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        L, U = self._num_bounds.get(col, (None, None))
        if L is None or U is None:
            L, U = self._dp_bounds_for_numeric(s, col)
            self._num_bounds[col] = (float(L), float(U))
            self._num_bins[col] = np.linspace(0.0, 1.0, int(self.n_bins) + 1)

        bins01 = self._num_bins[col]
        z = (x - L) / max(U - L, 1e-12)
        z = np.where(np.isfinite(z), z, 0.5)
        z = np.clip(z, 0.0, 1.0)
        idx = np.digitize(z, bins01, right=False) - 1
        idx = np.clip(idx, 0, len(bins01) - 2)
        return pd.Series(idx.astype(int), index=s.index)

    def _encode_categorical_col(self, s: pd.Series, col: str) -> pd.Series:
        pub = self.public_categories.get(col, [])
        if pub:
            cats = list(pub)
            self._cat_maps[col] = cats
            cat = pd.Categorical(s.astype("string"), categories=cats, ordered=False)
            codes = np.asarray(cat.codes, dtype=int)
            # unseen -> 0
            codes = np.where(codes < 0, 0, codes)
            return pd.Series(codes, index=s.index)

        if self.preprocess in {"dp", "public"} and self.strict_dp:
            raise ValueError(
                f"Categorical column '{col}' requires public_categories[{col}] for DP claims (strict_dp=True)."
            )

        warnings.warn(
            f"Non-DP categorical domain inference used for column '{col}'. This violates DP unless the domain is public.",
            stacklevel=1,
        )
        cats = pd.Series(s, copy=False).astype("string").dropna().unique().tolist()
        cats = sorted([str(v) for v in cats], key=lambda x: x)
        if not cats:
            cats = ["__UNK__"]
        self._cat_maps[col] = cats
        cat = pd.Categorical(s.astype("string").fillna(cats[0]), categories=cats, ordered=False)
        codes = np.asarray(cat.codes, dtype=int)
        codes = np.where(codes < 0, 0, codes)
        return pd.Series(codes, index=s.index)

    def _decode_synthetic(self, syn_disc: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in syn_disc.columns:
            if col in self._num_bounds:
                L, U = self._num_bounds[col]
                bins01 = self._num_bins[col]
                z = pd.to_numeric(syn_disc[col], errors="coerce").to_numpy(dtype=float)
                z = np.where(np.isfinite(z), z, 0.0).astype(int)
                z = np.clip(z, 0, len(bins01) - 2)
                left = bins01[z]
                right = bins01[np.minimum(z + 1, len(bins01) - 1)]
                mid01 = (left + right) * 0.5
                out[col] = (L + mid01 * (U - L)).astype(float)
            elif col in self._cat_maps:
                cats = self._cat_maps[col]
                z = pd.to_numeric(syn_disc[col], errors="coerce").to_numpy(dtype=float)
                z = np.where(np.isfinite(z), z, 0.0).astype(int)
                z = np.clip(z, 0, len(cats) - 1)
                decoded = np.array(cats, dtype=object)[z]
                # If this column originated from a numeric discrete column, cast back to numeric.
                cast_kind = self._cat_decode_numeric.get(col)
                if cast_kind:
                    ser = pd.to_numeric(pd.Series(decoded), errors="coerce")
                    if cast_kind == "int":
                        out[col] = np.round(ser).astype("Int64")
                    else:
                        out[col] = ser.astype(float)
                else:
                    out[col] = decoded
            else:
                out[col] = syn_disc[col]
        return pd.DataFrame(out, columns=syn_disc.columns)


def build_model(config: dict):
    """Build DPMM PrivBayes model from config (registry interface)."""
    return DPMMPrivBayesAdapter(
        epsilon=config.get("epsilon", 1.0),
        delta=config.get("delta", 1e-5),
        degree=config.get("degree", 2),
        n_bins=config.get("n_bins", 'auto'),
        seed=config.get("random_seed", 42),
        n_iters=config.get("n_iters", 5000),
        n_jobs=config.get("n_jobs", -1),
        compress=config.get("compress", True),
        max_model_size=config.get("max_model_size"),
        preprocess=config.get("preprocess", "dp"),
        eps_disc=config.get("eps_disc"),
        dp_quantile_alpha=config.get("dp_quantile_alpha", 0.01),
        public_bounds=config.get("public_bounds"),
        public_categories=config.get("public_categories"),
        strict_dp=config.get("strict_dp", True),
    )

