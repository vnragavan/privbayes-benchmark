"""Enhanced PrivBayes adapter for benchmarking comparison."""

from typing import Optional
import pandas as pd
import numpy as np


class EnhancedPrivBayesAdapter:
    """Adapter for Enhanced PrivBayes with QI-linkage reduction features."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: Optional[float] = 1e-6,
        seed: int = 42,
        temperature: float = 1.0,
        cpt_smoothing: float = 1.5,  # DP-safe CPT smoothing (post-processing)
        label_columns: Optional[list] = None,
        public_categories: Optional[dict] = None,
        cat_keep_all_nonzero: bool = True,  # Universal strategy: keep all categories
        **kwargs
    ):
        """
        Initialize Enhanced PrivBayes.
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter (default 1e-6)
            seed: Random seed
            temperature: Temperature for sampling (T>1 reduces linkage, default 1.0)
            label_columns: List of column names to treat as labels (no hashing, no UNK)
            public_categories: Dict mapping column names to their public category lists
        """
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            from external.privbayes_enhanced import PrivBayesSynthesizerEnhanced
        except ImportError:
            raise ImportError(
                "Enhanced PrivBayes not found. Ensure privbayes_enhanced.py is in external/"
            )
        
        self.epsilon = epsilon
        self.delta = delta or 1e-6
        self.seed = seed
        self.temperature = temperature
        self.cpt_smoothing = cpt_smoothing
        self.kwargs = kwargs
        
        # Merge public_categories into kwargs if provided
        model_kwargs = dict(kwargs)
        if public_categories:
            if 'public_categories' in model_kwargs:
                model_kwargs['public_categories'].update(public_categories)
            else:
                model_kwargs['public_categories'] = public_categories
        
        # Add label_columns if provided
        if label_columns:
            model_kwargs['label_columns'] = label_columns
        
        # Initialize Enhanced PrivBayes with universal strategy (keeps all categories)
        model_kwargs['cat_keep_all_nonzero'] = cat_keep_all_nonzero
        self.model = PrivBayesSynthesizerEnhanced(
            epsilon=epsilon,
            delta=self.delta,
            seed=seed,
            temperature=temperature,
            cpt_smoothing=cpt_smoothing,
            **model_kwargs
        )
        
        self._fitted = False
        self._real_data = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit Enhanced PrivBayes model to data.
        
        Handles datetime conversion, numeric coercion, and categorical detection.
        Delegates to PrivBayesSynthesizerEnhanced for actual fitting.
        """
        X2 = X.copy()
        for c in X2.columns:
            # Convert datetime/timedelta to nanoseconds
            if pd.api.types.is_datetime64_any_dtype(X2[c]):
                X2[c] = X2[c].view('int64')
            elif pd.api.types.is_timedelta64_dtype(X2[c]):
                X2[c] = X2[c].view('int64')
            # Try parsing string columns as datetime (common in CSV exports)
            elif X2[c].dtype == 'object':
                try:
                    dt_parsed = pd.to_datetime(X2[c], errors='coerce')
                    if dt_parsed.notna().mean() >= 0.95:
                        X2[c] = dt_parsed.astype('int64')
                        continue
                except (ValueError, TypeError, OverflowError):
                    pass
                # Fall back to numeric coercion
                s = pd.to_numeric(X2[c], errors='coerce')
                if s.notna().mean() >= 0.95:
                    X2[c] = s
        
        self._real_data = X2
        
        # Fit Enhanced PrivBayes model
        self.model.fit(X2)
        
        self._fitted = True
        return self
    
    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic dataframe with same number of rows as real data by default."""
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if n_samples is None:
            n_samples = len(self._real_data)
        
        # Generate synthetic data
        synthetic_df = self.model.sample(n_samples, seed=self.seed)
        
        return synthetic_df
    
    def privacy_report(self) -> dict:
        """Return privacy parameters."""
        report = self.model.privacy_report()
        return {
            "epsilon_total": report.get("epsilon_total_actual", self.epsilon),
            "delta": report.get("delta_used", self.delta),
            "eps_struct": report.get("eps_struct", 0.0),
            "eps_cpt": report.get("eps_cpt", 0.0),
            "eps_disc": report.get("eps_disc_used", 0.0),
            "implementation": "Enhanced PrivBayes",
            "temperature": float(self.temperature),
            "note": "Enhanced PrivBayes with temperature-based sampling and QI-linkage reduction"
        }


def build_model(config: dict):
    """Build Enhanced PrivBayes model from config (registry interface)."""
    return EnhancedPrivBayesAdapter(
        epsilon=config.get("epsilon", 1.0),
        delta=config.get("delta", 1e-6),
        seed=config.get("random_seed", 42),
        temperature=config.get("temperature", 1.0)
    )

