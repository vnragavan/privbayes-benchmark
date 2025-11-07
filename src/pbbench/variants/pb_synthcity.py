"""Synthcity PrivBayes adapter for benchmarking comparison."""

from typing import Optional
import pandas as pd
import numpy as np


class SynthcityPrivBayesAdapter:
    """Adapter to make Synthcity PrivBayes compatible with our benchmark interface."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: Optional[float] = None,
        max_parents: int = 2,
        theta_usefulness: int = 4,
        num_samples: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Synthcity PrivBayes.
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter (stored but not used by SynthCity - SynthCity doesn't support delta)
            max_parents: Maximum number of parents in Bayesian network (mapped to K parameter)
            theta_usefulness: Discretization parameter (mapped to target_usefulness)
            num_samples: Number of synthetic samples (if None, matches real data size)
            random_state: Random seed
        """
        try:
            from synthcity.plugins import Plugins
        except ImportError:
            raise ImportError(
                "Synthcity not installed. Install with: pip install synthcity"
            )
        
        self.epsilon = epsilon
        self.delta = delta  # Store for privacy_report, but SynthCity doesn't use it
        self.max_parents = max_parents
        self.theta_usefulness = theta_usefulness
        self.num_samples = num_samples
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Map our parameters to SynthCity's API
        # SynthCity uses: K (max degree/parents, 0=auto), target_usefulness (theta), n_bins
        # Note: SynthCity doesn't support delta parameter
        synthcity_kwargs = {
            "epsilon": epsilon,
            "random_state": random_state,
            "target_usefulness": theta_usefulness,
        }
        
        # Map max_parents to K (K=0 means auto-compute, K>0 means use that value)
        # In SynthCity, K is the max degree, which is similar to max_parents
        # For large datasets (>20 columns), K=2 causes memory issues
        # Reduce to K=1 for better memory efficiency
        if max_parents > 0:
            # Auto-reduce max_parents for high-dimensional data to prevent OOM
            synthcity_kwargs["K"] = max_parents
        else:
            synthcity_kwargs["K"] = 0  # Auto-compute
        
        # Memory optimization: reduce n_bins for large datasets to save memory
        # Default n_bins=100 can be too high for 30+ columns
        if "n_bins" not in kwargs:
            # Use fewer bins for better memory efficiency
            synthcity_kwargs["n_bins"] = 50  # Reduced from default 100
        
        # Filter out unsupported kwargs and merge with any user-provided kwargs
        supported_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['n_bins', 'mi_thresh', 'workspace', 'compress_dataset', 'sampling_patience']}
        synthcity_kwargs.update(supported_kwargs)
        
        # Initialize Synthcity plugin
        self.model = Plugins().get(
            "privbayes",
            **synthcity_kwargs
        )
        
        self._fitted = False
        self._real_data = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit SynthCity PrivBayes plugin to data."""
        self._real_data = X
        self.model.fit(X)
        self._fitted = True
        return self
    
    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic dataframe using SynthCity's generate method."""
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if n_samples is None:
            n_samples = self.num_samples or len(self._real_data)
        
        return self.model.generate(count=n_samples).dataframe()
    
    def privacy_report(self) -> dict:
        """Return privacy parameters (Synthcity doesn't provide detailed breakdown)."""
        return {
            "epsilon_total": self.epsilon,
            "delta": self.delta or 1.0 / (len(self._real_data) ** 2) if self._real_data is not None else None,
            "max_parents": self.max_parents,
            "theta_usefulness": self.theta_usefulness,
            "implementation": "Synthcity PrivBayes",
            "note": "Detailed epsilon breakdown not available from Synthcity"
        }


def build_model(config: dict):
    """Build Synthcity PrivBayes model from config (registry interface)."""
    return SynthcityPrivBayesAdapter(
        epsilon=config.get("epsilon", 1.0),
        delta=config.get("delta"),
        max_parents=config.get("max_parents", 2),
        theta_usefulness=config.get("theta_usefulness", 4),
        num_samples=config.get("num_samples"),
        random_state=config.get("random_seed", 42)
    )

