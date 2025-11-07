"""DPMM PrivBayes adapter for benchmarking comparison."""

from typing import Optional
import pandas as pd
import numpy as np


class DPMMPrivBayesAdapter:
    """Adapter to make DPMM's PrivBayes compatible with our benchmark interface."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: Optional[float] = 1e-5,
        degree: int = 2,
        n_bins: int = 50,
        seed: int = 42,
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
        
        # Initialize DPMM PrivBayes pipeline
        # DPMM requires integer data, so we disable processing and handle it ourselves
        self.model = PrivBayesPipeline(
            epsilon=epsilon,
            delta=self.delta,
            disable_processing=True,  # We handle discretization manually
            gen_kwargs={'degree': degree}
        )
        
        self._fitted = False
        self._real_data = None
        self._discretized_data = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit DPMM PrivBayes after discretizing data to integers.
        
        Numeric columns: quantile-based binning. Categorical: integer codes.
        DPMM requires integer input, so preprocessing happens here.
        """
        self._real_data = X
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Discretize all data to integers (DPMM requirement)
        X_disc = X.copy()
        for col in X_disc.columns:
            if X_disc[col].dtype == 'float64':
                # Use quantile-based binning for numeric columns
                X_disc[col] = pd.qcut(X_disc[col], q=self.n_bins, labels=False, duplicates='drop')
            elif X_disc[col].dtype == 'object':
                # Convert categorical to integer codes
                X_disc[col] = pd.Categorical(X_disc[col]).codes
            # int64 columns are already ok
        
        self._discretized_data = X_disc
        
        # Fit DPMM model on discretized data
        self.model.fit(X_disc)
        
        self._fitted = True
        return self
    
    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic dataframe (returns discretized integer codes)."""
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if n_samples is None:
            n_samples = len(self._real_data)
        
        # Generate synthetic data (will be discretized integers)
        synthetic_df = self.model.generate(n_samples)
        
        return synthetic_df
    
    def privacy_report(self) -> dict:
        """Return privacy parameters."""
        return {
            "epsilon_total": self.epsilon,
            "delta": self.delta,
            "degree": self.degree,
            "n_bins": self.n_bins,
            "implementation": "DPMM PrivBayes",
            "note": "Uses Bayesian network with exponential mechanism; data is discretized"
        }


def build_model(config: dict):
    """Build DPMM PrivBayes model from config (registry interface)."""
    return DPMMPrivBayesAdapter(
        epsilon=config.get("epsilon", 1.0),
        delta=config.get("delta", 1e-5),
        degree=config.get("degree", 2),
        n_bins=config.get("n_bins", 'auto'),
        seed=config.get("random_seed", 42)
    )

