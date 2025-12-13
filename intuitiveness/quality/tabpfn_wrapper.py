"""
TabPFN Wrapper with Fallback Logic

Provides a unified interface to TabPFN with automatic fallback:
1. Primary: tabpfn-client (cloud API, no GPU required)
2. Fallback: tabpfn local (requires GPU for optimal speed)

Both versions implement sklearn-compatible interface.
Includes timeout handling for graceful degradation.
"""

import logging
import signal
import functools
from typing import Optional, Literal, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TabPFNTimeoutError(Exception):
    """Raised when TabPFN operation times out."""
    pass


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to a function.

    Uses ThreadPoolExecutor for cross-platform compatibility.

    Args:
        timeout_seconds: Maximum execution time.

    Returns:
        Decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    raise TabPFNTimeoutError(
                        f"{func.__name__} timed out after {timeout_seconds}s"
                    )
        return wrapper
    return decorator

# Backend availability flags
_TABPFN_CLIENT_AVAILABLE = False
_TABPFN_LOCAL_AVAILABLE = False

try:
    from tabpfn_client import TabPFNClassifier as ClientClassifier
    from tabpfn_client import TabPFNRegressor as ClientRegressor
    import tabpfn_client
    _TABPFN_CLIENT_AVAILABLE = True

    # Auto-load token from stored file
    from pathlib import Path
    _token_file = Path.home() / ".tabpfn" / "token"
    if _token_file.exists():
        try:
            _token = _token_file.read_text().strip()
            tabpfn_client.set_access_token(_token)
            logger.info("TabPFN token loaded from ~/.tabpfn/token")
        except Exception as e:
            logger.warning(f"Failed to load TabPFN token: {e}")
except ImportError:
    ClientClassifier = None
    ClientRegressor = None

try:
    from tabpfn import TabPFNClassifier as LocalClassifier
    from tabpfn import TabPFNRegressor as LocalRegressor
    _TABPFN_LOCAL_AVAILABLE = True
except ImportError:
    LocalClassifier = None
    LocalRegressor = None


class TabPFNWrapper:
    """
    Unified TabPFN interface with automatic backend selection.

    Tries tabpfn-client (cloud API) first, falls back to local tabpfn.
    Implements sklearn-compatible fit/predict interface.

    Attributes:
        task_type: "classification" or "regression"
        backend: Which backend is active ("client", "local", or None)
        model: The underlying TabPFN model instance
    """

    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        prefer_local: bool = False,
        timeout: float = 60.0,
    ):
        """
        Initialize TabPFN wrapper.

        Args:
            task_type: Type of prediction task.
            prefer_local: If True, try local TabPFN before cloud API.
            timeout: Timeout in seconds for API calls.
        """
        self.task_type = task_type
        self.prefer_local = prefer_local
        self.timeout = timeout
        self.backend: Optional[str] = None
        self.model: Optional[Any] = None
        self._fitted = False

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the appropriate TabPFN backend."""
        if self.prefer_local:
            backends = [("local", self._init_local), ("client", self._init_client)]
        else:
            backends = [("client", self._init_client), ("local", self._init_local)]

        for name, init_fn in backends:
            try:
                if init_fn():
                    self.backend = name
                    logger.info(f"TabPFN initialized with {name} backend")
                    return
            except Exception as e:
                logger.warning(f"Failed to initialize {name} backend: {e}")

        logger.error("No TabPFN backend available")
        raise RuntimeError(
            "No TabPFN backend available. Install tabpfn-client or tabpfn: "
            "pip install tabpfn-client"
        )

    def _init_client(self) -> bool:
        """Initialize tabpfn-client backend."""
        if not _TABPFN_CLIENT_AVAILABLE:
            return False

        try:
            if self.task_type == "classification":
                self.model = ClientClassifier()
            else:
                self.model = ClientRegressor()
            return True
        except Exception as e:
            logger.warning(f"tabpfn-client initialization failed: {e}")
            return False

    def _init_local(self) -> bool:
        """Initialize local tabpfn backend."""
        if not _TABPFN_LOCAL_AVAILABLE:
            return False

        try:
            if self.task_type == "classification":
                self.model = LocalClassifier()
            else:
                self.model = LocalRegressor()
            return True
        except Exception as e:
            logger.warning(f"Local tabpfn initialization failed: {e}")
            return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNWrapper":
        """
        Fit the TabPFN model with timeout handling.

        Args:
            X: Training features (n_samples, n_features).
            y: Training labels (n_samples,).

        Returns:
            self for method chaining.

        Raises:
            TabPFNTimeoutError: If fitting exceeds timeout.
        """
        if self.model is None:
            raise RuntimeError("TabPFN backend not initialized")

        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        try:
            self._fit_with_timeout(X, y)
            self._fitted = True
        except TabPFNTimeoutError:
            logger.warning(f"TabPFN fit timed out after {self.timeout}s")
            raise

        return self

    def _fit_with_timeout(self, X: np.ndarray, y: np.ndarray) -> None:
        """Internal fit with timeout using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.model.fit, X, y)
            try:
                future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                raise TabPFNTimeoutError(
                    f"TabPFN fit timed out after {self.timeout}s"
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or regression values.

        Args:
            X: Features to predict (n_samples, n_features).

        Returns:
            Predictions (n_samples,).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict (n_samples, n_features).

        Returns:
            Class probabilities (n_samples, n_classes).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute prediction score.

        Args:
            X: Test features.
            y: True labels.

        Returns:
            Accuracy for classification, R² for regression.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        return self.model.score(X, y)


def get_tabpfn_model(
    task_type: Literal["classification", "regression"] = "classification",
    prefer_local: bool = False,
) -> TabPFNWrapper:
    """
    Factory function to get a TabPFN model wrapper.

    Args:
        task_type: Type of prediction task.
        prefer_local: If True, prefer local TabPFN over cloud API.

    Returns:
        Configured TabPFNWrapper instance.
    """
    return TabPFNWrapper(task_type=task_type, prefer_local=prefer_local)


def is_tabpfn_available() -> Tuple[bool, str]:
    """
    Check if any TabPFN backend is available.

    Returns:
        Tuple of (available, backend_name).
    """
    if _TABPFN_CLIENT_AVAILABLE:
        return True, "client"
    if _TABPFN_LOCAL_AVAILABLE:
        return True, "local"
    return False, "none"
