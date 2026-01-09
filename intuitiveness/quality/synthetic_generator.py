"""
Quality Data Platform - Synthetic Data Generation

Generates synthetic samples using TabPFN's unsupervised model,
preserving statistical properties and feature dependencies.

Based on: https://docs.priorlabs.ai/capabilities/data-generation
"""

import logging
import time
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

from intuitiveness.quality.models import SyntheticDataMetrics

logger = logging.getLogger(__name__)


def generate_synthetic(
    df: pd.DataFrame,
    n_samples: int,
    temperature: float = 1.0,
    n_permutations: int = 3,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, SyntheticDataMetrics]:
    """
    Generate synthetic data samples using TabPFN's unsupervised model.

    Uses TabPFN's autoregressive generation that models joint probability
    distributions and feature dependencies for high-fidelity synthetic data.

    Args:
        df: Source DataFrame to mimic.
        n_samples: Number of synthetic samples to generate.
        temperature: Controls sampling diversity (default 1.0).
            - Higher values (>1.0) produce more diverse/varied samples.
            - Lower values (<1.0) produce more deterministic samples.
        n_permutations: Number of feature permutations for generation (default 3).
            More permutations provide more robust results but increase time.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (synthetic DataFrame, quality metrics).
    """
    start_time = time.time()

    if random_state is not None:
        np.random.seed(random_state)

    # Store original column info
    original_columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Prepare data for TabPFN (requires numeric data)
    df_encoded, encoders = _encode_categorical_features(df)

    try:
        # Use TabPFN unsupervised model for generation
        synthetic_data = _generate_with_tabpfn(
            df_encoded,
            n_samples=n_samples,
            temperature=temperature,
            n_permutations=n_permutations,
        )
    except Exception as e:
        logger.warning(f"TabPFN generation failed: {e}. Falling back to statistical method.")
        synthetic_data = _generate_fallback(df_encoded, n_samples)

    # Decode categorical features back
    synthetic_df = _decode_categorical_features(
        synthetic_data,
        original_columns,
        categorical_cols,
        encoders,
    )

    # Ensure column order matches original
    synthetic_df = synthetic_df[original_columns]

    # Compute quality metrics
    generation_time = time.time() - start_time
    metrics = validate_synthetic(df, synthetic_df)
    metrics.generation_time_seconds = generation_time

    logger.info(
        f"Generated {n_samples} synthetic samples in {generation_time:.2f}s "
        f"(correlation error: {metrics.mean_correlation_error:.3f})"
    )

    return synthetic_df, metrics


def _encode_categorical_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features to numeric for TabPFN.

    Args:
        df: DataFrame with mixed types.

    Returns:
        Tuple of (encoded DataFrame, encoder mappings).
    """
    df_encoded = df.copy()
    encoders = {}

    for col in df.select_dtypes(exclude=[np.number]).columns:
        # Create label encoding
        unique_values = df[col].dropna().unique()
        value_to_int = {v: i for i, v in enumerate(unique_values)}
        value_to_int[np.nan] = -1  # Handle NaN

        df_encoded[col] = df[col].map(lambda x: value_to_int.get(x, -1))
        encoders[col] = {v: k for k, v in value_to_int.items()}

    return df_encoded, encoders


def _decode_categorical_features(
    synthetic_data: pd.DataFrame,
    original_columns: List[str],
    categorical_cols: List[str],
    encoders: dict,
) -> pd.DataFrame:
    """
    Decode synthetic numeric data back to categorical values.

    Args:
        synthetic_data: Generated numeric DataFrame.
        original_columns: Original column names.
        categorical_cols: List of categorical column names.
        encoders: Encoder mappings from encoding step.

    Returns:
        DataFrame with decoded categorical values.
    """
    synthetic_df = synthetic_data.copy()

    for col in categorical_cols:
        if col in synthetic_df.columns and col in encoders:
            # Round to nearest integer and decode
            synthetic_df[col] = synthetic_df[col].round().astype(int)
            synthetic_df[col] = synthetic_df[col].map(
                lambda x: encoders[col].get(x, encoders[col].get(0, None))
            )

    return synthetic_df


def _generate_with_tabpfn(
    df: pd.DataFrame,
    n_samples: int,
    temperature: float,
    n_permutations: int,
) -> pd.DataFrame:
    """
    Generate synthetic data using TabPFN's unsupervised model.

    Requires HuggingFace authentication for the gated TabPFN v2.5 model.
    Set up authentication by:
    1. Visit https://huggingface.co/Prior-Labs/tabpfn_2_5 and accept terms
    2. Run: huggingface-cli login

    Args:
        df: Numeric DataFrame (categorical already encoded).
        n_samples: Number of samples to generate.
        temperature: Sampling temperature.
        n_permutations: Number of feature permutations.

    Returns:
        DataFrame with synthetic samples.

    Raises:
        Exception: If TabPFN model loading fails (authentication issue).
    """
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
    from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

    # Handle missing values for fitting
    df_clean = df.fillna(df.median())

    # Initialize and fit the unsupervised model
    # Use MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration
    model = TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(device='mps'),
        tabpfn_reg=TabPFNRegressor(device='mps'),
    )

    logger.info(f"Fitting TabPFN unsupervised model on {len(df_clean)} samples...")
    model.fit(df_clean.values)

    # Generate synthetic samples
    logger.info(f"Generating {n_samples} synthetic samples (temp={temperature})...")
    synthetic_tensor = model.generate_synthetic_data(
        n_samples=n_samples,
        t=temperature,
        n_permutations=n_permutations,
    )

    # Convert to DataFrame
    synthetic_array = synthetic_tensor.numpy()
    synthetic_df = pd.DataFrame(synthetic_array, columns=df.columns)

    return synthetic_df


def check_tabpfn_auth() -> Tuple[bool, str]:
    """
    Check if TabPFN/HuggingFace authentication is configured.

    Returns:
        Tuple of (is_authenticated, message).
    """
    import os
    from pathlib import Path

    # Check HuggingFace token
    hf_token_file = Path.home() / '.cache' / 'huggingface' / 'token'
    hf_env = os.environ.get('HF_TOKEN')

    try:
        from huggingface_hub import HfFolder
        hf_token = HfFolder.get_token()
    except Exception:
        hf_token = None

    if hf_token or hf_env or hf_token_file.exists():
        return True, "HuggingFace authentication configured."

    return False, (
        "HuggingFace authentication not configured. "
        "For TabPFN synthetic generation, visit "
        "https://huggingface.co/Prior-Labs/tabpfn_2_5 to accept terms, "
        "then run: huggingface-cli login. "
        "Using statistical fallback instead."
    )


def _generate_fallback(
    df: pd.DataFrame,
    n_samples: int,
) -> pd.DataFrame:
    """
    Fallback generation using Gaussian copula when TabPFN fails.

    Args:
        df: Numeric DataFrame.
        n_samples: Number of samples to generate.

    Returns:
        DataFrame with synthetic samples.
    """
    from scipy import stats as scipy_stats

    df_clean = df.fillna(df.median())

    # Gaussian copula approach
    uniform_data = np.zeros_like(df_clean.values, dtype=float)
    for i, col in enumerate(df_clean.columns):
        values = df_clean[col].values
        ranks = scipy_stats.rankdata(values) / (len(values) + 1)
        uniform_data[:, i] = ranks

    # Transform to normal
    normal_data = scipy_stats.norm.ppf(np.clip(uniform_data, 0.001, 0.999))

    # Compute correlation matrix
    corr_matrix = np.corrcoef(normal_data.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 0.001)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Generate correlated normal samples
    try:
        L = np.linalg.cholesky(corr_matrix)
        z = np.random.standard_normal((n_samples, len(df_clean.columns)))
        correlated_normal = z @ L.T
    except np.linalg.LinAlgError:
        correlated_normal = np.random.standard_normal((n_samples, len(df_clean.columns)))

    # Transform back through inverse CDF
    synthetic = {}
    for i, col in enumerate(df_clean.columns):
        original_values = df_clean[col].values
        uniform_samples = scipy_stats.norm.cdf(correlated_normal[:, i])
        synthetic[col] = np.quantile(original_values, uniform_samples)

    return pd.DataFrame(synthetic)


def validate_synthetic(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> SyntheticDataMetrics:
    """
    Validate synthetic data quality by comparing to original.

    Args:
        original_df: Original DataFrame.
        synthetic_df: Synthetic DataFrame.

    Returns:
        SyntheticDataMetrics with quality scores.
    """
    n_samples = len(synthetic_df)

    # Compute correlation error for numeric columns
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    mean_corr_error = 0.0

    if len(numeric_cols) >= 2:
        try:
            orig_corr = original_df[numeric_cols].corr().values
            synth_corr = synthetic_df[numeric_cols].corr().values

            # Handle NaN
            orig_corr = np.nan_to_num(orig_corr, nan=0.0)
            synth_corr = np.nan_to_num(synth_corr, nan=0.0)

            # Mean absolute difference in upper triangle
            corr_diff = np.abs(orig_corr - synth_corr)
            mean_corr_error = np.mean(corr_diff[np.triu_indices(len(numeric_cols), k=1)])
        except Exception as e:
            logger.warning(f"Correlation comparison failed: {e}")

    # Compute distribution similarity via KS test
    ks_scores = []
    for col in numeric_cols:
        if col in synthetic_df.columns:
            try:
                orig_vals = original_df[col].dropna().values
                synth_vals = synthetic_df[col].dropna().values
                if len(orig_vals) > 0 and len(synth_vals) > 0:
                    ks_stat, _ = stats.ks_2samp(orig_vals, synth_vals)
                    ks_scores.append(1 - ks_stat)  # Convert to similarity
            except Exception:
                pass

    distribution_similarity = np.mean(ks_scores) if ks_scores else 1.0

    return SyntheticDataMetrics(
        n_samples=n_samples,
        mean_correlation_error=float(mean_corr_error),
        distribution_similarity=float(distribution_similarity),
        generation_time_seconds=0.0,  # Will be set by caller
    )


def get_synthetic_summary(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metrics: SyntheticDataMetrics,
) -> str:
    """
    Generate a human-readable summary of synthetic data quality.

    Args:
        original_df: Original DataFrame.
        synthetic_df: Synthetic DataFrame.
        metrics: Quality metrics.

    Returns:
        Formatted summary string.
    """
    lines = [
        "=== SYNTHETIC DATA SUMMARY (TabPFN) ===",
        "",
        f"Generated: {metrics.n_samples} samples",
        f"Generation time: {metrics.generation_time_seconds:.2f}s",
        "",
        "Quality Metrics:",
        f"  Correlation preservation: {(1 - metrics.mean_correlation_error):.1%}",
        f"  Distribution similarity: {metrics.distribution_similarity:.1%}",
        "",
    ]

    # Per-column comparison
    lines.append("Per-Column Statistics:")
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols[:5]:  # Show top 5
        orig_mean = original_df[col].mean()
        synth_mean = synthetic_df[col].mean()
        orig_std = original_df[col].std()
        synth_std = synthetic_df[col].std()

        mean_diff = abs(orig_mean - synth_mean) / (orig_std + 1e-8) * 100
        std_diff = abs(orig_std - synth_std) / (orig_std + 1e-8) * 100

        lines.append(f"  {col}:")
        lines.append(f"    Mean: {orig_mean:.2f} -> {synth_mean:.2f} ({mean_diff:.1f}% diff)")
        lines.append(f"    Std:  {orig_std:.2f} -> {synth_std:.2f} ({std_diff:.1f}% diff)")

    return "\n".join(lines)
