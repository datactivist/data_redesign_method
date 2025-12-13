"""
Quality Data Platform - Synthetic Data Benchmark

Train-on-synthetic/test-on-real methodology to prove synthetic data quality
before using it for model training or data augmentation.
"""

import logging
import time
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from intuitiveness.quality.models import (
    SyntheticBenchmarkReport,
    ModelBenchmarkResult,
)

logger = logging.getLogger(__name__)

# Constants
MIN_ROWS_FOR_BENCHMARK = 50
DEFAULT_SYNTHETIC_RATIO = 1.0  # Generate same number of synthetic as real
SAFE_TRANSFER_GAP = 0.10  # 10% gap = safe to use
CAUTION_TRANSFER_GAP = 0.15  # 15% gap = use with caution

# Model configurations
BENCHMARK_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    BENCHMARK_MODELS["XGBoost"] = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
except ImportError:
    logger.info("XGBoost not available, using sklearn models only")


def _prepare_for_benchmark(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare DataFrame for benchmarking by handling missing values and encoding.

    Args:
        df: Input DataFrame.
        target_column: Target column name.

    Returns:
        Tuple of (X, y) ready for training.
    """
    feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                mode_val = X[col].mode()
                X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "missing")

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            X[col] = pd.Categorical(X[col]).codes

    return X, y


def _train_and_evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Train a model and evaluate on test set.

    Returns:
        Dictionary with accuracy, f1, precision, recall.
    """
    try:
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        }
    except Exception as e:
        logger.warning(f"Model training failed: {e}")
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }


def generate_balanced_synthetic(
    df: pd.DataFrame,
    target_column: str,
    samples_per_class: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data with equal samples per class.

    Uses existing synthetic generator with per-class stratification.

    Args:
        df: Original dataset.
        target_column: Name of target column.
        samples_per_class: Samples per class (default: max class size).

    Returns:
        Balanced dataset (original + synthetic samples).
    """
    from intuitiveness.quality.synthetic_generator import generate_synthetic

    classes = df[target_column].unique()
    class_counts = df[target_column].value_counts()

    if samples_per_class is None:
        samples_per_class = class_counts.max()

    balanced_dfs = [df]

    for cls in classes:
        class_df = df[df[target_column] == cls]
        current_count = len(class_df)
        n_to_generate = samples_per_class - current_count

        if n_to_generate > 0:
            try:
                # Generate synthetic samples for this class
                synthetic_df, _ = generate_synthetic(
                    class_df,
                    n_samples=n_to_generate,
                    temperature=1.0,
                )
                balanced_dfs.append(synthetic_df)
                logger.info(f"Generated {n_to_generate} synthetic samples for class '{cls}'")
            except Exception as e:
                logger.warning(f"Failed to generate synthetic for class '{cls}': {e}")

    return pd.concat(balanced_dfs, ignore_index=True)


def generate_targeted_synthetic(
    df: pd.DataFrame,
    target_column: str,
    target_class_value,
    n_samples: int,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate synthetic samples for a specific class only.

    Useful for augmenting rare/minority classes.

    Args:
        df: Original dataset.
        target_column: Name of target column.
        target_class_value: The class value to generate samples for.
        n_samples: Number of synthetic samples to generate.

    Returns:
        Tuple of (synthetic_df, validation_metrics).
    """
    from intuitiveness.quality.synthetic_generator import generate_synthetic

    # Filter to target class only
    class_df = df[df[target_column] == target_class_value]

    if len(class_df) < 10:
        raise ValueError(f"Need at least 10 samples of class '{target_class_value}' for generation")

    # Generate synthetic
    synthetic_df, metrics = generate_synthetic(
        class_df,
        n_samples=n_samples,
        temperature=1.0,
    )

    # Validate distribution similarity
    validation = {
        "n_generated": len(synthetic_df),
        "original_class_count": len(class_df),
        "correlation_preservation": 1 - metrics.mean_correlation_error,
        "distribution_similarity": metrics.distribution_similarity,
    }

    return synthetic_df, validation


def benchmark_synthetic(
    df: pd.DataFrame,
    target_column: str,
    n_synthetic: Optional[int] = None,
    models: Optional[List[str]] = None,
    class_balanced: bool = False,
    dataset_name: str = "dataset",
) -> SyntheticBenchmarkReport:
    """
    Benchmark synthetic data quality by training on synthetic and testing on real data.

    This is the core validation function that PROVES synthetic data works.

    Methodology:
    1. Split real data: 80% train, 20% test (held out)
    2. Generate synthetic data from real_train only
    3. Train models on both real_train and synthetic_train
    4. Evaluate both on the same held-out real_test
    5. Calculate transfer gap (accuracy drop from synthetic training)

    Args:
        df: Original dataset.
        target_column: Name of target column for supervised learning.
        n_synthetic: Number of synthetic samples (default: match original size).
        models: List of model names to benchmark (default: all available).
        class_balanced: Generate balanced synthetic data across classes.
        dataset_name: Name for reporting.

    Returns:
        SyntheticBenchmarkReport with full benchmark results.

    Raises:
        ValueError: If target_column not in DataFrame.
        ValueError: If dataset has fewer than 50 rows.
    """
    from intuitiveness.quality.synthetic_generator import generate_synthetic

    start_time = time.time()

    # Validate inputs
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    if len(df) < MIN_ROWS_FOR_BENCHMARK:
        raise ValueError(f"Dataset has {len(df)} rows, minimum {MIN_ROWS_FOR_BENCHMARK} required")

    # Default parameters
    if n_synthetic is None:
        n_synthetic = len(df)

    if models is None:
        models = list(BENCHMARK_MODELS.keys())

    logger.info(f"Starting benchmark: {len(df)} rows, {n_synthetic} synthetic, models={models}")

    # Step 1: Split real data
    X, y = _prepare_for_benchmark(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    # Reconstruct train DataFrame for synthetic generation
    train_df = pd.concat([X_train, y_train], axis=1)

    # Step 2: Generate synthetic data
    generation_method = "gaussian_copula"
    try:
        if class_balanced:
            synthetic_df = generate_balanced_synthetic(
                train_df, target_column, samples_per_class=n_synthetic // len(y.unique())
            )
        else:
            synthetic_df, metrics = generate_synthetic(
                train_df, n_samples=n_synthetic, temperature=1.0
            )
            if hasattr(metrics, "generation_method"):
                generation_method = metrics.generation_method
    except Exception as e:
        logger.warning(f"Synthetic generation failed: {e}")
        # Return empty report with error
        return SyntheticBenchmarkReport(
            dataset_name=dataset_name,
            target_column=target_column,
            n_synthetic_samples=0,
            recommendation="not_recommended",
            recommendation_reason=f"Synthetic generation failed: {e}",
        )

    X_synthetic, y_synthetic = _prepare_for_benchmark(synthetic_df, target_column)

    # Step 3: Benchmark each model
    model_results = []
    for model_name in models:
        if model_name not in BENCHMARK_MODELS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        logger.info(f"Benchmarking {model_name}...")

        # Real → Real (baseline)
        model_real = type(BENCHMARK_MODELS[model_name])(**BENCHMARK_MODELS[model_name].get_params())
        real_metrics = _train_and_evaluate(model_real, X_train, y_train, X_test, y_test)

        # Synthetic → Real (transfer)
        model_synthetic = type(BENCHMARK_MODELS[model_name])(**BENCHMARK_MODELS[model_name].get_params())
        synthetic_metrics = _train_and_evaluate(model_synthetic, X_synthetic, y_synthetic, X_test, y_test)

        result = ModelBenchmarkResult(
            model_name=model_name,
            real_accuracy=real_metrics["accuracy"],
            real_f1=real_metrics["f1"],
            real_precision=real_metrics["precision"],
            real_recall=real_metrics["recall"],
            synthetic_accuracy=synthetic_metrics["accuracy"],
            synthetic_f1=synthetic_metrics["f1"],
            synthetic_precision=synthetic_metrics["precision"],
            synthetic_recall=synthetic_metrics["recall"],
        )
        model_results.append(result)
        logger.info(f"  {model_name}: real={real_metrics['accuracy']:.3f}, synthetic={synthetic_metrics['accuracy']:.3f}, gap={result.transfer_gap_percent}")

    # Step 4: Calculate aggregate metrics
    if model_results:
        transfer_gaps = [r.transfer_gap for r in model_results]
        mean_gap = np.mean(transfer_gaps)
        max_gap = np.max(transfer_gaps)
        min_gap = np.min(transfer_gaps)
    else:
        mean_gap = max_gap = min_gap = 1.0

    # Step 5: Generate recommendation
    if mean_gap < SAFE_TRANSFER_GAP:
        recommendation = "safe_to_use"
        recommendation_reason = f"Mean transfer gap ({mean_gap:.1%}) is below {SAFE_TRANSFER_GAP:.0%} threshold. Safe for data augmentation."
    elif mean_gap < CAUTION_TRANSFER_GAP:
        recommendation = "use_with_caution"
        recommendation_reason = f"Mean transfer gap ({mean_gap:.1%}) is moderate. Use with caution and validate on your specific use case."
    else:
        recommendation = "not_recommended"
        recommendation_reason = f"Mean transfer gap ({mean_gap:.1%}) is too high. Synthetic data may not preserve important patterns."

    elapsed = time.time() - start_time
    logger.info(f"Benchmark complete in {elapsed:.1f}s: {recommendation}")

    return SyntheticBenchmarkReport(
        dataset_name=dataset_name,
        target_column=target_column,
        timestamp=datetime.now(),
        n_synthetic_samples=len(synthetic_df),
        generation_method=generation_method,
        class_balanced=class_balanced,
        model_results=model_results,
        mean_transfer_gap=mean_gap,
        max_transfer_gap=max_gap,
        min_transfer_gap=min_gap,
        recommendation=recommendation,
        recommendation_reason=recommendation_reason,
    )
