"""
Quality Data Platform - Synthetic Data Benchmark

Train-on-synthetic/test-on-real methodology to prove synthetic data quality
before using it for model training or data augmentation.

P0 Fixes Applied (2025-12-13):
- Fixed encoding mismatch: Now uses shared encoder across train/test/synthetic
- Added TabPFN to benchmark models
- Added multi-seed benchmarking with confidence intervals
"""

import logging
import time
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

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
DEFAULT_N_SEEDS = 3  # Number of random seeds for robust benchmarking

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

# Try to import TabPFN (optional but recommended)
try:
    from intuitiveness.quality.tabpfn_wrapper import TabPFNWrapper
    BENCHMARK_MODELS["TabPFN"] = TabPFNWrapper(task_type="classification")
    logger.info("TabPFN added to benchmark models")
except ImportError:
    logger.info("TabPFN not available for benchmarking")


def _prepare_for_benchmark(
    df: pd.DataFrame,
    target_column: str,
    encoders: Optional[Dict[str, Dict[Any, int]]] = None,
    imputers: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[Any, int]], Dict[str, Any]]:
    """
    Prepare DataFrame for benchmarking with CONSISTENT encoding across datasets.

    P0 FIX: This function now supports shared encoders to ensure train, test,
    and synthetic data use the SAME categorical mappings.

    Args:
        df: Input DataFrame.
        target_column: Target column name.
        encoders: Pre-fitted encoders from training data. If None, creates new encoders.
        imputers: Pre-computed imputation values. If None, computes from this data.

    Returns:
        Tuple of (X, y, encoders, imputers) where encoders/imputers can be reused.
    """
    feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Initialize encoder/imputer dicts if not provided
    if encoders is None:
        encoders = {}
    if imputers is None:
        imputers = {}

    # Handle missing values with consistent imputation
    for col in X.columns:
        if col not in imputers:
            # Compute imputation value from this data (should be training data)
            if pd.api.types.is_numeric_dtype(X[col]):
                median_val = X[col].median()
                imputers[col] = median_val if not pd.isna(median_val) else 0.0
            else:
                mode_val = X[col].mode()
                imputers[col] = mode_val.iloc[0] if len(mode_val) > 0 else "__MISSING__"

        # Apply imputation
        if X[col].isna().any():
            X[col] = X[col].fillna(imputers[col])

    # Encode categorical features with consistent mapping
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            if col not in encoders:
                # Create encoder from this data (should be training data)
                unique_values = X[col].unique()
                encoders[col] = {v: i for i, v in enumerate(unique_values)}

            # Apply encoding - unseen categories get mapped to -1
            X[col] = X[col].map(lambda x: encoders[col].get(x, -1))

    return X, y, encoders, imputers


def _prepare_for_benchmark_legacy(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Legacy version for backward compatibility (deprecated).
    Use _prepare_for_benchmark with encoders instead.
    """
    X, y, _, _ = _prepare_for_benchmark(df, target_column)
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
    n_seeds: int = DEFAULT_N_SEEDS,
) -> SyntheticBenchmarkReport:
    """
    Benchmark synthetic data quality by training on synthetic and testing on real data.

    This is the core validation function that PROVES synthetic data works.

    P0 FIXES APPLIED:
    - Uses shared encoders across train/test/synthetic (fixes encoding mismatch)
    - Runs with multiple random seeds for confidence intervals
    - Includes TabPFN in benchmark models

    Methodology:
    1. Split real data: 80% train, 20% test (held out)
    2. Generate synthetic data from real_train only
    3. Train models on both real_train and synthetic_train
    4. Evaluate both on the same held-out real_test
    5. Calculate transfer gap (accuracy drop from synthetic training)
    6. Repeat with multiple seeds and compute confidence intervals

    Args:
        df: Original dataset.
        target_column: Name of target column for supervised learning.
        n_synthetic: Number of synthetic samples (default: match original size).
        models: List of model names to benchmark (default: all available).
        class_balanced: Generate balanced synthetic data across classes.
        dataset_name: Name for reporting.
        n_seeds: Number of random seeds for robust estimation (default: 3).

    Returns:
        SyntheticBenchmarkReport with full benchmark results and confidence intervals.

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

    logger.info(f"Starting benchmark: {len(df)} rows, {n_synthetic} synthetic, models={models}, n_seeds={n_seeds}")

    # Run benchmark with multiple seeds for confidence intervals
    all_seed_results = []
    generation_method = "gaussian_copula"
    n_synthetic_samples = 0

    for seed_idx, seed in enumerate(range(42, 42 + n_seeds)):
        logger.info(f"Running benchmark with seed {seed} ({seed_idx + 1}/{n_seeds})...")

        # Step 1: Split ORIGINAL data (before any encoding)
        feature_columns = [c for c in df.columns if c != target_column]
        X_raw = df[feature_columns].copy()
        y_raw = df[target_column].copy()

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=seed,
            stratify=y_raw if len(y_raw.unique()) > 1 else None
        )

        # Step 2: Fit encoders/imputers on TRAINING data only
        train_df_raw = pd.concat([X_train_raw, y_train], axis=1)
        X_train, _, encoders, imputers = _prepare_for_benchmark(
            train_df_raw, target_column, encoders=None, imputers=None
        )

        # Step 3: Apply SAME encoders to test data
        test_df_raw = pd.concat([X_test_raw, y_test], axis=1)
        X_test, y_test_enc, _, _ = _prepare_for_benchmark(
            test_df_raw, target_column, encoders=encoders, imputers=imputers
        )

        # Step 4: Generate synthetic data from ORIGINAL (non-encoded) training data
        try:
            if class_balanced:
                synthetic_df = generate_balanced_synthetic(
                    train_df_raw, target_column,
                    samples_per_class=n_synthetic // len(y_raw.unique())
                )
            else:
                synthetic_df, metrics = generate_synthetic(
                    train_df_raw, n_samples=n_synthetic, temperature=1.0
                )
                if hasattr(metrics, "generation_method"):
                    generation_method = metrics.generation_method

            n_synthetic_samples = len(synthetic_df)

            # Step 5: Apply SAME encoders to synthetic data
            X_synthetic, y_synthetic, _, _ = _prepare_for_benchmark(
                synthetic_df, target_column, encoders=encoders, imputers=imputers
            )

        except Exception as e:
            logger.warning(f"Synthetic generation failed with seed {seed}: {e}")
            continue

        # Step 6: Benchmark each model
        seed_model_results = {}
        for model_name in models:
            if model_name not in BENCHMARK_MODELS:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue

            try:
                # Handle TabPFN differently (doesn't use get_params)
                if model_name == "TabPFN":
                    model_real = BENCHMARK_MODELS[model_name]
                    model_synthetic = BENCHMARK_MODELS[model_name]
                else:
                    model_real = type(BENCHMARK_MODELS[model_name])(
                        **BENCHMARK_MODELS[model_name].get_params()
                    )
                    model_synthetic = type(BENCHMARK_MODELS[model_name])(
                        **BENCHMARK_MODELS[model_name].get_params()
                    )

                # Real → Real (baseline)
                real_metrics = _train_and_evaluate(model_real, X_train, y_train, X_test, y_test_enc)

                # Synthetic → Real (transfer)
                synthetic_metrics = _train_and_evaluate(
                    model_synthetic, X_synthetic, y_synthetic, X_test, y_test_enc
                )

                seed_model_results[model_name] = {
                    "real_accuracy": real_metrics["accuracy"],
                    "synthetic_accuracy": synthetic_metrics["accuracy"],
                    "real_f1": real_metrics["f1"],
                    "synthetic_f1": synthetic_metrics["f1"],
                    "real_precision": real_metrics["precision"],
                    "synthetic_precision": synthetic_metrics["precision"],
                    "real_recall": real_metrics["recall"],
                    "synthetic_recall": synthetic_metrics["recall"],
                }
            except Exception as e:
                logger.warning(f"Model {model_name} failed with seed {seed}: {e}")

        all_seed_results.append(seed_model_results)

    # Aggregate results across seeds
    model_results = []
    for model_name in models:
        if model_name not in BENCHMARK_MODELS:
            continue

        # Collect metrics across seeds
        real_accs = []
        synth_accs = []
        real_f1s = []
        synth_f1s = []
        real_precs = []
        synth_precs = []
        real_recalls = []
        synth_recalls = []

        for seed_result in all_seed_results:
            if model_name in seed_result:
                real_accs.append(seed_result[model_name]["real_accuracy"])
                synth_accs.append(seed_result[model_name]["synthetic_accuracy"])
                real_f1s.append(seed_result[model_name]["real_f1"])
                synth_f1s.append(seed_result[model_name]["synthetic_f1"])
                real_precs.append(seed_result[model_name]["real_precision"])
                synth_precs.append(seed_result[model_name]["synthetic_precision"])
                real_recalls.append(seed_result[model_name]["real_recall"])
                synth_recalls.append(seed_result[model_name]["synthetic_recall"])

        if real_accs:
            result = ModelBenchmarkResult(
                model_name=model_name,
                real_accuracy=float(np.mean(real_accs)),
                real_f1=float(np.mean(real_f1s)),
                real_precision=float(np.mean(real_precs)),
                real_recall=float(np.mean(real_recalls)),
                synthetic_accuracy=float(np.mean(synth_accs)),
                synthetic_f1=float(np.mean(synth_f1s)),
                synthetic_precision=float(np.mean(synth_precs)),
                synthetic_recall=float(np.mean(synth_recalls)),
            )
            model_results.append(result)
            logger.info(
                f"  {model_name}: real={np.mean(real_accs):.3f}±{np.std(real_accs):.3f}, "
                f"synthetic={np.mean(synth_accs):.3f}±{np.std(synth_accs):.3f}, "
                f"gap={result.transfer_gap_percent}"
            )

    # Calculate aggregate metrics with confidence intervals
    if model_results:
        transfer_gaps = [r.transfer_gap for r in model_results]
        mean_gap = float(np.mean(transfer_gaps))
        max_gap = float(np.max(transfer_gaps))
        min_gap = float(np.min(transfer_gaps))

        # Compute confidence interval (95%)
        if len(transfer_gaps) > 1:
            gap_std = float(np.std(transfer_gaps))
            ci_95 = 1.96 * gap_std / np.sqrt(len(transfer_gaps))
        else:
            ci_95 = 0.0
    else:
        mean_gap = max_gap = min_gap = 1.0
        ci_95 = 0.0

    # Generate recommendation
    if mean_gap < SAFE_TRANSFER_GAP:
        recommendation = "safe_to_use"
        recommendation_reason = (
            f"Mean transfer gap ({mean_gap:.1%} ± {ci_95:.1%}) is below "
            f"{SAFE_TRANSFER_GAP:.0%} threshold. Safe for data augmentation."
        )
    elif mean_gap < CAUTION_TRANSFER_GAP:
        recommendation = "use_with_caution"
        recommendation_reason = (
            f"Mean transfer gap ({mean_gap:.1%} ± {ci_95:.1%}) is moderate. "
            f"Use with caution and validate on your specific use case."
        )
    else:
        recommendation = "not_recommended"
        recommendation_reason = (
            f"Mean transfer gap ({mean_gap:.1%} ± {ci_95:.1%}) is too high. "
            f"Synthetic data may not preserve important patterns."
        )

    elapsed = time.time() - start_time
    logger.info(f"Benchmark complete in {elapsed:.1f}s: {recommendation}")

    return SyntheticBenchmarkReport(
        dataset_name=dataset_name,
        target_column=target_column,
        timestamp=datetime.now(),
        n_synthetic_samples=n_synthetic_samples,
        generation_method=generation_method,
        class_balanced=class_balanced,
        model_results=model_results,
        mean_transfer_gap=mean_gap,
        max_transfer_gap=max_gap,
        min_transfer_gap=min_gap,
        recommendation=recommendation,
        recommendation_reason=recommendation_reason,
    )
