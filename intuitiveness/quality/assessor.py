"""
Quality Data Platform - Dataset Assessor

TabPFN-based quality assessment with usability scoring, feature importance,
and task type detection.
"""

import logging
import time
from typing import Optional, Literal, List, Tuple, Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from intuitiveness.quality.models import (
    QualityReport,
    FeatureProfile,
    FeatureSuggestion,
    TransformationResult,
    TransformationLog,
    ReadinessIndicator,
)
from intuitiveness.quality.tabpfn_wrapper import TabPFNWrapper, is_tabpfn_available

logger = logging.getLogger(__name__)

# Constants
MIN_ROWS_FOR_ASSESSMENT = 50
MAX_ROWS_FOR_TABPFN = 10000
MAX_FEATURES_FOR_TABPFN = 500
HIGH_CARDINALITY_THRESHOLD = 100  # Categorical with >100 unique values
SAMPLE_SIZE = 5000  # Default sample size for large datasets


class DatasetWarning:
    """Container for dataset warnings during assessment."""

    def __init__(self):
        self.warnings: List[str] = []

    def add(self, message: str):
        self.warnings.append(message)
        logger.warning(message)

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


def detect_task_type(y: pd.Series) -> Literal["classification", "regression"]:
    """
    Auto-detect whether the target column is for classification or regression.

    Args:
        y: Target column series.

    Returns:
        "classification" or "regression"
    """
    # Check if target is categorical or has few unique values
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"

    n_unique = y.nunique()
    n_total = len(y)

    # If less than 20 unique values or less than 5% of total, treat as classification
    if n_unique <= 20 or (n_unique / n_total) < 0.05:
        return "classification"

    return "regression"


def detect_feature_type(
    series: pd.Series,
) -> Literal["numeric", "categorical", "boolean", "datetime"]:
    """
    Detect the type of a feature column.

    Args:
        series: Feature column series.

    Returns:
        Feature type string.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's really categorical (few unique values)
        if series.nunique() <= 10:
            return "categorical"
        return "numeric"
    return "categorical"


def compute_feature_profile(
    df: pd.DataFrame,
    feature_name: str,
    importance_score: float = 0.0,
    shap_mean: float = 0.0,
) -> FeatureProfile:
    """
    Compute statistics for a single feature.

    Args:
        df: DataFrame containing the feature.
        feature_name: Column name.
        importance_score: Pre-computed importance score.
        shap_mean: Pre-computed mean SHAP value.

    Returns:
        FeatureProfile instance.
    """
    series = df[feature_name]
    feature_type = detect_feature_type(series)

    missing_count = series.isna().sum()
    missing_ratio = missing_count / len(series)
    unique_count = series.nunique()

    # Compute skewness for numeric features
    distribution_skew = 0.0
    suggested_transform = None
    if feature_type == "numeric":
        clean_series = series.dropna()
        if len(clean_series) > 0:
            try:
                distribution_skew = float(stats.skew(clean_series))
                # Suggest log transform for highly skewed distributions
                if abs(distribution_skew) > 2:
                    suggested_transform = "log"
                elif abs(distribution_skew) > 1:
                    suggested_transform = "sqrt"
            except Exception:
                pass

    return FeatureProfile(
        feature_name=feature_name,
        feature_type=feature_type,
        missing_count=int(missing_count),
        missing_ratio=float(missing_ratio),
        unique_count=int(unique_count),
        importance_score=importance_score,
        shap_mean=shap_mean,
        distribution_skew=distribution_skew,
        suggested_transform=suggested_transform,
    )


def compute_data_completeness(df: pd.DataFrame) -> float:
    """
    Compute data completeness score (0-100).

    Score = (1 - overall_missing_ratio) * 100

    Args:
        df: DataFrame to assess.

    Returns:
        Completeness score 0-100.
    """
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0
    missing_cells = df.isna().sum().sum()
    return (1 - missing_cells / total_cells) * 100


def compute_feature_diversity(df: pd.DataFrame, target_column: str) -> float:
    """
    Compute feature type diversity score (0-100).

    Based on entropy of feature type distribution.
    Higher diversity = better for ML (different types capture different aspects).

    Args:
        df: DataFrame to assess.
        target_column: Target column to exclude.

    Returns:
        Diversity score 0-100.
    """
    feature_columns = [c for c in df.columns if c != target_column]
    if not feature_columns:
        return 0.0

    # Count feature types
    type_counts = {"numeric": 0, "categorical": 0, "boolean": 0, "datetime": 0}
    for col in feature_columns:
        ftype = detect_feature_type(df[col])
        type_counts[ftype] += 1

    # Compute entropy
    total = sum(type_counts.values())
    if total == 0:
        return 0.0

    probs = [count / total for count in type_counts.values() if count > 0]
    if len(probs) <= 1:
        return 25.0  # Only one type = low diversity

    entropy = -sum(p * np.log2(p) for p in probs)
    max_entropy = np.log2(len(probs))  # Maximum possible entropy

    # Normalize to 0-100
    return (entropy / max_entropy) * 100 if max_entropy > 0 else 0.0


def compute_size_appropriateness(row_count: int) -> float:
    """
    Compute size appropriateness score (0-100).

    TabPFN works best with 50-10,000 rows.
    Score penalizes datasets outside this range.

    Args:
        row_count: Number of rows in dataset.

    Returns:
        Size score 0-100.
    """
    if row_count < MIN_ROWS_FOR_ASSESSMENT:
        # Linear penalty for too few rows
        return max(0, (row_count / MIN_ROWS_FOR_ASSESSMENT) * 50)
    elif row_count <= MAX_ROWS_FOR_TABPFN:
        # Optimal range
        return 100.0
    else:
        # Gradual penalty for large datasets (still usable via sampling)
        # Drops to 70 at 50k rows, 50 at 100k rows
        excess = row_count - MAX_ROWS_FOR_TABPFN
        penalty = min(50, excess / 2000)  # Max 50 point penalty
        return 100 - penalty


def compute_usability_score(
    prediction_quality: float,
    data_completeness: float,
    feature_diversity: float,
    size_appropriateness: float,
) -> float:
    """
    Compute composite usability score (0-100).

    Formula: 40% prediction + 30% completeness + 20% diversity + 10% size

    Args:
        prediction_quality: TabPFN cross-validation score (0-100).
        data_completeness: Missing value score (0-100).
        feature_diversity: Feature type entropy (0-100).
        size_appropriateness: Size penalty score (0-100).

    Returns:
        Usability score 0-100.
    """
    return (
        0.4 * prediction_quality
        + 0.3 * data_completeness
        + 0.2 * feature_diversity
        + 0.1 * size_appropriateness
    )


def handle_high_cardinality_categorical(
    series: pd.Series,
    threshold: int = HIGH_CARDINALITY_THRESHOLD,
) -> pd.Series:
    """
    Handle high-cardinality categorical features by binning rare values.

    Args:
        series: Categorical series.
        threshold: Maximum number of unique values to keep.

    Returns:
        Series with rare values grouped as 'other'.
    """
    value_counts = series.value_counts()

    if len(value_counts) <= threshold:
        return series

    # Keep top N-1 values, group rest as 'other'
    top_values = set(value_counts.head(threshold - 1).index)

    return series.apply(lambda x: x if x in top_values else "_other_")


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = MAX_FEATURES_FOR_TABPFN,
) -> pd.DataFrame:
    """
    Select top features when dataset has too many.

    Uses variance-based selection for efficiency.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        max_features: Maximum number of features to keep.

    Returns:
        DataFrame with selected features.
    """
    if X.shape[1] <= max_features:
        return X

    logger.info(f"Selecting top {max_features} features from {X.shape[1]}")

    # Use variance as simple feature importance proxy
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(max_features).index.tolist()

    return X[top_features]


def check_dataset_edge_cases(
    df: pd.DataFrame,
    target_column: str,
) -> DatasetWarning:
    """
    Check for edge cases and generate warnings.

    Args:
        df: DataFrame to check.
        target_column: Target column name.

    Returns:
        DatasetWarning with any warnings.
    """
    warnings = DatasetWarning()

    # Check row count
    if len(df) < MIN_ROWS_FOR_ASSESSMENT:
        warnings.add(
            f"Dataset has only {len(df)} rows. "
            f"Minimum {MIN_ROWS_FOR_ASSESSMENT} recommended for reliable assessment."
        )

    # Check feature count
    feature_count = len(df.columns) - 1  # Exclude target
    if feature_count > MAX_FEATURES_FOR_TABPFN:
        warnings.add(
            f"Dataset has {feature_count} features. "
            f"Top {MAX_FEATURES_FOR_TABPFN} will be selected for assessment."
        )

    # Check for only-categorical or only-numeric
    feature_cols = [c for c in df.columns if c != target_column]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) == 0 and len(categorical_cols) > 0:
        warnings.add(
            "Dataset contains only categorical features. "
            "Consider adding numeric features for better ML performance."
        )
    elif len(categorical_cols) == 0 and len(numeric_cols) > 0:
        warnings.add(
            "Dataset contains only numeric features. "
            "This is fine but diversity score will be lower."
        )

    # Check for high-cardinality categoricals
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > HIGH_CARDINALITY_THRESHOLD:
            warnings.add(
                f"Feature '{col}' has {n_unique} unique values (high cardinality). "
                f"Rare values will be grouped for encoding."
            )

    return warnings


def prepare_data_for_tabpfn(
    df: pd.DataFrame,
    target_column: str,
    warnings: Optional[DatasetWarning] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare DataFrame for TabPFN by handling missing values and encoding.

    Handles edge cases:
    - High-cardinality categoricals (>100 unique values)
    - Too many features (>500)
    - Only categorical or only numeric datasets

    Args:
        df: Input DataFrame.
        target_column: Target column name.
        warnings: Optional DatasetWarning to collect warnings.

    Returns:
        Tuple of (X, y) ready for TabPFN.
    """
    # Separate features and target
    feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    # Handle missing values in features
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "missing")

    # Handle high-cardinality categoricals
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            if X[col].nunique() > HIGH_CARDINALITY_THRESHOLD:
                X[col] = handle_high_cardinality_categorical(X[col])

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            # Simple label encoding
            X[col] = pd.Categorical(X[col]).codes

    # Handle too many features
    if X.shape[1] > MAX_FEATURES_FOR_TABPFN:
        X = select_top_features(X, y)

    return X, y


def compute_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: Literal["classification", "regression"],
    n_folds: int = 3,
) -> List[Tuple[str, float]]:
    """
    Compute feature importance via ablation study.

    For each feature, measure prediction drop when it's removed.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        task_type: Classification or regression.
        n_folds: Number of cross-validation folds.

    Returns:
        List of (feature_name, importance_score) tuples.
    """
    available, _ = is_tabpfn_available()
    if not available:
        # Return equal importance if TabPFN not available
        return [(col, 1.0 / len(X.columns)) for col in X.columns]

    try:
        # Get baseline score
        wrapper = TabPFNWrapper(task_type=task_type)

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        baseline_scores = cross_val_score(wrapper.model, X.values, y.values, cv=cv)
        baseline = np.mean(baseline_scores)

        # Compute importance for each feature
        importance_list = []
        for col in X.columns:
            X_ablated = X.drop(columns=[col])
            if X_ablated.shape[1] == 0:
                importance_list.append((col, 1.0))
                continue

            try:
                ablated_wrapper = TabPFNWrapper(task_type=task_type)
                ablated_scores = cross_val_score(
                    ablated_wrapper.model, X_ablated.values, y.values, cv=cv
                )
                ablated = np.mean(ablated_scores)
                # Importance = how much performance drops when feature is removed
                importance = max(0, baseline - ablated)
                importance_list.append((col, importance))
            except Exception:
                importance_list.append((col, 0.0))

        # Normalize to sum to 1
        total_importance = sum(imp for _, imp in importance_list)
        if total_importance > 0:
            importance_list = [
                (name, imp / total_importance) for name, imp in importance_list
            ]
        else:
            importance_list = [
                (name, 1.0 / len(importance_list)) for name, _ in importance_list
            ]

        return importance_list

    except Exception as e:
        logger.warning(f"Feature importance computation failed: {e}")
        return [(col, 1.0 / len(X.columns)) for col in X.columns]


def compute_shap_values(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: Literal["classification", "regression"],
    max_samples: int = 100,
) -> List[Tuple[str, float]]:
    """
    Compute mean absolute SHAP values for each feature.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        task_type: Classification or regression.
        max_samples: Maximum samples for SHAP computation.

    Returns:
        List of (feature_name, mean_shap) tuples.
    """
    try:
        import shap

        available, _ = is_tabpfn_available()
        if not available:
            return [(col, 0.0) for col in X.columns]

        # Sample data if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
        else:
            X_sample = X
            y_sample = y

        # Fit model
        wrapper = TabPFNWrapper(task_type=task_type)
        wrapper.fit(X_sample.values, y_sample.values)

        # Create SHAP explainer
        # Use a background sample for faster computation
        background = X_sample.values[:min(50, len(X_sample))]

        if task_type == "classification":
            explainer = shap.KernelExplainer(
                wrapper.model.predict_proba, background
            )
        else:
            explainer = shap.KernelExplainer(
                wrapper.model.predict, background
            )

        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample.values[:min(20, len(X_sample))])

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values = np.abs(shap_values)

        # Compute mean absolute SHAP per feature
        mean_shap = np.mean(shap_values, axis=0)

        return list(zip(X.columns, mean_shap))

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return [(col, 0.0) for col in X.columns]


def assess_dataset(
    df: pd.DataFrame,
    target_column: str,
    task_type: Literal["classification", "regression", "auto"] = "auto",
    compute_shap: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    timeout_seconds: int = 300,
) -> QualityReport:
    """
    Run quality assessment on a dataset.

    Handles edge cases:
    - Datasets with <50 rows (warning, not error)
    - Datasets with >500 features (auto feature selection)
    - Only categorical or only numeric datasets
    - High-cardinality categoricals (>100 unique values)

    Args:
        df: DataFrame to assess.
        target_column: Column to use as prediction target.
        task_type: Task type or "auto" for auto-detection.
        compute_shap: Whether to compute SHAP values (slower but more interpretable).
        progress_callback: Optional callback for progress updates (message, progress 0-1).
        timeout_seconds: Maximum time for TabPFN operations (default 300s).

    Returns:
        QualityReport with all assessment results.

    Raises:
        ValueError: If target column doesn't exist.
    """
    start_time = time.time()

    def report_progress(message: str, progress: float):
        if progress_callback:
            progress_callback(message, progress)
        logger.info(f"Assessment progress: {message} ({progress:.0%})")

    # Validate inputs
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Check edge cases and collect warnings
    dataset_warnings = check_dataset_edge_cases(df, target_column)

    original_row_count = len(df)
    sampled = False
    sample_size = None

    # Handle large datasets via sampling
    if original_row_count > MAX_ROWS_FOR_TABPFN:
        sample_size = SAMPLE_SIZE
        df = df.sample(n=sample_size, random_state=42)
        sampled = True
        logger.info(f"Sampled {sample_size} rows from {original_row_count} for assessment")

    # Handle small datasets gracefully (warning instead of error)
    if len(df) < MIN_ROWS_FOR_ASSESSMENT:
        logger.warning(
            f"Dataset has {len(df)} rows, below minimum {MIN_ROWS_FOR_ASSESSMENT}. "
            "Results may be less reliable."
        )

    report_progress("Preparing data", 0.1)

    # Auto-detect task type
    if task_type == "auto":
        task_type = detect_task_type(df[target_column])
    logger.info(f"Task type: {task_type}")

    # Prepare data
    X, y = prepare_data_for_tabpfn(df, target_column)
    feature_columns = list(X.columns)

    report_progress("Computing basic metrics", 0.2)

    # Compute basic metrics
    data_completeness = compute_data_completeness(df)
    feature_diversity = compute_feature_diversity(df, target_column)
    size_appropriateness = compute_size_appropriateness(original_row_count)

    report_progress("Computing prediction quality", 0.3)

    # Compute prediction quality via TabPFN cross-validation
    prediction_quality = 0.0
    available, backend = is_tabpfn_available()
    if available:
        try:
            wrapper = TabPFNWrapper(task_type=task_type)

            if task_type == "classification":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)

            scores = cross_val_score(wrapper.model, X.values, y.values, cv=cv)
            prediction_quality = np.mean(scores) * 100  # Convert to 0-100
            logger.info(f"Prediction quality: {prediction_quality:.1f}% (backend: {backend})")
        except Exception as e:
            logger.warning(f"TabPFN scoring failed: {e}")
            prediction_quality = 50.0  # Default to middle score
    else:
        logger.warning("TabPFN not available, using default prediction quality")
        prediction_quality = 50.0

    report_progress("Computing feature importance", 0.5)

    # Compute feature importance
    importance_list = compute_feature_importance(X, y, task_type)
    importance_dict = dict(importance_list)

    report_progress("Computing SHAP values", 0.7)

    # Compute SHAP values if requested
    shap_dict = {}
    if compute_shap:
        shap_list = compute_shap_values(X, y, task_type)
        shap_dict = dict(shap_list)

    report_progress("Building feature profiles", 0.9)

    # Build feature profiles
    feature_profiles = []
    for col in feature_columns:
        profile = compute_feature_profile(
            df,
            col,
            importance_score=importance_dict.get(col, 0.0),
            shap_mean=shap_dict.get(col, 0.0),
        )
        feature_profiles.append(profile)

    # Compute usability score
    usability_score = compute_usability_score(
        prediction_quality,
        data_completeness,
        feature_diversity,
        size_appropriateness,
    )

    assessment_time = time.time() - start_time
    report_progress("Assessment complete", 1.0)

    return QualityReport(
        usability_score=usability_score,
        prediction_quality=prediction_quality,
        data_completeness=data_completeness,
        feature_diversity=feature_diversity,
        size_appropriateness=size_appropriateness,
        target_column=target_column,
        task_type=task_type,
        feature_profiles=feature_profiles,
        assessment_time_seconds=assessment_time,
        row_count=original_row_count,
        feature_count=len(feature_columns),
        sampled=sampled,
        sample_size=sample_size,
    )


# ============================================================================
# NEW FUNCTIONS FOR 010-quality-ds-workflow
# Data Scientist Co-Pilot Feature
# ============================================================================


def quick_benchmark(
    df: pd.DataFrame,
    target_column: str,
    n_folds: int = 3,
) -> float:
    """
    Quick accuracy benchmark using RandomForest.

    Used for tracking accuracy before/after transformations.

    Args:
        df: DataFrame to evaluate.
        target_column: Target column name.
        n_folds: Number of cross-validation folds.

    Returns:
        Mean accuracy score (0-1).
    """
    from sklearn.ensemble import RandomForestClassifier

    X, y = prepare_data_for_tabpfn(df, target_column)

    if len(X) < 20:
        return 0.5  # Not enough data

    try:
        model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)

        if len(y.unique()) > 1:
            cv = StratifiedKFold(n_splits=min(n_folds, len(y.unique())), shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        scores = cross_val_score(model, X.values, y.values, cv=cv)
        return float(np.mean(scores))
    except Exception as e:
        logger.warning(f"Quick benchmark failed: {e}")
        return 0.5


def apply_all_suggestions(
    df: pd.DataFrame,
    suggestions: List[FeatureSuggestion],
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, TransformationLog]:
    """
    Apply all suggestions with accuracy tracking.

    This is the "one-click fix" function that applies all high-confidence
    transformations and tracks the impact of each change.

    Args:
        df: Dataset to transform.
        suggestions: List of suggestions to apply.
        target_column: Target column for accuracy tracking (optional).

    Returns:
        Tuple of (transformed_df, TransformationLog).
    """
    from intuitiveness.quality.feature_engineer import apply_suggestion
    from datetime import datetime

    results = []
    current_df = df.copy()
    original_shape = (len(df), len(df.columns))

    # Get initial accuracy if target column specified
    initial_accuracy = None
    if target_column and target_column in df.columns:
        initial_accuracy = quick_benchmark(df, target_column)

    baseline_accuracy = initial_accuracy

    # Sort by confidence (highest first) for more reliable transformations first
    sorted_suggestions = sorted(suggestions, key=lambda s: s.confidence, reverse=True)

    for suggestion in sorted_suggestions:
        accuracy_before = baseline_accuracy

        try:
            new_df = apply_suggestion(current_df, suggestion)

            # Calculate accuracy after
            accuracy_after = None
            if target_column and target_column in new_df.columns:
                accuracy_after = quick_benchmark(new_df, target_column)

            result = TransformationResult(
                suggestion_type=suggestion.suggestion_type,
                target_features=suggestion.target_features,
                description=suggestion.description,
                applied_at=datetime.now(),
                success=True,
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
            )

            current_df = new_df
            baseline_accuracy = accuracy_after

            logger.info(
                f"Applied {suggestion.suggestion_type} on {suggestion.target_features}: "
                f"accuracy {result.accuracy_delta_percent or 'N/A'}"
            )

        except Exception as e:
            result = TransformationResult(
                suggestion_type=suggestion.suggestion_type,
                target_features=suggestion.target_features,
                description=suggestion.description,
                applied_at=datetime.now(),
                success=False,
                error=str(e),
                accuracy_before=accuracy_before,
            )
            logger.warning(f"Failed to apply suggestion: {e}")

        results.append(result)

    # Calculate final accuracy
    final_accuracy = None
    if target_column and target_column in current_df.columns:
        final_accuracy = quick_benchmark(current_df, target_column)

    log = TransformationLog(
        dataset_name="",  # Will be set by caller
        original_shape=original_shape,
        final_shape=(len(current_df), len(current_df.columns)),
        results=results,
        total_applied=sum(1 for r in results if r.success),
        total_failed=sum(1 for r in results if not r.success),
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
    )

    return current_df, log


def get_readiness_indicator(
    score: float,
    n_suggestions: int = 0,
    estimated_improvement: float = 0.0,
) -> ReadinessIndicator:
    """
    Get traffic light readiness indicator from usability score.

    Provides instant go/no-go visual for data scientists.

    Thresholds:
    - 80+: Green (READY) - Export and start training
    - 60-79: Yellow (FIXABLE) - N automated fixes will help
    - <60: Red (NEEDS WORK) - Significant issues

    Args:
        score: Usability score (0-100).
        n_suggestions: Number of available suggestions.
        estimated_improvement: Estimated score improvement if all suggestions applied.

    Returns:
        ReadinessIndicator with status, color, and messaging.
    """
    return ReadinessIndicator.from_score(
        score=score,
        n_suggestions=n_suggestions,
        estimated_improvement=estimated_improvement,
    )
