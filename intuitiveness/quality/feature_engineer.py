"""
Quality Data Platform - Feature Engineering Suggestions

Generates actionable feature engineering recommendations based on
TabPFN ablation studies and statistical analysis.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from intuitiveness.quality.models import (
    QualityReport,
    FeatureProfile,
    FeatureSuggestion,
)

logger = logging.getLogger(__name__)

# Thresholds for suggestions
LOW_IMPORTANCE_THRESHOLD = 0.05  # Features below this are candidates for removal
HIGH_SKEW_THRESHOLD = 2.0  # Skewness above this suggests log transform
MODERATE_SKEW_THRESHOLD = 1.0  # Skewness above this suggests sqrt transform
HIGH_CORRELATION_THRESHOLD = 0.8  # Correlations above this suggest combining
HIGH_MISSING_THRESHOLD = 0.3  # Features with >30% missing need attention


def suggest_features(
    report: QualityReport,
    df: Optional[pd.DataFrame] = None,
    max_suggestions: int = 10,
) -> List[FeatureSuggestion]:
    """
    Generate feature engineering suggestions based on quality report.

    Args:
        report: QualityReport from assess_dataset().
        df: Optional DataFrame for correlation analysis.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of FeatureSuggestion objects, sorted by expected impact.
    """
    suggestions: List[FeatureSuggestion] = []

    # 1. Suggest removing low-importance features
    suggestions.extend(_suggest_removals(report))

    # 2. Suggest transforms for skewed distributions
    suggestions.extend(_suggest_transforms(report))

    # 3. Suggest combining highly correlated features
    if df is not None:
        suggestions.extend(_suggest_combinations(report, df))

    # 4. Suggest handling high-missing features
    suggestions.extend(_suggest_missing_handling(report))

    # Sort by expected impact (descending) and limit
    suggestions.sort(key=lambda s: s.expected_impact, reverse=True)
    return suggestions[:max_suggestions]


def _suggest_removals(report: QualityReport) -> List[FeatureSuggestion]:
    """Suggest removing low-importance features."""
    suggestions = []

    for fp in report.feature_profiles:
        if fp.importance_score < LOW_IMPORTANCE_THRESHOLD:
            # Estimate impact: removing noise can improve slightly
            expected_impact = min(3.0, (LOW_IMPORTANCE_THRESHOLD - fp.importance_score) * 20)

            suggestions.append(FeatureSuggestion(
                suggestion_type="remove",
                target_features=[fp.feature_name],
                description=(
                    f"Consider removing '{fp.feature_name}' - it has very low predictive "
                    f"importance ({fp.importance_score:.3f}) and may add noise."
                ),
                expected_impact=expected_impact,
                confidence=0.7 + (LOW_IMPORTANCE_THRESHOLD - fp.importance_score) * 3,
            ))

    return suggestions


def _suggest_transforms(report: QualityReport) -> List[FeatureSuggestion]:
    """Suggest transforms for skewed distributions."""
    suggestions = []

    for fp in report.feature_profiles:
        if fp.feature_type != "numeric":
            continue

        if abs(fp.distribution_skew) > HIGH_SKEW_THRESHOLD:
            transform = "log" if fp.distribution_skew > 0 else "square"
            expected_impact = min(5.0, abs(fp.distribution_skew) * 0.5) * fp.importance_score * 10

            suggestions.append(FeatureSuggestion(
                suggestion_type="transform",
                target_features=[fp.feature_name],
                description=(
                    f"Apply {transform} transform to '{fp.feature_name}' - "
                    f"it has high skewness ({fp.distribution_skew:.2f}) which may hurt model performance."
                ),
                expected_impact=expected_impact,
                confidence=0.75,
            ))

        elif abs(fp.distribution_skew) > MODERATE_SKEW_THRESHOLD:
            transform = "sqrt" if fp.distribution_skew > 0 else "square"
            expected_impact = min(3.0, abs(fp.distribution_skew) * 0.3) * fp.importance_score * 10

            suggestions.append(FeatureSuggestion(
                suggestion_type="transform",
                target_features=[fp.feature_name],
                description=(
                    f"Consider {transform} transform for '{fp.feature_name}' - "
                    f"it has moderate skewness ({fp.distribution_skew:.2f})."
                ),
                expected_impact=expected_impact,
                confidence=0.6,
            ))

    return suggestions


def _suggest_combinations(
    report: QualityReport,
    df: pd.DataFrame,
) -> List[FeatureSuggestion]:
    """Suggest combining highly correlated features."""
    suggestions = []

    # Get numeric features
    numeric_features = [
        fp.feature_name for fp in report.feature_profiles
        if fp.feature_type == "numeric" and fp.feature_name in df.columns
    ]

    if len(numeric_features) < 2:
        return suggestions

    # Compute correlations
    try:
        corr_matrix = df[numeric_features].corr().abs()

        # Find highly correlated pairs
        for i, f1 in enumerate(numeric_features):
            for f2 in numeric_features[i + 1:]:
                corr = corr_matrix.loc[f1, f2]

                if corr > HIGH_CORRELATION_THRESHOLD:
                    # Get importance of both features
                    imp1 = next(
                        (fp.importance_score for fp in report.feature_profiles
                         if fp.feature_name == f1), 0
                    )
                    imp2 = next(
                        (fp.importance_score for fp in report.feature_profiles
                         if fp.feature_name == f2), 0
                    )

                    # Suggest keeping the more important one or combining
                    if abs(imp1 - imp2) > 0.1:
                        # One is clearly more important
                        keep, drop = (f1, f2) if imp1 > imp2 else (f2, f1)
                        suggestions.append(FeatureSuggestion(
                            suggestion_type="remove",
                            target_features=[drop],
                            description=(
                                f"Consider removing '{drop}' - it's highly correlated "
                                f"({corr:.2f}) with '{keep}' which has higher importance."
                            ),
                            expected_impact=2.0,
                            confidence=0.7,
                        ))
                    else:
                        # Similar importance - suggest combining
                        suggestions.append(FeatureSuggestion(
                            suggestion_type="combine",
                            target_features=[f1, f2],
                            description=(
                                f"Consider combining '{f1}' and '{f2}' - "
                                f"they are highly correlated ({corr:.2f}) with similar importance."
                            ),
                            expected_impact=3.0,
                            confidence=0.65,
                        ))

    except Exception as e:
        logger.warning(f"Correlation analysis failed: {e}")

    return suggestions


def _suggest_missing_handling(report: QualityReport) -> List[FeatureSuggestion]:
    """Suggest handling strategies for high-missing features."""
    suggestions = []

    for fp in report.feature_profiles:
        if fp.missing_ratio > HIGH_MISSING_THRESHOLD:
            if fp.importance_score < 0.1:
                # Low importance + high missing = remove
                suggestions.append(FeatureSuggestion(
                    suggestion_type="remove",
                    target_features=[fp.feature_name],
                    description=(
                        f"Consider removing '{fp.feature_name}' - it has high missing rate "
                        f"({fp.missing_ratio:.0%}) and low importance ({fp.importance_score:.3f})."
                    ),
                    expected_impact=2.0,
                    confidence=0.8,
                ))
            else:
                # High importance + high missing = needs careful handling
                suggestions.append(FeatureSuggestion(
                    suggestion_type="transform",
                    target_features=[fp.feature_name],
                    description=(
                        f"Improve imputation for '{fp.feature_name}' - it's important "
                        f"({fp.importance_score:.3f}) but has {fp.missing_ratio:.0%} missing values. "
                        f"Consider domain-specific imputation or a missing indicator column."
                    ),
                    expected_impact=4.0,
                    confidence=0.6,
                ))

    return suggestions


def apply_suggestion(
    df: pd.DataFrame,
    suggestion: FeatureSuggestion,
) -> pd.DataFrame:
    """
    Apply a feature engineering suggestion to a DataFrame.

    Args:
        df: Input DataFrame.
        suggestion: FeatureSuggestion to apply.

    Returns:
        Transformed DataFrame.
    """
    df = df.copy()

    if suggestion.suggestion_type == "remove":
        # Remove the feature(s)
        for feature in suggestion.target_features:
            if feature in df.columns:
                df = df.drop(columns=[feature])
                logger.info(f"Removed feature: {feature}")

    elif suggestion.suggestion_type == "transform":
        feature = suggestion.target_features[0]
        if feature not in df.columns:
            return df

        # Determine transform type from description
        desc_lower = suggestion.description.lower()

        if "log" in desc_lower:
            # Log transform (handle zeros and negatives)
            min_val = df[feature].min()
            if min_val <= 0:
                df[feature] = np.log1p(df[feature] - min_val + 1)
            else:
                df[feature] = np.log(df[feature])
            logger.info(f"Applied log transform to: {feature}")

        elif "sqrt" in desc_lower:
            # Square root transform
            min_val = df[feature].min()
            if min_val < 0:
                df[feature] = np.sqrt(df[feature] - min_val)
            else:
                df[feature] = np.sqrt(df[feature])
            logger.info(f"Applied sqrt transform to: {feature}")

        elif "square" in desc_lower:
            # Square transform (for negative skew)
            df[feature] = df[feature] ** 2
            logger.info(f"Applied square transform to: {feature}")

        elif "imputation" in desc_lower or "missing" in desc_lower:
            # Add missing indicator and impute
            df[f"{feature}_missing"] = df[feature].isna().astype(int)
            if pd.api.types.is_numeric_dtype(df[feature]):
                df[feature] = df[feature].fillna(df[feature].median())
            else:
                df[feature] = df[feature].fillna(df[feature].mode().iloc[0])
            logger.info(f"Added missing indicator and imputed: {feature}")

    elif suggestion.suggestion_type == "combine":
        if len(suggestion.target_features) >= 2:
            f1, f2 = suggestion.target_features[:2]
            if f1 in df.columns and f2 in df.columns:
                # Create combined feature (simple average for numeric)
                if pd.api.types.is_numeric_dtype(df[f1]) and pd.api.types.is_numeric_dtype(df[f2]):
                    # Normalize and average
                    norm1 = (df[f1] - df[f1].mean()) / (df[f1].std() + 1e-8)
                    norm2 = (df[f2] - df[f2].mean()) / (df[f2].std() + 1e-8)
                    df[f"{f1}_{f2}_combined"] = (norm1 + norm2) / 2
                    logger.info(f"Created combined feature: {f1}_{f2}_combined")

    return df


def get_transformation_preview(
    df: pd.DataFrame,
    suggestion: FeatureSuggestion,
    sample_size: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get before/after preview of a transformation.

    Args:
        df: Input DataFrame.
        suggestion: FeatureSuggestion to preview.
        sample_size: Number of sample rows to show.

    Returns:
        Tuple of (before_sample, after_sample) DataFrames.
    """
    # Get sample
    sample_df = df.head(sample_size)

    # Get affected columns
    affected_cols = suggestion.target_features.copy()
    if suggestion.suggestion_type == "combine" and len(affected_cols) >= 2:
        # Add the new combined column name
        affected_cols.append(f"{affected_cols[0]}_{affected_cols[1]}_combined")

    # Apply transformation
    transformed_df = apply_suggestion(sample_df, suggestion)

    # Filter to relevant columns
    before_cols = [c for c in affected_cols if c in sample_df.columns]
    after_cols = [c for c in affected_cols if c in transformed_df.columns]

    return sample_df[before_cols], transformed_df[after_cols]
