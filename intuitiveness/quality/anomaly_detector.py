"""
Quality Data Platform - Anomaly Detection

TabPFN-based density estimation for detecting anomalous records
with interpretable feature attributions.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from intuitiveness.quality.models import AnomalyRecord

logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    percentile_threshold: float = 2.0,
    max_anomalies: int = 100,
) -> List[AnomalyRecord]:
    """
    Detect anomalous rows using density estimation.

    Uses Local Outlier Factor as TabPFN doesn't directly support
    density estimation. LOF measures local density deviation.

    Args:
        df: DataFrame to analyze.
        percentile_threshold: Flag rows below this percentile as anomalies.
        max_anomalies: Maximum number of anomalies to return.

    Returns:
        List of AnomalyRecord objects, sorted by anomaly score (most anomalous first).
    """
    # Prepare data - only use numeric columns for LOF
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.empty:
        logger.warning("No numeric columns found for anomaly detection")
        return []

    # Handle missing values
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Compute LOF scores
    n_neighbors = min(20, len(df) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
    lof_labels = lof.fit_predict(scaled_data)
    lof_scores = -lof.negative_outlier_factor_  # Higher = more anomalous

    # Compute percentiles
    percentiles = np.array([
        100 * (lof_scores <= score).mean() for score in lof_scores
    ])

    # Find anomalies below threshold
    anomaly_indices = np.where(percentiles <= percentile_threshold)[0]

    # Sort by score (most anomalous first)
    sorted_indices = anomaly_indices[np.argsort(lof_scores[anomaly_indices])[::-1]]

    # Create anomaly records
    anomalies = []
    for idx in sorted_indices[:max_anomalies]:
        row_data = df.iloc[idx]
        contributors = explain_anomaly(
            row_data,
            df,
            numeric_df.columns.tolist(),
            scaler,
        )

        anomalies.append(AnomalyRecord(
            row_index=int(idx),
            anomaly_score=float(lof_scores[idx]),
            percentile=float(percentiles[idx]),
            top_contributors=contributors,
        ))

    logger.info(f"Detected {len(anomalies)} anomalies (threshold: {percentile_threshold}th percentile)")
    return anomalies


def explain_anomaly(
    row: pd.Series,
    df: pd.DataFrame,
    numeric_columns: List[str],
    scaler: StandardScaler,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Explain why a row is anomalous by identifying contributing features.

    Args:
        row: The anomalous row.
        df: Full DataFrame for context.
        numeric_columns: List of numeric column names.
        scaler: Fitted StandardScaler.
        top_n: Number of top contributors to return.

    Returns:
        List of contributor dictionaries with feature, contribution, and reason.
    """
    contributors = []

    for col in numeric_columns:
        if col not in row.index:
            continue

        value = row[col]
        if pd.isna(value):
            continue

        # Compute z-score
        col_idx = list(numeric_columns).index(col)
        mean = scaler.mean_[col_idx]
        std = scaler.scale_[col_idx]

        if std == 0:
            continue

        z_score = abs((value - mean) / std)

        # Determine reason
        if z_score > 3:
            if value > mean:
                reason = f"Unusually high value ({value:.2f}, z={z_score:.1f})"
            else:
                reason = f"Unusually low value ({value:.2f}, z={z_score:.1f})"
        elif z_score > 2:
            reason = f"Outlier value ({value:.2f}, z={z_score:.1f})"
        else:
            reason = f"Notable deviation ({value:.2f}, z={z_score:.1f})"

        contributors.append({
            "feature": col,
            "contribution": float(z_score),
            "reason": reason,
        })

    # Sort by contribution and return top N
    contributors.sort(key=lambda x: x["contribution"], reverse=True)
    return contributors[:top_n]


def get_anomaly_summary(
    anomalies: List[AnomalyRecord],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate a summary of detected anomalies.

    Args:
        anomalies: List of detected anomalies.
        df: Original DataFrame.

    Returns:
        Summary dictionary with statistics and patterns.
    """
    if not anomalies:
        return {
            "total_anomalies": 0,
            "percentage": 0.0,
            "top_contributing_features": [],
            "severity_distribution": {},
        }

    # Count feature contributions
    feature_counts: Dict[str, int] = {}
    for a in anomalies:
        for contrib in a.top_contributors:
            feature = contrib["feature"]
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Sort by count
    top_features = sorted(
        feature_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Severity distribution
    severity_dist = {
        "severe": sum(1 for a in anomalies if a.percentile < 0.5),
        "moderate": sum(1 for a in anomalies if 0.5 <= a.percentile < 1.0),
        "mild": sum(1 for a in anomalies if a.percentile >= 1.0),
    }

    return {
        "total_anomalies": len(anomalies),
        "percentage": len(anomalies) / len(df) * 100,
        "top_contributing_features": [
            {"feature": f, "count": c} for f, c in top_features
        ],
        "severity_distribution": severity_dist,
    }
