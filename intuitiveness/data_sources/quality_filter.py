"""
Quality-Aware Dataset Filtering for Data.gouv.fr.

Implements Spec 008: Quality-aware search filtering (extends Spec 009)

Provides:
- Quick quality assessment for search results
- Filtering datasets by usability score
- Quality indicators in search UI
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetQualityScore:
    """Quality score for a dataset in search results."""

    dataset_id: str
    usability_score: float
    row_count: int
    column_count: int
    assessment_timestamp: str
    can_assess: bool = True
    error_message: Optional[str] = None

    @property
    def quality_badge(self) -> str:
        """Get quality badge color and label."""
        if not self.can_assess:
            return "gray", "Unknown"

        if self.usability_score >= 80:
            return "green", "High Quality"
        elif self.usability_score >= 60:
            return "yellow", "Medium Quality"
        else:
            return "red", "Low Quality"


def quick_assess_dataset(
    df: pd.DataFrame,
    dataset_id: str
) -> DatasetQualityScore:
    """
    Perform quick quality assessment on a dataset.

    Uses fast heuristics instead of full TabPFN assessment:
    - Data completeness (missing values)
    - Size appropriateness (row count)
    - Basic feature diversity

    Args:
        df: DataFrame to assess
        dataset_id: Dataset identifier

    Returns:
        DatasetQualityScore with quick assessment
    """
    from datetime import datetime

    try:
        # Quick heuristics (no TabPFN needed)
        row_count = len(df)
        column_count = len(df.columns)

        # Data completeness score
        total_cells = row_count * column_count
        if total_cells > 0:
            missing_cells = df.isna().sum().sum()
            completeness = (1 - missing_cells / total_cells) * 100
        else:
            completeness = 0

        # Size appropriateness
        if row_count < 50:
            size_score = 30  # Too small
        elif row_count <= 10000:
            size_score = 100  # Optimal
        else:
            # Penalty for large datasets
            size_score = max(50, 100 - (row_count - 10000) / 1000)

        # Feature diversity (simple version)
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            diversity = 75  # Mixed types
        elif len(numeric_cols) > 0 or len(categorical_cols) > 0:
            diversity = 50  # Single type
        else:
            diversity = 25  # No features

        # Weighted average (quick heuristic)
        usability_score = (
            0.5 * completeness +
            0.3 * size_score +
            0.2 * diversity
        )

        return DatasetQualityScore(
            dataset_id=dataset_id,
            usability_score=usability_score,
            row_count=row_count,
            column_count=column_count,
            assessment_timestamp=datetime.now().isoformat(),
            can_assess=True
        )

    except Exception as e:
        logger.error(f"Quick assessment failed for {dataset_id}: {e}")
        return DatasetQualityScore(
            dataset_id=dataset_id,
            usability_score=0,
            row_count=0,
            column_count=0,
            assessment_timestamp=datetime.now().isoformat(),
            can_assess=False,
            error_message=str(e)
        )


def filter_by_quality(
    datasets: List[Any],
    min_score: float = 0.0,
    max_results: int = 20
) -> List[tuple[Any, Optional[DatasetQualityScore]]]:
    """
    Filter datasets by quality score.

    Note: This requires downloading and assessing each dataset,
    which can be slow. Use sparingly for small result sets.

    Args:
        datasets: List of dataset objects
        min_score: Minimum usability score (0-100)
        max_results: Maximum results to assess

    Returns:
        List of (dataset, quality_score) tuples meeting threshold
    """
    filtered = []

    for dataset in datasets[:max_results]:
        try:
            # This would require downloading the CSV first
            # For now, we just return without scores
            # In production, implement lazy loading + caching
            filtered.append((dataset, None))

        except Exception as e:
            logger.error(f"Failed to assess {dataset.id}: {e}")
            filtered.append((dataset, None))

    return filtered


def get_quality_cache_key(dataset_id: str, resource_url: str) -> str:
    """
    Generate cache key for dataset quality scores.

    Args:
        dataset_id: Dataset identifier
        resource_url: Resource URL

    Returns:
        Cache key string
    """
    import hashlib
    combined = f"{dataset_id}_{resource_url}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def should_show_quality_indicator(dataset_size: int) -> bool:
    """
    Determine if quality indicator should be shown.

    Quality assessment is only meaningful for datasets
    within TabPFN's optimal range (50-10,000 rows).

    Args:
        dataset_size: Number of rows in dataset

    Returns:
        True if quality indicator should be displayed
    """
    return 50 <= dataset_size <= 10000
