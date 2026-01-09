"""
Quality Data Platform - Catalog Search

In-memory indexing and search functionality for the dataset catalog.
Supports filtering, sorting, and full-text search.
"""

import logging
from typing import List, Optional, Dict, Tuple
from uuid import UUID

from intuitiveness.catalog.models import Dataset, DatasetSummary
from intuitiveness.catalog.storage import get_storage, CatalogStorage

logger = logging.getLogger(__name__)


class CatalogIndex:
    """
    In-memory index for fast catalog queries.

    Maintains secondary indexes for efficient filtering:
    - by_domain: domain_tag -> dataset IDs
    - by_score: sorted list of (score, ID) tuples
    """

    def __init__(self, storage: Optional[CatalogStorage] = None):
        """
        Initialize catalog index.

        Args:
            storage: CatalogStorage instance. Uses global storage if None.
        """
        self.storage = storage or get_storage()
        self._by_domain: Dict[str, List[UUID]] = {}
        self._by_score: List[Tuple[float, UUID]] = []
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild all secondary indexes from storage."""
        self._by_domain.clear()
        self._by_score.clear()

        for dataset in self.storage.list_datasets():
            # Index by domain
            for tag in dataset.domain_tags:
                if tag not in self._by_domain:
                    self._by_domain[tag] = []
                self._by_domain[tag].append(dataset.id)

            # Index by score
            score = dataset.usability_score if dataset.usability_score is not None else -1
            self._by_score.append((score, dataset.id))

        # Sort by score descending
        self._by_score.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Rebuilt index with {len(self._by_score)} datasets")

    def refresh(self) -> None:
        """Refresh the index from storage."""
        self._rebuild_index()


def filter_datasets(
    min_score: Optional[float] = None,
    domains: Optional[List[str]] = None,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    query: Optional[str] = None,
    sort_by: str = "usability_score",
    sort_desc: bool = True,
    limit: int = 50,
    storage: Optional[CatalogStorage] = None,
) -> List[Dataset]:
    """
    Filter and sort datasets in catalog.

    Args:
        min_score: Minimum usability score (0-100).
        domains: Filter by domain tags (OR logic).
        min_rows: Minimum row count.
        max_rows: Maximum row count.
        query: Full-text search in name/description.
        sort_by: Sort field ("usability_score", "name", "created_at", "row_count").
        sort_desc: Sort descending if True.
        limit: Maximum results to return.
        storage: Optional CatalogStorage instance.

    Returns:
        List of matching Dataset objects.
    """
    storage = storage or get_storage()
    datasets = storage.list_datasets()

    # Apply filters
    if min_score is not None:
        datasets = [
            ds for ds in datasets
            if ds.usability_score is not None and ds.usability_score >= min_score
        ]

    if domains:
        datasets = [
            ds for ds in datasets
            if any(tag in domains for tag in ds.domain_tags)
        ]

    if min_rows is not None:
        datasets = [ds for ds in datasets if ds.row_count >= min_rows]

    if max_rows is not None:
        datasets = [ds for ds in datasets if ds.row_count <= max_rows]

    if query:
        query_lower = query.lower()
        datasets = [
            ds for ds in datasets
            if query_lower in ds.name.lower() or query_lower in ds.description.lower()
        ]

    # Sort
    sort_key_map = {
        "usability_score": lambda ds: ds.usability_score if ds.usability_score else 0,
        "name": lambda ds: ds.name.lower(),
        "created_at": lambda ds: ds.created_at,
        "row_count": lambda ds: ds.row_count,
    }
    sort_key = sort_key_map.get(sort_by, sort_key_map["usability_score"])
    datasets.sort(key=sort_key, reverse=sort_desc)

    # Limit
    return datasets[:limit]


def search_datasets(
    query: str,
    limit: int = 20,
    storage: Optional[CatalogStorage] = None,
) -> List[Dataset]:
    """
    Full-text search in dataset names and descriptions.

    Args:
        query: Search query string.
        limit: Maximum results to return.
        storage: Optional CatalogStorage instance.

    Returns:
        List of matching Dataset objects, sorted by relevance.
    """
    if not query or not query.strip():
        return []

    storage = storage or get_storage()
    query_terms = query.lower().split()

    # Score each dataset by term matches
    scored: List[Tuple[int, Dataset]] = []
    for dataset in storage.list_datasets():
        name_lower = dataset.name.lower()
        desc_lower = dataset.description.lower()
        tags_lower = " ".join(dataset.domain_tags).lower()

        score = 0
        for term in query_terms:
            # Name matches are worth more
            if term in name_lower:
                score += 3
            if term in desc_lower:
                score += 1
            if term in tags_lower:
                score += 2

        if score > 0:
            scored.append((score, dataset))

    # Sort by relevance score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [ds for _, ds in scored[:limit]]


def get_all_domains(storage: Optional[CatalogStorage] = None) -> List[str]:
    """
    Get all unique domain tags in the catalog.

    Args:
        storage: Optional CatalogStorage instance.

    Returns:
        Sorted list of unique domain tags.
    """
    storage = storage or get_storage()
    domains: set = set()
    for dataset in storage.list_datasets():
        domains.update(dataset.domain_tags)
    return sorted(domains)


def get_score_distribution(
    storage: Optional[CatalogStorage] = None,
) -> Dict[str, int]:
    """
    Get distribution of usability scores in the catalog.

    Returns counts by score range:
    - "excellent": 80-100
    - "good": 60-79
    - "fair": 40-59
    - "poor": 0-39
    - "unassessed": None

    Args:
        storage: Optional CatalogStorage instance.

    Returns:
        Dictionary of score range -> count.
    """
    storage = storage or get_storage()
    distribution = {
        "excellent": 0,
        "good": 0,
        "fair": 0,
        "poor": 0,
        "unassessed": 0,
    }

    for dataset in storage.list_datasets():
        score = dataset.usability_score
        if score is None:
            distribution["unassessed"] += 1
        elif score >= 80:
            distribution["excellent"] += 1
        elif score >= 60:
            distribution["good"] += 1
        elif score >= 40:
            distribution["fair"] += 1
        else:
            distribution["poor"] += 1

    return distribution
