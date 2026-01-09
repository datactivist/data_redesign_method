"""
Quality Data Platform - Dataset Catalog Module

This module provides dataset catalog management for browsing, filtering,
and discovering high-quality datasets.

Primary Components:
- models: Catalog data models (Dataset, DatasetSummary, DatasetDetail)
- storage: JSON-based catalog persistence
- search: Filtering, sorting, and full-text search
"""

from intuitiveness.catalog.models import (
    Dataset,
    DatasetSummary,
    DatasetDetail,
)

from intuitiveness.catalog.storage import (
    CatalogStorage,
    get_storage,
)

from intuitiveness.catalog.search import (
    filter_datasets,
    search_datasets,
    get_all_domains,
    get_score_distribution,
)

__all__ = [
    # Models
    "Dataset",
    "DatasetSummary",
    "DatasetDetail",
    # Storage
    "CatalogStorage",
    "get_storage",
    # Search functions
    "filter_datasets",
    "search_datasets",
    "get_all_domains",
    "get_score_distribution",
]
