"""Service layer for external API integrations."""

from intuitiveness.services.datagouv_client import (
    DataGouvSearchService,
    SearchResult,
    DatasetInfo,
    ResourceInfo,
    DataGouvAPIError,
    DataGouvLoadError,
)

__all__ = [
    "DataGouvSearchService",
    "SearchResult",
    "DatasetInfo",
    "ResourceInfo",
    "DataGouvAPIError",
    "DataGouvLoadError",
]
