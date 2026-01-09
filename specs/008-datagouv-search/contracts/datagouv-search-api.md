# API Contract: DataGouv Search Service

**Feature**: 008-datagouv-search
**Date**: 2025-12-12

## Overview

Internal service contract for the data.gouv.fr search integration. This is NOT a REST API - it's a Python module contract.

## Module: `intuitiveness.services.datagouv_client`

### Class: DataGouvSearchService

Wrapper around the skill library's `DataGouvAPI` providing simplified interface for UI.

```python
class DataGouvSearchService:
    """Service for searching and loading data.gouv.fr datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the search service.

        Args:
            cache_dir: Optional custom cache directory. Defaults to ~/.cache/datagouv
        """
        pass

    def search(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10
    ) -> SearchResult:
        """
        Search for datasets matching the query.

        Args:
            query: Natural language search query
            page: Page number (1-indexed)
            page_size: Results per page (max 20)

        Returns:
            SearchResult containing datasets and pagination info

        Raises:
            DataGouvAPIError: If API call fails
        """
        pass

    def get_dataset_resources(
        self,
        dataset_id: str,
        format_filter: str = "csv"
    ) -> List[ResourceInfo]:
        """
        Get resources for a specific dataset, filtered by format.

        Args:
            dataset_id: Dataset ID from search results
            format_filter: File format to filter (default: csv)

        Returns:
            List of resources matching the format filter
        """
        pass

    def load_resource(
        self,
        resource_url: str,
        filename: str
    ) -> pd.DataFrame:
        """
        Download and load a resource as a DataFrame.

        Args:
            resource_url: URL of the resource to download
            filename: Display name for the loaded data

        Returns:
            Loaded DataFrame ready for L4 workflow

        Raises:
            DataGouvLoadError: If download or parsing fails
        """
        pass

    def is_available(self) -> bool:
        """
        Check if data.gouv.fr API is reachable.

        Returns:
            True if API responds, False otherwise
        """
        pass
```

### Data Classes

```python
@dataclass
class SearchResult:
    """Result of a dataset search."""
    datasets: List[DatasetInfo]
    total: int
    page: int
    page_size: int
    has_more: bool

@dataclass
class DatasetInfo:
    """Summary info about a dataset."""
    id: str
    title: str
    description: str  # Truncated to 200 chars for display
    organization_name: str
    last_modified: datetime
    resource_count: int
    has_csv: bool

@dataclass
class ResourceInfo:
    """Info about a downloadable resource."""
    id: str
    title: str
    url: str
    format: str
    filesize_bytes: Optional[int]
    filesize_display: str  # Human-readable, e.g., "2.5 MB"
    last_modified: datetime

@dataclass
class DataGouvAPIError(Exception):
    """Raised when API call fails."""
    message: str
    status_code: Optional[int] = None

@dataclass
class DataGouvLoadError(Exception):
    """Raised when resource loading fails."""
    message: str
    url: str
```

## Module: `intuitiveness.ui.datagouv_search`

### Functions

```python
def render_search_interface() -> Optional[pd.DataFrame]:
    """
    Render the complete search interface.

    Returns:
        DataFrame if user selected and loaded a dataset, None otherwise

    Side Effects:
        - Updates st.session_state with search state
        - May trigger st.rerun() on state changes
    """
    pass

def render_search_bar() -> Optional[str]:
    """
    Render the search input bar with tagline.

    Returns:
        Search query if submitted, None otherwise
    """
    pass

def render_dataset_cards(datasets: List[DatasetInfo]) -> Optional[str]:
    """
    Render search result cards.

    Args:
        datasets: List of dataset summaries to display

    Returns:
        Selected dataset ID if clicked, None otherwise
    """
    pass

def render_resource_selector(resources: List[ResourceInfo]) -> Optional[str]:
    """
    Render resource selection within a dataset.

    Args:
        resources: List of available resources

    Returns:
        Selected resource URL if clicked, None otherwise
    """
    pass

def render_loading_state(message: str) -> None:
    """
    Render loading spinner with message.

    Args:
        message: Localized loading message
    """
    pass

def render_error_state(error: str) -> None:
    """
    Render error message with fallback to upload.

    Args:
        error: Localized error message
    """
    pass
```

## Session State Contract

### Keys Used

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `datagouv_query` | `str` | `""` | Current search query |
| `datagouv_results` | `SearchResult | None` | `None` | Latest search results |
| `datagouv_selected_dataset` | `str | None` | `None` | ID of selected dataset |
| `datagouv_selected_resource` | `str | None` | `None` | URL of selected resource |
| `datagouv_loading` | `bool` | `False` | Loading state flag |
| `datagouv_error` | `str | None` | `None` | Error message |

### State Transitions

```
Initial State:
  query=""
  results=None
  selected_dataset=None
  selected_resource=None
  loading=False
  error=None

After Search:
  query="vaccination"
  results=SearchResult(...)
  loading=False

After Dataset Click:
  selected_dataset="abc123"

After Resource Click:
  selected_resource="https://..."
  loading=True

After Load Success:
  -> Clear all datagouv_* keys
  -> Set raw_data with loaded DataFrame
  -> Proceed to L4 workflow

After Load Error:
  loading=False
  error="Could not load..."
```

## Error Handling Contract

| Error Scenario | User Message (EN) | User Message (FR) | Action |
|----------------|-------------------|-------------------|--------|
| API unreachable | "Could not connect to data.gouv.fr" | "Connexion impossible" | Show upload fallback |
| No results | "No datasets found" | "Aucun resultat" | Suggest different query |
| Download failed | "Could not download file" | "Echec du telechargement" | Show retry option |
| Parse failed | "Could not read file format" | "Format illisible" | Show format info |
