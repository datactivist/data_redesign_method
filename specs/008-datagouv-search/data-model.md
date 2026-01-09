# Data Model: Data.gouv.fr Search Integration

**Feature**: 008-datagouv-search
**Date**: 2025-12-12

## Entities

### SearchState (Session State)

Manages the search workflow state within Streamlit session.

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Current search query entered by user |
| `results` | `List[DatasetResult]` | List of datasets from search |
| `total_count` | `int` | Total number of matching datasets |
| `selected_dataset` | `DatasetResult | None` | Currently selected dataset |
| `selected_resource` | `Resource | None` | Resource selected for loading |
| `loading` | `bool` | Whether a search/load is in progress |
| `error` | `str | None` | Error message if any |
| `page` | `int` | Current results page (1-indexed) |

**State Transitions**:
```
IDLE -> SEARCHING -> RESULTS_DISPLAYED
RESULTS_DISPLAYED -> DATASET_SELECTED -> RESOURCE_SELECTED
RESOURCE_SELECTED -> LOADING -> LOADED (transitions to L4 workflow)
```

### DatasetResult

Represents a dataset returned from data.gouv.fr search.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique dataset identifier |
| `title` | `str` | Human-readable title |
| `description` | `str` | Full description (may be truncated for display) |
| `organization` | `Organization` | Publishing organization |
| `resources` | `List[Resource]` | Available files |
| `last_modified` | `datetime` | Last update timestamp |
| `tags` | `List[str]` | Associated tags |

**Validation Rules**:
- `title` must not be empty
- `resources` may be empty (no files available)

### Resource

Represents a downloadable file within a dataset.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique resource identifier |
| `title` | `str` | File title/name |
| `url` | `str` | Download URL |
| `format` | `str` | File format (csv, json, xlsx, etc.) |
| `filesize` | `int | None` | Size in bytes (may be None) |
| `last_modified` | `datetime` | Last update timestamp |

**Validation Rules**:
- `url` must be a valid HTTP/HTTPS URL
- `format` is case-insensitive (normalized to lowercase)

### Organization

Represents the publishing organization.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Organization identifier |
| `name` | `str` | Organization display name |
| `logo` | `str | None` | Logo URL (optional) |

## Entity Relationships

```
SearchState
    │
    ├── has many → DatasetResult
    │                  │
    │                  ├── belongs to → Organization
    │                  │
    │                  └── has many → Resource
    │
    └── references → selected_dataset (DatasetResult)
                   → selected_resource (Resource)
```

## Mapping to Existing Entities

When a resource is loaded, it enters the existing workflow:

| Search Entity | Existing Entity | Transformation |
|---------------|-----------------|----------------|
| `Resource.url` | `raw_data[filename]` | Download CSV → pandas DataFrame |
| `DatasetResult.title` | File name in L4 display | Used as display name |
| Multiple Resources | Multiple files in `raw_data` | Each selected resource becomes one entry |

## Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `datagouv_search_state` | `SearchState` | Main search state object |
| `datagouv_cache` | `Dict[str, bytes]` | Downloaded file cache |

## Data Flow

```
User Query → search_datasets() → DatasetResult[] → Display Cards
                                                         ↓
                                                  User clicks card
                                                         ↓
                                            Show Resources (filtered to CSV)
                                                         ↓
                                                  User selects Resource
                                                         ↓
                                            load_csv(resource.url)
                                                         ↓
                                              pandas DataFrame
                                                         ↓
                                     Store in st.session_state.raw_data
                                                         ↓
                                          Enter L4 workflow (same as upload)
```
