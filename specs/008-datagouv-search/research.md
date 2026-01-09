# Research: Data.gouv.fr Search Integration

**Feature**: 008-datagouv-search
**Date**: 2025-12-12
**Status**: Complete

## Research Tasks Completed

### 1. Data.gouv.fr API Capabilities

**Decision**: Use the existing `DataGouvAPI` class from `skills/data-gouv/lib/datagouv.py`

**Findings**:
- Base URL: `https://www.data.gouv.fr/api/1`
- Key endpoint: `GET /datasets/?q={query}&page_size={size}&page={page}`
- Response contains: `data` (list of datasets), `total` (count)
- Each dataset has: `id`, `title`, `description`, `organization`, `resources`, `last_modified`
- Each resource has: `id`, `url`, `format`, `title`, `filesize`, `last_modified`
- No authentication required for public API
- Rate limits: Not strict for reasonable usage (no explicit documentation)

**Rationale**: The skill library already implements all needed API calls with proper error handling and caching.

**Alternatives Considered**:
- Direct API calls: Rejected - would duplicate existing code
- MCP server: Rejected - overkill for simple search/download, adds complexity

### 2. Existing DataGouvAPI Usage

**Decision**: Import `DataGouvAPI` directly from `skills/data-gouv/lib/datagouv.py`

**Key Methods Available**:
```python
# Search datasets
api.search_datasets(query, organization=None, tag=None, page_size=20, page=1)
# Returns: {"data": [...], "total": int}

# Get specific dataset
api.get_dataset(dataset_id)
# Returns: Full dataset object with resources

# Get latest resource of format
api.get_latest_resource(dataset_id, format="csv", title_contains=None)
# Returns: Resource dict with URL

# Load CSV with auto-detection
api.load_csv(resource_url, sep=None, encoding=None, decimal=",", cache=True)
# Returns: pandas DataFrame
```

**Caching**: Uses `~/.cache/datagouv/` directory, keyed by resource URL filename.

**Rationale**: Library handles French CSV quirks (semicolon delimiters, comma decimals, various encodings).

### 3. Streamlit Search Patterns

**Decision**: Use `st.text_input` for search + `st.session_state` for results + expander cards for display

**Best Practices**:
- Use `st.text_input` with `on_change` callback for search submission
- Store search results in session state to persist across reruns
- Use `st.spinner` during API calls
- Display results as expandable cards using `st.expander` or custom HTML cards
- Implement pagination if >20 results

**Implementation Pattern**:
```python
# Search state
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# Search input
query = st.text_input("Search", key="search_query")
if st.button("Search") and query:
    with st.spinner("Searching..."):
        st.session_state.search_results = api.search_datasets(query)

# Display results
if st.session_state.search_results:
    for dataset in st.session_state.search_results["data"]:
        with st.expander(dataset["title"]):
            # Show resources
```

**Rationale**: Standard Streamlit patterns ensure consistent UX with existing app.

### 4. i18n Extension Pattern

**Decision**: Add new keys to `TRANSLATIONS` dict in `intuitiveness/ui/i18n.py`

**New Translation Keys Required**:
```python
# Search interface
"search_placeholder": {"en": "Search French open data...", "fr": "Rechercher des donnees ouvertes francaises..."}
"search_tagline": {"en": "Redesign any data for your intent", "fr": "Redesignez toute donnee selon votre intention"}
"search_button": {"en": "Search", "fr": "Rechercher"}
"no_results": {"en": "No datasets found for '{query}'", "fr": "Aucun jeu de donnees trouve pour '{query}'"}
"loading_dataset": {"en": "Loading dataset...", "fr": "Chargement du jeu de donnees..."}
"dataset_loaded": {"en": "Dataset loaded! Starting redesign workflow.", "fr": "Jeu de donnees charge! Demarrage du workflow de redesign."}
"api_error": {"en": "Could not connect to data.gouv.fr. Please try uploading files instead.", "fr": "Connexion a data.gouv.fr impossible. Veuillez essayer d'importer des fichiers."}

# Dataset card
"dataset_by": {"en": "by {org}", "fr": "par {org}"}
"last_updated": {"en": "Updated {date}", "fr": "Mis a jour le {date}"}
"resources_count": {"en": "{count} files available", "fr": "{count} fichiers disponibles"}
"select_resource": {"en": "Select a file to load:", "fr": "Selectionnez un fichier a charger:"}
"load_csv": {"en": "Load this CSV", "fr": "Charger ce CSV"}
"file_size": {"en": "Size: {size}", "fr": "Taille: {size}"}
"no_csv_available": {"en": "No CSV files available in this dataset", "fr": "Aucun fichier CSV disponible dans ce jeu de donnees"}

# Entry point choice
"or_upload_files": {"en": "Or upload your own files:", "fr": "Ou importez vos propres fichiers:"}
"upload_option": {"en": "Upload Files", "fr": "Importer des fichiers"}
"search_option": {"en": "Search Open Data", "fr": "Rechercher des donnees ouvertes"}
```

**Rationale**: Follows existing pattern in `i18n.py`; all user-facing strings internationalized.

## Technical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API Client | Use existing `DataGouvAPI` | Avoids duplication, has caching/error handling |
| Search UI | `st.text_input` + cards | Standard Streamlit pattern |
| Results Display | Expander cards | Compact, scalable, consistent with existing UI |
| i18n | Extend `TRANSLATIONS` dict | Existing pattern, minimal changes |
| CSV Loading | Use `api.load_csv()` then `smart_load_csv()` | Double-fallback for format detection |
| Caching | Rely on `DataGouvAPI` cache | Already implemented in `~/.cache/datagouv/` |

## Open Questions Resolved

1. **Q: How to handle very large datasets?**
   - A: Show file size in UI, warn >50MB, let user decide

2. **Q: What if dataset has no CSV?**
   - A: Display message "No CSV available" with available formats listed

3. **Q: Search result ordering?**
   - A: Use API default (relevance), no custom sorting needed

4. **Q: Pagination?**
   - A: Initially show 10 results with "Load more" button; page_size=10

## Dependencies Confirmed

- `requests` (already in requirements.txt)
- `pandas` (already in requirements.txt)
- `streamlit` (already in requirements.txt)
- No new dependencies required
