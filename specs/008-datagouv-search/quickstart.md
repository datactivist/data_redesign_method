# Quickstart: Data.gouv.fr Search Integration

**Feature**: 008-datagouv-search
**Date**: 2025-12-12

## Prerequisites

1. Python 3.11+ (existing `myenv311` virtual environment)
2. All existing dependencies already installed
3. No additional packages required

## Setup

```bash
# Activate virtual environment
source myenv311/bin/activate

# Verify data-gouv skill is available
ls skills/data-gouv/lib/datagouv.py
# Should show the file

# Run the app
streamlit run intuitiveness/streamlit_app.py
```

## Development Workflow

### 1. Implement the Service Layer

Create `intuitiveness/services/datagouv_client.py`:

```python
# See contracts/datagouv-search-api.md for full interface
from skills.data-gouv.lib.datagouv import DataGouvAPI

class DataGouvSearchService:
    def __init__(self):
        self._api = DataGouvAPI()

    def search(self, query: str, page: int = 1) -> SearchResult:
        raw = self._api.search_datasets(query, page_size=10, page=page)
        # Transform to SearchResult...
```

### 2. Add Translations

In `intuitiveness/ui/i18n.py`, add to `TRANSLATIONS`:

```python
"search_tagline": {
    "en": "Redesign any data for your intent",
    "fr": "Redesignez toute donnee selon votre intention",
},
"search_placeholder": {
    "en": "Search French open data...",
    "fr": "Rechercher des donnees ouvertes francaises...",
},
# ... (see research.md for full list)
```

### 3. Create UI Component

Create `intuitiveness/ui/datagouv_search.py`:

```python
import streamlit as st
from intuitiveness.ui import t
from intuitiveness.services.datagouv_client import DataGouvSearchService

def render_search_interface():
    service = DataGouvSearchService()

    # Tagline
    st.markdown(f"### {t('search_tagline')}")

    # Search input
    query = st.text_input(
        label=t('search_placeholder'),
        key="datagouv_query"
    )

    if st.button(t('search_button')) and query:
        with st.spinner(t('searching')):
            results = service.search(query)
            st.session_state.datagouv_results = results

    # Display results...
```

### 4. Integrate in Main App

In `intuitiveness/streamlit_app.py`, modify `render_methodology_intro()`:

```python
from intuitiveness.ui.datagouv_search import render_search_interface

def render_methodology_intro():
    # Replace existing methodology cards with search-first interface
    loaded_df = render_search_interface()

    if loaded_df is not None:
        # User loaded a dataset - proceed to workflow
        st.session_state.raw_data = {"search_result.csv": loaded_df}
        st.session_state.datasets['l4'] = Level4Dataset(st.session_state.raw_data)
        st.session_state.current_step = 1
        st.rerun()

    # Keep file upload as fallback
    st.markdown(f"### {t('or_upload_files')}")
    # ... existing file uploader code
```

## Testing

### Manual Testing

1. Start the app: `streamlit run intuitiveness/streamlit_app.py`
2. Navigate to homepage (no files uploaded)
3. Test search:
   - Enter "vaccination" → should see results
   - Enter "asdfghjkl" → should see "no results" message
4. Test dataset selection:
   - Click a result card → should expand to show resources
   - Click a CSV resource → should load and enter workflow
5. Test error handling:
   - Disconnect network → should show error with upload fallback

### E2E Testing (Playwright)

```python
# tests/e2e/test_datagouv_search.py
def test_search_and_load_dataset(page):
    page.goto("http://localhost:8502")

    # Search
    page.fill('[data-testid="stTextInput"] input', 'vaccination')
    page.click('button:has-text("Search")')

    # Wait for results
    page.wait_for_selector('.dataset-card')

    # Click first result
    page.click('.dataset-card:first-child')

    # Select CSV resource
    page.click('button:has-text("Load this CSV")')

    # Verify workflow started
    page.wait_for_selector('text=L4: Raw Files')
```

## File Checklist

| File | Action | Status |
|------|--------|--------|
| `intuitiveness/services/__init__.py` | Create | Pending |
| `intuitiveness/services/datagouv_client.py` | Create | Pending |
| `intuitiveness/ui/datagouv_search.py` | Create | Pending |
| `intuitiveness/ui/i18n.py` | Modify (add translations) | Pending |
| `intuitiveness/ui/__init__.py` | Modify (add exports) | Pending |
| `intuitiveness/styles/search.py` | Create | Pending |
| `intuitiveness/streamlit_app.py` | Modify (integrate search) | Pending |
| `tests/e2e/test_datagouv_search.py` | Create | Pending |
| `tests/unit/test_datagouv_client.py` | Create | Pending |

## Common Issues

### Import Error for skills/data-gouv

If `from skills.data-gouv.lib.datagouv import DataGouvAPI` fails:

```python
# Use sys.path instead
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "skills" / "data-gouv" / "lib"))
from datagouv import DataGouvAPI
```

### Session State Not Persisting

Ensure all state keys are prefixed with `datagouv_` to avoid conflicts:
```python
if "datagouv_results" not in st.session_state:
    st.session_state.datagouv_results = None
```

### French CSV Not Parsing

The `DataGouvAPI.load_csv()` handles French formats, but if issues persist:
```python
# Fallback to app's smart_load_csv
from intuitiveness.streamlit_app import smart_load_csv
```
