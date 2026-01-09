# Research: Session Persistence

**Feature**: 005-session-persistence
**Date**: 2025-12-04

## Research Question 1: Streamlit Persistence Options

### Findings

Streamlit provides several mechanisms for state management:

1. **`st.session_state`** (in-memory)
   - Persists across reruns within same session
   - Lost on browser refresh or tab close
   - Current implementation uses this extensively

2. **`st.cache_data` / `st.cache_resource`**
   - Caches function results across reruns
   - Persists in Streamlit server memory
   - Not browser-specific, shared across users on same server

3. **External storage options**
   - File system (pickle files)
   - Database
   - Browser localStorage (requires JavaScript bridge)

### Decision

Use **browser localStorage** via `streamlit-javascript` component for true client-side persistence. This provides:
- Per-browser session isolation
- Survives browser refresh and tab close
- No server-side storage complexity
- Works offline

### Rationale

- `st.session_state` alone cannot survive page refresh (core problem)
- Server-side caching doesn't provide per-user isolation in Streamlit
- localStorage is the standard web solution for this exact use case

---

## Research Question 2: localStorage Limitations

### Findings

| Browser | localStorage Limit | Notes |
|---------|-------------------|-------|
| Chrome | 10 MB | Per origin |
| Firefox | 10 MB | Per origin |
| Safari | 5 MB | Per origin, can prompt for more |
| Edge | 10 MB | Per origin |

**Key constraints**:
- Data must be string (JSON serialized)
- Synchronous API (can block for large data)
- No built-in compression

### Decision

Implement with:
1. JSON serialization for all data
2. Base64 encoding for binary data (DataFrames as CSV, Graphs as JSON)
3. Compression using browser's CompressionStream API or Python's zlib before storage
4. Chunking strategy for data > 5MB
5. Graceful degradation: warn user if data exceeds limits

### Rationale

10MB limit accommodates typical use cases (files < 50MB compress well). Chunking handles edge cases without requiring complex backends.

---

## Research Question 3: Serialization Strategy

### Findings

**pandas DataFrame serialization options**:
| Format | Size | Speed | Preserves dtypes |
|--------|------|-------|------------------|
| JSON (orient='records') | Large | Fast | Partial |
| CSV | Medium | Fast | No |
| Parquet | Small | Medium | Yes |
| Pickle | Small | Fast | Yes (security risk) |

**networkx Graph serialization options**:
| Format | Complexity | Preserves attributes |
|--------|------------|---------------------|
| node_link_data JSON | Low | Yes |
| adjacency_data JSON | Low | Partial |
| GraphML | Medium | Yes |

### Decision

1. **DataFrames**: Use `to_json(orient='split')` - preserves index/columns structure, reasonable size
2. **Graphs**: Use `nx.node_link_data()` → JSON - standard, preserves all attributes
3. **Compression**: Apply zlib compression before base64 encoding
4. **Version tag**: Include schema version for forward compatibility

### Rationale

JSON is universally compatible with localStorage. `orient='split'` preserves DataFrame structure better than 'records'. node_link format is the networkx standard for serialization.

---

## Research Question 4: Streamlit-JavaScript Integration

### Findings

**Options for executing JavaScript in Streamlit**:

1. **`streamlit-javascript`** package
   - Simple `st_javascript()` function
   - Returns values to Python
   - Well-maintained, 1000+ GitHub stars

2. **`streamlit.components.v1.html()`**
   - Embed arbitrary HTML/JS
   - One-way (can't easily return values)
   - Built into Streamlit

3. **Custom Streamlit component**
   - Full bidirectional communication
   - More complex to implement
   - Maximum flexibility

### Decision

Use **`streamlit-javascript`** package for localStorage access:

```python
from streamlit_javascript import st_javascript

# Save
st_javascript(f"localStorage.setItem('session_data', '{json_data}')")

# Load
data = st_javascript("localStorage.getItem('session_data')")
```

### Rationale

`streamlit-javascript` is the simplest solution that provides bidirectional communication. Custom component is overkill for localStorage access.

---

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              st.session_state                       │   │
│  │  (in-memory, fast access during session)            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↑↓ sync                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           SessionPersistence Module                 │   │
│  │  - serialize(): session_state → JSON                │   │
│  │  - deserialize(): JSON → session_state              │   │
│  │  - save_to_storage(): JSON → localStorage           │   │
│  │  - load_from_storage(): localStorage → JSON         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↑↓ st_javascript                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Browser localStorage                   │   │
│  │  (persistent, survives refresh)                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Session Data Schema

```json
{
  "version": "1.0.0",
  "timestamp": "2025-12-04T12:00:00Z",
  "wizard_step": 2,
  "datasets": {
    "l4": "<compressed_base64>",
    "l3": "<compressed_base64>",
    "l2": {},
    "l1": {},
    "l0": {}
  },
  "raw_data": {
    "filename1.csv": "<compressed_base64>",
    "filename2.csv": "<compressed_base64>"
  },
  "selections": {
    "wizard_step_1": { "selected_columns": [...] },
    "wizard_step_2": { "connections": [...], "semantic_results": {...} }
  },
  "form_values": {
    "key1": "value1",
    "checkbox_key": true
  }
}
```

## Dependencies to Add

```
streamlit-javascript>=0.1.5
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| localStorage quota exceeded | Compress data, warn user, offer partial save |
| Data corruption | Version tag + validation on load |
| Browser doesn't support localStorage | Graceful degradation, continue without persistence |
| Large files (>50MB) | Skip file content, store only metadata with warning |

---

## Known Issues & Troubleshooting

### Streamlit Deprecation Warning (2025-12)

**Warning message:**
```
For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
```

**Cause:** Streamlit API change in newer versions - the `use_container_width` parameter is deprecated in favor of `width='stretch'` or `width='content'`.

**Fix:** Search for `use_container_width=True` in codebase and replace with `width='stretch'`. Search for `use_container_width=False` and replace with `width='content'`.

### Semaphore Leak Warning (Python 3.13)

**Warning message:**
```
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

**Cause:** Known issue with `loky` (used by joblib/scikit-learn) on Python 3.13.

**Workaround:** This is a warning, not an error. It doesn't affect functionality. Can be suppressed if needed.

### streamlit-javascript Auto-Save Crash

**Symptom:** Streamlit app shuts down when clicking buttons (e.g., "Continue" on L4→L3 transition).

**Cause:** `st_javascript()` is a Streamlit component that **renders visible elements** in the page. When called on every rerun (auto-save pattern), it interferes with Streamlit's page rendering lifecycle and causes crashes.

**Solution:** Changed from auto-save to manual "Save" button in sidebar. The `st_javascript` call only executes when the user explicitly clicks Save, avoiding lifecycle conflicts.

**Lesson learned:** `streamlit-javascript` is NOT suitable for background operations. For true background persistence, would need:
- Server-side file storage (pickle files)
- Custom Streamlit component with proper lifecycle handling
- Or server-side database (Redis, SQLite)
