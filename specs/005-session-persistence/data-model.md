# Data Model: Session Persistence

**Feature**: 005-session-persistence
**Date**: 2025-12-04

## Entities

### PersistedSession

The root entity containing all persisted session data.

| Field | Type | Description |
|-------|------|-------------|
| version | string | Schema version for migration (e.g., "1.0.0") |
| timestamp | datetime | When session was last saved |
| wizard_step | integer | Current step (0-5) in guided mode |
| nav_mode | string | Navigation mode: "guided" or "free" |
| datasets | DatasetCollection | Persisted datasets at each level |
| raw_data | FileCollection | Original uploaded files |
| selections | SelectionState | User's wizard selections |
| form_values | dict | Generic form field values |
| entity_mapping | dict | Entity definitions from wizard |
| relationship_mapping | dict | Relationship definitions from wizard |
| semantic_results | dict | AI semantic matching results |

### DatasetCollection

Collection of datasets organized by abstraction level.

| Field | Type | Description |
|-------|------|-------------|
| l4 | SerializedDataset | Level 4 dataset (if exists) |
| l3 | SerializedDataset | Level 3 graph/dataset (if exists) |
| l2 | dict[str, SerializedDataset] | Level 2 datasets by domain |
| l1 | dict[str, SerializedDataset] | Level 1 vectors by domain |
| l0 | dict[str, number] | Level 0 datums by domain |

### SerializedDataset

A single serialized dataset.

| Field | Type | Description |
|-------|------|-------------|
| type | string | "dataframe", "graph", or "value" |
| data | string | Base64-encoded compressed JSON |
| metadata | dict | Additional info (columns, node count, etc.) |

### FileCollection

Collection of uploaded files.

| Field | Type | Description |
|-------|------|-------------|
| files | dict[str, SerializedFile] | Filename → file data mapping |

### SerializedFile

A single serialized file.

| Field | Type | Description |
|-------|------|-------------|
| name | string | Original filename |
| content | string | Base64-encoded compressed content |
| size | integer | Original file size in bytes |
| type | string | MIME type or file extension |

### SelectionState

User's selections organized by wizard step.

| Field | Type | Description |
|-------|------|-------------|
| step_1 | Step1Selections | Column discovery selections |
| step_2 | Step2Selections | Connection method selections |
| step_3 | Step3Selections | Category/domain selections |

### Step1Selections

| Field | Type | Description |
|-------|------|-------------|
| selected_files | list[str] | Filenames user selected to analyze |
| discovery_results | dict | Auto-discovered column relationships |

### Step2Selections

| Field | Type | Description |
|-------|------|-------------|
| connections | list[ConnectionDef] | Defined column connections |
| connection_methods | dict[str, str] | Column pair → method mapping |

### ConnectionDef

| Field | Type | Description |
|-------|------|-------------|
| file1 | string | Source file name |
| file2 | string | Target file name |
| col1 | string | Source column name |
| col2 | string | Target column name |
| method | string | "common_key", "embeddings", "exact_match", "force_match" |

### Step3Selections

| Field | Type | Description |
|-------|------|-------------|
| domains | list[str] | User-defined domain categories |
| categorizations | dict | Column → domain assignments |

---

## State Transitions

```
┌─────────────────┐
│   No Session    │
│   (fresh start) │
└────────┬────────┘
         │ user uploads files
         ▼
┌─────────────────┐
│  Session Active │◄────────────────┐
│  (in memory)    │                 │
└────────┬────────┘                 │
         │ auto-save (debounced)    │
         ▼                          │
┌─────────────────┐                 │
│ Session Saved   │                 │
│ (localStorage)  │                 │
└────────┬────────┘                 │
         │ browser refresh          │
         ▼                          │
┌─────────────────┐                 │
│Session Recovery │─────────────────┘
│  (on app load)  │     restore to session_state
└────────┬────────┘
         │ user clicks "Start Fresh"
         ▼
┌─────────────────┐
│ Session Cleared │
│ (back to start) │
└─────────────────┘
```

---

## Validation Rules

1. **Version compatibility**: On load, check `version` field. If major version differs, prompt user to start fresh.

2. **Timestamp expiry**: If `timestamp` > 7 days old, warn user that session may be stale.

3. **Data integrity**: Validate that deserializedDataFrames have expected columns, Graphs have valid structure.

4. **Size limits**: Before save, check total serialized size. If > 8MB, warn user; if > 10MB, skip large files.

---

## Serialization Format

### DataFrame Serialization

```python
# Serialize
json_str = df.to_json(orient='split', date_format='iso')
compressed = zlib.compress(json_str.encode('utf-8'))
base64_str = base64.b64encode(compressed).decode('ascii')

# Deserialize
compressed = base64.b64decode(base64_str)
json_str = zlib.decompress(compressed).decode('utf-8')
df = pd.read_json(StringIO(json_str), orient='split')
```

### Graph Serialization

```python
# Serialize
data = nx.node_link_data(G)
json_str = json.dumps(data)
compressed = zlib.compress(json_str.encode('utf-8'))
base64_str = base64.b64encode(compressed).decode('ascii')

# Deserialize
compressed = base64.b64decode(base64_str)
json_str = zlib.decompress(compressed).decode('utf-8')
data = json.loads(json_str)
G = nx.node_link_graph(data)
```
