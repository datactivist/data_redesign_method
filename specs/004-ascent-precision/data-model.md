# Data Model: Ascent Phase Precision

**Feature**: 004-ascent-precision
**Date**: 2025-12-04

## Entity Definitions

### Core Dataset Entities (Existing)

#### Level0Dataset
Single scalar value with aggregation metadata.

| Attribute | Type | Description |
|-----------|------|-------------|
| value | float/int | The aggregated scalar value |
| description | str | Human-readable description |
| aggregation_type | str | Method used (median, average, count, sum, etc.) |
| parent_data | pd.Series \| None | Source vector from which datum was aggregated |
| has_parent | bool | Property: True if parent_data exists |

**Key Methods**:
- `get_parent_data() -> pd.Series | None`: Returns the source vector for unfold operation

#### Level1Dataset
One-dimensional vector/series of values.

| Attribute | Type | Description |
|-----------|------|-------------|
| data | pd.Series | The vector values |
| name | str | Column/series name |
| dtype | str | Data type of values |

#### Level2Dataset
Two-dimensional table with domain categorization.

| Attribute | Type | Description |
|-----------|------|-------------|
| data | pd.DataFrame | The tabular data |
| domains | List[str] | Domain categories applied |
| categorization_method | str | "semantic" or "keyword" |
| similarity_threshold | float | Threshold used for semantic matching (0.1-0.9) |

#### Level3Dataset
Knowledge graph with typed nodes and relationships.

| Attribute | Type | Description |
|-----------|------|-------------|
| graph | nx.Graph | NetworkX graph object |
| node_types | Set[str] | Types of nodes in the graph |
| relationship_types | Set[str] | Types of edges in the graph |

---

### Ascent Operation Entities

#### AscentOperation
Tracks a single ascent transition between levels.

| Attribute | Type | Description |
|-----------|------|-------------|
| source_level | int | Starting level (0, 1, or 2) |
| target_level | int | Ending level (1, 2, or 3) |
| operation_type | str | "unfold", "domain_enrichment", or "graph_building" |
| parameters | Dict | Operation-specific parameters |
| timestamp | datetime | When operation was executed |

#### UnfoldParameters (L0→L1)
Parameters for the unfold operation.

| Attribute | Type | Description |
|-----------|------|-------------|
| aggregation_type | str | Original aggregation method being reversed |
| preserve_column_name | bool | Whether to keep original column name (default: True) |

**Note**: This operation is deterministic - no user input required beyond confirmation.

#### DomainEnrichmentParameters (L1→L2)
Parameters for domain categorization during ascent.

| Attribute | Type | Description |
|-----------|------|-------------|
| domains | List[str] | User-specified domain names |
| categorization_method | str | "semantic" or "keyword" |
| similarity_threshold | float | Threshold for semantic matching (0.1-0.9) |
| unmatched_label | str | Label for uncategorized values (default: "Unmatched") |

**Reuses**: Same structure as L3→L2 descent domain categorization.

#### GraphBuildingParameters (L2→L3)
Parameters for building graph from table.

| Attribute | Type | Description |
|-----------|------|-------------|
| entity_column | str | Column to extract as new entity type |
| entity_type_name | str | Name for the new entity type |
| relationship_type | str | Label for relationships connecting to new entities |
| source_entity_columns | List[str] | Columns identifying source entities (table rows) |

---

### UI Form Entities

#### AscentFormState
Tracks state of ascent form inputs in Streamlit session.

| Attribute | Type | Description |
|-----------|------|-------------|
| current_level | int | Current abstraction level |
| selected_operation | str \| None | Selected ascent operation type |
| form_valid | bool | Whether current form inputs are valid |
| validation_errors | List[str] | Current validation error messages |

#### L1ToL2FormState
State for L1→L2 domain enrichment form.

| Attribute | Type | Description |
|-----------|------|-------------|
| domain_input | str | Raw comma-separated domain input |
| parsed_domains | List[str] | Validated domain list |
| use_semantic | bool | Toggle for semantic vs keyword matching |
| threshold | float | Similarity threshold slider value |

#### L2ToL3FormState
State for L2→L3 graph building form.

| Attribute | Type | Description |
|-----------|------|-------------|
| available_columns | List[str] | Columns available for entity extraction |
| selected_column | str \| None | User-selected entity column |
| entity_type_name | str | User-defined name for new entity type |
| relationship_type | str | User-defined relationship label |

---

## Entity Relationships

```
Level0Dataset ─────unfold────► Level1Dataset
     │                              │
     │ parent_data                  │ domain_enrichment
     │ (stored reference)           │
     ▼                              ▼
[Source Vector]               Level2Dataset
                                    │
                                    │ graph_building
                                    │
                                    ▼
                              Level3Dataset
```

## Validation Rules

1. **L0→L1 Unfold**:
   - MUST have `parent_data` attribute populated
   - `has_parent` property MUST return True
   - Blocked if `parent_data` is None (orphan datum)

2. **L1→L2 Domain Enrichment**:
   - `domains` list MUST have at least 1 item
   - `similarity_threshold` MUST be between 0.1 and 0.9
   - `categorization_method` MUST be "semantic" or "keyword"

3. **L2→L3 Graph Building**:
   - `entity_column` MUST exist in source DataFrame
   - `entity_type_name` MUST not be empty
   - `relationship_type` MUST not be empty
   - Resulting graph MUST have no orphan nodes (all nodes connected)

## State Transitions

| From | To | Operation | Required Input | Deterministic |
|------|-----|-----------|----------------|---------------|
| L0 | L1 | Unfold | Confirmation only | Yes |
| L1 | L2 | Domain Enrichment | Domains, method, threshold | No |
| L2 | L3 | Graph Building | Entity column, relationship type | No |
