# Data Model: Dataset Redesign Package

**Feature**: 001-dataset-redesign-package
**Date**: 2025-12-02

## Entity Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Core Entities                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐         ┌──────────────────┐                           │
│  │    Dataset      │────────▶│  ComplexityLevel │                           │
│  │                 │         │                  │                           │
│  │ - data          │         │ L0 = DATUM       │                           │
│  │ - level         │         │ L1 = VECTOR      │                           │
│  │ - complexity    │         │ L2 = TABLE       │                           │
│  │ - lineage       │         │ L3 = LINKABLE    │                           │
│  └────────┬────────┘         │ L4 = UNLINKABLE  │                           │
│           │                  └──────────────────┘                           │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   DataLineage   │                                                        │
│  │                 │                                                        │
│  │ - source_ref    │                                                        │
│  │ - operation     │                                                        │
│  │ - parent        │──────────┐ (linked list)                               │
│  └─────────────────┘          │                                             │
│           ▲                   │                                             │
│           └───────────────────┘                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Operation Entities                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐       ┌─────────────────────┐                      │
│  │  DescentOperation   │       │   AscentOperation   │                      │
│  │                     │       │                     │                      │
│  │ + execute(Dataset)  │       │ + execute(Dataset)  │                      │
│  │   → Dataset         │       │   → Dataset         │                      │
│  └──────────┬──────────┘       └──────────┬──────────┘                      │
│             │                             │                                  │
│     ┌───────┴───────┬───────┐     ┌───────┴───────┐                         │
│     ▼               ▼       ▼     ▼               ▼                         │
│  ┌──────┐    ┌──────┐  ┌──────┐ ┌──────┐    ┌──────┐                        │
│  │Link  │    │Query │  │Select│ │Enrich│    │Group │                        │
│  │L4→L3 │    │L3→L2 │  │L2→L1 │ │L0→L1 │    │L1→L2 │                        │
│  └──────┘    └──────┘  └──────┘ └──────┘    └──────┘                        │
│                        ┌──────┐              ┌──────┐                        │
│                        │Agg   │              │Hier  │                        │
│                        │L1→L0 │              │L2→L3 │                        │
│                        └──────┘              └──────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Navigation Entities                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐       ┌─────────────────────┐                      │
│  │  NavigationSession  │──────▶│  NavigationHistory  │                      │
│  │                     │       │                     │                      │
│  │ - session_id        │       │ - steps: List[Step] │                      │
│  │ - state             │       │ + append(step)      │                      │
│  │ - current_position  │       │ + get_path()        │                      │
│  │ - history           │       └─────────────────────┘                      │
│  │                     │                                                    │
│  │ + descend()         │       ┌─────────────────────┐                      │
│  │ + ascend()          │       │  NavigationState    │                      │
│  │ + move_horizontal() │       │                     │                      │
│  │ + exit()            │       │ ENTRY               │                      │
│  │ + resume()          │       │ EXPLORING           │                      │
│  └─────────────────────┘       │ EXITED              │                      │
│                                └─────────────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Entity Definitions

### Dataset

The core wrapper for data at any complexity level.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `data` | `Any` | Raw data (value, list, DataFrame, Graph, dict) | Non-null |
| `level` | `ComplexityLevel` | Current abstraction level (L0-L4) | Must match data structure |
| `complexity_order` | `float` | Calculated complexity value | ≥ 0 for L0-L3, ∞ for L4 |
| `lineage` | `DataLineage \| None` | Transformation history | Optional |
| `metadata` | `dict` | User-defined attributes | Optional |

**Validation Rules**:
- L0: `data` must be a single scalar value
- L1: `data` must be a 1D sequence (list, Series, array)
- L2: `data` must be a 2D structure (DataFrame)
- L3: `data` must be a graph (networkx.Graph)
- L4: `data` must be a collection of disparate sources (dict of DataFrames/values)

---

### ComplexityLevel

Enumeration of the five abstraction levels.

| Value | Name | Description |
|-------|------|-------------|
| 0 | `DATUM` | Single entity-attribute-value triplet |
| 1 | `VECTOR` | Single dimension (1×N or N×1) |
| 2 | `TABLE` | Two-dimensional (N×M) |
| 3 | `LINKABLE` | Multi-level with defined relationships |
| 4 | `UNLINKABLE` | Multi-level without defined relationships |

---

### DataLineage

Linked list tracking transformation history.

| Field | Type | Description |
|-------|------|-------------|
| `source_ref` | `SourceReference` | Pointer to original data location |
| `operation` | `str` | Name of transformation applied |
| `parameters` | `dict` | Parameters used in transformation |
| `parent` | `DataLineage \| None` | Previous lineage entry |
| `timestamp` | `datetime` | When transformation occurred |

**SourceReference** (nested):
| Field | Type | Description |
|-------|------|-------------|
| `dataset_id` | `str` | UUID of source dataset |
| `row_indices` | `List[int] \| None` | Row positions (for L2+) |
| `column_name` | `str \| None` | Column name (for L1+) |
| `node_id` | `str \| None` | Graph node ID (for L3) |

---

### DescentOperation

Abstract base for complexity-reducing transformations.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `execute(dataset)` | `Dataset` | `Dataset` | Apply transformation, return new Dataset at lower level |
| `validate(dataset)` | `Dataset` | `bool` | Check if operation is valid for this dataset |

**Concrete Implementations**:

| Operation | Level Transition | Required Parameters |
|-----------|------------------|---------------------|
| `LinkOperation` | L4 → L3 | `linking_function: Callable[[dict], Graph]` |
| `QueryOperation` | L3 → L2 | `entity_type: str`, `filters: dict` |
| `SelectOperation` | L2 → L1 | `column: str`, `row_filter: Optional[Callable]` |
| `AggregateOperation` | L1 → L0 | `method: str` (count, sum, mean, min, max, custom) |

---

### AscentOperation

Abstract base for complexity-increasing transformations.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `execute(dataset, source)` | `Dataset`, `Dataset` | `Dataset` | Apply transformation using source data |
| `validate(dataset)` | `Dataset` | `bool` | Check if operation is valid |

**Concrete Implementations**:

| Operation | Level Transition | Required Parameters |
|-----------|------------------|---------------------|
| `EnrichOperation` | L0 → L1 | `source: Dataset`, `selection_criteria: dict` |
| `DimensionOperation` | L1 → L2 | `dimensions: List[str]`, `source: Dataset` |
| `HierarchyOperation` | L2 → L3 | `groupings: List[str]`, `relationships: dict` |

---

### NavigationSession

Stateful exploration context.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `UUID` | Unique session identifier |
| `state` | `NavigationState` | Current state (ENTRY, EXPLORING, EXITED) |
| `current_position` | `NavigationPosition` | Current level and node |
| `history` | `NavigationHistory` | Path of visited nodes |
| `data_context` | `Dataset` | The L4 dataset being explored |

| Method | Description |
|--------|-------------|
| `descend()` | Move down one level (L4→L3, L3→L2, L2→L1) |
| `ascend()` | Move up one level (L1→L2, L2→L3) - blocked at L3→L4 |
| `move_horizontal(node_id)` | Move to related node at same level |
| `exit()` | End navigation, preserve position |
| `resume()` | Continue from saved position |
| `get_available_moves()` | List valid moves from current position |

**State Transitions**:
| From State | Action | To State | Condition |
|------------|--------|----------|-----------|
| ENTRY | descend | EXPLORING | Always allowed |
| EXPLORING | descend | EXPLORING | level > L1 |
| EXPLORING | ascend | EXPLORING | level < L3 |
| EXPLORING | ascend | BLOCKED | level == L3 (would go to L4) |
| EXPLORING | horizontal | EXPLORING | Related nodes exist |
| EXPLORING | exit | EXITED | Always allowed |
| EXITED | resume | EXPLORING | Session not expired |

---

### NavigationHistory

Append-only log of navigation steps.

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `List[NavigationStep]` | Ordered list of visited positions |

**NavigationStep** (nested):
| Field | Type | Description |
|-------|------|-------------|
| `level` | `ComplexityLevel` | Level at this step |
| `node_id` | `str` | Identifier of node visited |
| `action` | `str` | Action taken (descend, ascend, horizontal) |
| `timestamp` | `datetime` | When step occurred |

---

## Relationships

```
Dataset 1──────1 ComplexityLevel    (each dataset has exactly one level)
Dataset 1──────0..1 DataLineage     (lineage optional, attached on transformation)
DataLineage 1──0..1 DataLineage     (parent reference, forms linked list)

NavigationSession 1──────1 Dataset             (explores one L4 dataset)
NavigationSession 1──────1 NavigationHistory   (tracks all steps)
NavigationSession 1──────1 NavigationState     (current state)

DescentOperation ──▷ Dataset → Dataset         (transforms, reducing level)
AscentOperation ──▷ Dataset → Dataset          (transforms, increasing level)
```

---

## Validation Rules Summary

| Entity | Rule | Error Message |
|--------|------|---------------|
| Dataset | Data structure matches level | "Data structure does not match level {level}" |
| Dataset | Non-empty data | "Dataset cannot be empty" |
| DescentOperation | Target level < source level | "Descent must reduce complexity" |
| AscentOperation | Target level > source level | "Ascent must increase complexity" |
| NavigationSession | Cannot return to L4 | "L4 is entry-only; cannot return" |
| NavigationSession | Must start at L4 | "Navigation must begin at L4" |
| NavigationHistory | Max 100 steps for display | Warning after 100 steps |
