# Data Model: Ascent Functionality

**Feature**: Ascent Functionality (Reverse Navigation)
**Date**: 2025-12-02
**Phase**: 1 - Design

## Entity Definitions

### EnrichmentFunction

Callable that transforms data from a lower level to a higher level.

```python
@dataclass
class EnrichmentFunction:
    """
    A callable that takes data from a lower level and produces enriched data
    for a higher level.
    """
    name: str                           # Unique identifier
    description: str                    # User-facing description
    source_level: ComplexityLevel       # L0, L1, or L2
    target_level: ComplexityLevel       # L1, L2, or L3 (source + 1)
    func: Callable[[Any], Any]          # The enrichment callable
    requires_context: bool = False      # Whether parent data is needed

    def __call__(self, data: Any, context: Optional[Any] = None) -> Any:
        """Execute the enrichment function."""
        if self.requires_context and context is None:
            raise ValueError(f"{self.name} requires context data")
        return self.func(data) if not self.requires_context else self.func(data, context)
```

**Relationships**:
- Used by `AscentOperation` (1:1 per operation)
- Registered in `EnrichmentRegistry` (many:1)

### DimensionDefinition

Specifies a categorical dimension to add during L1→L2 or L2→L3 ascent.

```python
@dataclass
class DimensionDefinition:
    """
    Specifies a categorical dimension to add during ascent.
    """
    name: str                                    # Column name
    description: str                             # User-facing description
    possible_values: List[str]                   # Known categories + "Unknown"
    classifier: Callable[[Any], str]             # Function to classify each item
    default_value: str = "Unknown"               # Fallback for unclassifiable items

    def classify(self, item: Any) -> str:
        """Classify a single item into a category."""
        try:
            result = self.classifier(item)
            return result if result in self.possible_values else self.default_value
        except Exception:
            return self.default_value
```

**Relationships**:
- Applied in `AscentOperation` (0..* dimensions per operation)
- Used by `Redesigner.increase_complexity()` for L1→L2 and L2→L3

### AscentOperation

Records an ascent action for history and traceability.

```python
@dataclass
class AscentOperation:
    """
    Records an ascent action including source level, target level,
    enrichment function used, and resulting data structure.
    """
    id: str                                      # UUID
    source_level: ComplexityLevel                # Where we started
    target_level: ComplexityLevel                # Where we ended
    enrichment_function: str                     # Name of EnrichmentFunction used
    dimensions_added: List[str]                  # Names of dimensions added (L1→L2, L2→L3)
    timestamp: datetime                          # When operation occurred
    source_data_hash: str                        # Hash of input data for integrity
    result_data_hash: str                        # Hash of output data
    row_count_before: int                        # Items before ascent
    row_count_after: int                         # Items after ascent (should match)

    def validate_integrity(self) -> bool:
        """Check that row counts match (ascent should not add/remove items)."""
        return self.row_count_before == self.row_count_after
```

**Relationships**:
- Logged in `NavigationHistory` via `NavigationStep`
- References `EnrichmentFunction` by name

### EnrichmentRegistry

Manages available enrichment functions and provides defaults.

```python
class EnrichmentRegistry:
    """
    Registry of available enrichment functions.
    Provides defaults and allows custom registration.
    """
    _functions: Dict[str, EnrichmentFunction]

    def register(self, func: EnrichmentFunction) -> None: ...
    def get(self, name: str) -> EnrichmentFunction: ...
    def list_for_transition(self, source: ComplexityLevel, target: ComplexityLevel) -> List[EnrichmentFunction]: ...
    def get_defaults(self, source: ComplexityLevel, target: ComplexityLevel) -> List[EnrichmentFunction]: ...
```

## Data Flow

### L0 → L1 Ascent

```
┌─────────────────┐     EnrichmentFunction      ┌──────────────────┐
│  Level0Dataset  │ ─────────────────────────▶  │  Level1Dataset   │
│  (scalar value) │     requires_context=True   │  (pd.Series)     │
│  + parent_ref   │                             │                  │
└─────────────────┘                             └──────────────────┘
        │                                                │
        │  parent_ref holds                              │
        ▼  reference to original                         ▼
┌─────────────────┐                             ┌──────────────────┐
│ Original L1     │  ◀───── re-expansion ────── │ Enriched vector  │
│ (before agg)    │                             │ with features    │
└─────────────────┘                             └──────────────────┘
```

### L1 → L2 Ascent

```
┌─────────────────┐     DimensionDefinition[]   ┌──────────────────┐
│  Level1Dataset  │ ─────────────────────────▶  │  Level2Dataset   │
│  (pd.Series)    │     + EnrichmentFunction    │  (pd.DataFrame)  │
│  n items        │                             │  n rows × m cols │
└─────────────────┘                             └──────────────────┘
        │                                                │
        │  Each item                                     │
        ▼  becomes a row                                 ▼
┌─────────────────┐                             ┌──────────────────┐
│ item_1          │    classify each            │ item_1 | cat_A   │
│ item_2          │    using dimension          │ item_2 | cat_B   │
│ item_3          │    classifiers              │ item_3 | cat_A   │
└─────────────────┘                             └──────────────────┘
```

### L2 → L3 Ascent

```
┌─────────────────┐     DimensionDefinition[]   ┌──────────────────┐
│  Level2Dataset  │ ─────────────────────────▶  │  Level3Dataset   │
│  (pd.DataFrame) │     analytic dimensions     │  (pd.DataFrame)  │
│  n × m          │                             │  n × (m + k)     │
└─────────────────┘                             └──────────────────┘
        │                                                │
        │  Add k analytic                                │
        ▼  dimension columns                             ▼
┌───────────────────────┐                       ┌───────────────────────────────┐
│ item | cat | ...      │                       │ item | cat | segment | view   │
│                       │   add hierarchical    │                               │
│                       │   grouping dimensions │                               │
└───────────────────────┘                       └───────────────────────────────┘
```

## Extended Level0Dataset

To support L0→L1 ascent, Level0Dataset needs to store a reference to parent data:

```python
class Level0Dataset(Dataset):
    """
    Level 0: Data Point.
    Extended to support ascent by storing parent reference.
    """
    def __init__(
        self,
        value: Any,
        description: str = "value",
        parent_data: Optional[pd.Series] = None,  # NEW: for ascent
        aggregation_method: Optional[str] = None   # NEW: how it was computed
    ):
        self._value = value
        self.description = description
        self._parent_data = parent_data
        self._aggregation_method = aggregation_method

    @property
    def has_parent(self) -> bool:
        return self._parent_data is not None

    def get_parent_data(self) -> Optional[pd.Series]:
        return self._parent_data
```

## Default Enrichment Functions

### L0 → L1 Defaults

| Name | Description | Implementation |
|------|-------------|----------------|
| `source_expansion` | Re-expand to original vector | Return `parent_data` from Level0Dataset |
| `naming_signatures` | Extract naming features | Parse each item name for patterns |

### L1 → L2 Defaults

| Name | Description | Implementation |
|------|-------------|----------------|
| `business_object_classification` | Classify by business type | Pattern match: revenue/volume/ETP/other |
| `pattern_categorization` | Group by naming patterns | First word, prefix, suffix analysis |

### L2 → L3 Defaults

| Name | Description | Implementation |
|------|-------------|----------------|
| `analytic_dimensions` | Add standard analytic dims | Client/Sales/Product segmentation template |
| `duplicate_detection` | Group by dimension similarity | Flag items with identical dimension values |

## UI Entities (Phase 0 Update: 2025-12-03)

### NavigationTreeNode

Node in the branching navigation tree for time-travel support.

```python
@dataclass
class NavigationTreeNode:
    """
    A single node in the navigation tree, supporting branching paths.
    """
    id: str                                      # UUID
    level: ComplexityLevel                       # L0, L1, L2, L3, or L4
    dataset_snapshot: Dataset                    # Full dataset at this point
    parent_id: Optional[str]                     # Parent node ID (None for root)
    children_ids: List[str]                      # Child node IDs (branches)
    action: str                                  # "entry", "descend", "ascend"
    timestamp: datetime                          # When this node was created
    metadata: Dict[str, Any]                     # Enrichment/dimension info

    @property
    def depth(self) -> int:
        """Depth in tree (for UI indentation)."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export (excludes dataset_snapshot)."""
        ...
```

**Relationships**:
- Part of `NavigationTree` (many:1)
- Contains `Dataset` snapshot (1:1)

### NavigationTree

Branching tree structure for time-travel navigation.

```python
class NavigationTree:
    """
    Branching tree structure tracking all navigation decisions.
    Supports time-travel by preserving multiple exploration branches.
    """
    nodes: Dict[str, NavigationTreeNode]         # All nodes by ID
    root_id: str                                 # Entry point node
    current_id: str                              # Currently active node

    def branch(self, action: str, dataset: Dataset, metadata: Dict = None) -> str:
        """Create a new child node from current position."""
        ...

    def restore(self, node_id: str) -> Dataset:
        """Restore state to a previous node (time-travel)."""
        ...

    def get_current_branch_path(self) -> List[NavigationTreeNode]:
        """Get path from root to current node."""
        ...

    def get_all_branches(self) -> List[List[NavigationTreeNode]]:
        """Get all paths (for tree visualization)."""
        ...

    def export_to_json(self) -> dict:
        """Export full tree for JSON Crack visualization."""
        ...
```

**Relationships**:
- Replaces linear `NavigationHistory` in `NavigationSession`
- Contains multiple `NavigationTreeNode` instances

### NavigationExport

JSON export format for navigation path and output.

```python
@dataclass
class NavigationExport:
    """
    Export format for navigation tree and current output.
    Compatible with JSON Crack visualization.
    """
    version: str = "1.0"
    feature: str = "002-ascent-functionality"
    exported_at: datetime = field(default_factory=datetime.now)
    navigation_tree: Dict[str, Any]              # Tree structure
    current_path: List[str]                      # Node IDs from root to current
    current_output: Dict[str, Any]               # Level, type, data summary

    def to_json(self) -> str:
        """Serialize to JSON string."""
        ...

    @classmethod
    def from_tree(cls, tree: NavigationTree) -> 'NavigationExport':
        """Create export from navigation tree."""
        ...
```

### RelationshipDefinition

User-defined relationship for L2→L3 drag-and-drop interface.

```python
@dataclass
class RelationshipDefinition:
    """
    A user-defined relationship between entities in the drag-and-drop interface.
    """
    source_entity: str                           # Source column/entity name
    target_entity: str                           # Target column/entity name
    relationship_type: str                       # User-provided label (e.g., "BELONGS_TO")
    bidirectional: bool = False                  # Whether edge is bidirectional

    def to_networkx_edge(self) -> Tuple[str, str, Dict]:
        """Convert to NetworkX edge format."""
        return (self.source_entity, self.target_entity, {"type": self.relationship_type})
```

**Relationships**:
- Created by drag-and-drop UI
- Used to construct L3 graph structure

## Entity Relationship Diagram

```
┌─────────────────────┐       ┌─────────────────────┐
│ NavigationTree      │ 1   * │ NavigationTreeNode  │
│                     │───────│                     │
│ - nodes             │       │ - id                │
│ - root_id           │       │ - level             │
│ - current_id        │       │ - dataset_snapshot  │
└─────────────────────┘       │ - parent_id         │
         │                    │ - children_ids      │
         │                    │ - action            │
         │                    │ - timestamp         │
         │                    │ - metadata          │
         │                    └─────────────────────┘
         │
         │ exports to
         ▼
┌─────────────────────┐       ┌─────────────────────┐
│ NavigationExport    │       │ AscentOperation     │
│                     │       │                     │
│ - version           │       │ - id                │
│ - navigation_tree   │       │ - source_level      │
│ - current_path      │       │ - target_level      │
│ - current_output    │       │ - enrichment_func   │
└─────────────────────┘       │ - dimensions_added  │
                              └─────────────────────┘
                                       │
                                       │ uses
                                       ▼
┌─────────────────────┐       ┌─────────────────────┐
│ EnrichmentRegistry  │ 1   * │ EnrichmentFunction  │
│                     │───────│                     │
│ - _functions        │       │ - name              │
│ - register()        │       │ - source_level      │
│ - get()             │       │ - target_level      │
│ - list_for_transi() │       │ - func              │
└─────────────────────┘       └─────────────────────┘

┌─────────────────────┐       ┌─────────────────────┐
│ DimensionRegistry   │ 1   * │ DimensionDefinition │
│                     │───────│                     │
│ - _dimensions       │       │ - name              │
│ - register()        │       │ - possible_values   │
│ - get()             │       │ - classifier        │
└─────────────────────┘       └─────────────────────┘

┌─────────────────────┐
│RelationshipDefinition│
│                     │
│ - source_entity     │
│ - target_entity     │
│ - relationship_type │
│ - bidirectional     │
└─────────────────────┘
```

## State Transitions

### Navigation State Machine (Updated)

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
┌──────┐ entry   ┌─────────┐  descend/ascend  ┌───────────┐  │ restore
│ L4   │────────▶│ ENTRY   │─────────────────▶│ EXPLORING │──┘
│      │         │ (root)  │                  │           │
└──────┘         └─────────┘                  └───────────┘
                                                   │  │
                                              exit │  │ branch (new path)
                                                   │  │
                                                   ▼  ▼
                                              ┌───────────┐
                                              │  EXITED   │
                                              │           │
                                              └───────────┘
```

### Ascent State Transitions per Level

```
L0 (Datum)
  ├── [Exit] → Export JSON with datum + path
  └── [Ascend to L1] → Unfold datum to vector
                       ├── source_expansion (re-expand original)
                       └── naming_signatures (extract features)

L1 (Vector)
  ├── [Exit] → Export JSON with vector + path
  ├── [Descend to L0] → Aggregate to datum
  └── [Ascend to L2] → Add domain dimensions
                       ├── business_object (revenue/volume/ETP)
                       └── pattern_type (aggregated/ratio/raw)

L2 (Domain Table)
  ├── [Exit] → Export JSON with table + path
  ├── [Descend to L1] → Extract column as vector
  └── [Ascend to L3] → Specify relationships (drag-drop)
                       ├── client_segment (B2B/B2C/Gov)
                       └── financial_view (Revenue/Cost/Margin)

L3 (Graph)
  ├── [Exit] → Export JSON with graph + path
  └── [Descend to L2] → Query subgraph as table
```

## Spec Updates (2025-12-04)

### NavigationTreeNode Enhanced for DAG Display (FR-021)

```python
@dataclass
class NavigationTreeNode:
    """
    A single node in the navigation tree, displayed as DAG.
    Enhanced to record decision descriptions and output snapshots.
    """
    id: str                                      # UUID
    level: ComplexityLevel                       # L0, L1, L2, L3, or L4
    dataset_snapshot: Dataset                    # Full dataset at this point
    parent_id: Optional[str]                     # Parent node ID (None for root)
    children_ids: List[str]                      # Child node IDs (branches)
    action: str                                  # "entry", "descend", "ascend"
    timestamp: datetime                          # When this node was created
    metadata: Dict[str, Any]                     # Enrichment/dimension info
    decision_description: str                    # NEW: Human-readable decision (e.g., "make graph with entity X")
    output_snapshot: Dict[str, Any]              # NEW: Summary of output at this step
```

### Cumulative Output Tracking (FR-019)

```python
class NavigationSession:
    """
    Extended to track accumulated outputs at each level.
    """
    tree: NavigationTree
    accumulated_outputs: Dict[int, Any]          # NEW: level -> output snapshot

    def on_transition(self, to_level: int, output: Dataset) -> None:
        """Called after each navigation transition."""
        self.accumulated_outputs[to_level] = self._create_output_snapshot(output)

    def get_cumulative_export(self) -> Dict[str, Any]:
        """
        Get all accumulated outputs for exit export.

        Returns:
            Dict with graph (L3), table (L2), vector (L1), datum (L0),
            and navigation_tree - only includes visited levels.
        """
        return {
            "graph": self.accumulated_outputs.get(3),
            "domain_table": self.accumulated_outputs.get(2),
            "vector": self.accumulated_outputs.get(1),
            "datum": self.accumulated_outputs.get(0),
            "navigation_tree": self.tree.export_to_json()
        }
```

### L2→L3 Entity Selection from Raw Data (FR-020)

```python
class NavigationSession:
    """
    Extended to preserve L4 column reference for L2→L3 ascent.
    """
    raw_data_columns: List[str]                  # NEW: Available columns from L4

    def set_raw_data_reference(self, columns: List[str]) -> None:
        """Store available columns from original L4 data."""
        self.raw_data_columns = columns

    def get_available_graph_entities(self) -> List[str]:
        """
        Get available columns for L2→L3 graph entity selection.

        Returns:
            List of column names from original raw data
        """
        return self.raw_data_columns
```

### Updated Exit Output Structure

On exit at any level, export ALL accumulated outputs:

| Exit Level | Exported Outputs |
|------------|------------------|
| L3 | `{ graph, navigation_tree }` |
| L2 | `{ graph, domain_table, navigation_tree }` |
| L1 | `{ graph, domain_table, vector, navigation_tree }` |
| L0 | `{ graph, domain_table, vector, datum, navigation_tree }` |

### NavigationTree as DAG Visualization

```python
class NavigationTree:
    """
    Branching tree structure displayed as Directed Acyclic Graph.
    """

    def render_as_dag(self) -> Tuple[List[Node], List[Edge]]:
        """
        Convert tree to streamlit-agraph format for DAG display.

        Returns:
            Tuple of (nodes, edges) for agraph rendering
        """
        nodes = []
        edges = []
        for node in self.nodes.values():
            nodes.append(Node(
                id=node.id,
                label=f"L{node.level.value}: {node.decision_description}",
                color="#4CAF50" if node.id == self.current_id else "#9E9E9E",
                title=f"Output: {node.output_snapshot.get('type', 'N/A')}"
            ))
            if node.parent_id:
                edges.append(Edge(
                    source=node.parent_id,
                    target=node.id,
                    color="#666666"
                ))
        return nodes, edges
```

### Infinite Exploration (FR-022)

No limit on navigation iterations. NavigationTree grows unbounded.
