# Research: Ascent Functionality

**Feature**: Ascent Functionality (Reverse Navigation)
**Date**: 2025-12-02
**Phase**: 0 - Research

## Background

The Data Redesign Method defines a Descent-Ascent cycle (Constitution Principle II) where:
- **Descent** reduces complexity from L4 to L0 by progressively isolating entities
- **Ascent** reconstructs complexity from L0 upward by intentionally adding dimensions

The descent path is fully implemented. This research explores the ascent path.

## Source: Research Paper Analysis

Based on Section 5.2 of the research paper "Intuitive Datasets" (v2_intuitive_datasets_revised.md):

### L0 → L1: Datum to Vector (Reconstruct Features)

**Case Study Finding**: Starting from an L0 metric like "count of indicators = 523", the ascent reconstructs the naming signatures of each indicator.

**Enrichment Pattern**:
```
Input:  Scalar (count = 523)
Output: Vector of feature tuples for each item
        e.g., [(first_word, word_count, has_underscore, char_count), ...]
```

**Default Enrichment Functions**:
1. **Naming Signature Extraction**: For each item in the source, extract:
   - First word of the name
   - Total word count
   - Character patterns (underscores, numbers)
   - String length

2. **Source Reference Expansion**: If the L0 datum tracks a reference to original data, re-expand to show all contributing items.

**Key Insight**: The L0→L1 transition requires knowledge of **what** was aggregated. The session must retain a reference to the original L1 data that produced the L0 metric.

### L1 → L2: Vector to Table (Add Categorical Dimensions)

**Case Study Finding**: The vector of indicator naming signatures is enriched with **business object categories**.

**Dimension Pattern**:
```
Input:  Vector (523 naming signatures)
Output: Table with categorical columns
        | indicator | first_word | business_object | is_calculated | has_weight |
```

**Default Dimension Types**:
1. **Business Object Classification**: revenue, volume, ETP, other
2. **Calculated Flag**: derived vs. raw indicator
3. **Weight Flag**: weighted vs. unweighted
4. **RSE Flag**: has relative standard error

**Classification Approaches**:
- Rule-based: Pattern matching on names (e.g., "ETP" prefix → ETP category)
- Lookup-based: Join against reference tables
- User-defined: Custom classification function

### L2 → L3: Table to Linkable (Add Analytic Dimensions)

**Case Study Finding**: The table gains **hierarchical analytic dimensions** that enable multi-level grouping and duplicate detection.

**Dimension Pattern**:
```
Input:  Table with basic categories
Output: Multi-level structure with analytic dimensions
        | indicator | ... | client_segment | sales_location | product_segment | financial_view | lifecycle_view |
```

**Default Analytic Dimensions**:
1. **Client Segmentation**: B2B, B2C, Government, etc.
2. **Sales Location**: Geographic breakdown
3. **Product Segmentation**: Product line groupings
4. **Financial View**: Revenue, Cost, Margin perspectives
5. **Lifecycle View**: Acquisition, Retention, Churn stages

**Use Case - Duplicate Detection**: Items sharing identical analytic dimensions are candidates for consolidation.

## Technical Findings

### Existing Implementation Analysis

**`intuitiveness/redesign.py`** already has `increase_complexity()` but with limitations:
- Only supports `enrichment_func` parameter (no default functions)
- L0→L1 transition not implemented (jumps directly to L2)
- No dimension definition structure

**`intuitiveness/navigation.py`** has `ascend()` method that:
- Correctly blocks L3→L4 (FR-004)
- Delegates to `Redesigner.increase_complexity()`
- Updates history appropriately

### Required Extensions

1. **L0→L1 Implementation**: Add `_increase_0_to_1()` method in Redesigner
2. **Default Enrichment Registry**: Provide out-of-box enrichment functions
3. **Dimension Definition**: Structured way to define and apply dimensions
4. **Session Context**: Preserve references to parent data for enrichment

## Design Considerations

### Data Integrity (FR-005)

Each ascent must preserve traceability:
- L1 vector must aggregate back to original L0 value
- L2 table rows must map 1:1 to L1 vector elements
- L3 dimensions must not create/destroy rows (only add columns)

### Default Enrichment Options (FR-006)

Provide at least 2 defaults per transition:
| Transition | Default 1 | Default 2 |
|------------|-----------|-----------|
| L0→L1 | Naming signature extraction | Source reference expansion |
| L1→L2 | Business object classification | Pattern-based categorization |
| L2→L3 | Analytic dimension template | Duplicate detection grouping |

### UI Integration (FR-009)

The navigation panel already shows ascent options via `get_available_moves()`. Need to add:
- Enrichment function selector
- Dimension configuration UI
- Preview of enriched data before committing

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| L0→L1 requires source data reference | High | Store parent dataset reference in Level0Dataset |
| Custom enrichment functions may fail | Medium | Validate output structure before accepting |
| Large datasets slow during enrichment | Medium | Add progress indicators, batch processing |
| Dimension classification may be inaccurate | Low | Allow "Unknown" category, manual override |

## Recommendations

1. **Extend Level0Dataset** to optionally store reference to parent L1 data
2. **Create EnrichmentRegistry** class with default functions and custom registration
3. **Create DimensionDefinition** dataclass with name, values, classifier function
4. **Add validation step** in Redesigner to verify output matches target level requirements
5. **Update UI** to show enrichment options and previews

## Phase 0 Update: UI Research (2025-12-03)

Based on clarification session, additional research on UI components:

### Decision-Tree Sidebar (FR-009, FR-017)

**Decision**: Use native Streamlit components with custom CSS for decision-tree visualization

**Rationale**:
- Streamlit's native sidebar supports expanders, buttons, and custom styling
- Decision-tree is simple (max 4 levels L0-L3)
- `st.button` with dynamic styling for clickable nodes
- `st.container` for visual hierarchy

**Implementation approach**:
```python
def render_decision_tree(navigation_tree: NavigationTree):
    with st.sidebar:
        st.subheader("Navigation Path")
        for node in navigation_tree.get_all_paths():
            indent = "  " * node.depth
            if node.id == navigation_tree.current_id:
                st.markdown(f"**{indent}→ {node.level.name}** (current)")
            else:
                if st.button(f"{indent}{node.level.name}", key=node.id):
                    navigation_tree.restore(node.id)
```

**Alternatives considered**:
- `streamlit-agraph`: More visual but heavier dependency
- `st.graphviz_chart`: Static, not clickable
- Custom React component: Overkill for this use case

### Time-Travel Navigation (FR-017, FR-018)

**Decision**: Extend NavigationHistory to NavigationTree with branching support

**Rationale**:
- Current NavigationHistory is linear (append-only)
- Need to preserve branches when user time-travels and takes different path
- Each node stores dataset snapshot for restoration

**Data structure**:
```python
@dataclass
class NavigationTreeNode:
    id: str
    level: ComplexityLevel
    dataset_snapshot: Dataset  # Full dataset at this point
    parent_id: Optional[str]
    children_ids: List[str]
    action: str  # "entry", "descend", "ascend"
    timestamp: datetime
    metadata: Dict[str, Any]  # Enrichment/dimension info

class NavigationTree:
    nodes: Dict[str, NavigationTreeNode]
    root_id: str
    current_id: str

    def branch(self, action: str, dataset: Dataset) -> str: ...
    def restore(self, node_id: str) -> Dataset: ...
    def get_current_branch_path(self) -> List[NavigationTreeNode]: ...
    def export_to_json(self) -> dict: ...
```

### JSON Crack-Style Export (FR-015)

**Decision**: Use `streamlit_json_viewer` for inline visualization, JSON file for export

**Rationale**:
- `streamlit_json_viewer` provides collapsible tree view in Streamlit
- JSON Crack is web-based and can't be easily embedded
- Export produces structured JSON that can be opened in JSON Crack separately

**Export format**:
```json
{
  "version": "1.0",
  "feature": "002-ascent-functionality",
  "exported_at": "2025-12-03T10:30:00Z",
  "navigation_tree": {
    "nodes": [
      {
        "id": "root",
        "level": 4,
        "level_name": "LEVEL_4",
        "action": "entry",
        "timestamp": "...",
        "parent_id": null,
        "children_ids": ["node_1"]
      },
      ...
    ],
    "current_path": ["root", "node_1", "node_2"]
  },
  "current_output": {
    "level": 2,
    "level_name": "LEVEL_2",
    "type": "dataframe",
    "columns": ["name", "business_object", "..."],
    "row_count": 523,
    "sample_rows": [...]
  }
}
```

**Dependencies**:
```
streamlit-json-viewer>=0.0.2
```

### Drag-and-Drop for L2→L3 (FR-016)

**Decision**: Use `streamlit-agraph` for visual relationship building

**Rationale**:
- `streamlit-agraph` supports bidirectional communication
- Can render entities as nodes, user draws edges
- Returns edge list that defines relationships

**Implementation approach**:
1. Extract unique entities from L2 DataFrame columns
2. Render as nodes in agraph
3. User clicks two nodes to create edge
4. User labels relationship type
5. Convert edge list to NetworkX graph for L3

**Example UI flow**:
```
[Entity A] -----> [Entity B]
                    ↓
        "Relationship type: BELONGS_TO"
```

**Dependencies**:
```
streamlit-agraph>=0.0.45
```

### Requirements.txt Additions

```
streamlit-agraph>=0.0.45
streamlit-json-viewer>=0.0.2
```

## Summary of All Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| L0→L1 ascent | Naming signature + source expansion | Research paper patterns |
| L1→L2 ascent | Business object + pattern dimensions | Categorical classification |
| L2→L3 ascent | Analytic dimensions + drag-drop | Hierarchical grouping |
| Decision-tree sidebar | Native Streamlit + CSS | Simple, maintainable |
| Time-travel | NavigationTree with snapshots | Branching support |
| JSON export | streamlit-json-viewer + file | Interactive + portable |
| Relationship builder | streamlit-agraph | Visual graph editing |

## Phase 0 Update: Spec Changes (2025-12-04)

### New Requirements Added

#### FR-019: Cumulative Output Export

**Decision**: Store accumulated outputs at each level in NavigationSession

**Rationale**:
- User needs ALL outputs accumulated during session, not just current level
- Exit at L0 should include Graph (from L3) + Table (from L2) + Vector (from L1) + Datum (from L0)

**Implementation approach**:
```python
class NavigationSession:
    accumulated_outputs: Dict[int, Any] = {}  # level -> output

    def on_level_change(self, new_level: int, output: Any):
        self.accumulated_outputs[new_level] = output

    def get_cumulative_export(self) -> Dict[str, Any]:
        return {
            "graph": self.accumulated_outputs.get(3),
            "table": self.accumulated_outputs.get(2),
            "vector": self.accumulated_outputs.get(1),
            "datum": self.accumulated_outputs.get(0),
            "navigation_tree": self.tree.export_to_json()
        }
```

#### FR-020: L2→L3 Entity Selection from Raw Data

**Decision**: Provide column selector from original L4 data for graph entity selection

**Rationale**:
- L2→L3 ascent recreates graph structure
- User needs to pick which columns become nodes/relationships
- Original raw data has all available columns

**Implementation approach**:
- Store reference to L4 dataset columns in session
- Present column picker during L2→L3 ascent
- Selected columns define node types for new graph

#### FR-021: NavigationTree as DAG

**Decision**: Update DecisionTree UI to render as proper DAG visualization

**Rationale**:
- DAG structure ensures no cycles (navigation is always forward in time)
- Branching must be visually clear when time-travel creates new paths
- Each node must show: step number, decision made, output snapshot

**Implementation approach**:
```python
class NavigationTreeNode:
    # Existing fields plus:
    decision_description: str  # "make graph with entity X"
    output_snapshot: Dict[str, Any]  # Summary of output at this step

def render_dag(tree: NavigationTree):
    nodes = []
    edges = []
    for node in tree.nodes.values():
        nodes.append(Node(
            id=node.id,
            label=f"L{node.level}: {node.decision_description}",
            color="green" if node.id == tree.current_id else "gray"
        ))
        if node.parent_id:
            edges.append(Edge(source=node.parent_id, target=node.id))

    return agraph(nodes=nodes, edges=edges, config=dag_config)
```

**Dependencies**: Already using `streamlit-agraph>=0.0.45`

#### FR-022: Infinite Exploration

**Decision**: No artificial limits on navigation iterations

**Rationale**:
- Users may need to explore many paths before finding optimal solution
- Constitution Principle I.4 states "number of navigation steps is not finite"

**Implementation approach**:
- Remove any iteration counters or limits
- NavigationTree can grow indefinitely
- Performance considerations: lazy loading of old branches

### Corrections

**streamlit-json-viewer**: Package does not exist on PyPI. Using Streamlit's built-in `st.json()` with `streamlit-agraph` for tree visualization instead.

## Unresolved Questions

All technical questions resolved.
