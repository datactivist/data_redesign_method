# Quickstart: Ascent Functionality

**Feature**: Ascent Functionality (Reverse Navigation)
**Date**: 2025-12-04 (Updated)

## Overview

The Ascent functionality enables reverse navigation through abstraction levels:
- **L0 → L1**: Enrich a datum back to a vector
- **L1 → L2**: Add categorical dimensions to create a table
- **L2 → L3**: Add analytic dimensions for hierarchical grouping

## Navigation Rules

**Entry**: Users ALWAYS enter at L4 and MUST descend to L3.

**Exploration**: From L3, users can explore infinitely within L0-L3 boundaries:

```
     L4 (entry-only)
      │
      ↓ descend (required)
     L3 ←──────────────────┐
      │                    │
      ↓ descend    ascend ↑
     L2 ───────────────────┤
      │                    │
      ↓ descend    ascend ↑
     L1 ───────────────────┤
      │                    │
      ↓ descend    ascend ↑
     L0 ───────────────────┘
```

**Exit Options**: Exit is available at ANY level (L0-L3) with cumulative outputs.

## Quick Usage

### Using NavigationSession

```python
from intuitiveness import NavigationSession, Level4Dataset

# Start at L4 (entry point)
sources = {"sales": df_sales, "products": df_products}
l4 = Level4Dataset(sources)
nav = NavigationSession(l4)

# Descend to L0
nav.descend(builder_func=link_sales_products)   # L4 → L3
nav.descend(query_func=get_sales_table)         # L3 → L2
nav.descend(column="revenue")                   # L2 → L1
nav.descend(aggregation="sum")                  # L1 → L0

# Now ascend back up
nav.ascend(enrichment_func="source_expansion")  # L0 → L1
nav.ascend(dimensions=["business_object"])      # L1 → L2
nav.ascend(dimensions=["client_segment"])       # L2 → L3

# Check current position
print(nav.current_level)  # ComplexityLevel.LEVEL_3
```

### Check Available Moves

```python
moves = nav.get_available_moves()
# {
#   "descend": [{"target": "L2", "description": "..."}],
#   "ascend": [{"target": "L3", "description": "..."}]  # Only if < L3
# }

# At L3, ascent is blocked (L4 is entry-only)
if nav.current_level == ComplexityLevel.LEVEL_3:
    print(moves["ascend"])  # []
```

### Using Default Enrichment Functions

```python
from intuitiveness.ascent import EnrichmentRegistry

registry = EnrichmentRegistry.get_instance()

# List available functions for L0→L1
functions = registry.list_for_transition(ComplexityLevel.LEVEL_0, ComplexityLevel.LEVEL_1)
# [EnrichmentFunction(name="source_expansion", ...),
#  EnrichmentFunction(name="naming_signatures", ...)]

# Use a default function
nav.ascend(enrichment_func="source_expansion")
```

### Adding Custom Dimensions

```python
from intuitiveness.ascent import DimensionDefinition

# Define a custom dimension
region_dim = DimensionDefinition(
    name="region",
    description="Geographic region",
    possible_values=["North", "South", "East", "West", "Unknown"],
    classifier=lambda item: classify_region(item),
    default_value="Unknown"
)

# Apply during ascent
nav.ascend(dimensions=[region_dim])
```

## UI Integration (Streamlit)

### Decision-Tree Sidebar (FR-021 - DAG Display)

In Free Navigation Mode, the sidebar displays a **Directed Acyclic Graph (DAG)** showing:
- Your current position (highlighted in green)
- All navigation history (clickable for time-travel)
- **Decision made at each step** (entity, label, operation)
- **Output snapshot at every step**
- Branches when time-travel creates new paths

```
[L4: Entry with raw dataset]
         │ (1000 items)
         ↓
[L3: Make graph with Indicator, Source]
         │ (600 nodes, 1200 edges)
         ↓
[L2: Filter domain Revenue]
         │ (523 rows, 3 columns)
         ↓
[L1: Extract column name]
         │ (523 items)
         ↓
[L0: Aggregate count = 523]
         │
    ┌────┴────┐ (branch)
    ↓         ↓
[L1: naming_signatures] ← CURRENT
[L1: source_expansion]
```

Click any node to time-travel back to that state.

### Free Navigation Mode Options

At each level, the sidebar shows your available options per FR-011 through FR-014:

**At L3 (graph output):**
- Exit → Export cumulative outputs (Graph + NavigationTree)
- Descend to L2 (domain table)

**At L2 (domain table output):**
- Exit → Export cumulative outputs (Graph + Table + NavigationTree)
- Descend to L1 (vector)
- Ascend to L3 (specify relationships using entity selection)

**At L1 (vector output):**
- Exit → Export cumulative outputs (Graph + Table + Vector + NavigationTree)
- Descend to L0 (datum)
- Ascend to L2 (add domain)

**At L0 (datum output):**
- Exit → Export cumulative outputs (Graph + Table + Vector + Datum + NavigationTree)
- Ascend to L1 (unfold datum)

### L2→L3 Entity Selection (FR-020)

When ascending from L2 to L3, you can select additional entities from the original raw data:

1. **Column picker** shows all available columns from original L4 data
2. Select columns that should become **node types** in your graph
3. Define relationships between selected entities
4. Click "Create Graph" to complete ascent

### Drag-and-Drop Relationships (L2→L3)

After entity selection, use the visual relationship builder:

1. Selected entities appear as nodes
2. Drag to draw connections between entities
3. Label each relationship (e.g., "BELONGS_TO")
4. Click "Create Graph" to complete ascent

### JSON Export (FR-019)

On exit, receive **cumulative outputs** from ALL levels visited:

```json
{
  "version": "1.0",
  "navigation_tree": {
    "nodes": [
      {
        "id": "node_1",
        "level_name": "LEVEL_3",
        "action": "descend",
        "decision_description": "Make graph with entities Indicator, Source",
        "output_snapshot": {"type": "graph", "node_count": 600}
      }
    ]
  },
  "current_path": ["root", "node_1", ...],
  "current_output": {
    "level": 1,
    "level_name": "LEVEL_1",
    "output_type": "vector",
    "row_count": 523
  },
  "cumulative_outputs": {
    "graph": { "level": 3, "output_type": "graph", "node_count": 600 },
    "table": { "level": 2, "output_type": "dataframe", "row_count": 523 },
    "vector": { "level": 1, "output_type": "vector", "row_count": 523 },
    "datum": { "level": 0, "output_type": "datum", "sample_data": 523 }
  }
}
```

Each node in the navigation tree records (per FR-021):
- **(a)** Navigation step taken (`action`)
- **(b)** Decision made (`decision_description`)
- **(c)** Output snapshot at that step (`output_snapshot`)

## Key Constraints

| Rule | Enforcement |
|------|-------------|
| L4 is entry-only | `NavigationError` if ascend at L3 |
| Entry requires descent | MUST descend from L4 to L3 on entry |
| One level at a time | Must ascend L0→L1→L2→L3 sequentially |
| Data integrity | Row count preserved across ascent |
| Infinite exploration (FR-022) | No limits on navigation iterations within L0-L3 |
| Cumulative export (FR-019) | Exit exports ALL outputs from visited levels |

## Troubleshooting

### "L4 is entry-only"
You're at L3 and trying to ascend. L4 is the entry point only. Work with your L3 data or descend to continue.

### "Ascent requires enrichment function"
Either provide an `enrichment_func` parameter or use default functions from the registry.

### "Row count changed"
Data integrity error. The enrichment function added or removed items. Use a function that preserves the original item count.

## Time-Travel Navigation

### Restoring Previous States

```python
# Get tree visualization
tree_viz = nav.get_tree_visualization()
for node in tree_viz.nodes:
    print(f"{' ' * node.depth}{node.level.name} {'← CURRENT' if node.is_current else ''}")

# Restore to a previous node
nav.restore("node_2")  # Time-travel back to node_2
print(nav.current_level)  # Level at node_2
```

### Branching Exploration

When you restore and take a different path, both branches are preserved:

```python
# At L0, ascend with naming_signatures
nav.ascend(enrichment_func="naming_signatures")

# Time-travel back to L0
nav.restore("node_4")  # Back to L0

# Take a different path (creates new branch)
nav.ascend(enrichment_func="source_expansion")

# Both branches exist in the tree
tree = nav.get_tree_visualization()
# node_4 now has two children: naming_signatures AND source_expansion
```

## Dependencies

Add to requirements.txt:
```
streamlit-agraph>=0.0.45
# Note: Using built-in st.json() for JSON visualization (streamlit-json-viewer doesn't exist on PyPI)
```

## Next Steps

- See [data-model.md](./data-model.md) for entity details
- See [contracts/](./contracts/) for interface specifications
- Run `/speckit.tasks` to generate implementation tasks
