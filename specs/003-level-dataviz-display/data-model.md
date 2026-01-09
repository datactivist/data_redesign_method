# Data Model: Level-Specific Data Visualization Display

**Feature**: 003-level-dataviz-display
**Date**: 2025-12-04
**Constitution**: v1.2.0 (Target User Assumption)

> **Design Principle**: All user-facing labels use domain language. Technical terms are internal-only.

## Entities

### LevelDisplay

Represents the visualization configuration for a specific abstraction level.

| Field | Type | Description |
|-------|------|-------------|
| level | int (0-4) | The abstraction level (L0, L1, L2, L3, L4) |
| display_type | string | Internal type: "file_list", "connected_info_tabs", "categorized_items", "value_list", "single_result" |
| user_title | string | Domain-friendly title (e.g., "Your Uploaded Files", "Browse by Category") |
| show_counts | boolean | Whether to show item/row counts |
| max_preview_rows | int | Maximum rows to show in preview (default: 50) |

**Validation Rules**:
- level must be 0-4
- display_type must match level (see mapping below)
- user_title must NOT contain technical terms (graph, table, vector, datum, entity, node, edge, schema)

**Level to Display Type Mapping**:

| Level | Internal Type | User-Facing Title | What User Sees |
|-------|--------------|-------------------|----------------|
| L4 | file_list | "Your Uploaded Files" | File names, item counts, column counts |
| L3 | connected_info_tabs | "Browse by Category" | Category tabs + connection tabs |
| L2 | categorized_items | "Items by Category" | Items grouped with category labels |
| L1 | value_list | "Your Selected Values" | List of values with context |
| L0 | single_result | "Your Computed Result" | Single value prominently displayed |

### NavigationDirection

Indicates whether user is ascending or descending.

| Field | Type | Description |
|-------|------|-------------|
| direction | enum | "exploring_deeper" (descent) or "building_up" (ascent) |
| source_level | int | Level user is coming from |
| target_level | int | Level user is going to |

**User-Facing Direction Labels**:
- Descent (L4→L0): "Exploring deeper" or "Getting more specific"
- Ascent (L0→L3): "Building up" or "Adding context"

**Validation Rules**:
- For "exploring_deeper": target_level = source_level - 1 (L4→L3→L2→L1→L0)
- For "building_up": target_level = source_level + 1 (L0→L1→L2→L3)
- Cannot ascend from L3 (L4 is entry-only)

### CategoryTab (formerly EntityTab)

A tabbed view showing all items of a single category from the connected information.

| Field | Type | Description |
|-------|------|-------------|
| category_name | string | The category/type of items in this tab (domain-friendly) |
| item_count | int | Number of items of this type |
| columns | list[string] | Column names to display |
| data | list[dict] | Item data rows |

**User-Facing Labels**:
- Tab icon: Use contextual emoji (e.g., 🏫 for College, 📊 for Performance)
- Tab title: Category name without technical prefix

**Validation Rules**:
- category_name must not be "Source" (excluded from display)
- Must have human-readable columns, avoid internal IDs unless necessary

### ConnectionTab (formerly RelationshipTab)

A tabbed view showing linked item pairs for a specific connection type.

| Field | Type | Description |
|-------|------|-------------|
| connection_key | string | Format: "{from_category} → {to_category}" |
| connection_type | string | The connection type (e.g., "has performance of", "enrolls") |
| connection_count | int | Number of connections |
| columns | list[string] | Column names to display |
| data | list[dict] | Connection data rows |

**User-Facing Labels**:
- Tab icon: 🔗 (link emoji)
- Tab title: e.g., "College → Performance" or "How colleges connect to performance"
- Column headers: "From", "Connection", "To" (not "start_name", "relationship", "end_name")

**Validation Rules**:
- Must have user-friendly column headers
- Excludes connections involving "Source" type items

## Relationships

```
LevelDisplay
    └── determines → NavigationDirection.display_level

NavigationDirection
    ├── "exploring_deeper" → shows target_level's LevelDisplay
    └── "building_up" → shows source_level's LevelDisplay

CategoryTab ────┐
                ├── part of → LevelDisplay (when level=3, display_type="connected_info_tabs")
ConnectionTab ──┘
```

## State Transitions

### Display Selection Flow

```
1. User at level X
2. User chooses action: explore deeper or build up
3. System determines:
   - If exploring deeper: show LevelDisplay for level X (what user is leaving)
   - If building up: show LevelDisplay for level X (what user is enriching)
4. After transition completes:
   - Update to new level's LevelDisplay
```

## Example Data

### CategoryTab Example (College category)

```json
{
  "category_name": "College",
  "item_count": 1200,
  "columns": ["Name", "Category", "Details"],
  "data": [
    {
      "Name": "COLLEGE JEAN MOULIN",
      "Category": "College",
      "Details": "Source: fr-en-indicateurs-valeur-ajoutee-colleges.csv"
    }
  ]
}
```

**Note**: Internal fields like `id`, `key_value`, `source_file` are still stored but displayed with user-friendly labels.

### ConnectionTab Example

```json
{
  "connection_key": "College → Performance",
  "connection_type": "has performance of",
  "connection_count": 3500,
  "columns": ["From", "Connection", "To"],
  "data": [
    {
      "From": "COLLEGE JEAN MOULIN",
      "Connection": "has performance of",
      "To": "96% success rate"
    }
  ]
}
```

## Constitution v1.2.0 Compliance Checklist

- [ ] All `user_title` values avoid technical terms
- [ ] Column headers use plain language ("From", "To", not "start_node", "end_node")
- [ ] Direction labels are action-oriented ("Exploring deeper", "Building up")
- [ ] Category/connection types use domain language
- [ ] No UI element exposes internal data structure concepts
