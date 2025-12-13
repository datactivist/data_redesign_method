"""
Entity and Relationship Tab Components

Feature: 003-level-dataviz-display

This module provides functions for extracting and displaying entity/relationship
tabs from a NetworkX graph. Used at the L3→L2 transition.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd
import networkx as nx

from intuitiveness.ui.i18n import t


@dataclass
class EntityTabData:
    """
    Data structure for an entity type tab.

    FR-004: Display tabbed views with one tab per entity type
    FR-007: Each entity tab shows id, name, type, and properties
    """
    entity_type: str
    entity_count: int
    columns: List[str]
    data: List[Dict[str, Any]]

    def __post_init__(self):
        # Validation: must have required columns
        required = {"id", "name", "type"}
        if not required.issubset(set(self.columns)):
            raise ValueError(f"Entity tab must have columns: {required}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entity data to a pandas DataFrame."""
        return pd.DataFrame(self.data, columns=self.columns)


@dataclass
class RelationshipTabData:
    """
    Data structure for a relationship type tab.

    FR-005: Display one tab per relationship type showing linked entity pairs
    """
    relationship_key: str  # Format: "{start_type} → {end_type}"
    relationship_type: str
    relationship_count: int
    columns: List[str]
    data: List[Dict[str, Any]]

    def __post_init__(self):
        # Validation: must have required columns
        required = {"start_name", "relationship", "end_name"}
        if not required.issubset(set(self.columns)):
            raise ValueError(f"Relationship tab must have columns: {required}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert relationship data to a pandas DataFrame."""
        return pd.DataFrame(self.data, columns=self.columns)


@dataclass
class CombinedTabData:
    """
    Data structure for a combined "All" tab showing all entities or relationships.

    This provides a unified view of all data at L3 level for categorization.
    """
    tab_type: str  # "all_entities" or "all_relationships"
    label: str
    count: int
    columns: List[str]
    data: List[Dict[str, Any]]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert combined data to a pandas DataFrame."""
        return pd.DataFrame(self.data, columns=self.columns)


def extract_entity_tabs(graph_or_df) -> List[EntityTabData]:
    """
    Extract entity data grouped by type from a NetworkX graph or pandas DataFrame.

    FR-004: Display tabbed views with one tab per entity type
    FR-007: Each entity tab shows id, name, type, and properties

    Args:
        graph_or_df: NetworkX graph with nodes containing 'type' attribute,
                     OR pandas DataFrame (from OOM fix - L3 now stores DataFrame)

    Returns:
        List of EntityTabData, one per entity type (excluding "Source")
    """
    # Handle DataFrame input (from OOM Fix #1 - Level3Dataset stores DataFrame)
    if isinstance(graph_or_df, pd.DataFrame):
        df = graph_or_df
        if df.empty:
            return []

        # Create a single "Data" entity tab with all rows
        # Use first column as name if available, otherwise row index
        name_col = df.columns[0] if len(df.columns) > 0 else None

        entities = []
        for idx, row in df.iterrows():
            entity_record = {
                "id": str(idx),
                "name": str(row[name_col]) if name_col else str(idx),
                "type": "Data",
            }
            # Add all columns as properties
            for col in df.columns:
                if col not in entity_record:
                    entity_record[col] = row[col]
            entities.append(entity_record)

        # Build columns list
        columns = ["id", "name", "type"]
        columns.extend([c for c in df.columns if c not in columns])

        return [EntityTabData(
            entity_type="Data",
            entity_count=len(entities),
            columns=columns,
            data=entities
        )]

    # Original NetworkX graph handling
    graph = graph_or_df
    entities_by_type: Dict[str, List[Dict[str, Any]]] = {}

    for node_id, attrs in graph.nodes(data=True):
        entity_type = attrs.get("type", "Unknown")

        # Skip source nodes
        if entity_type == "Source":
            continue

        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []

        # Build entity record with id, name, type, and all properties
        entity_record = {
            "id": node_id,
            "name": attrs.get("name", node_id),
            "type": entity_type,
        }

        # Add additional properties
        for key, value in attrs.items():
            if key not in ["id", "name", "type"]:
                entity_record[key] = value

        entities_by_type[entity_type].append(entity_record)

    # Build EntityTabData for each type
    result: List[EntityTabData] = []
    for entity_type, entities in entities_by_type.items():
        if not entities:
            continue

        # Get all columns from the first entity (assuming uniform structure)
        all_columns = list(entities[0].keys())
        # Ensure required columns are first
        columns = ["id", "name", "type"]
        columns.extend([c for c in all_columns if c not in columns])

        result.append(EntityTabData(
            entity_type=entity_type,
            entity_count=len(entities),
            columns=columns,
            data=entities
        ))

    return result


def extract_relationship_tabs(graph_or_df) -> List[RelationshipTabData]:
    """
    Extract relationship data grouped by type from a NetworkX graph or pandas DataFrame.

    FR-005: Display one tab per relationship type showing linked entity pairs

    Args:
        graph_or_df: NetworkX graph with edges containing relationship info,
                     OR pandas DataFrame (from OOM fix - returns empty list for DataFrame)

    Returns:
        List of RelationshipTabData, one per relationship pattern
        (Empty list for DataFrame input since flat data has no relationships)
    """
    # Handle DataFrame input (from OOM Fix #1 - Level3Dataset stores DataFrame)
    # DataFrames are flat and have no relationships
    if isinstance(graph_or_df, pd.DataFrame):
        return []

    # Original NetworkX graph handling
    graph = graph_or_df
    relationships_by_key: Dict[str, List[Dict[str, Any]]] = {}
    relationship_types: Dict[str, str] = {}

    for start_node, end_node, edge_attrs in graph.edges(data=True):
        # Get node attributes
        start_attrs = graph.nodes.get(start_node, {})
        end_attrs = graph.nodes.get(end_node, {})

        start_type = start_attrs.get("type", "Unknown")
        end_type = end_attrs.get("type", "Unknown")

        # Skip relationships involving Source nodes
        if start_type == "Source" or end_type == "Source":
            continue

        # Get relationship type
        rel_type = edge_attrs.get("type", edge_attrs.get("label", "RELATED_TO"))

        # Create relationship key
        rel_key = f"{start_type} → {end_type}"

        if rel_key not in relationships_by_key:
            relationships_by_key[rel_key] = []
            relationship_types[rel_key] = rel_type

        # Build relationship record
        rel_record = {
            "start_name": start_attrs.get("name", start_node),
            "relationship": rel_type,
            "end_name": end_attrs.get("name", end_node),
            "start_id": start_node,
            "end_id": end_node,
            "start_type": start_type,
            "end_type": end_type,
        }

        # Add additional edge properties
        for key, value in edge_attrs.items():
            if key not in ["type", "label"]:
                rel_record[key] = value

        relationships_by_key[rel_key].append(rel_record)

    # Build RelationshipTabData for each pattern
    result: List[RelationshipTabData] = []
    for rel_key, relationships in relationships_by_key.items():
        if not relationships:
            continue

        # Get all columns
        all_columns = list(relationships[0].keys())
        # Ensure required columns are first
        columns = ["start_name", "relationship", "end_name"]
        columns.extend([c for c in all_columns if c not in columns])

        result.append(RelationshipTabData(
            relationship_key=rel_key,
            relationship_type=relationship_types[rel_key],
            relationship_count=len(relationships),
            columns=columns,
            data=relationships
        ))

    return result


def render_entity_relationship_tabs(
    entity_tabs: List[EntityTabData],
    relationship_tabs: List[RelationshipTabData],
    graph: Optional[nx.Graph] = None,
    max_rows: int = 50,
    show_summary: bool = True,
    enable_selection: bool = False,
    selection_key_prefix: str = "tab_select"
) -> Optional[Dict[str, Any]]:
    """
    Render entity and relationship tabs using Streamlit tabs.

    FR-004: Display tabbed views with one tab per entity type
    FR-005: Display one tab per relationship type
    FR-006: Show entity tables side-by-side or adjacent to graph
    FR-007: Each entity tab shows id, name, type, and properties

    Enhanced: Includes combined tables:
    - "All (Entities + Relationships)": All entities with relationships as extra columns
    - "All Entities": All entities combined
    - "All Relationships": All relationships combined

    Enhanced: Allows selection of any table for categorization.

    Args:
        entity_tabs: List of EntityTabData from extract_entity_tabs()
        relationship_tabs: List of RelationshipTabData from extract_relationship_tabs()
        graph: Optional NetworkX graph (required for combined all table with relationships as columns)
        max_rows: Maximum rows to display per tab (SC-004)
        show_summary: Whether to show summary counts
        enable_selection: Whether to show selection buttons for categorization
        selection_key_prefix: Prefix for selection button keys

    Returns:
        If enable_selection is True and user selects a table:
            Dict with 'table_type', 'table_name', 'dataframe' keys
        Otherwise: None
    """
    selected_table = None

    # Show summary - simplified for minimal design (007-streamlit-design-makeup)
    # Skip summary for simple DataFrame cases - the tab label shows the count
    if show_summary:
        total_entities = sum(tab.entity_count for tab in entity_tabs)
        total_relationships = sum(tab.relationship_count for tab in relationship_tabs)
        # Only show summary if we have relationships or multiple entity types
        if total_relationships > 0 or len(entity_tabs) > 1:
            st.markdown(
                f"**{t('found_items_connections', items=total_entities, categories=len(entity_tabs), connections=total_relationships)}**"
            )

    # Handle empty state (no connections)
    if not entity_tabs and not relationship_tabs:
        st.info(t("no_items_connections"))
        return None

    # Create combined tables
    # Use explicit None check - DataFrames raise ValueError with "if df" truthiness check
    combined_all = create_combined_all_table(graph) if graph is not None else None
    combined_relationships = create_combined_relationship_table(relationship_tabs)

    # Build tab labels
    tab_labels = []
    tab_data_list = []  # Track which data corresponds to which tab

    # SIMPLE CASE: DataFrame with single entity type and no relationships
    # Just show one "Data" tab - no redundant tabs (007-streamlit-design-makeup)
    is_simple_dataframe = (
        combined_all is not None and
        combined_all.tab_type == "all_data" and
        len(entity_tabs) == 1 and
        entity_tabs[0].entity_type == "Data" and
        not relationship_tabs
    )

    if is_simple_dataframe:
        # Show single clean tab for DataFrame data
        tab_labels.append(f"📊 Data ({combined_all.count})")
        tab_data_list.append(("all_combined", combined_all))
    else:
        # COMPLEX CASE: Graph with multiple entity types and relationships
        # Add combined "All" tab first (items + connections as columns)
        if combined_all:
            tab_labels.append(f"📊 All ({combined_all.count})")
            tab_data_list.append(("all_combined", combined_all))

        # Add combined "All Connections" tab (if we have connections)
        if combined_relationships:
            tab_labels.append(f"🔗 All Connections ({combined_relationships.count})")
            tab_data_list.append(("combined_relationships", combined_relationships))

        # Add individual category tabs (only if multiple types)
        if len(entity_tabs) > 1:
            for entity_tab in entity_tabs:
                tab_labels.append(f"📦 {entity_tab.entity_type} ({entity_tab.entity_count})")
                tab_data_list.append(("entity", entity_tab))

        # Add individual connection tabs
        for rel_tab in relationship_tabs:
            tab_labels.append(f"🔗 {rel_tab.relationship_key} ({rel_tab.relationship_count})")
            tab_data_list.append(("relationship", rel_tab))

    if not tab_labels:
        st.info(t("no_data_to_display"))
        return None

    # Create tabs
    tabs = st.tabs(tab_labels)

    # Render all tabs - Constitution v1.2.0: Use domain-friendly labels
    for i, (tab_type, tab_data) in enumerate(tab_data_list):
        with tabs[i]:
            if tab_type == "all_combined":
                st.markdown(f"**{t('items_with_connections', count=tab_data.count)}**")
                st.caption(t("each_record_shows"))
                df = tab_data.to_dataframe()
            elif tab_type == "combined_entities":
                st.markdown(f"**{t('total_items_categories', count=tab_data.count)}**")
                df = tab_data.to_dataframe()
            elif tab_type == "combined_relationships":
                st.markdown(f"**{t('total_connections_types', count=tab_data.count)}**")
                df = tab_data.to_dataframe()
            elif tab_type == "entity":
                st.markdown(f"**{t('entity_items_count', count=tab_data.entity_count, entity_type=tab_data.entity_type)}**")
                df = tab_data.to_dataframe()
            elif tab_type == "relationship":
                st.markdown(f"**{t('relationship_connections_count', count=tab_data.relationship_count, rel_type=tab_data.relationship_type)}**")
                df = tab_data.to_dataframe()

            display_df = df.head(max_rows) if len(df) > max_rows else df
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if len(df) > max_rows:
                st.caption(t("showing_first_of", first=max_rows, total=len(df)))

            # Selection button for categorization
            if enable_selection:
                table_name = _get_table_name(tab_type, tab_data)
                btn_key = f"{selection_key_prefix}_{tab_type}_{i}"

                if st.button(t("use_this_data"), key=btn_key, type="secondary"):
                    selected_table = {
                        "table_type": tab_type,
                        "table_name": table_name,
                        "dataframe": df
                    }

    return selected_table


def _get_table_name(tab_type: str, tab_data: Any) -> str:
    """Get a human-readable name for the table."""
    # Constitution v1.2.0: Use domain-friendly labels
    if tab_type == "all_combined":
        return t("all_items_connections")
    elif tab_type == "combined_entities":
        return t("all_items_label")
    elif tab_type == "combined_relationships":
        return t("all_connections_label")
    elif tab_type == "entity":
        return f"{tab_data.entity_type} {t('items_suffix')}"
    elif tab_type == "relationship":
        return f"{tab_data.relationship_key} {t('connections_suffix')}"
    return "Unknown"


@dataclass
class CombinedTabData:
    """
    Data structure for a combined "All" tab showing all entities or relationships.

    This provides a unified view of all data at L3 level for categorization.
    """
    tab_type: str  # "all_entities" or "all_relationships"
    label: str
    count: int
    columns: List[str]
    data: List[Dict[str, Any]]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert combined data to a pandas DataFrame."""
        return pd.DataFrame(self.data, columns=self.columns)


def create_combined_entity_table(entity_tabs: List[EntityTabData]) -> Optional[CombinedTabData]:
    """
    Create a combined table of all entities from all entity types.

    Args:
        entity_tabs: List of EntityTabData from extract_entity_tabs()

    Returns:
        CombinedTabData with all entities, or None if no entities
    """
    if not entity_tabs:
        return None

    all_data = []
    all_columns = set(["id", "name", "type"])

    for entity_tab in entity_tabs:
        for entity in entity_tab.data:
            all_data.append(entity)
            all_columns.update(entity.keys())

    # Order columns: required first, then others
    columns = ["id", "name", "type"]
    columns.extend([c for c in sorted(all_columns) if c not in columns])

    return CombinedTabData(
        tab_type="all_entities",
        label="All Entities",
        count=len(all_data),
        columns=columns,
        data=all_data
    )


def create_combined_relationship_table(relationship_tabs: List[RelationshipTabData]) -> Optional[CombinedTabData]:
    """
    Create a combined table of all relationships from all relationship types.

    Args:
        relationship_tabs: List of RelationshipTabData from extract_relationship_tabs()

    Returns:
        CombinedTabData with all relationships, or None if no relationships
    """
    if not relationship_tabs:
        return None

    all_data = []
    all_columns = set(["start_name", "relationship", "end_name"])

    for rel_tab in relationship_tabs:
        for rel in rel_tab.data:
            all_data.append(rel)
            all_columns.update(rel.keys())

    # Order columns: required first, then others
    columns = ["start_name", "relationship", "end_name"]
    columns.extend([c for c in sorted(all_columns) if c not in columns])

    return CombinedTabData(
        tab_type="all_relationships",
        label="All Relationships",
        count=len(all_data),
        columns=columns,
        data=all_data
    )


def create_combined_all_table(graph_or_df) -> Optional[CombinedTabData]:
    """
    Create a combined table of all entities with relationships as extra fields.

    Each row represents an entity, and relationships are added as extra columns
    showing what each entity is connected to.

    Example: For entity "College A" with relationships:
    - BELONGS_TO -> "Region X"
    - HAS_PERFORMANCE -> "Performance 1"

    The row would be:
    | id | name | type | BELONGS_TO | HAS_PERFORMANCE |
    | 1  | College A | College | Region X | Performance 1 |

    Args:
        graph_or_df: NetworkX graph with nodes and edges, or pandas DataFrame

    Returns:
        CombinedTabData with all entities + relationships as columns, or None if empty
    """
    if graph_or_df is None:
        return None

    # Handle DataFrame input (from OOM Fix #1 - Level3Dataset stores DataFrame)
    if isinstance(graph_or_df, pd.DataFrame):
        df = graph_or_df
        if df.empty:
            return None

        # Convert DataFrame to CombinedTabData format
        columns = list(df.columns)
        data = df.to_dict('records')

        return CombinedTabData(
            tab_type="all_data",
            label="All Data",
            count=len(data),
            columns=columns,
            data=data
        )

    # Original NetworkX graph handling
    graph = graph_or_df
    if graph.number_of_nodes() == 0:
        return None


    # Build entity data with relationships as extra fields
    entity_data: Dict[str, Dict[str, Any]] = {}
    relationship_columns: set = set()

    # First pass: collect all entities
    for node_id, attrs in graph.nodes(data=True):
        entity_type = attrs.get("type", "Unknown")

        # Skip source nodes
        if entity_type == "Source":
            continue

        entity_data[node_id] = {
            "id": node_id,
            "name": attrs.get("name", node_id),
            "type": entity_type,
        }

        # Add any additional node properties
        for key, value in attrs.items():
            if key not in ["id", "name", "type"]:
                entity_data[node_id][key] = value

    # Second pass: count relationships and collect samples (max 3 names per relationship type)
    # Using count + sample approach to handle many relationships gracefully
    MAX_SAMPLE = 3

    for start_node, end_node, edge_attrs in graph.edges(data=True):
        start_attrs = graph.nodes.get(start_node, {})
        end_attrs = graph.nodes.get(end_node, {})

        start_type = start_attrs.get("type", "Unknown")
        end_type = end_attrs.get("type", "Unknown")

        # Skip relationships involving Source nodes
        if start_type == "Source" or end_type == "Source":
            continue

        # Get relationship type
        rel_type = edge_attrs.get("type", edge_attrs.get("label", "RELATED_TO"))

        # Add relationship to start entity (outgoing relationship)
        if start_node in entity_data:
            count_col = f"{rel_type}_to_count"
            sample_col = f"{rel_type}_to_sample"
            relationship_columns.add(count_col)
            relationship_columns.add(sample_col)

            # Increment count
            entity_data[start_node][count_col] = entity_data[start_node].get(count_col, 0) + 1

            # Collect sample (first MAX_SAMPLE names)
            current_sample = entity_data[start_node].get(sample_col, [])
            if isinstance(current_sample, list) and len(current_sample) < MAX_SAMPLE:
                end_name = end_attrs.get("name", end_node)
                current_sample.append(str(end_name))
                entity_data[start_node][sample_col] = current_sample

        # Add relationship to end entity (incoming relationship)
        if end_node in entity_data:
            count_col = f"{rel_type}_from_count"
            sample_col = f"{rel_type}_from_sample"
            relationship_columns.add(count_col)
            relationship_columns.add(sample_col)

            # Increment count
            entity_data[end_node][count_col] = entity_data[end_node].get(count_col, 0) + 1

            # Collect sample (first MAX_SAMPLE names)
            current_sample = entity_data[end_node].get(sample_col, [])
            if isinstance(current_sample, list) and len(current_sample) < MAX_SAMPLE:
                start_name = start_attrs.get("name", start_node)
                current_sample.append(str(start_name))
                entity_data[end_node][sample_col] = current_sample

    # Convert sample lists to strings
    for entity in entity_data.values():
        for key in list(entity.keys()):
            if key.endswith("_sample") and isinstance(entity[key], list):
                entity[key] = ", ".join(entity[key])

    if not entity_data:
        return None

    # Build final data list
    all_data = list(entity_data.values())

    # Build columns: id, name, type first, then relationship columns, then others
    base_columns = ["id", "name", "type"]
    rel_columns_sorted = sorted(relationship_columns)
    other_columns = set()
    for entity in all_data:
        other_columns.update(entity.keys())
    other_columns = other_columns - set(base_columns) - relationship_columns
    other_columns_sorted = sorted(other_columns)

    columns = base_columns + rel_columns_sorted + other_columns_sorted

    # Ensure all entities have all columns (fill missing with empty string)
    for entity in all_data:
        for col in columns:
            if col not in entity:
                entity[col] = ""

    return CombinedTabData(
        tab_type="all_combined",
        label="All (Entities + Relationships)",
        count=len(all_data),
        columns=columns,
        data=all_data
    )


def get_graph_summary(graph: nx.Graph) -> Dict[str, Any]:
    """
    Get summary statistics for a graph.

    Args:
        graph: NetworkX graph

    Returns:
        Dict with node_count, edge_count, entity_types, relationship_types
    """
    entity_types = set()
    relationship_types = set()

    for _, attrs in graph.nodes(data=True):
        entity_type = attrs.get("type", "Unknown")
        if entity_type != "Source":
            entity_types.add(entity_type)

    for _, _, attrs in graph.edges(data=True):
        rel_type = attrs.get("type", attrs.get("label", "RELATED_TO"))
        relationship_types.add(rel_type)

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "entity_types": list(entity_types),
        "relationship_types": list(relationship_types),
    }
