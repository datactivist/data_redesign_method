"""
Drag-and-Drop Relationship Builder for L2→L3 Ascent

This module provides a visual interface for defining relationships between
entities using streamlit-agraph for graph visualization and interaction.

Feature: 002-ascent-functionality
Date: 2025-12-03
"""

from typing import Any, Callable, Dict, List, Optional
import streamlit as st

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False
    Node = None
    Edge = None
    Config = None


class DragDropRelationshipBuilder:
    """
    Visual drag-and-drop interface for defining relationships.

    Implements IDragDropRelationshipBuilder contract from navigation_api.py.
    Uses streamlit-agraph for interactive graph visualization.
    """

    def __init__(self):
        """Initialize the relationship builder."""
        self._relationships: List[Dict[str, str]] = []

    def render(
        self,
        entities: List[str],
        on_relationship_created: Optional[Callable[[str, str, str], None]] = None
    ) -> List[Dict[str, str]]:
        """
        Render the drag-and-drop interface for defining relationships.

        Args:
            entities: List of entity names from L2 columns
            on_relationship_created: Callback(source, target, type)

        Returns:
            List of relationship definitions
        """
        if not AGRAPH_AVAILABLE:
            st.warning(
                "streamlit-agraph not available. "
                "Install with: pip install streamlit-agraph>=0.0.45"
            )
            return self._render_fallback(entities, on_relationship_created)

        st.subheader("Define Connections")
        st.caption("Draw lines to show how things connect")

        # Create graph nodes from entities
        nodes = [
            Node(
                id=entity,
                label=entity,
                size=25,
                color="#4CAF50" if i % 2 == 0 else "#2196F3"
            )
            for i, entity in enumerate(entities)
        ]

        # Create edges from existing relationships
        edges = [
            Edge(
                source=rel["source_entity"],
                target=rel["target_entity"],
                label=rel["relationship_type"]
            )
            for rel in self._relationships
            if rel["source_entity"] in entities and rel["target_entity"] in entities
        ]

        # Configure graph display - width reduced to avoid overlap with right sidebar
        config = Config(
            width=600,
            height=400,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
            node={'labelProperty': 'label'},
            link={'labelProperty': 'label', 'renderLabel': True}
        )

        # Render graph
        agraph(nodes=nodes, edges=edges, config=config)

        # Manual connection input form
        st.subheader("Add Connection")

        col1, col2, col3 = st.columns(3)

        with col1:
            source = st.selectbox(
                "From",
                options=entities,
                key="drag_drop_source"
            )

        with col2:
            target = st.selectbox(
                "To",
                options=[e for e in entities if e != source] if source else entities,
                key="drag_drop_target"
            )

        with col3:
            rel_type = st.text_input(
                "Connection Type",
                value="BELONGS_TO",
                key="drag_drop_rel_type"
            )

        col_add, col_clear = st.columns(2)

        with col_add:
            if st.button("Add Connection", key="add_relationship"):
                if source and target and rel_type:
                    new_rel = {
                        "source_entity": source,
                        "target_entity": target,
                        "relationship_type": rel_type.upper().replace(" ", "_"),
                        "bidirectional": False
                    }
                    self._relationships.append(new_rel)

                    if on_relationship_created:
                        on_relationship_created(source, target, rel_type)

                    st.success(f"Added: {source} → [{rel_type}] → {target}")
                    st.rerun()

        with col_clear:
            if st.button("Clear All", key="clear_relationships"):
                self._relationships = []
                st.rerun()

        # Display current connections
        if self._relationships:
            st.subheader("Current Connections")
            for i, rel in enumerate(self._relationships):
                st.text(
                    f"{i+1}. {rel['source_entity']} "
                    f"→ [{rel['relationship_type']}] → "
                    f"{rel['target_entity']}"
                )

        return self._relationships

    def _render_fallback(
        self,
        entities: List[str],
        on_relationship_created: Optional[Callable[[str, str, str], None]] = None
    ) -> List[Dict[str, str]]:
        """
        Fallback rendering when streamlit-agraph is not available.

        Uses simple form-based input instead of visual drag-and-drop.
        """
        st.subheader("Define Connections (Form Mode)")
        st.caption(
            "Visual drag-and-drop requires streamlit-agraph. "
            "Using form-based input instead."
        )

        # Display items as chips
        st.write("**Available Items:**")
        entity_cols = st.columns(min(len(entities), 4))
        for i, entity in enumerate(entities):
            with entity_cols[i % 4]:
                st.info(entity)

        # Connection input form
        with st.form("relationship_form"):
            source = st.selectbox("From", options=entities)
            target = st.selectbox(
                "To",
                options=[e for e in entities if e != source]
            )
            rel_type = st.text_input("Connection Type", value="BELONGS_TO")

            if st.form_submit_button("Add Connection"):
                if source and target and rel_type:
                    new_rel = {
                        "source_entity": source,
                        "target_entity": target,
                        "relationship_type": rel_type.upper().replace(" ", "_"),
                        "bidirectional": False
                    }
                    self._relationships.append(new_rel)

                    if on_relationship_created:
                        on_relationship_created(source, target, rel_type)

        # Display current connections
        if self._relationships:
            st.subheader("Current Connections")
            for i, rel in enumerate(self._relationships):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(
                        f"{rel['source_entity']} → [{rel['relationship_type']}] → "
                        f"{rel['target_entity']}"
                    )
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        self._relationships.pop(i)
                        st.rerun()

        return self._relationships

    def get_relationships(self) -> List[Dict[str, str]]:
        """Get the current list of defined relationships."""
        return self._relationships

    def set_relationships(self, relationships: List[Dict[str, str]]) -> None:
        """Set the relationships from an external source."""
        self._relationships = relationships

    def clear_relationships(self) -> None:
        """Clear all defined relationships."""
        self._relationships = []


def get_entities_from_dataframe(df, columns: List[str] = None) -> List[str]:
    """
    Extract entity names from a DataFrame for the drag-drop interface.

    Args:
        df: pandas DataFrame
        columns: Specific columns to use (if None, uses all columns)

    Returns:
        List of unique entity names
    """
    if columns:
        return [col for col in columns if col in df.columns]
    return list(df.columns)
