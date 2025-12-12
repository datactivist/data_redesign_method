"""
Decision Tree Sidebar Component for Navigation

This module provides a visual decision-tree sidebar showing the navigation
history with clickable nodes for time-travel navigation.

Feature: 002-ascent-functionality
Date: 2025-12-03
"""

from typing import Any, Callable, Dict, List, Optional
import streamlit as st


class DecisionTreeComponent:
    """
    Decision tree sidebar component for navigation visualization.

    Implements IDecisionTreeComponent contract from navigation_api.py.
    Displays the navigation tree with clickable nodes for time-travel.
    """

    def __init__(self):
        """Initialize the decision tree component."""
        # T064: Level-specific colors for non-current nodes
        self._level_colors = {
            "LEVEL_0": "#E57373",  # Red
            "LEVEL_1": "#FFB74D",  # Orange
            "LEVEL_2": "#4FC3F7",  # Light Blue
            "LEVEL_3": "#81C784",  # Green
            "LEVEL_4": "#9575CD",  # Purple
        }
        self._level_icons = {
            "LEVEL_0": "0",
            "LEVEL_1": "1",
            "LEVEL_2": "2",
            "LEVEL_3": "3",
            "LEVEL_4": "4",
        }
        # T064: Styling colors
        self._current_node_color = "#2ECC71"  # Green for current node
        self._path_node_color = "#3498DB"  # Blue for nodes on current path
        self._branch_node_color = "#95A5A6"  # Gray for branch nodes

    def render(
        self,
        tree_visualization: Dict[str, Any],
        on_node_click: Callable[[str], None],
        available_options: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Render the decision tree in Streamlit sidebar.

        Args:
            tree_visualization: Tree visualization data from NavigationSession.get_tree_visualization()
            on_node_click: Callback when user clicks a node (for time-travel)
            available_options: Current navigation options from get_available_options()
        """
        nodes = tree_visualization.get("nodes", [])
        current_path = tree_visualization.get("current_path", [])
        branches = tree_visualization.get("branches", [])

        if not nodes:
            st.sidebar.info("No navigation history yet. Start navigating to see your path.")
            return

        st.sidebar.subheader("Navigation Path")

        # Sort nodes by depth and organize for tree display
        nodes_by_id = {node["id"]: node for node in nodes}

        # Display tree structure
        self._render_tree(nodes_by_id, current_path, branches, on_node_click)

        # Display available options for current position
        if available_options:
            st.sidebar.divider()
            st.sidebar.subheader("Available Actions")
            self._render_options(available_options)

    def _render_tree(
        self,
        nodes_by_id: Dict[str, Dict[str, Any]],
        current_path: List[str],
        branches: List[str],
        on_node_click: Callable[[str], None]
    ) -> None:
        """Render the tree structure with indentation."""
        # Find root node
        root_id = None
        for node_id, node in nodes_by_id.items():
            if node.get("depth", 0) == 0:
                root_id = node_id
                break

        if root_id is None:
            return

        # Recursive render
        self._render_node(
            root_id,
            nodes_by_id,
            current_path,
            branches,
            on_node_click,
            depth=0
        )

    def _render_node(
        self,
        node_id: str,
        nodes_by_id: Dict[str, Dict[str, Any]],
        current_path: List[str],
        branches: List[str],
        on_node_click: Callable[[str], None],
        depth: int = 0
    ) -> None:
        """
        Render a single node and its children.

        Per FR-021, displays:
        - (a) Navigation step taken (action)
        - (b) Decision made at each step (decision_description)
        - (c) Output snapshot at every step (output_snapshot)
        """
        node = nodes_by_id.get(node_id)
        if not node:
            return

        level = node.get("level", "UNKNOWN")
        action = node.get("action", "")
        is_current = node.get("is_current", False)
        is_on_path = node_id in current_path
        is_branch_point = node_id in branches

        # FR-021: Get decision description and output snapshot
        decision_description = node.get("decision_description", "")
        output_snapshot = node.get("output_snapshot", {})

        # Build display string
        indent = "  " * depth
        level_num = level.replace("LEVEL_", "L")

        # Style based on state
        if is_current:
            style = "**"
            suffix = " <- CURRENT"
        elif is_on_path:
            style = ""
            suffix = ""
        else:
            style = ""
            suffix = " (branch)"

        # Icon based on action
        action_icon = {
            "entry": "entry",
            "descend": "descend",
            "ascend": "ascend",
            "restore": "restore"
        }.get(action, action)

        # FR-021: Build display with decision_description
        if decision_description:
            node_text = f"{decision_description}"
        else:
            node_text = f"{action_icon}"

        # Create unique key for button
        button_key = f"nav_node_{node_id}"

        # T064: Determine node color based on state
        if is_current:
            node_color = self._current_node_color  # Green
        elif is_on_path:
            node_color = self._path_node_color  # Blue
        else:
            node_color = self._branch_node_color  # Gray

        # Render based on whether it's current or clickable
        if is_current:
            # T064: Current node highlighted in green with bold
            st.sidebar.markdown(
                f'{indent}<span style="color: {node_color}; font-weight: bold;">'
                f'[{level_num}: {node_text}]{suffix}</span>',
                unsafe_allow_html=True
            )
        else:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                # T064: Non-current nodes with appropriate color
                st.sidebar.markdown(
                    f'{indent}<span style="color: {node_color};">'
                    f'[{level_num}: {node_text}]{suffix}</span>',
                    unsafe_allow_html=True
                )
            with col2:
                if st.sidebar.button("↩", key=button_key, help=f"Restore to {level_num}"):
                    on_node_click(node_id)

        # FR-021: Render output_snapshot summary if available
        if output_snapshot:
            snapshot_info = self._format_output_snapshot(output_snapshot)
            if snapshot_info:
                st.sidebar.caption(f"{indent}  {snapshot_info}")

        # Render branch indicator
        if is_branch_point:
            st.sidebar.caption(f"{indent}  (branch point)")

        # Render children
        children_ids = node.get("children_ids", [])
        # Get children from nodes_by_id that have this node as parent
        for child_id in children_ids:
            if child_id in nodes_by_id:
                self._render_node(
                    child_id,
                    nodes_by_id,
                    current_path,
                    branches,
                    on_node_click,
                    depth + 1
                )

    def _format_output_snapshot(self, output_snapshot: Dict[str, Any]) -> str:
        """
        Format output_snapshot for display in sidebar (FR-021).

        Args:
            output_snapshot: Output snapshot dict from NavigationTreeNode

        Returns:
            Formatted string like "(600 nodes, 1200 edges)" or "(523 rows)"
        """
        output_type = output_snapshot.get("type", "")

        if output_type == "graph":
            node_count = output_snapshot.get("node_count")
            edge_count = output_snapshot.get("edge_count")
            row_count = output_snapshot.get("row_count")
            if node_count is not None and edge_count is not None:
                return f"({node_count} nodes, {edge_count} edges)"
            elif row_count is not None:
                return f"({row_count} rows)"
        elif output_type == "dataframe":
            row_count = output_snapshot.get("row_count")
            columns = output_snapshot.get("columns", [])
            if row_count is not None:
                col_info = f", {len(columns)} cols" if columns else ""
                return f"({row_count} rows{col_info})"
        elif output_type == "vector":
            length = output_snapshot.get("length")
            if length is not None:
                return f"({length} items)"
        elif output_type == "datum":
            value = output_snapshot.get("value", "")
            if value:
                # Truncate long values
                display_val = value[:20] + "..." if len(str(value)) > 20 else value
                return f"(= {display_val})"
        elif output_type == "unlinkable":
            source_count = output_snapshot.get("source_count")
            if source_count is not None:
                return f"({source_count} sources)"

        return ""

    def _render_options(self, options: List[Dict[str, Any]]) -> None:
        """Render available navigation options."""
        for option in options:
            action = option.get("action", "")
            description = option.get("description", "")
            target = option.get("target_level", "")

            if action == "exit":
                if st.sidebar.button(f"Exit: {description}", key=f"opt_exit"):
                    st.session_state["nav_action"] = "exit"
            elif action == "descend":
                if st.sidebar.button(f"Descend to {target}", key=f"opt_descend_{target}"):
                    st.session_state["nav_action"] = "descend"
                    st.session_state["nav_target"] = target
            elif action == "ascend":
                if st.sidebar.button(f"Ascend to {target}", key=f"opt_ascend_{target}"):
                    st.session_state["nav_action"] = "ascend"
                    st.session_state["nav_target"] = target

                # Show enrichment/dimension options if available
                enrichment_opts = option.get("enrichment_options", [])
                dimension_opts = option.get("dimension_options", [])

                if enrichment_opts:
                    st.sidebar.caption("Available enrichments:")
                    for e in enrichment_opts[:3]:
                        name = e.get("name", "") if isinstance(e, dict) else str(e)
                        st.sidebar.text(f"  - {name}")

                if dimension_opts:
                    st.sidebar.caption("Available dimensions:")
                    for d in dimension_opts[:3]:
                        name = d.get("name", "") if isinstance(d, dict) else str(d)
                        st.sidebar.text(f"  - {name}")


def render_simple_tree(tree_visualization: Dict[str, Any]) -> None:
    """
    Render a simple text-based tree without interactivity.

    Per FR-021, displays decision_description and output_snapshot at each node.
    Useful for display-only contexts.
    """
    nodes = tree_visualization.get("nodes", [])
    current_path = tree_visualization.get("current_path", [])

    if not nodes:
        st.text("No navigation history")
        return

    # Sort by depth
    sorted_nodes = sorted(nodes, key=lambda n: n.get("depth", 0))

    for node in sorted_nodes:
        depth = node.get("depth", 0)
        level = node.get("level", "UNKNOWN").replace("LEVEL_", "L")
        action = node.get("action", "")
        is_current = node.get("is_current", False)

        # FR-021: Get decision_description and output_snapshot
        decision_description = node.get("decision_description", "")
        output_snapshot = node.get("output_snapshot", {})

        indent = "  " * depth
        marker = "" if depth == 0 else ""
        current_marker = " <- CURRENT" if is_current else ""

        # FR-021: Display decision_description if available
        if decision_description:
            st.text(f"{indent}{marker}[{level}: {decision_description}]{current_marker}")
        else:
            st.text(f"{indent}{marker}[{level}: {action}]{current_marker}")

        # FR-021: Display output_snapshot summary
        if output_snapshot:
            snapshot_info = _format_snapshot_summary(output_snapshot)
            if snapshot_info:
                st.caption(f"{indent}  {snapshot_info}")


def _format_snapshot_summary(output_snapshot: Dict[str, Any]) -> str:
    """Format output snapshot for simple tree display (FR-021)."""
    output_type = output_snapshot.get("type", "")

    if output_type == "graph":
        node_count = output_snapshot.get("node_count")
        edge_count = output_snapshot.get("edge_count")
        if node_count is not None:
            return f"({node_count} nodes, {edge_count or 0} edges)"
    elif output_type == "dataframe":
        row_count = output_snapshot.get("row_count")
        if row_count is not None:
            return f"({row_count} rows)"
    elif output_type == "vector":
        length = output_snapshot.get("length")
        if length is not None:
            return f"({length} items)"
    elif output_type == "datum":
        value = output_snapshot.get("value", "")
        if value:
            display_val = str(value)[:15] + "..." if len(str(value)) > 15 else value
            return f"(= {display_val})"

    return ""
