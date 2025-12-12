"""
JSON Visualizer Component for Navigation Export

This module provides a visual tree visualization for navigation exports
using streamlit-agraph for interactive hierarchical display.

Feature: 002-ascent-functionality
Date: 2025-12-03
"""

from typing import Any, Dict, List, Optional
import json
import streamlit as st

# Try to import streamlit-agraph for visual tree
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False


class JsonVisualizer:
    """
    Visual tree visualization for navigation exports.

    Uses streamlit-agraph for interactive hierarchical tree display
    with nodes and edges.
    """

    def __init__(self):
        """Initialize the JSON visualizer."""
        self._default_expand_depth = 3
        self._node_counter = 0

    def render(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        expand_depth: Optional[int] = None
    ) -> None:
        """
        Render JSON data as interactive tree visualization.

        Args:
            data: JSON-serializable dict to visualize
            title: Optional title to display above visualization
            expand_depth: How deep to expand tree by default (None = use default)
        """
        if title:
            st.subheader(title)

        if not data:
            st.info("No data to display")
            return

        depth = expand_depth if expand_depth is not None else self._default_expand_depth

        if AGRAPH_AVAILABLE:
            self._render_as_tree(data, depth)
        else:
            self._render_fallback(data, depth)

    def _render_as_tree(self, data: Dict[str, Any], max_depth: int) -> None:
        """Render JSON as visual tree using streamlit-agraph."""
        nodes: List[Node] = []
        edges: List[Edge] = []
        self._node_counter = 0

        def get_node_color(depth: int, is_leaf: bool) -> str:
            """Get node color based on depth and type."""
            if is_leaf:
                return "#90EE90"  # Light green for leaves
            colors = ["#4169E1", "#6495ED", "#87CEEB", "#B0E0E6", "#E0FFFF"]
            return colors[min(depth, len(colors) - 1)]

        def get_node_size(depth: int, is_leaf: bool) -> int:
            """Get node size based on depth."""
            if is_leaf:
                return 15
            return max(30 - depth * 5, 15)

        def truncate_value(value: Any, max_len: int = 25) -> str:
            """Truncate long values for display."""
            str_val = str(value)
            if len(str_val) > max_len:
                return str_val[:max_len] + "..."
            return str_val

        def add_nodes_recursive(
            obj: Any,
            parent_id: Optional[str] = None,
            depth: int = 0,
            key_name: str = "root"
        ) -> None:
            """Recursively add nodes and edges for the data structure."""
            if depth > max_depth:
                return

            self._node_counter += 1
            node_id = f"node_{self._node_counter}"

            # Determine if this is a leaf or container
            is_dict = isinstance(obj, dict)
            is_list = isinstance(obj, list)
            is_container = is_dict or is_list
            is_leaf = not is_container

            # Build label
            if is_leaf:
                # For leaf nodes, show key: value
                value_str = truncate_value(obj)
                label = f"{key_name}: {value_str}"
            elif is_dict:
                label = f"{key_name} {{{len(obj)}}}"
            elif is_list:
                label = f"{key_name} [{len(obj)}]"
            else:
                label = str(key_name)

            # Create node
            nodes.append(Node(
                id=node_id,
                label=label,
                size=get_node_size(depth, is_leaf),
                color=get_node_color(depth, is_leaf),
                font={"size": 12, "color": "#333333"}
            ))

            # Create edge to parent
            if parent_id:
                edges.append(Edge(
                    source=parent_id,
                    target=node_id,
                    color="#888888"
                ))

            # Recurse into children
            if is_dict and depth < max_depth:
                for key, value in list(obj.items())[:20]:  # Limit children
                    add_nodes_recursive(value, node_id, depth + 1, str(key))
            elif is_list and depth < max_depth:
                for i, item in enumerate(obj[:10]):  # Limit list items
                    add_nodes_recursive(item, node_id, depth + 1, f"[{i}]")

        # Build the tree
        add_nodes_recursive(data)

        if not nodes:
            st.info("No data to visualize")
            return

        # Configure the graph
        config = Config(
            directed=True,
            hierarchical=True,
            physics=False,
            height=500,
            width=800,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
            node={"highlightStrokeColor": "blue"},
            link={"highlightColor": "lightblue"}
        )

        # Render the graph
        try:
            agraph(nodes=nodes, edges=edges, config=config)
        except Exception as e:
            st.warning(f"Tree visualization error: {e}")
            self._render_fallback(data, max_depth)

    def _render_fallback(self, data: Dict[str, Any], expand_depth: int) -> None:
        """
        Fallback rendering when streamlit-agraph is not available.

        Uses native Streamlit JSON display with expandable sections.
        """
        st.json(data, expanded=expand_depth)

    def get_download_button(
        self,
        data: Dict[str, Any],
        filename: str = "export.json",
        button_label: str = "Download JSON"
    ) -> None:
        """
        Render download button for JSON export.

        Args:
            data: Data to export
            filename: Default filename for download
            button_label: Text to display on button
        """
        if not data:
            st.warning("No data available for export")
            return

        try:
            json_str = json.dumps(data, indent=2, default=str)
            st.download_button(
                label=button_label,
                data=json_str,
                file_name=filename,
                mime="application/json",
                key=f"download_{filename.replace('.', '_')}"
            )
        except (TypeError, ValueError) as e:
            st.error(f"Error serializing data to JSON: {e}")

    def render_with_download(
        self,
        data: Dict[str, Any],
        title: str = "Navigation Export",
        filename: str = "navigation_export.json"
    ) -> None:
        """
        Render JSON visualization with download button.

        Convenience method that combines render() and get_download_button().

        Args:
            data: JSON-serializable dict to visualize
            title: Title for the visualization
            filename: Default filename for download
        """
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(title)

        with col2:
            self.get_download_button(data, filename, "Download")

        if data:
            if AGRAPH_AVAILABLE:
                self._render_as_tree(data, self._default_expand_depth)
            else:
                self._render_fallback(data, self._default_expand_depth)
        else:
            st.info("No data to display")


def render_navigation_export(export_data: Dict[str, Any]) -> None:
    """
    Render a navigation export with appropriate formatting.

    Utility function for displaying NavigationExport data.

    Args:
        export_data: Export data from NavigationSession.export()
    """
    visualizer = JsonVisualizer()

    if not export_data:
        st.info("No navigation export data available")
        return

    # Header with metadata
    metadata = export_data.get("metadata", {})
    if metadata:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Session ID", metadata.get("session_id", "N/A")[:8] + "...")
        with col2:
            st.metric("Current Level", metadata.get("current_level", "N/A"))
        with col3:
            total_nodes = len(export_data.get("navigation_tree", {}).get("nodes", []))
            st.metric("Tree Nodes", total_nodes)

    st.divider()

    # Tabs for different sections
    tabs = st.tabs(["Navigation Path", "Cumulative Outputs", "Current Output", "Full Export"])

    with tabs[0]:
        # Check for tree data first (branching mode)
        tree_data = export_data.get("navigation_tree", {})
        tree_nodes = tree_data.get("nodes", []) if tree_data else []

        # Check for linear history (default mode)
        history_data = export_data.get("navigation_history", [])

        if tree_nodes:
            st.markdown("**Navigation Tree (branching mode):**")
            visualizer.render(tree_data, expand_depth=3)
        elif history_data:
            st.markdown("**Navigation Path:**")
            # Display linear history as path
            for i, step in enumerate(history_data):
                level = step.get("level", "?")
                action = step.get("action", "?")
                timestamp = step.get("timestamp", "")[:19] if step.get("timestamp") else ""
                st.markdown(f"**{i+1}.** L{level} - {action} {'(' + timestamp + ')' if timestamp else ''}")

                # Show step metadata if available
                metadata = step.get("metadata", {})
                if metadata:
                    with st.expander("Details"):
                        st.json(metadata)
        else:
            st.info("No navigation path data")

        # Show current_path summary
        current_path = export_data.get("current_path", [])
        if current_path:
            st.markdown(f"**Path Depth:** {len(current_path)} steps")

    with tabs[1]:
        # Show cumulative outputs from all levels visited (FR-019)
        cumulative_data = export_data.get("cumulative_outputs", {})
        if cumulative_data:
            st.markdown("**Data accumulated at each level:**")
            visualizer.render(cumulative_data, expand_depth=3)
        else:
            st.info("No cumulative output data")

    with tabs[2]:
        output_data = export_data.get("current_output", {}) or export_data.get("output", {})
        if output_data:
            visualizer.render(output_data, expand_depth=2)
        else:
            st.info("No output data")

    with tabs[3]:
        visualizer.render_with_download(
            export_data,
            title="Complete Export",
            filename="navigation_export.json"
        )
