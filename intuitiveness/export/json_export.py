"""
Export API for Ascent Functionality

This module defines the JSON export format for navigation sessions,
compatible with JSON Crack visualization.

Feature: 002-ascent-functionality
Date: 2025-12-03
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


@dataclass
class NavigationNodeExport:
    """
    Serialized navigation tree node for export.

    Excludes dataset_snapshot (too large), includes summary info.

    Per FR-021, each node must record:
    - (a) The navigation step taken (action)
    - (b) Decision made at each step (decision_description)
    - (c) Generated output snapshot (output_snapshot)
    """
    id: str
    level: int
    level_name: str
    action: str
    timestamp: str  # ISO format
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]
    decision_description: str = ""  # FR-021: e.g., "make graph with entity X"
    output_snapshot: Dict[str, Any] = None  # FR-021: summary of output at this step

    def __post_init__(self):
        if self.output_snapshot is None:
            self.output_snapshot = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "level_name": self.level_name,
            "action": self.action,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "decision_description": self.decision_description,
            "output_snapshot": self.output_snapshot
        }


@dataclass
class OutputSummary:
    """
    Summary of the current output at export time.

    Does not include full data, just structure info.
    """
    level: int
    level_name: str
    output_type: str  # "datum", "vector", "dataframe", "graph"
    row_count: Optional[int] = None
    column_names: Optional[List[str]] = None
    node_count: Optional[int] = None  # For graphs
    edge_count: Optional[int] = None  # For graphs
    sample_data: Optional[Any] = None  # First few rows/items

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "level": self.level,
            "level_name": self.level_name,
            "output_type": self.output_type
        }
        if self.row_count is not None:
            result["row_count"] = self.row_count
        if self.column_names is not None:
            result["column_names"] = self.column_names
        if self.node_count is not None:
            result["node_count"] = self.node_count
        if self.edge_count is not None:
            result["edge_count"] = self.edge_count
        if self.sample_data is not None:
            result["sample_data"] = self.sample_data
        return result


@dataclass
class CumulativeOutputs:
    """
    Cumulative outputs accumulated during navigation session (FR-019).

    On exit at any level, ALL accumulated outputs are exported:
    - Exit at L3: Graph + NavigationTree
    - Exit at L2: Graph + Table + NavigationTree
    - Exit at L1: Graph + Table + Vector + NavigationTree
    - Exit at L0: Graph + Table + Vector + Datum + NavigationTree
    """
    graph: Optional[OutputSummary] = None      # L3 output
    table: Optional[OutputSummary] = None      # L2 output
    vector: Optional[OutputSummary] = None     # L1 output
    datum: Optional[OutputSummary] = None      # L0 output

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.graph:
            result["graph"] = self.graph.to_dict()
        if self.table:
            result["table"] = self.table.to_dict()
        if self.vector:
            result["vector"] = self.vector.to_dict()
        if self.datum:
            result["datum"] = self.datum.to_dict()
        return result


@dataclass
class NavigationExport:
    """
    Complete export format for navigation session.

    Per FR-019, includes cumulative outputs from ALL levels visited.

    JSON Schema:
    {
      "version": "1.0",
      "feature": "002-ascent-functionality",
      "exported_at": "2025-12-03T10:30:00Z",
      "session_id": "uuid",
      "navigation_tree": {
        "nodes": [...],
        "root_id": "...",
        "current_id": "..."
      },
      "current_path": ["root", "node_1", ...],
      "current_output": {...},
      "cumulative_outputs": {...}
    }
    """
    version: str = "1.0"
    feature: str = "002-ascent-functionality"
    exported_at: datetime = field(default_factory=datetime.utcnow)
    session_id: str = ""
    navigation_tree: Dict[str, Any] = field(default_factory=dict)
    current_path: List[str] = field(default_factory=list)
    current_output: Dict[str, Any] = field(default_factory=dict)
    cumulative_outputs: CumulativeOutputs = field(default_factory=CumulativeOutputs)  # FR-019

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "feature": self.feature,
            "exported_at": self.exported_at.isoformat() + "Z",
            "session_id": self.session_id,
            "navigation_tree": self.navigation_tree,
            "current_path": self.current_path,
            "current_output": self.current_output,
            "cumulative_outputs": self.cumulative_outputs.to_dict()  # FR-019
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def create(
        cls,
        session_id: str,
        nodes: List[NavigationNodeExport],
        root_id: str,
        current_id: str,
        output_summary: OutputSummary,
        cumulative: Optional[CumulativeOutputs] = None  # FR-019
    ) -> 'NavigationExport':
        """
        Factory method to create export from components.

        Args:
            session_id: Navigation session UUID
            nodes: List of exported nodes
            root_id: Root node ID
            current_id: Current node ID
            output_summary: Summary of current output
            cumulative: Accumulated outputs from all levels visited (FR-019)

        Returns:
            Complete NavigationExport instance
        """
        # Build current path by traversing parent links
        current_path = []
        node_map = {n.id: n for n in nodes}

        current = current_id
        while current:
            current_path.insert(0, current)
            node = node_map.get(current)
            if node:
                current = node.parent_id
            else:
                break

        return cls(
            session_id=session_id,
            navigation_tree={
                "nodes": [n.to_dict() for n in nodes],
                "root_id": root_id,
                "current_id": current_id
            },
            current_path=current_path,
            current_output=output_summary.to_dict(),
            cumulative_outputs=cumulative or CumulativeOutputs()
        )


def convert_to_jsoncrack_format(export: NavigationExport) -> Dict[str, Any]:
    """
    Convert NavigationExport to a format optimized for JSON Crack visualization.

    JSON Crack works best with nested structures, so we convert the flat
    node list into a nested tree structure.

    Per FR-021, includes decision_description and output_snapshot at each node.

    Args:
        export: NavigationExport instance

    Returns:
        Dict with nested tree structure for JSON Crack
    """
    nodes = export.navigation_tree.get("nodes", [])
    root_id = export.navigation_tree.get("root_id")

    # Build node lookup
    node_map = {n["id"]: n for n in nodes}

    def build_nested(node_id: str) -> Dict[str, Any]:
        node = node_map.get(node_id, {})
        children_ids = node.get("children_ids", [])

        result = {
            "level": node.get("level_name", ""),
            "action": node.get("action", ""),
            "decision": node.get("decision_description", ""),  # FR-021
            "timestamp": node.get("timestamp", ""),
            "is_current": node_id == export.navigation_tree.get("current_id")
        }

        if node.get("output_snapshot"):  # FR-021
            result["output"] = node["output_snapshot"]

        if node.get("metadata"):
            result["details"] = node["metadata"]

        if children_ids:
            result["branches"] = [
                build_nested(child_id) for child_id in children_ids
            ]

        return result

    return {
        "version": export.version,
        "feature": export.feature,
        "exported_at": export.exported_at.isoformat() + "Z",
        "navigation": build_nested(root_id) if root_id else {},
        "current_output": export.current_output,
        "cumulative_outputs": export.cumulative_outputs.to_dict()  # FR-019
    }
