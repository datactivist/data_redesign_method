"""
Export API Contract for Ascent Functionality

This module defines the JSON export format for navigation sessions,
compatible with JSON Crack visualization.

Feature: 002-ascent-functionality
Date: 2025-12-03
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
import json


# =============================================================================
# Export Format Schema
# =============================================================================

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
    output_snapshot: Dict[str, Any] = field(default_factory=dict)  # FR-021: summary of output at this step

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
      "cumulative_outputs": {
        "graph": {...},
        "table": {...},
        "vector": {...},
        "datum": {...}
      }
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


# =============================================================================
# Export Protocol
# =============================================================================

class IExporter(Protocol):
    """Contract for navigation session exporter."""

    def export_session(self, session: Any) -> NavigationExport:
        """
        Export a navigation session to NavigationExport format.

        Args:
            session: NavigationSession instance

        Returns:
            NavigationExport ready for serialization
        """
        ...

    def export_to_file(self, session: Any, filepath: str) -> None:
        """
        Export a navigation session directly to a JSON file.

        Args:
            session: NavigationSession instance
            filepath: Output file path
        """
        ...


# =============================================================================
# JSON Crack Compatibility
# =============================================================================

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


# =============================================================================
# Example Export
# =============================================================================

EXAMPLE_EXPORT = """
{
  "version": "1.0",
  "feature": "002-ascent-functionality",
  "exported_at": "2025-12-03T10:30:00Z",
  "session_id": "abc123-def456",
  "navigation_tree": {
    "nodes": [
      {
        "id": "root",
        "level": 4,
        "level_name": "LEVEL_4",
        "action": "entry",
        "timestamp": "2025-12-03T10:00:00Z",
        "parent_id": null,
        "children_ids": ["node_1"],
        "metadata": {},
        "decision_description": "Entry with raw dataset",
        "output_snapshot": {"type": "unlinkable", "item_count": 1000}
      },
      {
        "id": "node_1",
        "level": 3,
        "level_name": "LEVEL_3",
        "action": "descend",
        "timestamp": "2025-12-03T10:05:00Z",
        "parent_id": "root",
        "children_ids": ["node_2"],
        "metadata": {"entities_defined": ["Indicator", "Source"]},
        "decision_description": "Make graph with entities Indicator, Source",
        "output_snapshot": {"type": "graph", "node_count": 600, "edge_count": 1200}
      },
      {
        "id": "node_2",
        "level": 2,
        "level_name": "LEVEL_2",
        "action": "descend",
        "timestamp": "2025-12-03T10:10:00Z",
        "parent_id": "node_1",
        "children_ids": ["node_3"],
        "metadata": {"domain": "Revenue"},
        "decision_description": "Filter domain Revenue",
        "output_snapshot": {"type": "dataframe", "row_count": 523, "columns": ["name", "source", "domain"]}
      },
      {
        "id": "node_3",
        "level": 1,
        "level_name": "LEVEL_1",
        "action": "descend",
        "timestamp": "2025-12-03T10:15:00Z",
        "parent_id": "node_2",
        "children_ids": ["node_4"],
        "metadata": {"column": "name"},
        "decision_description": "Extract column name",
        "output_snapshot": {"type": "vector", "length": 523}
      },
      {
        "id": "node_4",
        "level": 0,
        "level_name": "LEVEL_0",
        "action": "descend",
        "timestamp": "2025-12-03T10:20:00Z",
        "parent_id": "node_3",
        "children_ids": ["node_5"],
        "metadata": {"aggregation": "count", "value": 523},
        "decision_description": "Aggregate count",
        "output_snapshot": {"type": "datum", "value": 523}
      },
      {
        "id": "node_5",
        "level": 1,
        "level_name": "LEVEL_1",
        "action": "ascend",
        "timestamp": "2025-12-03T10:25:00Z",
        "parent_id": "node_4",
        "children_ids": [],
        "metadata": {"enrichment": "naming_signatures"},
        "decision_description": "Ascend with naming_signatures enrichment",
        "output_snapshot": {"type": "vector", "length": 523}
      }
    ],
    "root_id": "root",
    "current_id": "node_5"
  },
  "current_path": ["root", "node_1", "node_2", "node_3", "node_4", "node_5"],
  "current_output": {
    "level": 1,
    "level_name": "LEVEL_1",
    "output_type": "vector",
    "row_count": 523,
    "sample_data": [
      {"original": "CA_B2B_FR", "first_word": "CA", "word_count": 3},
      {"original": "VOL_B2C_INT", "first_word": "VOL", "word_count": 3}
    ]
  },
  "cumulative_outputs": {
    "graph": {
      "level": 3,
      "level_name": "LEVEL_3",
      "output_type": "graph",
      "node_count": 600,
      "edge_count": 1200
    },
    "table": {
      "level": 2,
      "level_name": "LEVEL_2",
      "output_type": "dataframe",
      "row_count": 523,
      "column_names": ["name", "source", "domain"]
    },
    "vector": {
      "level": 1,
      "level_name": "LEVEL_1",
      "output_type": "vector",
      "row_count": 523
    },
    "datum": {
      "level": 0,
      "level_name": "LEVEL_0",
      "output_type": "datum",
      "sample_data": 523
    }
  }
}
"""
