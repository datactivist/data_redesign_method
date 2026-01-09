"""
Navigation API Contract for Ascent Functionality

This module defines the interface contracts for the navigation system
with decision-tree support and time-travel navigation.

Feature: 002-ascent-functionality
Date: 2025-12-03
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol


# =============================================================================
# Enums
# =============================================================================

class ComplexityLevel(Enum):
    """Abstraction levels in the Data Redesign Method."""
    LEVEL_0 = 0  # Datum
    LEVEL_1 = 1  # Vector
    LEVEL_2 = 2  # Table
    LEVEL_3 = 3  # Linkable/Graph
    LEVEL_4 = 4  # Unlinkable (entry-only)


class NavigationAction(Enum):
    """Types of navigation actions."""
    ENTRY = "entry"
    DESCEND = "descend"
    ASCEND = "ascend"
    RESTORE = "restore"
    EXIT = "exit"


# =============================================================================
# Data Contracts
# =============================================================================

@dataclass
class NavigationOption:
    """
    Available navigation option at current level.

    Attributes:
        action: Type of action (descend/ascend/exit)
        target_level: Resulting level after action
        description: User-facing description
        enrichment_options: Available enrichment functions (for ascend)
        dimension_options: Available dimensions (for ascend to L2/L3)
    """
    action: NavigationAction
    target_level: Optional[ComplexityLevel]
    description: str
    enrichment_options: List[str] = None
    dimension_options: List[str] = None


@dataclass
class NavigationNodeInfo:
    """
    Information about a navigation tree node (for UI rendering).

    Attributes:
        id: Unique node identifier
        level: Complexity level at this node
        action: Action that created this node
        timestamp: When node was created
        depth: Depth in tree (for indentation)
        is_current: Whether this is the active node
        has_children: Whether this node has branches
        decision_description: Human-readable decision at this step (FR-021)
        output_snapshot: Summary of output at this step (FR-021)
    """
    id: str
    level: ComplexityLevel
    action: NavigationAction
    timestamp: datetime
    depth: int
    is_current: bool
    has_children: bool
    decision_description: str = ""  # NEW: FR-021 - e.g., "make graph with entity X"
    output_snapshot: Dict[str, Any] = None  # NEW: FR-021 - summary of output


@dataclass
class TreeVisualization:
    """
    Data for rendering the decision tree in sidebar.

    Attributes:
        nodes: All nodes in display order
        current_path: Node IDs from root to current
        branches: List of branch points
    """
    nodes: List[NavigationNodeInfo]
    current_path: List[str]
    branches: List[str]


# =============================================================================
# Protocol Contracts (Interface Definitions)
# =============================================================================

class INavigationTree(Protocol):
    """
    Contract for navigation tree with branching support.

    The navigation tree maintains the complete history of navigation
    decisions, supporting time-travel by preserving all branches.
    """

    @property
    def root_id(self) -> str:
        """Get the root node ID."""
        ...

    @property
    def current_id(self) -> str:
        """Get the current node ID."""
        ...

    def branch(
        self,
        action: NavigationAction,
        dataset: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new branch from current position.

        Args:
            action: The navigation action taken
            dataset: The dataset at this new position
            metadata: Additional info (enrichment used, dimensions added)

        Returns:
            ID of the newly created node
        """
        ...

    def restore(self, node_id: str) -> Any:
        """
        Restore navigation state to a previous node.

        Args:
            node_id: ID of the node to restore

        Returns:
            The dataset at the restored node

        Raises:
            KeyError: If node_id not found
        """
        ...

    def get_visualization(self) -> TreeVisualization:
        """
        Get tree structure for UI rendering.

        Returns:
            TreeVisualization with all node info for sidebar display
        """
        ...

    def get_current_path(self) -> List[str]:
        """
        Get node IDs from root to current position.

        Returns:
            List of node IDs in path order
        """
        ...


class INavigationSession(Protocol):
    """
    Contract for navigation session with ascent support.

    Extends the existing NavigationSession to support:
    - Ascent operations (L0�L1�L2�L3)
    - Decision-tree visualization
    - Time-travel navigation
    - JSON export
    """

    @property
    def session_id(self) -> str:
        """Get unique session identifier."""
        ...

    @property
    def current_level(self) -> ComplexityLevel:
        """Get current complexity level."""
        ...

    @property
    def current_dataset(self) -> Any:
        """Get current dataset."""
        ...

    def get_available_options(self) -> List[NavigationOption]:
        """
        Get all available navigation options at current level.

        Returns:
            List of NavigationOption including exit, descend, and ascend options
            based on current level per FR-011 through FR-014
        """
        ...

    def descend(self, **params) -> 'INavigationSession':
        """
        Move down one level.

        Returns:
            Self for method chaining

        Raises:
            NavigationError: If at L0 or invalid params
        """
        ...

    def ascend(
        self,
        enrichment_function: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        relationships: Optional[List[Dict]] = None,
        **params
    ) -> 'INavigationSession':
        """
        Move up one level.

        Args:
            enrichment_function: Name of enrichment to use (L0�L1)
            dimensions: Dimension names to add (L1�L2, L2�L3)
            relationships: Relationship definitions for drag-drop (L2�L3)

        Returns:
            Self for method chaining

        Raises:
            NavigationError: If at L3/L4 or invalid params
        """
        ...

    def restore(self, node_id: str) -> 'INavigationSession':
        """
        Time-travel to a previous navigation state.

        Args:
            node_id: ID of the tree node to restore

        Returns:
            Self for method chaining

        Raises:
            KeyError: If node_id not found
        """
        ...

    def get_tree_visualization(self) -> TreeVisualization:
        """
        Get decision tree for sidebar rendering.

        Returns:
            TreeVisualization for UI
        """
        ...

    def export(self) -> Dict[str, Any]:
        """
        Export current state as JSON-serializable dict.

        Per FR-015, includes navigation tree and current output.

        Returns:
            Dict suitable for JSON export and JSON Crack visualization
        """
        ...

    def exit(self) -> Dict[str, Any]:
        """
        Exit navigation and return export data.

        Returns:
            Same as export() but also marks session as exited
        """
        ...


class IEnrichmentRegistry(Protocol):
    """Contract for enrichment function registry."""

    def register(
        self,
        func: Callable,
        name: str,
        source_level: ComplexityLevel,
        target_level: ComplexityLevel,
        description: str,
        requires_context: bool = False,
        is_default: bool = False
    ) -> None:
        """Register an enrichment function."""
        ...

    def get(self, name: str) -> Callable:
        """Get enrichment function by name."""
        ...

    def list_for_transition(
        self,
        source: ComplexityLevel,
        target: ComplexityLevel
    ) -> List[Dict[str, Any]]:
        """
        List available functions for a transition.

        Returns:
            List of dicts with 'name', 'description', 'requires_context'
        """
        ...


class IDimensionRegistry(Protocol):
    """Contract for dimension definition registry."""

    def register(
        self,
        name: str,
        description: str,
        possible_values: List[str],
        classifier: Callable[[Any], str],
        default_value: str = "Unknown",
        applicable_levels: Optional[List[tuple]] = None,
        is_default: bool = False
    ) -> None:
        """Register a dimension definition."""
        ...

    def get(self, name: str) -> Any:
        """Get dimension definition by name."""
        ...

    def list_for_transition(
        self,
        source: ComplexityLevel,
        target: ComplexityLevel
    ) -> List[Dict[str, Any]]:
        """
        List available dimensions for a transition.

        Returns:
            List of dicts with 'name', 'description', 'possible_values'
        """
        ...


# =============================================================================
# UI Component Contracts
# =============================================================================

class IDecisionTreeComponent(Protocol):
    """Contract for decision tree sidebar component."""

    def render(
        self,
        tree: TreeVisualization,
        on_node_click: Callable[[str], None]
    ) -> None:
        """
        Render the decision tree in Streamlit sidebar.

        Args:
            tree: Tree visualization data
            on_node_click: Callback when user clicks a node (for time-travel)
        """
        ...


class IDragDropRelationshipBuilder(Protocol):
    """Contract for drag-and-drop relationship builder (L2�L3)."""

    def render(
        self,
        entities: List[str],
        on_relationship_created: Callable[[str, str, str], None]
    ) -> List[Dict[str, str]]:
        """
        Render the drag-and-drop interface for defining relationships.

        Args:
            entities: List of entity names from L2 columns
            on_relationship_created: Callback(source, target, type)

        Returns:
            List of relationship definitions
        """
        ...


class IJsonVisualizer(Protocol):
    """Contract for JSON Crack-style visualization."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render JSON data as interactive tree visualization.

        Args:
            data: JSON-serializable dict to visualize
        """
        ...

    def get_download_button(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> None:
        """
        Render download button for JSON export.

        Args:
            data: Data to export
            filename: Default filename for download
        """
        ...
