"""
Display API Contracts for Level-Specific Visualization

Feature: 003-level-dataviz-display
Date: 2025-12-04

These interfaces define the contracts for level-specific display components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NavigationDirection(Enum):
    """Direction of navigation between levels."""
    DESCEND = "descend"  # L4→L3→L2→L1→L0
    ASCEND = "ascend"    # L0→L1→L2→L3


class DisplayType(Enum):
    """Type of visualization for each level."""
    FILE_LIST = "file_list"           # L4: Raw data files
    GRAPH_WITH_TABS = "graph_with_tabs"  # L3: Graph + entity/relationship tabs
    DOMAIN_TABLE = "domain_table"     # L2: Domain-categorized table
    VECTOR = "vector"                 # L1: Series/list of values
    DATUM = "datum"                   # L0: Single value


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


@dataclass
class LevelDisplayConfig:
    """
    Configuration for level-specific visualization.

    SC-001: Users can identify which abstraction level within 3 seconds
    """
    level: int  # 0-4
    display_type: DisplayType
    title: str
    show_counts: bool = True
    max_preview_rows: int = 50


# Level to display type mapping (FR-001 through FR-011)
LEVEL_DISPLAY_MAPPING: Dict[int, DisplayType] = {
    4: DisplayType.FILE_LIST,
    3: DisplayType.GRAPH_WITH_TABS,
    2: DisplayType.DOMAIN_TABLE,
    1: DisplayType.VECTOR,
    0: DisplayType.DATUM,
}


class ILevelDisplay(ABC):
    """
    Interface for level-specific display components.

    FR-014: Guided Mode and Free Navigation Mode MUST display identical visualizations
    """

    @abstractmethod
    def render(self, data: Any, config: LevelDisplayConfig) -> None:
        """
        Render the visualization for this level.

        Args:
            data: Level-appropriate data (DataFrame, Graph, Series, scalar)
            config: Display configuration
        """
        pass

    @abstractmethod
    def get_summary(self, data: Any) -> Dict[str, Any]:
        """
        Get summary statistics for the data.

        Returns:
            Dict with counts, metadata, etc.
        """
        pass


class IGraphTabDisplay(ABC):
    """
    Interface for graph visualization with entity/relationship tabs.

    FR-003: Display knowledge graph as interactive visualization
    FR-004: Display tabbed views with one tab per entity type
    FR-005: Display one tab per relationship type
    FR-006: Show entity tables side-by-side or adjacent to graph
    """

    @abstractmethod
    def extract_entity_tabs(self, graph: Any) -> List[EntityTabData]:
        """
        Extract entity data grouped by type.

        Args:
            graph: NetworkX graph

        Returns:
            List of EntityTabData, one per entity type (excluding "Source")
        """
        pass

    @abstractmethod
    def extract_relationship_tabs(self, graph: Any) -> List[RelationshipTabData]:
        """
        Extract relationship data grouped by type.

        Args:
            graph: NetworkX graph

        Returns:
            List of RelationshipTabData, one per relationship pattern
        """
        pass

    @abstractmethod
    def render_tabs(
        self,
        entity_tabs: List[EntityTabData],
        relationship_tabs: List[RelationshipTabData]
    ) -> None:
        """
        Render the tabbed display.

        Args:
            entity_tabs: List of entity tab data
            relationship_tabs: List of relationship tab data
        """
        pass


def get_display_level(
    source_level: int,
    target_level: int,
    direction: NavigationDirection
) -> int:
    """
    Determine which level's visualization to show.

    FR-012: During ascent, show visualization from LOWER level (source)
    SC-002: 100% of descent transitions show higher level's data
    SC-003: 100% of ascent transitions show lower level's data

    Args:
        source_level: Level user is coming from
        target_level: Level user is going to
        direction: Navigation direction

    Returns:
        Level whose visualization should be displayed
    """
    if direction == NavigationDirection.DESCEND:
        # Descent: show source level (what user is leaving/transforming)
        return source_level
    else:
        # Ascent: show source level (what user is enriching FROM)
        return source_level
