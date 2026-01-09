"""
API Contracts for Ascent Phase Precision (004-ascent-precision)

This module defines the interfaces and contracts for ascent operations.
These are CONTRACTS only - implementation goes in intuitiveness/ package.

Date: 2025-12-04
"""

from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Optional
from enum import Enum
import pandas as pd


# =============================================================================
# Enums
# =============================================================================

class AscentOperationType(Enum):
    """Types of ascent operations between abstraction levels."""
    UNFOLD = "unfold"  # L0 → L1
    DOMAIN_ENRICHMENT = "domain_enrichment"  # L1 → L2
    GRAPH_BUILDING = "graph_building"  # L2 → L3


class CategorizationMethod(Enum):
    """Methods for domain categorization."""
    SEMANTIC = "semantic"  # AI-powered semantic similarity
    KEYWORD = "keyword"  # Simple text matching


# =============================================================================
# Parameter Dataclasses
# =============================================================================

@dataclass
class UnfoldParameters:
    """
    Parameters for L0→L1 unfold operation.

    This operation is deterministic - it simply restores the parent vector.
    No user input required beyond confirmation.

    Attributes:
        preserve_column_name: Keep original column name from parent vector
    """
    preserve_column_name: bool = True


@dataclass
class DomainEnrichmentParameters:
    """
    Parameters for L1→L2 domain enrichment operation.

    Reuses the same logic as L3→L2 descent domain categorization (FR-009).

    Attributes:
        domains: List of domain names to categorize into
        categorization_method: Semantic or keyword matching
        similarity_threshold: Threshold for semantic matching (0.1-0.9)
        unmatched_label: Label for values that don't match any domain
    """
    domains: List[str]
    categorization_method: CategorizationMethod = CategorizationMethod.SEMANTIC
    similarity_threshold: float = 0.5
    unmatched_label: str = "Unmatched"

    def __post_init__(self):
        if not 0.1 <= self.similarity_threshold <= 0.9:
            raise ValueError("similarity_threshold must be between 0.1 and 0.9")
        if not self.domains:
            raise ValueError("At least one domain must be specified")


@dataclass
class GraphBuildingParameters:
    """
    Parameters for L2→L3 graph building operation.

    Attributes:
        entity_column: Column to extract as new entity type
        entity_type_name: Name for the new entity type
        relationship_type: Label for connecting relationships
        source_entity_columns: Columns identifying source entities (optional)
    """
    entity_column: str
    entity_type_name: str
    relationship_type: str
    source_entity_columns: Optional[List[str]] = None

    def __post_init__(self):
        if not self.entity_column:
            raise ValueError("entity_column must be specified")
        if not self.entity_type_name:
            raise ValueError("entity_type_name must be specified")
        if not self.relationship_type:
            raise ValueError("relationship_type must be specified")


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class AscentResult:
    """
    Result of an ascent operation.

    Attributes:
        success: Whether the operation succeeded
        target_level: The level reached (1, 2, or 3)
        dataset: The resulting dataset object
        message: Human-readable status message
        errors: List of error messages if failed
    """
    success: bool
    target_level: int
    dataset: Any  # Level1Dataset, Level2Dataset, or Level3Dataset
    message: str
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ValidationResult:
    """
    Result of validating ascent preconditions.

    Attributes:
        can_ascend: Whether ascent is possible
        reason: Explanation if ascent is blocked
        available_operations: List of valid operation types
    """
    can_ascend: bool
    reason: str
    available_operations: List[AscentOperationType]


# =============================================================================
# Protocol Definitions (Interfaces)
# =============================================================================

class AscentOperationHandler(Protocol):
    """
    Protocol for handling ascent operations.

    Implementations should be in intuitiveness/redesign.py or ascent/ modules.
    """

    def validate_ascent(self, current_level: int, current_dataset: Any) -> ValidationResult:
        """
        Check if ascent is possible from current state.

        Args:
            current_level: Current abstraction level (0, 1, or 2)
            current_dataset: Current dataset object

        Returns:
            ValidationResult with can_ascend status and available operations
        """
        ...

    def execute_unfold(
        self,
        dataset: Any,  # Level0Dataset
        params: UnfoldParameters
    ) -> AscentResult:
        """
        Execute L0→L1 unfold operation.

        Restores the parent vector from which the datum was aggregated.

        Args:
            dataset: Level0Dataset with parent_data
            params: UnfoldParameters

        Returns:
            AscentResult with Level1Dataset

        Raises:
            ValueError: If dataset has no parent_data (orphan datum)
        """
        ...

    def execute_domain_enrichment(
        self,
        dataset: Any,  # Level1Dataset
        params: DomainEnrichmentParameters
    ) -> AscentResult:
        """
        Execute L1→L2 domain enrichment operation.

        Categorizes vector values into domains to create a 2D table.

        Args:
            dataset: Level1Dataset to enrich
            params: DomainEnrichmentParameters with domains and method

        Returns:
            AscentResult with Level2Dataset
        """
        ...

    def execute_graph_building(
        self,
        dataset: Any,  # Level2Dataset
        params: GraphBuildingParameters
    ) -> AscentResult:
        """
        Execute L2→L3 graph building operation.

        Creates a graph by extracting an entity column and defining relationships.

        Args:
            dataset: Level2Dataset to transform
            params: GraphBuildingParameters with entity and relationship config

        Returns:
            AscentResult with Level3Dataset

        Raises:
            ValueError: If entity_column not in dataset
            ValueError: If resulting graph would have orphan nodes
        """
        ...


class AscentFormRenderer(Protocol):
    """
    Protocol for rendering ascent UI forms.

    Implementations should be in intuitiveness/ui/ascent_forms.py.
    """

    def render_l0_to_l1_form(self, dataset: Any) -> Optional[UnfoldParameters]:
        """
        Render L0→L1 unfold confirmation form.

        Shows the aggregation type and source vector preview.
        Returns parameters if user confirms, None if cancelled.
        """
        ...

    def render_l1_to_l2_form(self, dataset: Any) -> Optional[DomainEnrichmentParameters]:
        """
        Render L1→L2 domain enrichment form.

        Shows domain input, categorization method toggle, threshold slider.
        Returns parameters if user submits valid form, None if cancelled.
        """
        ...

    def render_l2_to_l3_form(self, dataset: Any) -> Optional[GraphBuildingParameters]:
        """
        Render L2→L3 graph building form.

        Shows entity column selector, entity type name input, relationship type input.
        Returns parameters if user submits valid form, None if cancelled.
        """
        ...


# =============================================================================
# Form State Dataclasses (for Streamlit session state)
# =============================================================================

@dataclass
class L1ToL2FormState:
    """State for L1→L2 domain enrichment form."""
    domain_input: str = ""
    parsed_domains: List[str] = None
    use_semantic: bool = True
    threshold: float = 0.5

    def __post_init__(self):
        if self.parsed_domains is None:
            self.parsed_domains = []

    def parse_domains(self) -> List[str]:
        """Parse comma-separated domain input into list."""
        if not self.domain_input:
            return []
        return [d.strip() for d in self.domain_input.split(",") if d.strip()]

    def is_valid(self) -> bool:
        """Check if form state is valid for submission."""
        return len(self.parse_domains()) > 0


@dataclass
class L2ToL3FormState:
    """State for L2→L3 graph building form."""
    available_columns: List[str] = None
    selected_column: Optional[str] = None
    entity_type_name: str = ""
    relationship_type: str = ""

    def __post_init__(self):
        if self.available_columns is None:
            self.available_columns = []

    def is_valid(self) -> bool:
        """Check if form state is valid for submission."""
        return (
            self.selected_column is not None
            and self.entity_type_name.strip() != ""
            and self.relationship_type.strip() != ""
        )


# =============================================================================
# Constants
# =============================================================================

# Default similarity threshold for semantic matching
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# Valid range for similarity threshold
MIN_SIMILARITY_THRESHOLD = 0.1
MAX_SIMILARITY_THRESHOLD = 0.9

# Default label for unmatched values
DEFAULT_UNMATCHED_LABEL = "Unmatched"

# Session state keys for ascent forms
SESSION_KEY_L1_TO_L2_FORM = "ascent_l1_to_l2_form_state"
SESSION_KEY_L2_TO_L3_FORM = "ascent_l2_to_l3_form_state"
